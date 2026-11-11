"""Explicit cached normalized GLiDE fresh/FULL/RAW entry; no encoder loader."""
from copy import deepcopy
from pathlib import Path
import hashlib,re,math,torch
from glide_particle_sliders import binding as original
from glide_particle_sliders.binding_normalized_v3 import NormalizedNativeResponse
from glide_particle_sliders.binding_residual_v2 import ResidualCritic
from particle_sliders.pg095 import ConditionalPG095
from . import PROFILE,ownership,policy
from .cache import load_cache
from .util import canonical,equal,new_directory,read_packet,save_packet,write_json
FULL_SCHEMA='GLiDE-normalized-cached-explicit-fresh95-v1-FULL'
RAW_SCHEMA='GLiDE-normalized-cached-explicit-fresh95-v1-RAW'

def identity_valid(value):
    if(not isinstance(value,dict)or value.get('model_id')!=original.MODEL_ID
            or re.fullmatch('[0-9a-f]{40}',str(value.get('revision','')))is None
            or re.fullmatch('[0-9a-f]{64}',str(value.get('readout_weight_sha256','')))is None):
        raise ValueError('explicit frozen native model/revision/readout byte identity required')
    canonical(value)

def configuration(value):
    keys={'seed','steps','batch','lr','d_lr_mult','prior_lr_mult','model_identity','ema_decay','device','pg_version','pg_source_profile','penalty_input_domain'}
    if(not isinstance(value,dict)or set(value)!=keys or value['ema_decay']is not None or value['device']!='cpu'
            or(value['pg_version'],value['pg_source_profile'],value['penalty_input_domain'])!=('0.9.5','release-0.9.5','continuous')):
        raise ValueError('exact original CPU FP32/PG095 continuous/noEMA configuration required')
    for key,low,high in(('seed',0,2**63-101),('steps',1,10000),('batch',2,512)):
        if type(value[key])is not int or not low<=value[key]<=high:raise ValueError('invalid '+key)
    for key in('lr','d_lr_mult','prior_lr_mult'):
        if type(value[key])not in(int,float)or not math.isfinite(value[key])or value[key]<=0:raise ValueError('positive explicit original rate required:'+key)
    identity_valid(value['model_identity']);canonical(value);return deepcopy(value)

def intent(mode,resume,cohort_id,ema_decay):
    if mode not in('fresh','resume')or not isinstance(cohort_id,str)or not cohort_id.strip():raise ValueError('explicit fresh|resume/new cohort required')
    if(mode=='fresh'and resume is not None)or(mode=='resume'and resume is None):raise ValueError('fresh refuses state; strict resume requires new FULL')
    if ema_decay is not None:raise ValueError('weight EMA refused')

class ReceiptBackend:
    def __init__(self,native,receipt,sources,cache):
        self.backend=native;self.network=native.network;self.particles=native.particles
        self.receipt=deepcopy(receipt);self.sources=deepcopy(sources);self.cache=deepcopy(cache)
    def __getattr__(self,key):return getattr(self.backend,key)
    def metadata(self):return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,receipt=self.receipt,sources=self.sources),
        completed_original_cache=self.cache,new_encoder_or_teacher_calls=0,selection='raw',ema_decay=None)

class NativeOwner(ownership.Owner):
    def _components(self,request,component_factory,center,rms):
        self.model=component_factory(deepcopy(request))
        if original._readout_identity(self.config['model_identity'],self.model.weight)!=self.config['model_identity']:
            raise ValueError('actual frozen readout bytes contradict declared identity')
        self.attachment=ownership.Attachment(self.model)
        self.slider=self.attachment.attach(self.config['model_identity'],center,rms,self.config['seed'])
        self.native=NormalizedNativeResponse(self.slider);self.network=self.native.network
    def response(self,hidden,*,strength=1.,seed=7):
        from particle_sliders.lifecycle import _network_observation
        self.require_open();original.capabilities().check('response',strength=strength,precision='float32')
        if type(seed)is not int or not 0<=seed<2**63-100:raise ValueError('explicit native observation seed required')
        with _network_observation(self.network,seed,devices=[]),torch.no_grad():
            return self.slider.response(hidden,strength=strength).detach().clone()

class Training(NativeOwner):
    def __init__(self,config,cache,directory,*,mode,cohort_id,component_factory,owner_admission,resume=None,ema_decay=None):
        super().__init__();intent(mode,resume,cohort_id,ema_decay)
        if not callable(component_factory)or not callable(owner_admission):raise ValueError('readout owner factory/admission required')
        self.config=configuration(config);ownership.guard();self.sources=ownership.sources()
        self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new immutable owner directory required')
        self.cohort_id=cohort_id;self.options=dict(config=self.config,original_profile='normalized-v3/residual64',rank=8,alpha=8,bank=[128,4])
        self.construction=policy.Construction(mode,cohort_id,hashlib.sha256(canonical(self.options).encode()).hexdigest())
        packet=read_packet(resume)if mode=='resume'else None
        if packet is not None:
            keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','game','ema_decay'}
            if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=FULL_SCHEMA or packet['profile']!=PROFILE
                    or packet['cohort_id']!=cohort_id or packet['options']!=self.options or packet['sources']!=self.sources or packet['ema_decay']is not None):
                raise ValueError('exact typed new-profile FULL required; no old migration')
            policy.validate_receipt(packet['initialization'],self.construction,self.options,self.sources)
        rows,splits,center,rms,self.cache_identity=load_cache(cache,self.config['model_identity'])
        if packet is not None and packet['cache_identity']!=self.cache_identity:raise ValueError('whole completed cache differs')
        request=dict(operation=mode,profile=PROFILE,cohort_id=cohort_id,options=deepcopy(self.options),completed_cache=deepcopy(self.cache_identity),
            new_encoder_or_teacher_calls=0,fresh_PG_initialization_calls=int(mode=='fresh'),ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not authenticate/admit operation')
        try:
            with torch.random.fork_rng(devices=[]):
                self._components(request,component_factory,center,rms)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(self.config['seed']+1);self.critic=ResidualCritic()
                self.initialization=(policy.initialize(self.slider,self.critic,self.construction,self.options,self.sources)
                    if mode=='fresh'else deepcopy(packet['initialization']))
                self.backend=ReceiptBackend(self.native,self.initialization,self.sources,self.cache_identity)
                train=[r for r,s in zip(rows,splits)if s=='train']
                self.game=ConditionalPG095(self.backend,train,self.critic,conditions=torch.cat([r.row.hidden for r in train]),
                    particles=self.slider.particles,response_domain='continuous_vector',response_scale=1.,batch_size=self.config['batch'],
                    total_steps=self.config['steps'],seed=self.config['seed'],
                    recipe_overrides={k:self.config[k]for k in('lr','d_lr_mult','prior_lr_mult')},
                    pg_source_profile='release-0.9.5',penalty_input_domain='continuous')
                if self.game.backend.particles is not self.slider.particles:raise RuntimeError('actual bank identity differs')
                if packet is not None:
                    self.game.load_state_dict(packet['game'])
                    if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('whole game/optimizers/RNG restore differs')
            self.inherited_steps=self.game.completed_steps
            new_directory(self.directory);write_json(self.directory/'initialization.json',self.initialization)
            self.entry_FULL=self.save(self.directory/f'FULL-{self.inherited_steps:06d}.pt')
            self.entry_RAW=self.export(self.directory/f'RAW-{self.inherited_steps:06d}.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('GLiDE entry and cleanup failed',[primary,cleanup])
            raise
    def update(self):self.require_open();return self.game.step()
    def save(self,path):
        self.require_open();return save_packet(path,dict(schema=FULL_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,options=deepcopy(self.options),
            sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),cache_identity=deepcopy(self.cache_identity),game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open();state=self.game.export_state_dict()
        return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,options=deepcopy(self.options),
            sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),cache_identity=deepcopy(self.cache_identity),network=state['network'],
            completed_steps=self.game.completed_steps,selection='raw',ema_decay=None))

class Inference(NativeOwner):
    def __init__(self,raw,*,component_factory,owner_admission):
        super().__init__();value=read_packet(raw)
        keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','network','completed_steps','selection','ema_decay'}
        if(not isinstance(value,dict)or set(value)!=keys or value['schema']!=RAW_SCHEMA or value['profile']!=PROFILE
                or value['ema_decay']is not None or value['selection']!='raw'or type(value['completed_steps'])is not int
                or value['completed_steps']<0 or not callable(component_factory)or not callable(owner_admission)):
            raise ValueError('exact typed new RAW and original owner required')
        self.config=configuration(value['options']['config']);ownership.guard();self.sources=ownership.sources()
        expected=dict(config=self.config,original_profile='normalized-v3/residual64',rank=8,alpha=8,bank=[128,4])
        if(value['options']!=expected or value['sources']!=self.sources or value['cache_identity']['model_identity']!=self.config['model_identity']
                or value['completed_steps']>self.config['steps']):raise ValueError('native Source/config/update identity differs')
        con=policy.Construction('inference',value['cohort_id'],hashlib.sha256(canonical(expected).encode()).hexdigest())
        policy.validate_receipt(value['initialization'],con,expected,self.sources);self.initialization=deepcopy(value['initialization'])
        request=dict(operation='inference',profile=PROFILE,cohort_id=value['cohort_id'],options=expected,
            completed_cache_metadata_only=deepcopy(value['cache_identity']),new_encoder_or_teacher_calls=0,fresh_PG_initialization_calls=0,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not admit RAW response')
        try:
            with torch.random.fork_rng(devices=[]):
                self._components(request,component_factory,torch.zeros(original.HIDDEN),torch.ones(()))
                self.network.load_state_dict(value['network'],strict=True)
                if not equal(self.network.state_dict(),value['network']):raise RuntimeError('strict RAW added tensors/buffers differ')
            self.completed_steps=value['completed_steps']
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('GLiDE RAW entry and cleanup failed',[primary,cleanup])
            raise
