"""Public PG095 same-Parameter downQR and declared std1 routed bank R2 policy."""
from dataclasses import dataclass
from importlib import metadata
import hashlib,math
import torch
from torch import nn
from particlegan import init as pg_init
from . import PROFILE
from .util import canonical,source_record

INIT_SHA='3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b'
SCHEMA='GLiDE-normalized-readout-explicit-downQR-bankR2std1-initialization-v1'
MARKER='_glide_explicit_fresh95_receipt'

@dataclass(frozen=True)
class Construction:
    mode:str
    cohort:str
    options_sha256:str

class DeclaredOwned(nn.Module):
    def __init__(self,rows):
        super().__init__();self.specs={}
        for index,row in enumerate(rows):
            key='p'+str(index);self.register_parameter(key,row['parameter']);self.specs[key]=row['distribution']

pg_init.register(DeclaredOwned,lambda module:module.specs)

def tensor_sha(value):
    flat=torch.empty(value.numel(),dtype=value.dtype,device='cpu')
    flat.copy_(value.detach().reshape(-1).cpu())
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()

def seed_for(seed,name):
    return int.from_bytes(hashlib.sha256(f'{PROFILE}:{seed}:{name}'.encode()).digest()[:8],'big')%2**63

def declarations(slider,critic):
    from glide_particle_sliders.binding_normalized_v3 import NormalizedReadoutSlider
    from glide_particle_sliders.binding_residual_v2 import ResidualCritic
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    from hndl.operators.linear import Linear
    if type(slider)is not NormalizedReadoutSlider or type(critic)is not ResidualCritic or type(slider.adapter)is not HNDLParticleProjection:
        raise ValueError('exact original normalized readout/ResidualCritic64/HNDL projection required')
    if tuple(slider.particles.shape)!=(128,4)or slider.network.particles is not slider.particles:
        raise ValueError('same original separately owned128x4 bank required')
    rows=[];seen=set();storages=set()
    for name,p in zip(slider.network.names,slider.network.weights):
        rows.append(dict(name='network.'+name,parameter=p,distribution=pg_init.KEEP,fan_in=None,role='original-router-bridge-constructor-KEEP'))
    by_id={id(r['parameter']):r for r in rows}
    for role,leaf in(('down',slider.adapter.lora_down),('up',slider.adapter.lora_up)):
        if(type(leaf)is not Linear or leaf.bias is not None or leaf.equalized is not False
                or any(leaf.children())or list(leaf.buffers())or nn.utils.parametrize.is_parametrized(leaf)):
            raise ValueError('original bias-free HNDL down/up required')
        if id(leaf.weight)not in by_id:raise ValueError('exact existing native down/up Parameter required')
        if role=='up'and torch.count_nonzero(leaf.weight).item():raise ValueError('genuine fresh zero-UP required')
        by_id[id(leaf.weight)].update(distribution=pg_init.Uniform(-1/math.sqrt(leaf.in_features),1/math.sqrt(leaf.in_features))if role=='down'else pg_init.KEEP,
            fan_in=leaf.in_features,role='host-asqrt5-down-QR'if role=='down'else'protected-zero-UP-KEEP')
    by_id[id(slider.particles)].update(distribution=pg_init.R2Normal(0.,1.),role='declared-bank-R2std1')
    rows.extend(dict(name='critic.'+name,parameter=p,distribution=pg_init.KEEP,fan_in=None,
        role='original-residual64-HNDL-critic-constructor-KEEP')for name,p in critic.named_parameters())
    for row in rows:
        p=row['parameter'];storage=(str(p.device),p.untyped_storage().data_ptr())
        if(type(p)is not nn.Parameter or not p.requires_grad or p.grad is not None or p.dtype!=torch.float32
                or p.device.type!='cpu'or not p.is_contiguous()or not torch.isfinite(p).all()
                or id(p)in seen or p.numel()and storage in storages or hasattr(p,MARKER)):
            raise ValueError('unique finite fresh original CPU FP32 owned Parameters required')
        seen.add(id(p));storages.add(storage)
    actual={id(p)for p in slider.network.parameters()};d={id(p)for p in critic.parameters()}
    if seen!=actual|d or actual&d:raise ValueError('exact adapter/bank/router/critic coverage required')
    return rows

def initialize(slider,critic,construction,options,sources):
    if (type(construction)is not Construction or construction.mode!='fresh'or not construction.cohort
            or construction.options_sha256!=hashlib.sha256(canonical(options).encode()).hexdigest()):
        raise ValueError('explicit new fresh intent before optimizers required')
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise ValueError('exact public PG095 initializer Source required')
    rows=declarations(slider,critic);before=[tensor_sha(x['parameter'])for x in rows]
    ids=[id(x['parameter'])for x in rows];holder=DeclaredOwned(rows)
    seeds={'p'+str(i):seed_for(options['config']['seed'],row['name'])for i,row in enumerate(rows)}
    pg_init.deterministic_orthogonal_(holder,strict=True,parameter_seeds=seeds)
    if ids!=[id(x['parameter'])for x in rows]:raise RuntimeError('original Parameter replaced')
    parameters=[]
    for index,row in enumerate(rows):
        after=tensor_sha(row['parameter'])
        if row['distribution']is pg_init.KEEP and before[index]!=after:raise RuntimeError('zero-UP/router/critic KEEP changed')
        parameters.append(dict(name=row['name'],role=row['role'],seed=seeds['p'+str(index)],
            shape=list(row['parameter'].shape),dtype=str(row['parameter'].dtype),fan_in=row['fan_in'],
            distribution=repr(row['distribution']),before_sha256=before[index],after_sha256=after))
    receipt=dict(schema=SCHEMA,profile=PROFILE,cohort=construction.cohort,
        options_sha256=construction.options_sha256,completed_steps=0,restored_from=None,
        source_identity=sources,seed=options['config']['seed'],initializer=source_record(pg_init.__file__),
        parameters=parameters,parameter_identity_preserved=True,particle_bank=dict(shape=[128,4],policy='R2Normal(0,1)'),router='original-constructor-KEEP',
        critic_policy='original-residual64-HNDL-constructor-KEEP',host_down_kaiming_a=5**.5,
        host_down_target_mean_square='1/(3*fan_in)',selection='raw',ema_decay=None)
    marker=hashlib.sha256(canonical(receipt).encode()).hexdigest()
    for row in rows:setattr(row['parameter'],MARKER,marker)
    return receipt

def validate_receipt(receipt,construction,options,sources):
    if (not isinstance(receipt,dict)or receipt.get('schema')!=SCHEMA or receipt.get('profile')!=PROFILE
            or receipt.get('cohort')!=construction.cohort or receipt.get('options_sha256')!=construction.options_sha256
            or receipt.get('source_identity')!=sources or receipt.get('seed')!=options['config']['seed']
            or receipt.get('completed_steps')!=0 or receipt.get('restored_from')is not None
            or receipt.get('particle_bank')!=dict(shape=[128,4],policy='R2Normal(0,1)')or receipt.get('router')!='original-constructor-KEEP'
            or receipt.get('ema_decay')is not None or receipt.get('selection')!='raw'
            or receipt.get('initializer',{}).get('sha256')!=INIT_SHA):
        raise ValueError('exact new-profile original initialization receipt required')
