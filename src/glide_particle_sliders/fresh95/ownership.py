"""Genuine installed guards and exclusive reversible normalized-readout ownership."""
from pathlib import Path
from importlib import metadata
import base64,hashlib
from .util import source_record
MARKER='_glide_explicit_fresh95_owner'

def guard():
    from glide_particle_sliders import binding as original
    from particle_sliders import pg09
    from particlegan import init as pg_init
    from .baseline import ORIGINAL_SOURCES,CORE_SOURCES
    from .policy import INIT_SHA
    root=Path(original.__file__).absolute().parent;core=Path(pg09.__file__).absolute().parent
    for name,record in ORIGINAL_SOURCES.items():
        if source_record(root/name)!=record:raise RuntimeError('original GLiDE Source differs:'+name)
    for name,record in CORE_SOURCES.items():
        if source_record(core/name)!=record:raise RuntimeError('exact original core013 Source differs:'+name)
    original._runtime()  # Original noneditable core Git pin and real HNDL/HyperGAN guard.
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise RuntimeError('exact public PG095 initializer required')
    owners=[d for d in metadata.distributions()if(d.metadata.get('Name')or'').lower().replace('_','-')=='glide-particle-sliders']
    if len(owners)!=1 or owners[0].version!='0.1.0':raise RuntimeError('one genuine owning GLiDE distribution required')
    owner=owners[0];records={str(r):r for r in owner.files or ()}
    for p in [*(root/n for n in ORIGINAL_SOURCES),*sorted(Path(__file__).absolute().parent.glob('*.py'))]:
        relative='glide_particle_sliders/'+str(p.relative_to(root));r=records.get(relative)
        if r is None or Path(owner.locate_file(relative)).absolute()!=p.absolute()or r.hash is None or r.hash.mode!='sha256':
            raise RuntimeError('genuine original/new namespace RECORD ownership required:'+relative)
        data=p.read_bytes();digest=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        if r.size!=len(data)or r.hash.value!=digest:raise RuntimeError('GLiDE installed Source/RECORD differs:'+relative)
    return dict(original_core='0.13.0@3dd672a7',PG='0.9.5',original_guards=True,genuine_owning_RECORD=True)

def sources():
    from glide_particle_sliders import binding as original
    from particle_sliders import pg09
    from .baseline import ORIGINAL_SOURCES,CORE_SOURCES
    root=Path(original.__file__).absolute().parent;core=Path(pg09.__file__).absolute().parent
    return dict(own={p.name:source_record(p)for p in sorted(Path(__file__).absolute().parent.glob('*.py'))},
        original={n:source_record(root/n)for n in ORIGINAL_SOURCES},core={n:source_record(core/n)for n in CORE_SOURCES})

class Attachment:
    def __init__(self,model):
        from torch import nn
        if(type(model)is not nn.Linear or model.in_features!=1024 or model.out_features!=255 or model.bias is not None
                or any(model._forward_hooks.values())or any(model._forward_pre_hooks.values())
                or 'forward'in vars(model)or hasattr(model,MARKER)):
            raise ValueError('unowned original bias-free1024-to255 native readout required')
        self.model=model;self.parameter=model.weight;self.slider=None;self.original_forward=None;self.closed=self.closing=False
        self.forward_before=('forward'in vars(model),vars(model).get('forward'));self.owned_forward=None
        self.flag=model.weight.requires_grad;self.mode=model.training
        self.owned_flag=self.flag;self.owned_mode=self.mode
        self.pending_forward=False;self.pending_flag=self.pending_mode=self.pending_marker=True
        self.pending_hook=False;self.hook=None;self.hooks=None;self.callback=None;self.extras=[]
        setattr(model,MARKER,self)
    def attach(self,identity,center,rms,seed):
        from glide_particle_sliders.binding_normalized_v3 import NormalizedReadoutSlider
        self.slider=NormalizedReadoutSlider.__new__(NormalizedReadoutSlider)
        try:NormalizedReadoutSlider.__init__(self.slider,self.model,model_identity=identity,center=center,rms=rms,seed=seed)
        finally:
            adapter=getattr(self.slider,'adapter',None)
            if adapter is not None:self.original_forward=adapter.org_forward
            current=vars(self.model).get('forward')
            self.owned_flag=self.model.weight.requires_grad;self.owned_mode=self.model.training
            original_restored=getattr(self.slider,'_forward',None)
            if adapter is not None and (getattr(current,'__self__',None)is adapter or current is original_restored):
                self.owned_forward=current;self.pending_forward=True
            handle=getattr(self.slider,'_center_handle',None)
            if handle is not None:
                hooks=handle.hooks_dict_ref()
                if hooks is None or handle.id not in hooks:raise RuntimeError('owned normalization callback missing')
                self.hook=handle;self.hooks=hooks;self.callback=hooks[handle.id];self.pending_hook=True
                self.extras=[(extra,handle.id in extra,extra.get(handle.id))for ref in getattr(handle,'extra_dict_ref',())if(extra:=ref())is not None]
        self.ids=[(n,id(p))for n,p in self.slider.network.named_parameters()]
        return self.slider
    def require_active(self):
        if self.closed or self.closing:raise RuntimeError('GLiDE owner closed/cleanup pending')
        if vars(self.model).get(MARKER)is not self:raise RuntimeError('foreign GLiDE owner marker')
        if self.model.weight is not self.parameter:raise RuntimeError('foreign native readout Parameter retained')
        if self.slider.adapter.org_forward is not self.original_forward:raise RuntimeError('foreign original readout callback retained')
        if not self.pending_forward or vars(self.model).get('forward')is not self.owned_forward:
            raise RuntimeError('foreign native readout callback')
        if self.slider.closed or [(n,id(p))for n,p in self.slider.network.named_parameters()]!=self.ids:
            raise RuntimeError('original GLiDE slider/Parameter identity changed')
        if self.model.weight.requires_grad or self.model.training:raise RuntimeError('frozen native readout state changed')
        self._require_hook()
    def _require_hook(self):
        if(self.hook is None or getattr(self.slider,'_center_handle',None)is not self.hook
                or self.hook.hooks_dict_ref()is not self.hooks or self.hooks.get(self.hook.id)is not self.callback
                or any((self.hook.id in extra)!=had or extra.get(self.hook.id)is not value for extra,had,value in self.extras)):
            raise RuntimeError('foreign normalization handle/callback/extra-dictionary retained')
    def _restore_flag(self):
        if self.model.weight.requires_grad is not self.owned_flag:raise RuntimeError('foreign native requires-grad retained')
        self.model.weight.requires_grad_(self.flag)
    def _restore_mode(self):
        if self.model.training is not self.owned_mode:raise RuntimeError('foreign native training mode retained')
        self.model.training=self.mode
    def close(self):
        if self.closed:return
        self.closing=True
        if not self.pending_marker or vars(self.model).get(MARKER)is not self:
            raise RuntimeError('foreign GLiDE owner marker retained before cleanup')
        if self.model.weight is not self.parameter:raise RuntimeError('foreign native readout Parameter retained before cleanup')
        adapter=getattr(self.slider,'adapter',None)
        if adapter is not None and adapter.org_forward is not self.original_forward:
            raise RuntimeError('foreign original readout callback retained before cleanup')
        if self.slider is not None:self.slider.closed=True
        errors=[]
        if self.pending_hook:
            try:self._require_hook();self.hook.remove();self.pending_hook=False
            except BaseException as error:errors.append(error)
        if not self.pending_hook and self.pending_forward:
            try:
                if vars(self.model).get('forward')is not self.owned_forward:raise RuntimeError('foreign native readout callback retained')
                had,old=self.forward_before
                if had:self.model.forward=old
                else:delattr(self.model,'forward')
                self.pending_forward=False
            except BaseException as error:errors.append(error)
        if not self.pending_hook and not self.pending_forward:
            for flag,operation in(('pending_flag',self._restore_flag),('pending_mode',self._restore_mode)):
                if getattr(self,flag):
                    try:operation();setattr(self,flag,False)
                    except BaseException as error:errors.append(error)
        if not any((self.pending_hook,self.pending_forward,self.pending_flag,self.pending_mode)):
            if vars(self.model).get(MARKER)is not self:errors.append(RuntimeError('foreign GLiDE owner marker retained'))
            else:delattr(self.model,MARKER);self.pending_marker=False
        if errors:
            if len(errors)==1:raise errors[0]
            raise BaseExceptionGroup('GLiDE cleanup pending',errors)
        if self.pending_marker:raise RuntimeError('GLiDE cleanup remains pending')
        self.closed,self.closing=True,False

class Owner:
    def __init__(self):self.closed=self.closing=False;self.attachment=None
    def require_open(self):
        if self.closed or self.closing:raise RuntimeError('GLiDE owner closed/cleanup pending')
        if self.attachment is not None:self.attachment.require_active()
    def close(self):
        if self.closed:return
        self.closing=True
        if self.attachment is not None:self.attachment.close()
        self.closed,self.closing=True,False
    def __enter__(self):self.require_open();return self
    def __exit__(self,kind,primary,traceback):
        try:self.close()
        except BaseException as cleanup:
            if primary is not None:raise BaseExceptionGroup('GLiDE work and cleanup failed',[primary,cleanup])
            raise
