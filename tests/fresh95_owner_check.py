"""Four NEW authored/UNRUN GLiDE readout cases; ONE tiny software update total."""
from copy import deepcopy
import math,pytest,torch
from torch import nn
from glide_particle_sliders import MODEL_ID,MODEL_REVISION
from glide_particle_sliders import binding as original,binding_normalized_v3 as normalized
from glide_particle_sliders.fresh95 import open_training,open_inference,save_prepared_cache,file_pin,Checkpoint
from glide_particle_sliders.fresh95 import api,cache,policy,ownership
from glide_particle_sliders.fresh95.util import equal

@pytest.fixture
def staged(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('no original preparation/encoder/teacher collector in NEW software cases')
    monkeypatch.setattr(original,'prepare',forbidden);monkeypatch.setattr(normalized,'prepare',forbidden)
    monkeypatch.setattr(original,'load_readout_file',forbidden)
    g=torch.Generator().manual_seed(34)
    weight=torch.randn(255,1024,generator=g)/32
    identity=original._readout_identity(dict(model_id=MODEL_ID,revision=MODEL_REVISION,scope='synthetic-software-only'),weight)
    def factory(request):
        assert request['operation']in('fresh','resume','inference')and request['new_encoder_or_teacher_calls']==0
        with torch.random.fork_rng(devices=[]):
            model=nn.Linear(1024,255,bias=False)
            with torch.no_grad():model.weight.copy_(weight)
            return model
    hidden=torch.randn(4,1024,generator=g)*.1+3
    neutral=torch.nn.functional.linear(hidden,weight)[:,:2]
    rows=[original.PreparedReadout(original.ReadoutRow(hidden[i:i+1],f'NEW-{i}'),neutral[i:i+1]+.2,neutral[i:i+1])for i in range(4)]
    original_file=tmp_path/'synthetic-completed-native-inputs.pt';torch.save([cache.dump(x)for x in rows],original_file)
    whole=file_pin(original_file);source=file_pin(__file__)
    origins=[dict(scope='synthetic-software-only',row_id=x.row.row_id,complete_prepared_output=True,target_semantics='synthetic-software-only',
        whole_output_files=[dict(path=whole.path,bytes=whole.bytes,sha256=whole.sha256)],
        source_files=[dict(path=source.path,bytes=source.bytes,sha256=source.sha256)],real_encoder_or_teacher_calls=0)for x in rows]
    frozen=save_prepared_cache(tmp_path/'completed-cache.pt',prepared=rows,splits=['train','train','train','dev'],
        teacher_origins=origins,model_identity=identity,cache_id='NEW-GLiDE-normalized-synthetic',owner_admission=lambda request:True)
    config=dict(seed=7,steps=3,batch=2,lr=.001,d_lr_mult=18.,prior_lr_mult=30.,model_identity=identity,
        ema_decay=None,device='cpu',pg_version='0.9.5',pg_source_profile='release-0.9.5',penalty_input_domain='continuous')
    return dict(config=config,cache=frozen,cohort_id='NEW-GLiDE-fresh95-software-v1',component_factory=factory,
        owner_admission=lambda request:True),tmp_path,weight,rows

def opened(staged,name='fresh'):
    options,root,_,_=staged
    return open_training(directory=root/name,mode='fresh',**options)

def test_new_same_Parameters_bank_before_optimizer_receipt_zeroUP_router_critic_KEEP(staged,monkeypatch):
    init=policy.initialize;game=api.ConditionalPG095;seen=[]
    def initialize(slider,critic,*args):
        ids=[id(p)for p in slider.network.parameters()]+[id(p)for p in critic.parameters()]
        old_critic=deepcopy(critic.state_dict());old_buffers={n:t.clone()for n,t in slider.network.named_buffers()}
        receipt=init(slider,critic,*args)
        assert ids==[id(p)for p in slider.network.parameters()]+[id(p)for p in critic.parameters()]
        assert equal(old_critic,critic.state_dict())and equal(old_buffers,dict(slider.network.named_buffers()))
        assert torch.count_nonzero(slider.adapter.lora_up.weight)==0
        assert all(r['before_sha256']==r['after_sha256']for r in receipt['parameters']if r['role'].endswith('KEEP'))
        seen.append(receipt);return receipt
    def construct(backend,*args,**kwargs):
        assert seen and backend.metadata()['fresh_initialization']['receipt']==seen[-1]
        assert kwargs['particles']is backend.particles and kwargs['response_scale']==1.
        return game(backend,*args,**kwargs)
    monkeypatch.setattr(policy,'initialize',initialize);monkeypatch.setattr(api,'ConditionalPG095',construct)
    with opened(staged)as owner:
        assert owner.game.completed_steps==0 and owner.game.backend.particles is owner.slider.particles
        assert tuple(owner.slider.particles.shape)==(128,4)
        assert owner.game.state_dict()['config']['backend']['fresh_initialization']['receipt']==seen[0]
        assert (owner.directory/'FULL-000000.pt').is_file()and(owner.directory/'RAW-000000.pt').is_file()
        assert torch.equal(owner.model.weight,staged[2])

def test_new_host_asqrt5_down_Gram_bank_R2std1_determinism_and_caller_RNG(staged):
    before=torch.get_rng_state().clone()
    with opened(staged,'a')as owner:
        first=deepcopy(owner.network.state_dict());receipt=deepcopy(owner.initialization)
        down=owner.slider.adapter.lora_down.weight.detach();fan=down.shape[1]
        assert torch.allclose(down@down.T,torch.eye(8)/3,rtol=1e-5,atol=1e-6)
        assert math.isclose(float(down.square().mean()),1/(3*fan),rel_tol=1e-5)
        assert receipt['particle_bank']==dict(shape=[128,4],policy='R2Normal(0,1)')
        assert torch.isfinite(owner.slider.particles).all()and torch.count_nonzero(owner.slider.particles)>0
        model=owner.model
    assert 'forward'not in vars(model)and not hasattr(model,ownership.MARKER)
    assert torch.equal(before,torch.get_rng_state())
    with opened(staged,'b')as again:assert equal(first,again.network.state_dict())and receipt==again.initialization
    assert torch.equal(before,torch.get_rng_state())

def test_new_ONE_update_strict_whole_FULL_optimizers_RNG_RAW_no_initializer_or_cache(staged,monkeypatch):
    options,root,weight,rows=staged
    with opened(staged)as trained:
        trained.update()  # Exactly ONE total successful tiny update across these four nodes.
        full=trained.save(root/'FULL1.pt');raw=trained.export(root/'RAW1.pt')
        state=deepcopy(trained.game.state_dict());network=deepcopy(trained.network.state_dict())
        assert torch.equal(weight,trained.model.weight)
    def forbidden(*args,**kwargs):raise AssertionError('restore/RAW cannot initialize or collect original data')
    monkeypatch.setattr(policy,'initialize',forbidden)
    with open_training(directory=root/'resume',mode='resume',resume=full,**options)as resumed:
        assert resumed.inherited_steps==1 and equal(state,resumed.game.state_dict())
    monkeypatch.setattr(api,'load_cache',forbidden);monkeypatch.setattr(api,'ConditionalPG095',forbidden)
    with open_inference(raw,component_factory=options['component_factory'],owner_admission=lambda request:True)as inferred:
        assert not hasattr(inferred,'game')and not hasattr(inferred,'critic')
        assert equal(network,inferred.network.state_dict())
        before=torch.get_rng_state().clone();output=inferred.response(rows[0].row.hidden,strength=1.,seed=17)
        assert tuple(output.shape)==(1,2)and torch.isfinite(output).all()and torch.equal(before,torch.get_rng_state())
        assert torch.equal(weight,inferred.model.weight)

def test_new_early_intent_cache_EMA_refusals_and_foreign_retryable_markerLAST(staged,monkeypatch):
    options,root,_,_=staged;calls=[]
    def forbidden(request):calls.append(request);raise AssertionError('invalid intent reached readout factory')
    bad=dict(options,component_factory=forbidden)
    for mode,resume,ema in(('fresh',Checkpoint('unused',1,'0'*64),None),('resume',None,None),('fresh',None,.9)):
        with pytest.raises(ValueError):open_training(directory=root/'bad',mode=mode,resume=resume,ema_decay=ema,**bad)
    legacy=root/'legacy.pt';torch.save(dict(schema='old-GLiDE-FULL'),legacy)
    with pytest.raises(ValueError):open_training(directory=root/'legacy',mode='resume',resume=file_pin(legacy),**bad)
    from dataclasses import replace
    corrupt=replace(options['cache'],sha256='0'*64)
    with pytest.raises(ValueError):open_training(directory=root/'bad-cache',mode='fresh',**dict(bad,cache=corrupt))
    with pytest.raises(ValueError):open_training(directory=root/'denied',mode='fresh',**dict(bad,owner_admission=lambda request:False))
    assert calls==[]
    owner=opened(staged,'retry');handle=owner.attachment.hook;remove=handle.remove;attempts=[]
    def once():
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('deliberate fail-once normalization cleanup')
        return remove()
    monkeypatch.setattr(handle,'remove',once)
    with pytest.raises(RuntimeError,match='fail-once'):owner.close()
    assert owner.closing and not owner.closed and getattr(owner.model,ownership.MARKER)is owner.attachment
    with pytest.raises(RuntimeError,match='cleanup'):owner.update()
    owner.close();assert owner.closed and len(attempts)==2
    marked=opened(staged,'foreign-marker');a=marked.attachment;callback=marked.model.forward;hooks=dict(a.hooks)
    foreign=object();setattr(marked.model,ownership.MARKER,foreign)
    with pytest.raises(RuntimeError,match='marker'):marked.close()
    assert vars(marked.model)['forward']is callback and dict(a.hooks)==hooks and getattr(marked.model,ownership.MARKER)is foreign
    setattr(marked.model,ownership.MARKER,a);marked.close();assert marked.closed
    hooked=opened(staged,'foreign-hook');a=hooked.attachment;owned=a.hooks[a.hook.id];foreign=lambda *args:None;a.hooks[a.hook.id]=foreign
    with pytest.raises(RuntimeError,match='foreign normalization'):hooked.close()
    assert a.hooks[a.hook.id]is foreign and getattr(hooked.model,ownership.MARKER)is a
    a.hooks[a.hook.id]=owned;hooked.close();assert hooked.closed
    flagged=opened(staged,'foreign-flag');a=flagged.attachment;flagged.model.weight.requires_grad_(True)
    with pytest.raises(RuntimeError,match='requires-grad'):flagged.close()
    assert flagged.model.weight.requires_grad and getattr(flagged.model,ownership.MARKER)is a
    flagged.model.weight.requires_grad_(a.owned_flag);flagged.close();assert flagged.closed
    modes=opened(staged,'foreign-mode');a=modes.attachment;modes.model.training=True
    with pytest.raises(RuntimeError,match='training mode'):modes.close()
    assert modes.model.training and getattr(modes.model,ownership.MARKER)is a
    modes.model.training=a.owned_mode;modes.close();assert modes.closed

    replaced=opened(staged,'foreign-base-Parameter');a=replaced.attachment;original_parameter=replaced.model.weight
    foreign_parameter=nn.Parameter(original_parameter.detach().clone(),requires_grad=False);replaced.model.weight=foreign_parameter
    with pytest.raises(RuntimeError,match='readout Parameter'):replaced.close()
    assert replaced.model.weight is foreign_parameter and getattr(replaced.model,ownership.MARKER)is a
    replaced.model.weight=original_parameter;replaced.close();assert replaced.closed
    forwarded=opened(staged,'foreign-original-callback');a=forwarded.attachment;original_forward=forwarded.slider.adapter.org_forward
    foreign_forward=lambda value:value;forwarded.slider.adapter.org_forward=foreign_forward
    with pytest.raises(RuntimeError,match='original readout callback'):forwarded.close()
    assert forwarded.slider.adapter.org_forward is foreign_forward and getattr(forwarded.model,ownership.MARKER)is a
    forwarded.slider.adapter.org_forward=original_forward;forwarded.close();assert forwarded.closed
