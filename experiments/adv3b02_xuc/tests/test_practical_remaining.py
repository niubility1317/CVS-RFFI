from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from leo_practical import Config, SCENARIOS, apply_leo_practical_channel_batch
from leo_practical.execution import FAST, REFERENCE, tracking

VARIANTS=[('full',False,'zf'),('full',True,'zf'),('full',True,'mmse'),('residual',False,'zf')]

@pytest.mark.parametrize('scene',list(SCENARIOS))
@pytest.mark.parametrize('variant',VARIANTS)
def test_individual_execution_equivalence(scene,variant):
    cfg=Config(fs_hz=25e6,fc_hz=2.462e9,scenario=scene,processing_route=variant[0],
        equalization_enabled=variant[1],equalizer_method=variant[2],zf_regularization=1e-6,equalizer_max_gain_db=20)
    x=np.random.default_rng(74).normal(size=(3,2,256))
    kw=dict(seed=392005,sample_ids=['a','b','c'],session_ids=['s']*3,realization_namespace='source_dynamic_E1_L')
    old,meta,states=apply_leo_practical_channel_batch(x,cfg,**kw)
    for option in [replace(REFERENCE,**{key:True}) for key in REFERENCE.__dataclass_fields__]+[FAST]:
        new,records,end=apply_leo_practical_channel_batch(x,cfg,execution=option,**kw)
        assert np.array_equal(old,new)
        assert np.array_equal(states,end)
        for a,b in zip(meta,records):
            if not option.light_metadata:
                assert a==b
            else:
                for key in b.keys()-{'metadata_level','geometry'}:assert a[key]==b[key]
                for key in b['geometry']:assert a['geometry'][key]==b['geometry'][key]


def test_compiled_tracking_stream_state():
    draws=np.random.default_rng(5).normal(size=8192)
    for rho in (0.,.99999999999,.7):
        a,z=tracking(.73,rho,.003,draws,.127,False)
        b,w=tracking(.73,rho,.003,draws,.127,True)
        assert np.array_equal(a,b) and z==w


def test_prefetch_order_exception_and_bound():
    from cvsrffi.bounded_prefetch import prefetch_map
    assert list(prefetch_map(lambda x:x*x,range(8)))==[i*i for i in range(8)]
    def fail(x):
        if x==2:raise ValueError('producer failure')
        return x
    with pytest.raises(ValueError,match='producer failure'):list(prefetch_map(fail,range(5)))
    from contextlib import closing
    import threading
    with pytest.raises(RuntimeError,match='consumer failure'):
        with closing(prefetch_map(lambda x:x,range(5))) as values:
            for value in values:
                raise RuntimeError('consumer failure')
    assert not any(t.name.startswith('practical-view') for t in threading.enumerate())


def test_actual_clean_geometry_reuse_and_rng():
    import torch
    from scripts.train_rc4_practical import native,build_args
    from cvsrffi.eval import evaluate_loader
    from cvsrffi.source_validation_reuse import SourceValidationReuse
    args=build_args(ROOT/'configs/rc4_practical_full_noeq_20260918.json','unused','unused','unused','unused','unused','synthetic')
    args.rc4_satellite_family='original'
    args.eval_max_batches=2
    device=torch.device('cpu')
    ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},args,input_len=256,num_domains=15)
    model=native.build_baseline_model(native._apply_model_cli_args(ma,args),device).eval()
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(18,2,256),torch.arange(18)%6,torch.arange(18)%3),batch_size=6)
    context={'val_loader':loader,'domain_label_map':{i:i for i in range(3)}}
    state=torch.get_rng_state().clone()
    before={k:v.clone() for k,v in model.state_dict().items()}
    base=evaluate_loader(model,loader,device,context['domain_label_map'],max_batches=2)
    old=native._evaluate_source_val_tail_geometry(model,context,device,args)
    rng=torch.get_rng_state().clone()
    torch.set_rng_state(state)
    cache=SourceValidationReuse(model)
    newbase=evaluate_loader(model,loader,device,context['domain_label_map'],max_batches=2,feature_cache=cache)
    new=native._evaluate_source_val_tail_geometry(model,context,device,args,feature_cache=cache)
    assert cache.hits==2
    assert repr(base)==repr(newbase) and repr(old)==repr(new)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    x=next(iter(loader))[0]
    with torch.no_grad():z=model(x,y_tx=None,return_aux=True)['z_id']
    cache=SourceValidationReuse(model);cache.record(0,x,z)
    assert cache.take(0,x+1) is None
    cache=SourceValidationReuse(model);cache.record(0,x,z)
    with torch.no_grad():next(model.parameters()).add_(.001)
    assert cache.take(0,x) is None
