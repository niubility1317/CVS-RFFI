"""Focused signal, objective and source-permission checks; synthetic IQ only."""
import copy
import math
import torch


def signal_checks():
    from .augmentation import delayed, draw_delays
    torch.set_num_threads(2)
    x=torch.randn(8,2,256,dtype=torch.float64)
    assert torch.equal(delayed(x,torch.zeros(8,dtype=x.dtype))[0],x)
    constant=torch.ones_like(x)
    assert torch.allclose(delayed(constant,torch.full((8,),.125,dtype=x.dtype))[0],constant,atol=1e-12)
    y,detail=delayed(x,torch.linspace(-.125,.125,8,dtype=x.dtype))
    assert y.shape==x.shape and torch.isfinite(y).all()
    assert torch.allclose(x.square().mean((1,2)),y.square().mean((1,2)),atol=1e-12)
    impulse=torch.zeros(1,2,256,dtype=x.dtype);impulse[0,0,0]=1
    assert float(delayed(impulse,torch.tensor([.125],dtype=x.dtype))[0][:,:,32:].abs().max())==0
    # Boundary choice cannot change the unnormalized fixed reference [80,160).
    reflect=delayed(x,torch.full((8,),.125,dtype=x.dtype),boundary='reflect',restore_rms=False)[0]
    edge=delayed(x,torch.full((8,),.125,dtype=x.dtype),boundary='replicate',restore_rms=False)[0]
    assert torch.equal(reflect[:,:,16:240],edge[:,:,16:240])
    n=torch.arange(256,dtype=x.dtype);omega=.2*math.pi
    tone=torch.stack((torch.cos(omega*n),torch.sin(omega*n)))[None]
    moved=delayed(tone,torch.tensor([.125],dtype=x.dtype),restore_rms=False)[0]
    expected=torch.stack((torch.cos(omega*(n-.125)),torch.sin(omega*(n-.125))))[None]
    assert float((moved[:,:,32:224]-expected[:,:,32:224]).abs().max())<2e-4
    state=torch.random.get_rng_state().clone()
    a=draw_delays(8,step=1);b=draw_delays(8,step=1)
    assert torch.equal(a,b) and not torch.equal(a,draw_delays(8,step=2))
    assert torch.equal(state,torch.random.get_rng_state()) and a.abs().max()<=.125
    negatives=0
    for xx,dd in ((x[:,:,:255],a),(x,a[:2]),(x,torch.full((8,),.126)),(x,torch.full((8,),float('nan'))),(x.long(),a)):
        try:delayed(xx,dd)
        except (ValueError,TypeError):negatives+=1
        else:raise AssertionError('Invalid IQ/delay accepted')
    return dict(status='PASS',negative_cases=negatives,reference_boundary_exact=True,same_RMS=True,private_RNG=True,detail=detail)


def objective_checks():
    from .objective import objective,rd
    y=torch.arange(8)%6
    logits=[torch.randn(8,6,requires_grad=True) for _ in range(6)]
    loss,parts=objective(logits,y)
    expected=.5*rd(*logits[:2],y)[0]+.25*rd(*logits[2:4],y)[0]+.25*rd(*logits[4:],y)[0]
    assert torch.allclose(loss,expected)
    loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in logits)
    try:objective(logits,torch.full((8,),-1,dtype=torch.long))
    except ValueError:pass
    else:raise AssertionError('U truth placeholder accepted as supervised label')
    return dict(status='PASS',weights=[.5,.25,.25],pairs=3,alpha=1.,label_smoothing=.01,finite_gradients=True)


def run():
    return dict(signal=signal_checks(),objective=objective_checks())


def real_network(device='cpu'):
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
    from SSDG import train_ssdg as actual_native
    from . import design as d,model as api
    from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
    from experiments.cvs_rdrop_sinc_units.model import export
    from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
    torch.set_num_threads(2)
    configs=[d.config(row) for row in d.rows()]
    models=[api.build(c,device) for c in configs]
    a,b=(m.state_dict() for m in models)
    assert set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items())
    negatives=0
    for key,value in (('epochs',50),('test_views',['clean']),('model_initialization','pretrained'),('separate_head_training',True)):
        bad=copy.deepcopy(configs[0]);bad[key]=value
        try:d.validate(bad)
        except ValueError:negatives+=1
        else:raise AssertionError('Illegal budget/data/initialization accepted')
    reports=[]
    for m,c in zip(models,configs):
        m.train();x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6
        observed=[]
        handle=m.id_backbone.encoder.register_forward_pre_hook(lambda mod,args:observed.append(args[0].detach().clone()))
        # IdentityAdapter.features is called directly, so observe its actual input.
        original_features=m.id_backbone.encoder._timing_features
        def traced(inp):
            observed.append(inp.detach().clone());return original_features(inp)
        m.id_backbone.encoder._timing_features=traced
        W='id_backbone.encoder.core.id_backbone.cls_head.weight'
        before={k:p.detach().clone() for k,p in m.named_parameters()}
        opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001)
        with m._phase1_labeled_feature_context():out=m(x,return_aux=True)
        ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra={},domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
        loss=ce+correction;loss.backward();clip_physical(opt,m.parameters(),5.);opt.step()
        handle.remove();m.id_backbone.encoder._timing_features=original_features
        assert len(observed)==6 and torch.equal(observed[0],observed[1]) and torch.equal(observed[2],observed[3]) and torch.equal(observed[4],observed[5])
        assert torch.equal(observed[0],observed[4])
        assert (not torch.equal(observed[0],observed[2]))==c['training_mechanism']['Ag']
        changed={k:not torch.equal(v,dict(m.named_parameters())[k]) for k,v in before.items()}
        assert changed[W] and any(v for k,v in changed.items() if k!=W)
        assert m._timing_calls==m._feature_l_steps==1 and m._timing_L_packets==768
        m.eval();probe=x[:2]
        with torch.no_grad(),fresh_parameter_caches(m):
            f=m.id_backbone.encoder.features(probe).clone();z=m.forward_identity_only(probe)['tx_logits']
            standalone=export(m).eval();ef=standalone(probe)
            assert f.shape==(2,160) and torch.equal(f,ef)
            m.get_parameter(W).zero_()
            with fresh_parameter_caches(m):assert torch.equal(f,m.id_backbone.encoder.features(probe))
        reports.append(dict(arm=c['arm'],parameters=sum(p.numel() for p in m.parameters()),default_W_updated=changed[W],same_IQ_pairs=True,L_forwards=6,joint160_export_exact=True,W_removal_leaves_feature=True,loss=float(loss.detach())))
    return dict(status='PASS',protocol_negative_cases=negatives,rows=reports,target_access=False,head_fit_updates=0,synthetic_training_L_forward_packets=1536,synthetic_feature_check_packets=16)


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
