"""Exact native control, new geometry, source permissions and actual feature gradients."""
import copy,json,sys
import torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as f

def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError,KeyError):return
    raise AssertionError('Illegal action accepted')

def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    device='cuda:0' if '--cuda' in sys.argv else 'cpu'
    if device!='cpu':
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    z1=torch.randn(32,160,dtype=torch.float64,device=device,requires_grad=True)
    z2=torch.randn_like(z1,requires_grad=True)
    W=torch.randn(6,160,dtype=torch.float64,device=device,requires_grad=True)
    a=30*F.normalize(z1,dim=1,eps=1e-4)@F.normalize(W,dim=1,eps=1e-4).T
    b=30*F.normalize(z2,dim=1,eps=1e-4)@F.normalize(W,dim=1,eps=1e-4).T
    geometry=f.geometry(a,b,W)
    B,T,rank,error=geometry
    assert rank==5 and error<1e-12 and not B.requires_grad and not T.requires_grad
    A=F.normalize(W,dim=1,eps=1e-4).detach();C=A-A.mean(0)
    P=B@B.T;I=torch.eye(160,dtype=W.dtype,device=device)
    assert torch.allclose(P@P,P,atol=1e-13,rtol=1e-12)
    assert (C@(I-P)).abs().max()<1e-12
    S=torch.diag_embed(.5*(a.softmax(1)+b.softmax(1)).detach())
    p=.5*(a.softmax(1)+b.softmax(1)).detach()
    S=S-p[:,:,None]*p[:,None,:]
    dense=900*torch.einsum('kd,bkl,ld->b',A,S,A)
    assert torch.allclose(T,dense,atol=1e-12,rtol=1e-12)
    traces=[torch.trace(P)/rank,torch.trace(I-P)/(160-rank),torch.trace(I)/160]
    assert all(abs(float(t)-1)<1e-12 for t in traces)
    perm=torch.tensor([2,5,0,3,1,4],device=device)
    for arm in d.ARMS:
        loss,m=f.consistency(a,b,z1,z2,W,arm,f.new_audit())
        swapped,_=f.consistency(b,a,z2,z1,W,arm,f.new_audit())
        permuted,_=f.consistency(a[:,perm],b[:,perm],z1,z2,W[perm],arm,f.new_audit())
        assert torch.allclose(loss,swapped,atol=1e-13,rtol=1e-12) and torch.allclose(loss,permuted,atol=1e-12,rtol=1e-12)
        same,_=f.consistency(a,a,z1,z1,W,arm,f.new_audit())
        assert abs(float(same.detach()))<1e-15
        assert torch.autograd.grad(same,z1,retain_graph=True)[0].abs().max()<1e-15
        g=torch.autograd.grad(loss,(z1,z2,W),retain_graph=True)
        assert all(torch.isfinite(x).all() and x.norm()>0 for x in g)
        if arm!='FULL':
            aux=f.penalty(z1,z2,*geometry[:3],arm)[0]
            gg=torch.autograd.grad(aux,(z1,z2,W),retain_graph=True,allow_unused=True)
            assert gg[2] is None and all(torch.isfinite(x).all() and x.norm()>0 for x in gg[:2])
    # Exact zero response of each projector to a direction in the other subspace.
    delta=torch.randn(32,160,dtype=W.dtype,device=device)
    span=delta@P;null=delta@(I-P)
    assert (null@P).abs().max()<1e-12 and (span@(I-P)).abs().max()<1e-12
    negative=0
    for bad in [dict(arm='INVALID'),dict(W=W[:5]),dict(a=a[:,:5]),dict(z1=z1[:,:159]),dict(z2=z2*float('nan'))]:
        kw=dict(a=a,b=b,z1=z1,z2=z2,W=W,arm='NULL',audit=f.new_audit());kw.update(bad)
        reject(lambda:f.consistency(**kw));negative+=1
    from .provenance import validate_args
    for row in d.rows():
        c=d.config(row);args=vars(d.make_args(c,'cpu'));validate_args(args,c)
        for key,value in [('source_fold_held_rx',None),('source_fold_training_receivers','1,3,4,6,8'),('baseline_ckpt','unknown.pt'),('teacher_ckpt','unknown.pt'),('epochs',50),('use_aug',False),('label_smoothing',0.)]:
            bad=dict(args);bad[key]=value;reject(lambda:validate_args(bad,c));negative+=1
        for key,value in [('test_views',['clean']),('head_fit_updates',1),('model_seed',2026092703),('fold',{})]:
            bad=dict(c);bad[key]=value;reject(lambda:d.validate(bad));negative+=1
    models=[api.build(d.config(row),device) for row in d.rows()]
    states=[m.state_dict() for m in models]
    assert all(set(s)==set(states[0]) and all(torch.equal(v,s[k]) for k,v in states[0].items()) for s in states[1:])
    x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;domains=torch.arange(128,device=device)%12
    names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
    rng=torch.get_rng_state if device=='cpu' else torch.cuda.get_rng_state
    logs,CE,ends,grads,norms,audits=[],[],[],{}, {},{}
    for row,m in zip(d.rows(),models):
        m.train();m._completion_capture_aux=True;torch.manual_seed(2026101605)
        with m._phase1_labeled_feature_context():out=m(x,y_tx=y,domain_labels=domains,return_aux=True)
        assert torch.equal(m._legacy_actual_L_input,x)
        ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=None,domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128)
        assert m._legacy_actual_L_input is None and m._completion_calls==m._legacy_calls==m._feature_l_steps==1
        named=dict(m.named_parameters());params=[named[n] for n in names]+[api.classifier(m).weight]
        g=torch.autograd.grad(ce+correction,params,retain_graph=True)
        assert all(torch.isfinite(v).all() and v.norm()>0 for v in g)
        if row['arm']!='FULL':
            ga=torch.autograd.grad(m._completion_last_aux,params,allow_unused=True)
            assert ga[-1] is None and all(torch.isfinite(v).all() and v.norm()>0 for v in ga[:-1])
        m._completion_last_aux=None
        logs.append(out['tx_logits'].detach());CE.append(metrics['rdrop_completion/paired_CE']);ends.append(rng().clone());grads[row['arm']]=g
        norms[row['arm']]=[float(v.norm()) for v in g];audits[row['arm']]=m._completion_audit.copy()
        teacher=copy.deepcopy(m).eval()
        reject(lambda:teacher._source_labeled_feature_hook(model=teacher,labeled_output=out,labels=y,extra=None,domains=domains,optimizer_step=2,identity_loss=ce,labeled_count=128));negative+=1
    assert all(torch.equal(v,logs[0]) for v in logs[1:]) and all(torch.equal(v,CE[0]) for v in CE[1:]) and all(torch.equal(v,ends[0]) for v in ends[1:])
    native=api.original.build(d.native_config(),device).train();torch.manual_seed(2026101605)
    with native._phase1_labeled_feature_context():out=native(x,y_tx=y,domain_labels=domains,return_aux=True)
    second=native.forward_identity_only(x,domain_labels=domains)['tx_logits']
    loss=api.original.terms(out['tx_logits'],second,y,1.)[0];named=dict(native.named_parameters())
    reference=torch.autograd.grad(loss,[named[n] for n in names]+[api.classifier(native).weight])
    assert torch.equal(rng(),ends[0]) and torch.equal(out['tx_logits'].detach(),logs[0])
    assert all(torch.allclose(g,h,atol=2e-7,rtol=2e-5) for g,h in zip(grads['FULL'],reference))
    fixture={arm:dict(source_health=True,evidence_audit=dict(min_rank=5,max_rank=5,aux_positive_calls=4440),exports={'FINAL':dict(count=1260,errors=100,mean_CE=1.)}) for arm in d.ARMS}
    assert not d.qualify(fixture)['qualified'];fixture['NULL']['exports']['FINAL']['errors']=99;assert d.qualify(fixture)['qualified']
    for arm in ('FULL','SPAN','ISOTROPIC'):
        bad=copy.deepcopy(fixture);bad[arm]['exports']['FINAL']['errors']=98;assert not d.qualify(bad)['qualified']
    for key in ('CE','health','inactive'):
        bad=copy.deepcopy(fixture)
        if key=='CE':bad['NULL']['exports']['FINAL']['mean_CE']=1.01
        elif key=='health':bad['FULL']['source_health']=False
        else:bad['NULL']['evidence_audit']['aux_positive_calls']=3995
        assert not d.qualify(bad)['qualified']
    # Exercise the new source observer's real progress shape without any data.
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from types import SimpleNamespace
    from .source import Observer
    with TemporaryDirectory() as root:
        c=d.config(d.rows()[0]);observer=Observer(c);observer.out=Path(root)
        dummy=SimpleNamespace(_completion_audit=dict(f.new_audit(),calls=222,packets=222*128),_feature_l_steps=222,_legacy_calls=222,_completion_calls=222,_legacy_extra_packets=222*128,_legacy_actual_L_input=None)
        observer.epoch(dummy,None,[{}],root,device)
        assert d.read(Path(root)/'rdrop_completion_progress.json')['actual_steps']==222
    print(json.dumps(dict(status='PASS',device=device,negative_cases=negative,qualification_negative_cases=7,actual_initial_state_tensors=len(states[0]),
        source_only=True,exact_Fisher_trace=True,equal_added_metric_trace=True,class_permutation_and_view_swap=True,
        classifier_reconstruction=True,auxiliary_default_W_gradient_none=True,aux_actual_t1_t2_SIG_gradients_nonzero=True,
        actual_CE_logits_RNG_equal=True,FULL_native_gradient_close=True,actual_parameter_gradient_norms=norms,geometry_audits=audits,
        synthetic_identity_forwards=1280,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,head_fit_updates=0,target_access=False)))

if __name__=='__main__':main()
