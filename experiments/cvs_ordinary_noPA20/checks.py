import copy
import math
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .source import prohibit_PA
from experiments.cvs_ordinary_fold20 import model as old_api
from experiments.cvs_ordinary_fold20.streams import PrivateAugment,stream

def run(reference):
    from SSDG import train_ssdg as native
    torch.set_num_threads(2);c=d.config(d.rows()[0]);a=d.make_args(c,'cpu')
    oldc=d.old.config(next(r for r in d.old.rows() if r['arm']=='N1'));olda=d.old.make_args(oldc,'cpu')
    def equal(x,y):return x==y or (isinstance(x,float) and isinstance(y,float) and math.isnan(x) and math.isnan(y))
    changed={k for k in vars(a) if not equal(vars(a)[k],vars(olda)[k])}
    assert changed=={'run_id','candidate_id','output_dir','aug_enable_pa_normal'} and a.use_aug and not a.aug_enable_pa_normal
    assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing)==(20,13,7,.01)
    m=api.build(c,'cpu');initial={k:v.clone() for k,v in m.state_dict().items()};oldm=old_api.build(oldc,'cpu')
    assert all(torch.equal(v,oldm.state_dict()[k]) for k,v in initial.items())
    ema=copy.deepcopy(m);assert ema._source_labeled_feature_hook.__self__ is ema
    x=torch.randn(128,2,256);y=torch.arange(128)%6;meta=dict(rx_i=torch.tensor([3,4,6,8]*32),day_i=torch.arange(128)%3+1)
    cfg=native.build_aug_base_cfg(a)
    with stream(20261016051,0):inner=prohibit_PA(native.make_augmentor(cfg))
    aug=PrivateAugment(inner,20261016051);native.configure_augmentor_for_epoch(aug,cfg,13,a)
    assert aug.p_pa>0 and aug.p_awgn>0 and aug.p_multipath>0
    before=torch.get_rng_state().clone();actual=aug(x.clone(),labels=y,no_pa=not a.aug_enable_pa_normal)
    assert torch.equal(before,torch.get_rng_state()) and torch.isfinite(actual).all()
    m.train();seen=[];original=m.forward_identity_only
    def tracked(value,*args,**kwargs):seen.append(value.detach().clone());return original(value,*args,**kwargs)
    m.forward_identity_only=tracked
    with m._phase1_labeled_feature_context():out=m(actual,y_tx=y,return_aux=True)
    ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
    correction,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,meta),domains=torch.zeros(128,dtype=torch.long),optimizer_step=1,identity_loss=ce,labeled_count=128)
    (ce+correction).backward()
    assert len(seen)==1 and torch.equal(seen[0],actual) and m._legacy_calls==1 and m.id_backbone.encoder._ordinary_train_packets==256
    for suffix in ('cls_head.weight','cls_head.gain','sequence_pool.local_projection.weight'):
        p=next(p for k,p in m.named_parameters() if k.endswith(suffix));assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0
    assert all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
    negatives=0
    try:inner.simulate_pa(x);raise AssertionError('PA call accepted')
    except ValueError:negatives+=1
    for change in (dict(test_views=['clean']),dict(checkpoint_sources=['old']),dict(epochs=50)):
        try:d.validate(dict(c,**change));raise AssertionError('Illegal training config accepted')
        except ValueError:negatives+=1
    value=copy.deepcopy(reference['N1']);assert d.qualify(value,reference)['status']=='PASS'
    for section,key,replacement in [('held_V','accuracy',reference['N1']['held_V']['accuracy']-.006),('held_V','macro_f1',reference['N1']['held_V']['macro_f1']-.006),('seen_V','source_val_accuracy',reference['N0']['seen_V']['source_val_accuracy']-.006)]:
        v=copy.deepcopy(value);v[section][key]=replacement;assert d.qualify(v,reference)['status']=='FAIL';negatives+=1
    value['U_truth_read']=True
    try:d.qualify(value,reference);raise AssertionError('U truth use accepted')
    except ValueError:negatives+=1
    return dict(status='PASS',actual_native_arg_differences=sorted(changed),synthetic_network_forwards=256,synthetic_augmentation_packets=128,
        initial_state_tensors=len(initial),same_old_N1_initial_state=True,same_IQ_RDrop_pair=True,PA_calls_rejected=True,other_ordinary_enabled=True,
        default_head_SIG_gradients=True,protocol_negatives=negatives,real_IQ_packets=0,checkpoint_loads=0,optimizer_updates=0,target_access=False)
