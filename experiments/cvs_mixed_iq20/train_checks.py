"""Actual SIG_TIME graph and native loss entry, synthetic IQ only."""
import copy
import json
import torch
from torch.nn import functional as F
from . import train_design as d,train_model as api
from .objective import mixed_objective,rdrop_per_sample


def run():
    torch.set_num_threads(2);initial=None;forwards=0;records=[]
    x=torch.randn(128,2,256);y=torch.arange(128)%6
    rx=torch.tensor([1,3,4,6,8]*26)[:128];day=torch.arange(128)%3+1
    extra=(None,dict(rx_i=rx,day_i=day,eq_i=torch.ones(128,dtype=torch.long),base_index=torch.arange(128)))
    for row in d.rows():
        c=d.config(row);a=d.make_args(c,'cpu')
        if a.epochs!=20 or a.label_epochs!=13 or a.pseudo_epochs!=7 or a.use_aug!=(row['arm']=='R_LEGACY'):
            raise ValueError('Actual native schedule/ordinary augmentor switch differs')
        m=api.build(c,'cpu');state={k:v.clone() for k,v in m.state_dict().items()}
        if initial is None:initial=state
        elif any(not torch.equal(v,initial[k]) for k,v in state.items()):raise ValueError('All five initial states must match')
        ema=copy.deepcopy(m)
        if ema._mixed_original_labeled_scope.__self__ is not ema or ema._phase1_labeled_feature_context.__self__ is not ema:
            raise ValueError('Deepcopy EMA methods must bind to the EMA, not student')
        if ema._mixed_actual_input is not None:raise ValueError('No L packet carried into own EMA')
        if ema.id_backbone.encoder._mixed_original_features.__self__ is not ema.id_backbone.encoder:
            raise ValueError('EMA feature counter must bind to own encoder')
        m.train();seen=[];scope=[]
        handle=m.register_forward_pre_hook(lambda model,args:seen.append(args[0].detach().clone()))
        second_method=m.forward_identity_only
        def tracked(value,*args,**kwargs):
            scope.append(value.detach().clone())
            return second_method(value,*args,**kwargs)
        m.forward_identity_only=tracked
        try:
            with m._phase1_labeled_feature_context():out=m(x,y_tx=y,return_aux=True)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=extra,
                domains=torch.zeros(128,dtype=torch.long),optimizer_step=1,identity_loss=ce,labeled_count=128)
            (ce+correction).backward()
        finally:handle.remove()
        if any(module._forward_hooks or module._forward_pre_hooks for module in m.modules()):
            raise ValueError('No persistent forward hooks may disable native validation reuse')
        expected=1 if row['arm']=='R_LEGACY' else 5
        if len(seen)!=1 or len(scope)!=expected or not torch.equal(scope[0],x):raise ValueError('Actual two/six L forward count or captured IQ differs')
        if expected==5:
            if not torch.equal(scope[1],scope[2]) or not torch.equal(scope[3],scope[4]):raise ValueError('RDrop pair redrew IQ')
            if row['arm'] in ('B00','B01') and not torch.equal(scope[1],x):raise ValueError('Inactive G must be raw')
            if row['arm'] in ('B00','B10') and not torch.equal(scope[3],x):raise ValueError('Inactive S must be raw')
            if row['arm'] in ('B10','B11') and torch.equal(scope[1],x):raise ValueError('Active G must change waveform')
            if row['arm'] in ('B01','B11') and torch.equal(scope[3],x):raise ValueError('Active S must change waveform')
        for suffix in ('cls_head.weight','cls_head.gain','sequence_pool.local_projection.weight'):
            pairs=[(n,p) for n,p in m.named_parameters() if n.endswith(suffix)]
            if len(pairs)!=1 or pairs[0][1].grad is None or not bool(torch.isfinite(pairs[0][1].grad).all()) or float(pairs[0][1].grad.norm())<=0:
                raise ValueError('Real SIG/default W/fusion gradients must be finite and nonzero: '+suffix)
        if m._mixed_calls!=1 or m._feature_l_steps!=1 or m._mixed_actual_input is not None:raise ValueError('One actual native L step required')
        if m.id_backbone.encoder._mixed_train_packets!=128*(1+expected) or ema.id_backbone.encoder._mixed_train_packets:
            raise ValueError('Actual original features forward accounting differs')
        if any(not torch.equal(v,ema.state_dict()[k]) for k,v in initial.items()):raise ValueError('Student training changed own EMA without an EMA update')
        for bad_y in (torch.full_like(y,-1),y[:127]):
            try:m._source_labeled_feature_hook(model=m,labeled_output=out,labels=bad_y,extra=extra,domains=None,
                 optimizer_step=2,identity_loss=ce,labeled_count=len(bad_y))
            except ValueError:pass
            else:raise ValueError('Invalid labels/sequence must be rejected')
        forwards+=128*(1+expected)
        records.append(dict(arm=row['arm'],forward_packets=128*(1+expected),same_IQ_pairs=True,
                            default_W_and_SIG_gradients=True,own_EMA_state_unchanged=True))
    result=dict(status='PASS',actual_graph_rows=records,initial_state_tensors=len(initial),
                synthetic_forward_packets=forwards,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,
                protocol_negative_cases=10,network='original SIG_TIME/RAW_HZ/NoExplicitCFO/defaultcos')
    print(json.dumps(result))
    return result
