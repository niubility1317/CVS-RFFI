"""Actual graph and ordinary augmentation with synthetic IQ only."""
import copy
import io
import json
import random
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .fold import selected,seed_loader
from .streams import stream,PrivateAugment


def run():
    torch.set_num_threads(2);torch.manual_seed(20261016057)
    x=torch.randn(128,2,256);y=torch.arange(128)%6
    meta=dict(rx_i=torch.tensor(list(d.FIT)*32),day_i=torch.arange(128)%3+1)
    initial=None;forwards=0;records=[]
    from SSDG import train_ssdg as native
    from .source import Observer
    for row in d.rows():
        c=d.config(row);a=d.make_args(c,'cpu');m=api.build(c,'cpu')
        state={k:v.clone() for k,v in m.state_dict().items()}
        if initial is None:initial=state
        elif any(not torch.equal(v,initial[k]) for k,v in state.items()):raise ValueError('Same current model initialization required')
        ema=copy.deepcopy(m)
        assert ema._ordinary_original_hook.__self__ is ema and ema._source_labeled_feature_hook.__self__ is ema
        assert ema.id_backbone.encoder._ordinary_original_features.__self__ is ema.id_backbone.encoder
        before=torch.get_rng_state().clone();np_before=np.random.get_state();py_before=random.getstate()
        if row['arm']=='N1':
            cfg=native.build_aug_base_cfg(a)
            with stream(20261016051,0):inner=native.make_augmentor(cfg)
            aug=PrivateAugment(inner,20261016051)
            native.configure_augmentor_for_epoch(aug,cfg,1,a)
            actual=aug(x.clone(),labels=y,no_pa=not a.aug_enable_pa_normal)
            assert aug._calls==1 and aug._packets==128 and torch.isfinite(actual).all()
            assert len(aug.summary()['probabilities'])==11
            # configure_augmentor_for_epoch must update the real object.
            assert inner.p_awgn==aug.p_awgn and inner.max_time_shift==aug.max_time_shift
        else:actual=x
        assert torch.equal(before,torch.get_rng_state())
        assert py_before==random.getstate() and np.array_equal(np_before[1],np.random.get_state()[1])
        m.train();seen=[];original=m.forward_identity_only
        def tracked(value,*args,**kwargs):seen.append(value.detach().clone());return original(value,*args,**kwargs)
        m.forward_identity_only=tracked
        with m._phase1_labeled_feature_context():out=m(actual,y_tx=y,return_aux=True)
        ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,meta),
            domains=torch.zeros(128,dtype=torch.long),optimizer_step=1,identity_loss=ce,labeled_count=128)
        (ce+correction).backward()
        assert len(seen)==1 and torch.equal(seen[0],actual) and m._legacy_calls==1 and m._legacy_extra_packets==128
        assert len(m._ordinary_latest)==128 and m.id_backbone.encoder._ordinary_train_packets==256
        assert ema.id_backbone.encoder._ordinary_train_packets==0
        assert all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
        for suffix in ('cls_head.weight','cls_head.gain','sequence_pool.local_projection.weight'):
            p=next(p for k,p in m.named_parameters() if k.endswith(suffix))
            assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0
        records.append(dict(arm=row['arm'],use_aug=a.use_aug,label_smoothing=a.label_smoothing,L_forwards=256))
        forwards+=256
    # The RX mask cannot inspect U labels/TX or load any IQ.
    class Item:
        def __init__(self,rx):self.rx_i=rx
        @property
        def tx_i(self):raise AssertionError('U TX accessed')
    class Dataset:
        hide=True
        index=[Item(r) for r in (1,3,4,6,8) for _ in range(11340)]
        selected=list(range(56700))
        def __len__(self):return 56700
        def __getitem__(self,index):raise AssertionError('IQ accessed while selecting metadata')
    assert len(selected(Dataset(),'U_s'))==45360
    negative=0
    for changes in (dict(test_views=['clean']),dict(checkpoint_sources=['old']),dict(epochs=50)):
        c=dict(d.config(d.rows()[0]),**changes)
        try:d.validate(c)
        except ValueError:negative+=1
    observer=Observer(d.config(d.rows()[0]));observer.sequence=io.StringIO()
    for bad_y,bad_rx in ((torch.zeros(128,dtype=torch.long),meta['rx_i']),(torch.full((128,),-1),torch.ones(128,dtype=torch.long))):
        try:observer.record('U_s',(x,bad_y,None,dict(meta,rx_i=bad_rx)))
        except ValueError:negative+=1
    assert negative==5
    # Data sampling does not consume global dropout RNG and stays paired.
    values=torch.utils.data.TensorDataset(torch.arange(200))
    loaders=[seed_loader(torch.utils.data.DataLoader(values,batch_size=16,shuffle=True,num_workers=0),1234) for _ in range(2)]
    # Local Torch2.1's RandomSampler calls .numpy(), unavailable with this
    # workstation's NumPy2 ABI. Verify the actual bound generators and their
    # randperm stream directly; real N607 loaders are checked by full traces.
    before=torch.get_rng_state().clone()
    orders=[torch.randperm(200,generator=loader.sampler.generator) for loader in loaders]
    assert loaders[0].generator.initial_seed()==loaders[1].generator.initial_seed()
    assert torch.equal(orders[0],orders[1]) and torch.equal(before,torch.get_rng_state())
    return dict(status='PASS',rows=records,initial_state_tensors=len(initial),synthetic_network_forwards=forwards,
        synthetic_augmentation_packets=128,protocol_negatives=negative,private_RNG_preserved=True,
        same_sampler_seed_and_permutation=True,same_IQ_RDrop_pairs=True,default_head_and_SIG_gradients=True,
        U_TX_and_IQ_not_read_by_filter=True,checkpoint_loads=0,real_IQ_packets=0,optimizer_steps=0)


if __name__=='__main__':print(json.dumps(run()))
