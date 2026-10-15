"""All five RX masks without IQ, plus original/current real graph regression."""
import copy
import json
import math
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d,model as api
from experiments.cvs_ordinary_fold20 import design as old
from experiments.cvs_ordinary_fold20.fold import prepare


def fixture_masks():
    class Item:
        def __init__(self,rx):self.rx_i=rx
        @property
        def tx_i(self):raise AssertionError('Hidden TX inspected')
    class Dataset:
        def __init__(self,role):
            self.split_source=role;self.hide=role=='U_s';self.base=self
            total={'L_s':6300,'U_s':56700,'V':27000}[role]
            self.index=[Item(rx) for rx in d.RX for _ in range(total//5)];self.selected=list(range(total))
        def __len__(self):return len(self.selected)
        def __getitem__(self,index):raise AssertionError('IQ read during RX mask')
    class Subset:
        def __init__(self,base,indices,split_source,transform):
            self.base=base;self.selected=indices;self.index=[base.index[i] for i in indices]
            self.split_source=split_source;self.hide=split_source=='U_s'
        def __len__(self):return len(self.selected)
        def __getitem__(self,index):raise AssertionError('IQ read during RX mask')
    class Loader:
        def __init__(self,ds,batch,shuffle=False,drop_last=False):
            self.dataset=ds;self.batch_size=batch;self.drop_last=drop_last
            self.sampler=torch.utils.data.RandomSampler(ds) if shuffle else torch.utils.data.SequentialSampler(ds)
        def __len__(self):return len(self.dataset)//self.batch_size if self.drop_last else math.ceil(len(self.dataset)/self.batch_size)
    native=SimpleNamespace(WiSigSubsetDataset=Subset,
        make_loader=lambda ds,batch,shuffle,workers,device,drop_last,threads:Loader(ds,batch,shuffle,drop_last))
    original={role:Dataset(role) for role in ('L_s','U_s','V')};proof=[]
    for held in d.RX:
        ctx=dict(named_test_loaders={},balanced_train_sampler=None,split_info={},
            train_loader=Loader(original['L_s'],128),probe_train_loader=Loader(original['L_s'],128),
            unlabeled_loader=Loader(original['U_s'],256),val_loader=Loader(original['V'],512),source_calibration_loader=Loader(original['V'],512))
        evaluation=prepare(native,ctx,lambda *args:None,held_rx=held)
        assert len(evaluation.dataset)==5400 and {v.rx_i for v in evaluation.dataset.index}=={held}
        for name in ('train_loader','probe_train_loader','unlabeled_loader','val_loader','source_calibration_loader'):
            assert {v.rx_i for v in ctx[name].dataset.index}==set(d.RX)-{held}
        assert len(ctx['train_loader'].dataset)==5040 and len(ctx['unlabeled_loader'].dataset)==45360 and len(ctx['val_loader'].dataset)==21600
        assert ctx['unlabeled_loader'].dataset.hide and len(ctx['unlabeled_loader'])==222 and len(ctx['unlabeled_loader'].original)==178
        proof.append(dict(held_RX=held,all_five_loaders_exclude_held=True,held_V_only=True,U_TX_and_IQ_reads=0))
    return proof


def qualify_checks():
    def result(acc=.7,f1=.7,seen=.98):return dict(held_V=dict(count=5400,accuracy=acc,macro_f1=f1),
        seen_V=dict(source_val_accuracy=seen),complete_health=True,target_access=False,U_truth_read=False,head_fit_updates=0)
    base={rx:dict(N0=result(),N1=result(.72,.72,.979)) for rx in d.RX}
    assert d.qualify(base)['status']=='PASS'
    cases=[]
    bad=copy.deepcopy(base);bad[3]['N1']['held_V']['accuracy']=.68;cases.append(('worst_fold',bad))
    bad=copy.deepcopy(base)
    for rx in (3,4):bad[rx]['N1']['held_V']['accuracy']=.699
    cases.append(('positive_folds',bad))
    bad=copy.deepcopy(base)
    for rx in d.RX:bad[rx]['N1']['held_V']['accuracy']=.705
    cases.append(('mean_accuracy',bad))
    bad=copy.deepcopy(base)
    for rx in d.RX:bad[rx]['N1']['held_V']['macro_f1']=.69
    cases.append(('mean_macroF1',bad))
    bad=copy.deepcopy(base);bad[3]['N1']['seen_V']['source_val_accuracy']=.97;cases.append(('seen_health',bad))
    for check,value in cases:
        q=d.qualify(value);assert q['status']=='FAIL' and not q['checks'][check]
    return len(cases)


def run():
    from experiments.cvs_ordinary_fold20.checks import run as regression
    prior=regression();proof=fixture_masks();negative=qualify_checks();torch.set_num_threads(2)
    x=torch.randn(128,2,256);y=torch.arange(128)%6;initial=None;forward=prior['synthetic_network_forwards']
    for row in d.rows():
        c=d.config(row);a=d.make_args(c,'cpu');fit=c['fold']['fit_RX']
        assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing)==(20,13,7,.01)
        assert a.source_fold_held_rx==row['held_RX'] and a.source_fold_training_receivers==','.join(map(str,fit))
        assert a.use_aug==(row['arm']=='N1') and not a.use_concat_sat_channel_aug and not a.baseline_ckpt and not a.teacher_ckpt
        if row['held_RX']!=3:continue
        m=api.build(c,'cpu');state={k:v.clone() for k,v in m.state_dict().items()}
        if initial is None:
            reference=api.base.build(old.config(old.rows()[0]),'cpu')
            assert state.keys()==reference.state_dict().keys() and all(torch.equal(v,reference.state_dict()[k]) for k,v in state.items())
            initial=state
        else:assert all(torch.equal(v,initial[k]) for k,v in state.items())
        ema=copy.deepcopy(m);assert ema._ordinary_original_hook.__self__ is ema and ema._ordinary_fit_RX==tuple(fit)
        meta=dict(rx_i=torch.tensor(fit*32),day_i=torch.arange(128)%3+1)
        m.train()
        with m._phase1_labeled_feature_context():out=m(x,y_tx=y,return_aux=True)
        ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,parts=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,meta),
            domains=torch.zeros(128,dtype=torch.long),optimizer_step=1,identity_loss=ce,labeled_count=128)
        (ce+correction).backward();forward+=256
        assert len(m._ordinary_latest)==128 and {v[1] for v in m._ordinary_latest}==set(fit)
        for suffix in ('cls_head.weight','cls_head.gain','sequence_pool.local_projection.weight'):
            p=next(p for n,p in m.named_parameters() if n.endswith(suffix))
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm())>0
    for changes in (dict(test_views=['clean']),dict(held_RX=1),dict(epochs=50),dict(checkpoint_sources=['old'])):
        c=dict(d.config(d.rows()[0]),**changes)
        try:d.validate(c)
        except ValueError:negative+=1
        else:raise AssertionError('Illegal confirmation config accepted')
    return dict(status='PASS',all_RX_loader_masks=proof,all_eight_native_configs=True,original_RX1_regression=prior,
        actual_graph_held_RX=3,initial_state_tensors=len(initial),same_original_initial_weights=True,
        synthetic_network_forwards=forward,synthetic_augmentation_packets=128,negative_cases=negative+prior['protocol_negatives'],
        default_head_SIG_gradients=True,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,target_access=False)


if __name__=='__main__':print(json.dumps(run()))
