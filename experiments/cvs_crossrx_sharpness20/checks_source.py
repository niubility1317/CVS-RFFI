"""Focused synthetic source/native wiring checks; no actual IQ/training launch."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from . import design as d
from .training_observer import TrainingObserver, build_bank, LengthOnlyUnlabeled
from experiments.cvs_crossrx_sharpness20 import training_observer as api
from experiments.cvs_crossrx_sharpness50 import observe as callback_api
from experiments.cvs_feature_sequence50.source import source_budget, require_budget
from experiments.cvs_phase1_overlay.model import native_modules


class Dataset:
    def __init__(self):
        self.hide = False; self.index = []; self.selected = list(range(6300)); self.reads = []
        for y in range(6):
            for rx in (1,3,4,6,8):
                for day in (1,2,3):
                    for ordinal in range(70):
                        self.index.append(SimpleNamespace(tx_i=y,rx_i=rx,day_i=day,eq_i=0,sig_i=ordinal))
    def __len__(self): return len(self.index)
    def __getitem__(self,index):
        self.reads.append(index); item = self.index[index]
        torch.rand(())  # Bank construction must preserve caller RNG.
        return torch.zeros(2,256), item.tx_i, 0, dict(eq_i=0,rx_i=item.rx_i,day_i=item.day_i,
            sig_i=item.sig_i,base_index=self.selected[index])


class ULoader:
    def __init__(self): self.dataset = object(); self.sampler = object(); self.iterations = 0
    def __len__(self): return 222
    def __iter__(self): self.iterations += 1; raise AssertionError('U was consumed')


class Sinc(nn.Module):
    def __init__(self):
        super().__init__(); self.low_coordinate=nn.Parameter(torch.full((24,1),.2))
        self.band_coordinate=nn.Parameter(torch.full((24,1),.1))
        self._filter_cache=self._filter_cache_key=self._coordinate_cache_key=None
    @property
    def low_hz_(self): return self.low_coordinate*25000+100000
    @property
    def band_hz_(self): return self.band_coordinate*25000+100000
    def _filters(self,device,dtype):
        t=torch.arange(79,device=device,dtype=dtype).reshape(1,1,79)/79
        return self.low_coordinate[:,:,None]+self.band_coordinate[:,:,None]*t


class Group(nn.Module):
    def __init__(self,count):
        super().__init__()
        for i in range(count): self.register_parameter(f'p{i}',nn.Parameter(torch.tensor(.1+i*.02)))


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.id_backbone=nn.Module(); self.id_backbone.encoder=nn.Module()
        self.id_backbone.encoder.core=nn.Module(); self.id_backbone.encoder.core.id_backbone=nn.Module()
        primary=self.id_backbone.encoder.core.id_backbone
        primary.sinc=Sinc();primary.time_fuse=Group(3);primary.t1=Group(4);primary.t2=Group(4)
        self.id_backbone.encoder.sequence_pool=Group(4)
        self.id_backbone.encoder.sequence_pool.gain_raw=nn.Parameter(torch.tensor(.2))
        self.head=nn.Module();self.head.weight=nn.Parameter(torch.eye(6,160))
        self.other=nn.Parameter(torch.tensor(.3));self._feature_l_steps=0


class Native:
    def compute_core_losses(self,*args,**kwargs): return dict(unchanged_core=True)
    def _meta_from_extra(self,extra): return extra[1]


def _raises(call):
    try: call()
    except (ValueError,RuntimeError): return
    raise AssertionError('Expected direct budget/source rejection')


def check():
    torch.set_num_threads(2)
    require_budget(11100,11100)
    require_budget(4440,4440,4440)
    _raises(lambda:require_budget(4440,4440))
    assert source_budget(SimpleNamespace(epochs=50,label_epochs=33,pseudo_epochs=17),SimpleNamespace(UPDATES=11100)) == (50,33,17,11100)
    for arm in d.ARMS:
        c=d.config(next(row for row in d.rows() if row['arm']==arm));args=d.make_args(c,'cpu')
        assert source_budget(args,d)==(20,c['label_epochs'],c['pseudo_epochs'],4440)
        assert not args.use_aug and args.use_unlabeled==(arm!='BASE_CE') and args.use_ema_teacher==(arm!='BASE_CE')
    assert sum(d.active(step) for step in range(1,4441))==250
    assert all(d.active(step) for step in d.PROBE_STEPS)
    dataset=Dataset();rng=torch.get_rng_state().clone();bank=build_bank(dataset)
    assert torch.equal(rng,torch.get_rng_state()) and len(dataset.reads)==180
    assert dataset.reads==bank['dataset_indices'] and all(index%70 in (68,69) for index in dataset.reads)
    assert {int(rx):int((bank['rx']==rx).sum()) for rx in (1,3,4,6,8)}=={1:36,3:36,4:36,6:36,8:36}
    hidden=Dataset();hidden.hide=True;_raises(lambda:build_bank(hidden));assert not hidden.reads
    bad=Dataset();bad.index[0].rx_i=2;_raises(lambda:build_bank(bad));assert not bad.reads
    u=ULoader();length_only=LengthOnlyUnlabeled(u);native_real=native_modules()
    pairs=list(native_real._muse_epoch_pairs([1,2,3],length_only,use_muse=False,use_unlabeled_step_budget=True))
    assert len(pairs)==222 and all(second is None for first,second in pairs) and u.iterations==0
    assert not native_real._legacy_unlabeled_active(SimpleNamespace(use_unlabeled=False,use_muse_ssdg=False),muse_state=None,phase='pseudo')
    _raises(lambda:iter(length_only))

    c=d.config(next(row for row in d.rows() if row['arm']=='CROSS_GAP'))
    native=Native();core=native.compute_core_losses
    observer=TrainingObserver(c);model=Model().train();optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    original_step=torch.optim.AdamW.step  # First optimizer construction installs Torch's own profiler wrapper.
    current_indices=[68]+[i*70 for i in range(1,90)]+[i*70+1 for i in range(38)]
    assert len(current_indices)==128
    items=[dataset.index[i] for i in current_indices]
    y=torch.tensor([item.tx_i for item in items])
    meta={key:torch.tensor([getattr(item,key) for item in items]) for key in ('eq_i','rx_i','day_i','sig_i')}
    meta['base_index']=torch.tensor(current_indices)
    calls=[]
    def fake_objective(m,logits,labels,rx,hx,hy,hr,ids,hids,names,weight,mode,*,role):
        assert len(hx)==179 and {p[-1] for p in ids}.isdisjoint(p[-1] for p in hids)
        assert set(hr.tolist())=={1,3,4,6,8} and role=='L_s' and mode=='CROSS_GAP'
        assert m._feature_l_steps in d.PROBE_STEPS
        calls.append(m._feature_l_steps)
        named=dict(m.named_parameters());loss=.0001*sum(named[name].square().sum() for name in names)
        return dict(loss=loss,summary=dict(loss=float(loss.detach()),extra_IQ_forwards=895,functional_forwards=6,
            by_RX={str(r):dict(clean_CE=1.,pert_CE=1.01,gap=.01,positive_gap=.01) for r in (1,3,4,6,8)}))
    with TemporaryDirectory(prefix='sharpness20_mock_') as temp:
        out=Path(temp)
        ctx=dict(train_loader=SimpleNamespace(dataset=dataset),unlabeled_loader=ULoader())
        with patch.object(api.model_api,'classifier',lambda m:m.head),patch.object(callback_api,'objective',fake_objective):
            with observer.native_scope(native):
                observer.data(native,ctx,out);observer.initialize(model,c,out)
                teacher=copy.deepcopy(model)
                assert not hasattr(teacher,'_source_labeled_feature_hook') and not observer.bound
                for step in d.PROBE_STEPS:
                    model._feature_l_steps=step
                    selected=sum(p.mean() for name,p in model.named_parameters() if name in api.select_names(model))
                    logits=model.head.weight[:,:6].T.repeat(22,1)[:128]+selected*torch.arange(6)
                    native.compute_core_losses(logits)
                    assert observer.bound and not hasattr(teacher,'_source_optimizer_update_probe_hook')
                    auxiliary,metrics=model._source_labeled_feature_hook(model=model,labeled_output={'tx_logits':logits},
                        labels=y,extra=(None,meta),domains=None,optimizer_step=step,identity_loss=logits.sum(),labeled_count=128)
                    epoch=(step-1)//222+1;batch=(step-1)%222+1;phase='label' if epoch<=13 else 'pseudo'
                    base=(selected+model.head.weight.square().sum()*.01+model.other).square()
                    u_loss=selected*.07+model.head.weight.sum()*.001 if phase=='pseudo' else selected*0
                    total=base+u_loss+auxiliary
                    result=model._source_optimizer_update_probe_hook(model=model,optimizer=optimizer,epoch=epoch,
                        batch_idx=batch,phase=phase,optimizer_step=step,total_loss=total,unlabeled_loss=u_loss,
                        labeled_count=128,unlabeled_count=256 if phase=='pseudo' else 0)
                    assert result is None and all(p.grad is None for p in model.parameters())
                    total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();optimizer.zero_grad(set_to_none=True)
                    assert observer.pending_update is None and model._feature_l_steps==step
                observer._write_report(0)
                report=json.loads((out/'source_training_mechanism.json').read_text())
                assert report['optimizer_probe_steps']==list(d.PROBE_STEPS) and report['active_gradient_probe_steps']==list(d.PROBE_STEPS)
                assert report['optimizer_steps']==4 and report['head_fit_updates']==0 and report['extra_actual_optimizer_steps']==0
                assert all(p['selected_W_gain_gradients_None'] and p['actual_update']['actual_optimizer_steps']==1
                    and p['actual_update']['actual_shadow_max_abs_error']<=p['actual_update']['actual_shadow_FP32_tolerance'] for p in report['actual_probes'])
                assert all(p['actual_unlabeled_loss']!=0 for p in report['actual_probes'] if p['phase']=='pseudo')
                _raises(lambda:observer.epoch(model,ctx,[{}]*20,out,torch.device('cpu')))
            assert native.compute_core_losses==core and torch.optim.AdamW.step is original_step
            assert not hasattr(model,'_source_labeled_feature_hook') and not hasattr(model,'_source_optimizer_update_probe_hook')
            assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
        ce=d.config(next(row for row in d.rows() if row['arm']=='BASE_CE'));ceobs=TrainingObserver(ce)
        cectx=dict(train_loader=SimpleNamespace(dataset=dataset),unlabeled_loader=ULoader())
        ceobs.data(native,cectx,out)
        assert isinstance(cectx['unlabeled_loader'],LengthOnlyUnlabeled) and len(cectx['unlabeled_loader'])==222
        _raises(lambda:iter(cectx['unlabeled_loader']))
    return dict(status='PASS',torch=torch.__version__,device='cpu',default50_budget_compatible=True,
        scratch20_budget=4440,scheduled_calls=250,bank_mock_IQ_reads=180,bank_original_ordinals=[68,69],
        BASE_CE_U_length_only_no_iteration=True,BASE_CE_EMA_args_disabled=True,
        callbacks_installed_after_teacher_deepcopy=True,auxiliary_loss_single_consumption=True,
        actual_mock_AdamW_updates=4,temporary_mock_AdamW_steps=8,probe_steps=list(d.PROBE_STEPS),
        full_pseudo_base_includes_U=True,protected_auxiliary_gradients_None=True,
        actual_shadow_matches=True,live_L_counter_unchanged_by_auxiliary=True,native_scope_restored=True,
        source_IQ_reads=0,training_launches=0,boundary='Synthetic source/native integration only; no efficacy evidence')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
