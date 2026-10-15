"""Synthetic actual-callback/native-step checks, without IQ or checkpoints."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from . import design as d, callback as callback_api, training_observer as api
from .callback import Callback, selected_names, protected_names, HEAD_PATH
from .training_observer import TrainingObserver, LengthOnlyUnlabeled, build_bank
from experiments.cvs_sigtime_heldcredit.coordinate import PREFIX
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _rng_snapshot, _equal


class Dataset:
    def __init__(self):
        self.hide=False; self.split_source='ssdg_labeled_tx_visible'; self.transform=None
        self.index=[]; self.selected=list(range(6300)); self.reads=[]
        for y in range(6):
            for rx in (1,3,4,6,8):
                for day in (1,2,3):
                    for ordinal in range(70):
                        self.index.append(SimpleNamespace(tx_i=y,rx_i=rx,day_i=day,eq_i=0,sig_i=ordinal))
    def __len__(self): return len(self.index)
    def __getitem__(self,index):
        self.reads.append(index); item=self.index[index]; torch.rand(())
        x=torch.zeros(2,256); x[0,:4]=torch.tensor([item.tx_i,item.rx_i,item.day_i,item.sig_i])
        return x,item.tx_i,0,dict(eq_i=0,tx_i=item.tx_i,rx_i=item.rx_i,day_i=item.day_i,sig_i=item.sig_i,base_index=self.selected[index])


class ForbiddenDataset:
    def __len__(self): raise AssertionError('Baseline touched auxiliary L dataset')
    def __getitem__(self,index): raise AssertionError('Baseline read auxiliary L IQ')


class ULoader:
    def __init__(self): self.dataset=object(); self.sampler=object(); self.iterations=0
    def __len__(self): return 222
    def __iter__(self): self.iterations+=1; raise AssertionError('Hidden U was iterated by observer')


class Sinc(nn.Module):
    def __init__(self):
        super().__init__(); self.low_coordinate=nn.Parameter(torch.full((24,1),.2))
        self.band_coordinate=nn.Parameter(torch.full((24,1),.1))
        self._filter_cache=torch.tensor([7.]); self._filter_cache_key=('original',); self._coordinate_cache_key=None
    @property
    def low_hz_(self): return self.low_coordinate*25000+100000
    @property
    def band_hz_(self): return self.band_coordinate*25000+100000
    def _filters(self,device,dtype):
        t=torch.arange(79,device=device,dtype=dtype).reshape(1,1,79)/79
        return self.low_coordinate[:,:,None]+self.band_coordinate[:,:,None]*t


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.id_backbone=nn.Module(); encoder=nn.Module(); self.id_backbone.encoder=encoder
        encoder.core=nn.Module(); encoder.core.id_backbone=nn.Module(); core=encoder.core.id_backbone
        core.sinc=Sinc(); core.features=nn.ParameterList([nn.Parameter(torch.tensor(.01+i*.0001)) for i in range(66)])
        core.cls_head=nn.Module(); generator=torch.Generator().manual_seed(18068)
        core.cls_head.weight=nn.Parameter(torch.randn(6,160,generator=generator)*.1)
        core.cls_head.gain=nn.Parameter(torch.full((160,),.3)); encoder.response_gain=nn.Parameter(torch.full((160,),.2))
        encoder.sequence_pool=nn.Module(); encoder.sequence_pool.gain_raw=nn.Parameter(torch.full((160,),.25))
        self.other=nn.ParameterList([nn.Parameter(torch.tensor(.01+i*.00001)) for i in range(116)])
        self._feature_l_steps=0


class Native:
    def compute_core_losses(self,*args,**kwargs): return dict(unchanged_core=True)


def _raises(call):
    try: call()
    except (ValueError,RuntimeError): return
    raise AssertionError('Expected direct source/step/scope rejection')


def _fake_capture(model,x):
    assert all(not module.training for module in model.modules())
    with fresh_parameter_caches(model):
        torch.rand(())  # Callback must restore caller RNG even with this fixture.
        sinc=model.get_submodule(PREFIX); sinc._filter_cache=torch.tensor([99.]); sinc._filter_cache_key=('temporary',)
        named=dict(model.named_parameters()); s=sum(named[name].mean() for name in selected_names(model))
        grid=torch.arange(160,device=x.device,dtype=x.dtype)[None,:]+1
        y=x[:,0,0:1]+1
        raw=dict(identity=(grid*y*.071).sin()+s*(grid*.017).cos(),
            physical=.3*(grid*y*.037).cos()+s*(grid*.031).sin(),
            reference_raw=.2*(grid*y*.027).sin()+s*(grid*.013).cos(),
            sequence_raw=.4*(grid*y*.043).cos()+s*(grid*.023).sin())
        encoder=model.id_backbone.encoder; head=model.get_submodule(HEAD_PATH)
        joint=((raw['identity']+head.gain.tanh()*raw['physical'])+encoder.response_gain.tanh()*raw['reference_raw'])
        joint=joint+(.2*encoder.sequence_pool.gain_raw.tanh())*raw['sequence_raw']
        return dict(raw,joint=joint)


def _oracle_regression():
    model=Model(); parameters=tuple(model.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.002,weight_decay=.01)
    step=torch.optim.AdamW.step
    for p in parameters: p.grad=torch.ones_like(p)*.2
    step(optimizer)  # Existing moments, not an empty-state-only oracle check.
    for index,p in enumerate(parameters):
        p.grad=None if index==0 else torch.zeros_like(p) if index==1 else torch.ones_like(p)*.3
    torch.nn.utils.clip_grad_norm_(parameters,5.)
    before=api._live_snapshot(optimizer); rng=_rng_snapshot()
    oracle=api.actual_postclip_oracle(optimizer,step)
    assert all(api._live_readback(optimizer,before).values()) and _equal(rng,_rng_snapshot())
    assert oracle['summary']['parameter_count']==188 and not oracle['summary']['second_gradient_clip']
    assert oracle['summary']['actual_postclip_gradients_copied'] and oracle['summary']['temporary_AdamW_steps']==1
    step(optimizer)
    assert all(torch.equal(p,q) for p,q in zip(parameters,oracle['after']))
    return dict(parameters=188,initialized_moments=True,None_and_zero_preserved=True,
        live_parameters_gradients_state_RNG_unchanged=True,no_second_clip=True,following_actual_step_bitwise_match=True)


def _damaged_actual_step_must_fail(out):
    c=d.config(next(row for row in d.rows() if row['arm']=='BASE_CE'))
    model=Model().train(); native=Native(); observer=TrainingObserver(c)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    original=torch.optim.AdamW.step; first_parameter=next(model.parameters())
    def corrupt_actual(used_optimizer,*args,**kwargs):
        result=original(used_optimizer,*args,**kwargs)
        if used_optimizer is optimizer:
            with torch.no_grad(): first_parameter.add_(.001)
        return result
    ctx=dict(train_loader=SimpleNamespace(dataset=ForbiddenDataset()),unlabeled_loader=ULoader())
    with patch.object(api.model_api,'classifier',lambda m:m.get_submodule(HEAD_PATH)), \
         patch.object(torch.optim.AdamW,'step',corrupt_actual):
        with observer.native_scope(native):
            observer.data(native,ctx,out); observer.initialize(model,c,out)
            model._feature_l_steps=16; native.compute_core_losses()
            base=(model.other[0]+model.get_submodule(HEAD_PATH).weight.square().mean()).square()
            auxiliary,_=model._source_labeled_feature_hook(model=model,labeled_output={},labels=None,extra=None,
                domains=None,optimizer_step=16,identity_loss=base,labeled_count=128)
            total=base+auxiliary
            model._source_optimizer_update_probe_hook(model=model,optimizer=optimizer,epoch=1,batch_idx=16,
                phase='label',optimizer_step=16,total_loss=total,unlabeled_loss=base*0,
                labeled_count=128,unlabeled_count=0)
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            _raises(lambda:optimizer.step())
            # The failing scalar record must exist before the exception escapes.
            path=out/'actual_responsibility_optimizer_increment.jsonl'
            record=json.loads(path.read_text(encoding='utf-8').splitlines()[-1]); update=record['actual_update']
            first=update['first_bad_parameter']
            assert update['actual_optimizer_steps']==1 and not update['actual_shadow_per_element_allclose']
            assert update['actual_shadow_reference']==d.ACTUAL_SHADOW_REFERENCE
            assert first['parameter']==next(iter(dict(model.named_parameters())))
            assert first['max_abs_error']>first['max_allowed_tolerance'] and first['mismatched_coordinate_count']>0
            assert observer.probe_steps==[] and observer.actual_steps==1
            assert set(first)=={'parameter','max_abs_error','max_allowed_tolerance','max_error_to_allowed_tolerance_ratio',
                'mismatched_coordinate_count','actual_nonfinite_count','reference_nonfinite_count'}
            return dict(rejected=True,scalar_failure_saved_before_raise=True,
                first_bad_parameter=first['parameter'],max_abs_error=first['max_abs_error'],
                max_error_to_allowed_tolerance_ratio=first['max_error_to_allowed_tolerance_ratio'])


def check():
    torch.set_num_threads(2)
    assert sum(d.active(step) for step in range(1,d.UPDATES+1))==180 and all(d.active(step) for step in d.PROBE_STEPS)
    dataset=Dataset(); rng=_rng_snapshot(); built=build_bank(dataset)
    assert _equal(rng,_rng_snapshot()) and len(dataset.reads)==900
    assert built['summary']['current_IQ_count']==720 and built['summary']['held_IQ_count']==180
    hidden=Dataset(); hidden.hide=True; _raises(lambda:build_bank(hidden)); assert hidden.reads==[]
    fake_model=Model().train()
    assert len(list(fake_model.parameters()))==188 and len(selected_names(fake_model))==68 and len(protected_names(fake_model))==4
    assert set(selected_names(fake_model)).isdisjoint(protected_names(fake_model))
    fake_model.id_backbone.encoder.core.id_backbone.features[0].requires_grad_(False)
    _raises(lambda:selected_names(fake_model))
    baseline=d.config(next(row for row in d.rows() if row['arm']=='BASE_CE'))
    authority=Callback(None,baseline,None)
    for role in ('U_s','V','target'):
        _raises(lambda role=role:authority(model=object(),labeled_output=None,labels=object(),extra=None,
            domains=None,optimizer_step=1,identity_loss=None,labeled_count=128,role=role))
    assert authority.counts['calls']==0 and authority.model is None
    oracle_regression=_oracle_regression()
    capture_counts={}; schedule_counts={}; total_actual_steps=0
    original_objective=callback_api.losses
    original_simulate=api.simulate
    with TemporaryDirectory(prefix='responsibility20_observer_') as temp:
        for arm in d.ARMS:
            c=d.config(next(row for row in d.rows() if row['arm']==arm)); out=Path(temp)/arm; out.mkdir()
            native=Native(); core=native.compute_core_losses; observer=TrainingObserver(c); model=Model().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01); original_step=torch.optim.AdamW.step
            u=ULoader(); data=ForbiddenDataset() if arm in callback_api.BASELINES else Dataset()
            ctx=dict(train_loader=SimpleNamespace(dataset=data),unlabeled_loader=u)
            captured=[]; actual_objective_steps=[]
            def fake_capture(m,x):
                assert m is model and len(x)==180
                captured.append((m._feature_l_steps,int(x[0,0,3])))
                return _fake_capture(m,x)
            def focused_objective(components,bank,weight,gains,source_role='L_s'):
                assert source_role=='L_s' and set(bank['rx'].tolist())=={1,3,4,6,8}
                assert {int(v) for v in bank['x'][:,0,3].tolist()}.issubset({2,3,19,20,37,38,54,55})
                if model._feature_l_steps in d.PROBE_STEPS:
                    actual_objective_steps.append(model._feature_l_steps)
                    return original_objective(components,bank,weight,gains,source_role=source_role)
                # Only nonprobe budget iterations use this small fixture. The
                # four actual-step probes consume the real reviewed objective.
                loss=.0001*sum(value.square().mean() for name,value in components.items() if name in ('identity','physical','reference_raw','sequence_raw'))
                modes=('ZERO','JOINT_GAP','ROUTED_GAP')
                return {mode:loss*0 if mode=='ZERO' else loss for mode in modes}, {
                    mode:dict(by_RX={str(rx):dict(mean_positive_burden=.01) for rx in (1,3,4,6,8)}) for mode in modes}
            def proxy_disagreement(*args,**kwargs):
                result=original_simulate(*args,**kwargs)
                if arm=='BASE_CE':
                    values=list(result['total_after']); values[0]=values[0].clone()+.001
                    result['total_after']=tuple(values)
                return result
            before_rng=_rng_snapshot(); before_flags=[module.training for module in model.modules()]
            with patch.object(api.model_api,'classifier',lambda m:m.get_submodule(HEAD_PATH)), \
                 patch.object(callback_api,'capture',fake_capture), patch.object(callback_api,'losses',focused_objective), \
                 patch.object(api,'simulate',proxy_disagreement):
                with observer.native_scope(native):
                    observer.data(native,ctx,out); observer.initialize(model,c,out)
                    assert observer.bank is None if arm in callback_api.BASELINES else observer.bank is not None
                    teacher=copy.deepcopy(model)
                    assert not observer.bound and not hasattr(teacher,'_source_labeled_feature_hook')
                    assert not hasattr(teacher,'_source_optimizer_update_probe_hook')
                    for step in range(1,d.UPDATES+1):
                        model._feature_l_steps=step
                        if step==1: assert native.compute_core_losses()==dict(unchanged_core=True)
                        selected=sum(p.mean() for name,p in model.named_parameters() if name in selected_names(model)) if step in d.PROBE_STEPS else None
                        base=(selected+model.get_submodule(HEAD_PATH).weight.square().mean()+sum(p for p in model.other)*.01).square() if selected is not None else model.other[0]*0
                        loss,metrics=model._source_labeled_feature_hook(model=model,labeled_output={},labels=None,
                            extra=None,domains=None,optimizer_step=step,identity_loss=base,labeled_count=128)
                        epoch=(step-1)//222+1; batch=(step-1)%222+1
                        phase='label' if epoch<=c['label_epochs'] else 'pseudo'
                        u_count=0 if phase=='label' else 124 if batch==222 else 256
                        u_loss=model.other[0]*.03 if phase=='pseudo' else model.other[0]*0
                        result=model._source_optimizer_update_probe_hook(model=model,optimizer=optimizer,epoch=epoch,
                            batch_idx=batch,phase=phase,optimizer_step=step,total_loss=base+u_loss+loss,
                            unlabeled_loss=u_loss,labeled_count=128,unlabeled_count=u_count)
                        assert result is None and model._feature_l_steps==step
                        if step in d.PROBE_STEPS:
                            total=base+u_loss+loss; total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                            optimizer.step(); optimizer.zero_grad(set_to_none=True); total_actual_steps+=1
                            assert observer.pending_update is None
                    observer._write_report(0)
                    report=json.loads((out/'source_training_mechanism.json').read_text())
                    expected=d.scheduled(c)
                    assert report['calls']==report['late_calls']==4440 and report['optimizer_steps']==4
                    assert report['auxiliary_calls']==report['scheduled_calls']==report['eligible_calls']==expected
                    assert report['extra_IQ_forwards']==expected*180 and report['extra_model_forward_calls']==expected
                    assert report['U_packet_observations']==(0 if arm=='BASE_CE' else 7*56700)
                    assert report['optimizer_probe_steps']==list(d.PROBE_STEPS) and len(report['actual_probes'])==4
                    assert report['extra_temporary_AdamW_steps']==12
                    assert report['head_fit_updates']==report['extra_actual_optimizer_steps']==0
                    assert all(p['selected_W_gain_gradients_None'] and p['actual_update']['actual_optimizer_steps']==1
                        and p['actual_update']['actual_shadow_per_element_allclose']
                        and p['actual_update']['actual_shadow_reference']==d.ACTUAL_SHADOW_REFERENCE
                        and p['actual_update']['actual_shadow_per_element_atol']==2e-7
                        and p['actual_update']['actual_shadow_per_element_rtol']==2e-6
                        and p['actual_update']['actual_gradient_oracle']['parameter_count']==188
                        and p['actual_update']['actual_gradient_oracle']['live_parameters_unchanged']
                        and p['actual_update']['actual_gradient_oracle']['live_gradients_unchanged']
                        and p['actual_update']['actual_gradient_oracle']['live_optimizer_state_unchanged']
                        and p['actual_update']['actual_gradient_oracle']['rng_unchanged']
                        and not p['actual_update']['actual_gradient_oracle']['second_gradient_clip']
                        and p['temporary_AdamW_steps']==3 and p['autograd_proxy_is_approximate']
                        and p['not_exact_actual_auxiliary_increment']
                        and p['actual_update']['actual_shadow_max_abs_error']<=p['actual_update']['actual_shadow_FP32_tolerance']
                        for p in report['actual_probes'])
                    if arm=='BASE_CE':
                        assert all(not p['actual_update']['autograd_proxy_per_element_allclose']
                            and p['actual_update']['autograd_proxy_max_abs_error']>2e-7
                            for p in report['actual_probes'])
                    if arm in callback_api.BASELINES:
                        assert not captured and not actual_objective_steps and report['bank_source_IQ_reads']==0
                        assert all(p['selected_auxiliary_gradient_norm']==0 for p in report['actual_probes'])
                    else:
                        assert len(data.reads)==900 and len(captured)==180 and actual_objective_steps==list(d.PROBE_STEPS)
                        assert report['by_current_ordinal']=={str(key):45 for key in callback_api.CURRENT_ORDINALS}
                        assert [ordinal for _,ordinal in captured]==list(callback_api.CURRENT_ORDINALS)*45
                        assert any(p['selected_auxiliary_gradient_norm']>1e-8 for p in report['actual_probes'])
                    assert u.iterations==0 and (isinstance(ctx['unlabeled_loader'],LengthOnlyUnlabeled)==(arm=='BASE_CE'))
                    _raises(lambda:observer.callback(model=model,labeled_output={},labels=None,extra=None,
                        domains=None,optimizer_step=d.UPDATES,identity_loss=None,labeled_count=128))
                    _raises(lambda:observer.callback(model=model,labeled_output={},labels=None,extra=None,
                        domains=None,optimizer_step=d.UPDATES+1,identity_loss=None,labeled_count=127))
                    assert observer.callback.counts['calls']==4440
                    _raises(lambda:observer.epoch(model,ctx,[{}]*20,out,torch.device('cpu')))
                    capture_counts[arm]=len(captured); schedule_counts[arm]=expected
                assert native.compute_core_losses==core and torch.optim.AdamW.step is original_step
                assert not hasattr(model,'_source_labeled_feature_hook') and not hasattr(model,'_source_optimizer_update_probe_hook')
                assert not hasattr(teacher,'_source_labeled_feature_hook') and not hasattr(teacher,'_source_optimizer_update_probe_hook')
            assert _equal(before_rng,_rng_snapshot()) and before_flags==[module.training for module in model.modules()]
            assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
            assert model.get_submodule(PREFIX)._filter_cache_key==('original',)
            assert torch.equal(model.get_submodule(PREFIX)._filter_cache,torch.tensor([7.]))
        damaged=Path(temp)/'damaged_actual_step'; damaged.mkdir()
        actual_failure=_damaged_actual_step_must_fail(damaged)
    return dict(status='PASS',device='cpu',actual_callback_scope_tensors=68,original_parameter_traversal188=True,
        four_arms_full_L_callback_and_late_calls=4440,scheduled_by_arm=schedule_counts,capture_calls_by_arm=capture_counts,
        active_bank_builder_synthetic_IQ_reads=900,active_auxiliary_IQ_packets=32400,held_bank_not_in_main_graph=True,
        baseline_bank_reads_and_extra_capture_zero=True,BASE_CE_U_length_only_no_iteration=True,
        callbacks_bound_after_teacher_deepcopy=True,W_three_gain_auxiliary_gradients_None=True,
        nonL_role_rejected_before_model_or_labels=True,duplicate_step_and_wrong_L_count_rejected=True,
        actual_reviewed_objective_at_four_probes=True,synthetic_actual_AdamW_steps=total_actual_steps,
        temporary_AdamW_steps=total_actual_steps*3,temporary_steps_per_probe=3,
        actual_shadow_reference=d.ACTUAL_SHADOW_REFERENCE,once_only_update_per_element_shadow_pass=True,
        exact_input_oracle_regression=oracle_regression,approximate_proxy_disagreement_nonblocking=True,
        deliberately_damaged_actual_step=actual_failure,
        full_U_exposure_count_preserved=True,native_L_counter_flags_RNG_caches_hooks_restored=True,
        true_IQ_reads=0,checkpoint_reads=0,native_constructors=0,training_launches=0,
        limitation='Synthetic full callback schedule; only four real mock optimizer updates per arm, not full native training')


if __name__=='__main__': print(json.dumps(check(),allow_nan=False,indent=2))
