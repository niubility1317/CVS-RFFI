"""Late student-only responsibility callbacks and once-only native AdamW probes."""
from contextlib import contextmanager
from pathlib import Path
import json

import torch

from . import design as d, model as model_api
from .callback import Callback, BASELINES, selected_names
from experiments.cvs_identity_correspondence20.bank import banks
from experiments.cvs_crossrx_sharpness20.training_observer import _sinc_snapshot, _norm, _auxiliary_names
from experiments.cvs_u_schedule50.counterfactual import simulate, _preserve_rng


class LengthOnlyUnlabeled:
    """Keep222 L steps without fetching any hidden U packet in BASE_CE."""
    def __init__(self,loader):
        if len(loader)!=d.STEPS: raise ValueError('Original U length differs from222')
        self.original=loader; self.dataset=loader.dataset; self.sampler=getattr(loader,'sampler',None)
    def __len__(self): return d.STEPS
    def __iter__(self): raise RuntimeError('BASE_CE must never iterate U')


def build_bank(dataset):
    with _preserve_rng(): return banks(dataset)


class TrainingObserver:
    def __init__(self,config):
        d.validate(config); self.c=config
        self.model=self.callback=self.bank=self.native=None
        self.bound=False; self.pending_aux=self.pending_update=None
        self.optimizer=None; self.actual_steps=self.late_calls=self.U_packets=0
        self.positive_gap_calls=0; self.actual_probes=[]; self.active_gradient_probe_steps=[]
        self.probe_steps=[]; self.initial_sinc=self.final_sinc=None

    def data(self,native,ctx,out):
        if self.c['arm']=='BASE_CE':
            if self.c['execution_overrides']['use_unlabeled'] or self.c['execution_overrides']['use_ema_teacher']:
                raise ValueError('BASE_CE must actually disable U and EMA')
            ctx['unlabeled_loader']=LengthOnlyUnlabeled(ctx['unlabeled_loader'])
        if self.c['arm'] not in BASELINES: self.bank=build_bank(ctx['train_loader'].dataset)
        d.write(Path(out)/'responsibility_L_bank.json',dict(role='L_s',
            built=self.bank is not None,source_IQ_reads=0 if self.bank is None else 900,
            current_IQ_count=0 if self.bank is None else 720,held_IQ_reads=0 if self.bank is None else 180,
            held_bank_used_for_main_gradient=False,current_ordinal_pairs=[[2,3],[19,20],[37,38],[54,55]],
            summary=None if self.bank is None else self.bank['summary'],U_IQ_reads=0,V_IQ_reads=0,target_access=False))

    def initialize(self,model,config,out):
        if config!=self.c or (self.c['arm'] not in BASELINES and self.bank is None):
            raise ValueError('Actual config and validated active-arm L banks required')
        if any(hasattr(model,key) for key in ('_source_labeled_feature_hook','_source_optimizer_update_probe_hook')):
            raise ValueError('Only an unbound own scratch student is permitted')
        self.model,self.out=model,Path(out)
        selected_names(model)
        weight=model_api.classifier(model).weight
        matching=[name for name,p in model.named_parameters() if p is weight]
        if len(matching)!=1: raise ValueError('Original default W traversal differs')
        self.callback=Callback(self.native,self.c,self.bank,default_weight_name=matching[0],
            metrics_path=self.out/'responsibility_training_steps.jsonl')
        self.initial_sinc=_sinc_snapshot(model)
        torch.save(self.initial_sinc,self.out/'source_sinc_initial.pt')

    def _bind(self):
        if self.bound: return
        if self.model is None or self.callback is None: raise ValueError('Student initialization incomplete')
        self.model._source_labeled_feature_hook=self.auxiliary
        self.model._source_optimizer_update_probe_hook=self.probe
        self.bound=True

    def auxiliary(self,**kwargs):
        if kwargs['model'] is not self.model or self.model._feature_l_steps!=kwargs['optimizer_step']:
            raise ValueError('Current main L context/step mismatch')
        loss,metrics=self.callback(**kwargs); record=self.callback.last_record
        self.positive_gap_calls+=int(any(row['positive_gap']>0 for row in record['by_RX'].values()))
        if kwargs['optimizer_step'] in d.PROBE_STEPS:
            if self.pending_aux is not None: raise ValueError('Unconsumed actual auxiliary probe')
            self.pending_aux=dict(step=kwargs['optimizer_step'],loss=loss,callback_record=dict(record))
        return loss,metrics

    auxiliary._accepts_labeled_inputs=False

    def probe(self,*,model,optimizer,epoch,batch_idx,phase,optimizer_step,total_loss,
              unlabeled_loss,labeled_count,unlabeled_count):
        if (model is not self.model or self.callback.last_step!=optimizer_step
                or model._feature_l_steps!=optimizer_step or optimizer_step!=(epoch-1)*d.STEPS+batch_idx):
            raise ValueError('Actual main L/late global step mismatch')
        expected_phase='label' if epoch<=self.c['label_epochs'] else 'pseudo'
        expected_u=0 if expected_phase=='label' else 124 if batch_idx==d.STEPS else 256
        if phase!=expected_phase or unlabeled_count!=expected_u or labeled_count!=128:
            raise ValueError('Declared native L/U phase or exposure differs')
        if self.optimizer is not None and optimizer is not self.optimizer: raise ValueError('Actual optimizer changed')
        self.optimizer=optimizer; self.late_calls+=1; self.U_packets+=unlabeled_count
        if optimizer_step not in d.PROBE_STEPS: return
        pending=self.pending_aux
        if pending is None or pending['step']!=optimizer_step or self.pending_update is not None:
            raise ValueError('Actual auxiliary probe alignment differs')
        try:
            params=[p for group in optimizer.param_groups for p in group['params']]
            names_by_id={id(p):name for name,p in model.named_parameters()}; names=[names_by_id[id(p)] for p in params]
            auxiliary=torch.autograd.grad(pending['loss'],params,retain_graph=True,allow_unused=True)
            if any(g is not None for name,g in zip(names,auxiliary) if name not in self.callback.selected_names):
                raise ValueError('Auxiliary connects to protected W/gain or nonencoder parameters')
            norm=_norm(auxiliary)
            if norm>1e-8: self.active_gradient_probe_steps.append(optimizer_step)
            total=torch.autograd.grad(total_loss,params,retain_graph=True,allow_unused=True)
            base=torch.autograd.grad(total_loss-pending['loss'],params,retain_graph=True,allow_unused=True)
            shadow=simulate(optimizer,base,total,names=names,max_grad_norm=5.,step_fn=self.original_step)
            record=dict(step=optimizer_step,epoch=epoch,phase=phase,selected_auxiliary_gradient_norm=norm,
                selected_W_gain_gradients_None=True,protected_parameter_names=[name for name in names if name not in self.callback.selected_names],
                main_base_includes_actual_U=True,actual_unlabeled_loss=float(unlabeled_loss.detach()),
                auxiliary_added_once_via_standard_Lhook=True,callback=pending['callback_record'],
                temporary_AdamW_steps=2,actual_AdamW_increment=_auxiliary_names(shadow['summary']))
            self.actual_probes.append(record); self.pending_update=dict(optimizer=optimizer,params=params,shadow=shadow,record=record)
        finally: self.pending_aux=None

    def _write_report(self,epoch):
        summary=self.callback.summary(); summary.pop('EMA_updates',None)
        value=dict(summary,epoch=epoch,arm=self.c['arm'],counts=dict(summary),
            L_context_steps=self.model._feature_l_steps,optimizer_steps=self.actual_steps,late_calls=self.late_calls,
            U_packet_observations=self.U_packets,auxiliary_calls=summary['scheduled_calls'],
            eligible_calls=summary['eligible_calls'],positive_gap_calls=self.positive_gap_calls,
            optimizer_probe_steps=self.probe_steps,active_gradient_probe_steps=self.active_gradient_probe_steps,
            actual_probes=self.actual_probes,extra_temporary_AdamW_steps=2*len(self.probe_steps),
            label_epochs=self.c['label_epochs'],pseudo_epochs=self.c['pseudo_epochs'],baseline_recipe=self.c['training_recipe'],
            default_head_joint_training=True,original_L_loader_unchanged=True,original_U_sampler_unchanged=True,
            BASE_CE_U_length_only=self.c['arm']=='BASE_CE',own_EMA_enabled=self.c['execution_overrides']['use_ema_teacher'],
            main_base_includes_actual_U=True,auxiliary_added_once_via_standard_Lhook=True,
            teacher_callbacks_installed=False,extra_EMA_updates=0,head_fit_updates=0,extra_actual_optimizer_steps=0,
            U_truth_read=False,target_access=False,source_Sinc_initial_final=self.final_sinc,
            previously_trained_source_L=True,never_trained_holdout=False,
            U_strong_view='Original oldgate Gaussian strong view retained in SSL arms; ordinary IQ augmentation disabled')
        d.write(self.out/'source_training_mechanism.json',value)

    def epoch(self,model,ctx,rows,out,device):
        epoch=len(rows); expected=epoch*d.STEPS
        if (model is not self.model or self.pending_aux is not None or self.pending_update is not None
                or self.actual_steps!=expected or self.late_calls!=expected
                or self.callback.counts['calls']!=expected or model._feature_l_steps!=expected):
            raise ValueError('Actual optimizer/L context/callback budget or pending observation differs')
        expected_calls=0 if self.c['arm'] in BASELINES else sum(d.active(step) for step in range(1,expected+1))
        if (self.callback.counts['scheduled_calls']!=expected_calls
                or self.callback.counts['extra_IQ_forwards']!=expected_calls*180):
            raise ValueError('Fixed scheduled responsibility/L180 forward budget differs')
        if self.U_packets!=max(0,epoch-self.c['label_epochs'])*56700: raise ValueError('Full original U epoch exposure differs')
        if epoch==d.EPOCHS:
            if expected_calls!=d.scheduled(self.c) or self.probe_steps!=list(d.PROBE_STEPS):
                raise ValueError('Final scheduled auxiliary or four actual probe budget differs')
            final=_sinc_snapshot(model); initial=self.initial_sinc; change=final['filters']-initial['filters']
            self.final_sinc=dict(kernel_count=24,changed_kernels=int((change.flatten(1)!=0).any(1).sum()),
                kernel_change_norm=float(change.double().norm()),kernel_change_max=float(change.abs().max()),
                low_hz_change_max=float((final['low_hz']-initial['low_hz']).abs().max()),
                band_hz_change_max=float((final['band_hz']-initial['band_hz']).abs().max()),
                initial_artifact='source_sinc_initial.pt',final_artifact='source_sinc_final.pt')
            torch.save(final,self.out/'source_sinc_final.pt')
        self._write_report(epoch)

    @contextmanager
    def native_scope(self,native):
        self.native=native; core,step=native.compute_core_losses,torch.optim.AdamW.step; self.original_step=step
        def losses_wrapper(*args,**kwargs):
            result=core(*args,**kwargs)
            self._bind()  # Native own EMA deepcopy precedes the first real L core loss.
            return result
        def actual_step(optimizer,*args,**kwargs):
            result=step(optimizer,*args,**kwargs)
            if optimizer is self.optimizer: self.actual_steps+=1
            pending=self.pending_update
            if pending is not None and optimizer is pending['optimizer']:
                expected=pending['shadow']['total_after']; atol,rtol=2e-7,2e-6
                per_element=all(torch.allclose(p.detach(),q,atol=atol,rtol=rtol) for p,q in zip(pending['params'],expected))
                error=max(float((p.detach()-q).abs().max()) for p,q in zip(pending['params'],expected))
                legacy_tolerance=max(float((atol+rtol*q.abs()).max()) for q in expected)
                if not per_element: raise ValueError('Once-only actual AdamW differs from shadow per element')
                pending['record']['actual_update']=dict(actual_optimizer_steps=1,actual_shadow_max_abs_error=error,
                    actual_shadow_FP32_tolerance=legacy_tolerance,actual_shadow_per_element_allclose=True,
                    actual_shadow_per_element_atol=atol,actual_shadow_per_element_rtol=rtol,
                    bitwise_equality_claimed=False,main_base_includes_actual_U=True)
                self.probe_steps.append(pending['record']['step'])
                with (self.out/'actual_responsibility_optimizer_increment.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(pending['record'],allow_nan=False)+'\n')
                self.pending_update=None
            return result
        native.compute_core_losses,torch.optim.AdamW.step=losses_wrapper,actual_step
        try: yield
        finally:
            native.compute_core_losses,torch.optim.AdamW.step=core,step
            if self.bound:
                delattr(self.model,'_source_labeled_feature_hook'); delattr(self.model,'_source_optimizer_update_probe_hook')
            self.bound=False; self.pending_aux=self.pending_update=None; self.native=None
