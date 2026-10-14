"""Fixed L bank, standard L callback and four actual-step observations only."""
from contextlib import contextmanager
from pathlib import Path
import json

import torch

from . import design as d, model as model_api
from experiments.cvs_crossrx_sharpness50.observe import Observer as LCallback
from experiments.cvs_crossrx_sharpness50.objective import select_names
from experiments.cvs_sigtime_heldcredit.coordinate import PREFIX
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches, physical_ids
from experiments.cvs_u_schedule50.counterfactual import simulate, _preserve_rng


class LengthOnlyUnlabeled:
    """BASE_CE keeps the original222 budget without fetching any U packet."""
    def __init__(self, loader):
        if len(loader) != d.STEPS:
            raise ValueError('Original U length differs from222')
        self.original = loader
        self.dataset = loader.dataset
        self.sampler = getattr(loader, 'sampler', None)
    def __len__(self): return d.STEPS
    def __iter__(self): raise RuntimeError('BASE_CE must never iterate U')


def build_bank(dataset):
    """Read only180 clean L packets, ordinal68/69 of each original TX/RX/day cell."""
    if len(dataset) != 6300 or getattr(dataset, 'hide', True):
        raise ValueError('Bank requires the actual visible source L6300 dataset')
    cells, metadata = {}, []
    for index, item in enumerate(dataset.index):
        y, rx, day = int(item.tx_i), int(item.rx_i), int(item.day_i)
        identity = (int(item.eq_i), rx, day, int(item.sig_i), int(dataset.selected[index]))
        if y not in range(6) or rx not in (1,3,4,6,8) or day not in (1,2,3):
            raise ValueError('Bank metadata is outside visible source TX/RX/day')
        metadata.append((y, identity)); cells.setdefault((y,rx,day), []).append(index)
    physical_ids([identity for _,identity in metadata], 6300)
    if len(cells) != 90 or any(len(indices) != 70 for indices in cells.values()):
        raise ValueError('Actual L dataset must have90 original cells of70 packets')
    selected = [cells[key][ordinal] for key in sorted(cells) for ordinal in (68,69)]
    packets, labels, receivers, identities = [], [], [], []
    with _preserve_rng():
        for index in selected:
            x, label, _, meta = dataset[index]
            y, identity = metadata[index]
            actual = tuple(int(meta[key]) for key in ('eq_i','rx_i','day_i','sig_i','base_index'))
            if actual != identity or int(label) != y:
                raise ValueError('Clean source bank sample and original physical metadata disagree')
            if x.shape != (2,256) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):
                raise ValueError('Clean source bank IQ shape/finite contract differs')
            packets.append(x.detach().cpu()); labels.append(y); receivers.append(identity[1]); identities.append(identity)
    return dict(x=torch.stack(packets), y=torch.tensor(labels), rx=torch.tensor(receivers),
                physical_ids=identities, role='L_s', dataset_indices=selected)


def _sinc_snapshot(model):
    sinc = model.get_submodule(PREFIX); flags = [(m,m.training) for m in sinc.modules()]
    with fresh_parameter_caches(sinc), torch.no_grad():
        try:
            sinc.eval(); physical = sinc.low_hz_
            filters = sinc._filters(physical.device, physical.dtype).detach().cpu().clone()
            low, band = sinc.low_hz_.detach().cpu().clone(), sinc.band_hz_.detach().cpu().clone()
        finally:
            for module, flag in flags: module.training = flag
    if filters.shape != (24,1,79) or not bool(torch.isfinite(filters).all()):
        raise ValueError('Actual primary Sinc kernel contract differs')
    return dict(filters=filters, low_hz=low, band_hz=band)


def _norm(grads):
    value = sum(float(g.detach().double().square().sum()) for g in grads if g is not None)**.5
    if not torch.isfinite(torch.tensor(value)): raise ValueError('Auxiliary probe gradient is nonfinite')
    return value


def _auxiliary_names(value):
    if isinstance(value,dict):
        names = {'U_increment_norm':'aux_increment_norm','U_to_total_step_ratio':'aux_to_total_step_ratio'}
        return {names.get(key,key):_auxiliary_names(item) for key,item in value.items()}
    if isinstance(value,list): return [_auxiliary_names(item) for item in value]
    return value


class TrainingObserver:
    def __init__(self, config):
        d.validate(config); self.c = config
        self.model = self.callback = self.bank = self.native = None
        self.bound = False; self.pending_aux = self.pending_update = None
        self.optimizer = None; self.actual_steps = self.late_calls = self.U_packets = 0
        self.positive_gap_calls = 0; self.actual_probes = []; self.active_gradient_probe_steps = []
        self.probe_steps = []; self.initial_sinc = self.final_sinc = None

    def data(self, native, ctx, out):
        self.bank = build_bank(ctx['train_loader'].dataset)
        if self.c['arm'] == 'BASE_CE':
            if self.c['execution_overrides']['use_unlabeled'] or self.c['execution_overrides']['use_ema_teacher']:
                raise ValueError('BASE_CE must actually disable U and EMA')
            ctx['unlabeled_loader'] = LengthOnlyUnlabeled(ctx['unlabeled_loader'])
        d.write(Path(out)/'sharpness_L_bank.json', dict(role='L_s', bank_count=180,
            ordinals=[68,69], cells=90, packets_per_cell=2, dataset_indices=self.bank['dataset_indices'],
            physical_ids=self.bank['physical_ids'], source_L_count=6300, source_IQ_reads=180,
            U_IQ_reads=0, U_truth_read=False, V_IQ_reads=0, target_access=False))

    def initialize(self, model, config, out):
        if config != self.c or self.bank is None: raise ValueError('Validated L bank must precede student initialization')
        if any(hasattr(model,key) for key in ('_source_labeled_feature_hook','_source_optimizer_update_probe_hook')):
            raise ValueError('Only an unbound own scratch student is permitted')
        self.model, self.out = model, Path(out)
        names = select_names(model)
        if len(names) != 17: raise ValueError('Actual fixed feature scope must have17 parameter tensors')
        weight = model_api.classifier(model).weight
        matching = [name for name,p in model.named_parameters() if p is weight]
        if len(matching) != 1: raise ValueError('Default W parameter traversal differs')
        self.callback = LCallback(self.native, self.c['training_mechanism'], self.bank,
                                 default_weight_name=matching[0], metrics_path=self.out/'sharpness_training_steps.jsonl')
        self.initial_sinc = _sinc_snapshot(model)
        torch.save(self.initial_sinc, self.out/'source_sinc_initial.pt')

    def _bind(self):
        if self.bound: return
        if self.model is None or self.callback is None: raise ValueError('Student/L callback initialization incomplete')
        self.model._source_labeled_feature_hook = self.auxiliary
        self.model._source_optimizer_update_probe_hook = self.probe
        self.bound = True

    def auxiliary(self, **kwargs):
        if kwargs['model'] is not self.model or self.model._feature_l_steps != kwargs['optimizer_step']:
            raise ValueError('Auxiliary current main L context/step mismatch')
        if kwargs['labeled_count'] != 128: raise ValueError('Original main L128 sampler must remain unchanged')
        loss, metrics = self.callback(**kwargs)
        record = self.callback.last_record
        self.positive_gap_calls += int(any(row['positive_gap'] > 0 for row in record['by_RX'].values()))
        if kwargs['optimizer_step'] in d.PROBE_STEPS:
            if self.pending_aux is not None: raise ValueError('Unconsumed actual auxiliary probe')
            self.pending_aux = dict(step=kwargs['optimizer_step'], loss=loss, callback_record=dict(record))
        return loss, metrics

    auxiliary._accepts_labeled_inputs = False

    def probe(self, *, model, optimizer, epoch, batch_idx, phase, optimizer_step, total_loss,
              unlabeled_loss, labeled_count, unlabeled_count):
        if (model is not self.model or self.callback.last_step != optimizer_step
                or model._feature_l_steps != optimizer_step
                or optimizer_step != (epoch-1)*d.STEPS+batch_idx):
            raise ValueError('Actual main L/late global step mismatch')
        expected_phase = 'label' if epoch <= self.c['label_epochs'] else 'pseudo'
        expected_u = 0 if expected_phase == 'label' else 124 if batch_idx == d.STEPS else 256
        if phase != expected_phase or unlabeled_count != expected_u or labeled_count != 128:
            raise ValueError('Declared native L/U stage/exposure differs')
        if self.optimizer is not None and optimizer is not self.optimizer: raise ValueError('Actual optimizer changed')
        self.optimizer = optimizer; self.late_calls += 1; self.U_packets += unlabeled_count
        if optimizer_step not in d.PROBE_STEPS: return
        pending = self.pending_aux
        if pending is None or pending['step'] != optimizer_step or self.pending_update is not None:
            raise ValueError('Actual auxiliary probe alignment differs')
        try:
            params = [p for group in optimizer.param_groups for p in group['params']]
            name_by_id = {id(p):name for name,p in model.named_parameters()}
            names = [name_by_id[id(p)] for p in params]
            auxiliary = torch.autograd.grad(pending['loss'], params, retain_graph=True, allow_unused=True)
            if any(g is not None for name,g in zip(names,auxiliary) if name not in self.callback.selected_names):
                raise ValueError('Auxiliary probe connects to W/gain/unselected parameters')
            norm = _norm(auxiliary)
            if norm > 1e-8: self.active_gradient_probe_steps.append(optimizer_step)
            total = torch.autograd.grad(total_loss, params, retain_graph=True, allow_unused=True)
            base = torch.autograd.grad(total_loss-pending['loss'], params, retain_graph=True, allow_unused=True)
            shadow = simulate(optimizer, base, total, names=names, max_grad_norm=5., step_fn=self.original_step)
            record = dict(step=optimizer_step, epoch=epoch, phase=phase,
                selected_auxiliary_gradient_norm=norm, selected_W_gain_gradients_None=True,
                protected_parameter_names=[name for name in names if name not in self.callback.selected_names],
                main_base_includes_actual_U=True, actual_unlabeled_loss=float(unlabeled_loss.detach()),
                auxiliary_added_once_via_standard_Lhook=True, callback=pending['callback_record'],
                temporary_AdamW_steps=2, actual_AdamW_increment=_auxiliary_names(shadow['summary']))
            self.actual_probes.append(record)
            self.pending_update = dict(optimizer=optimizer, params=params, shadow=shadow, record=record)
        finally:
            self.pending_aux = None

    def _write_report(self, epoch):
        summary = self.callback.summary()
        summary.pop('EMA_updates',None)
        value = dict(summary, epoch=epoch, arm=self.c['arm'], counts=dict(summary),
            L_context_steps=self.model._feature_l_steps, optimizer_steps=self.actual_steps,
            late_calls=self.late_calls, U_packet_observations=self.U_packets,
            auxiliary_calls=summary['scheduled_calls'], eligible_calls=summary['eligible_calls'],
            positive_gap_calls=self.positive_gap_calls,
            optimizer_probe_steps=self.probe_steps, active_gradient_probe_steps=self.active_gradient_probe_steps,
            actual_probes=self.actual_probes, extra_temporary_AdamW_steps=2*len(self.probe_steps),
            label_epochs=self.c['label_epochs'], pseudo_epochs=self.c['pseudo_epochs'],
            baseline_recipe=self.c['training_recipe'], default_head_joint_training=True,
            original_L_loader_unchanged=True, original_U_sampler_unchanged=True,
            BASE_CE_U_length_only=self.c['arm']=='BASE_CE',
            own_EMA_enabled=self.c['execution_overrides']['use_ema_teacher'],
            main_base_includes_actual_U=True, auxiliary_added_once_via_standard_Lhook=True,
            teacher_callbacks_installed=False, extra_EMA_updates=0,
            head_fit_updates=0, extra_actual_optimizer_steps=0, U_truth_read=False, target_access=False,
            source_Sinc_initial_final=self.final_sinc,
            previously_trained_source_L=True, never_trained_holdout=False,
            U_strong_view='original oldgate Gaussian view retained in SSL arms; use_augFalse only disables ordinary IQ augmentation')
        d.write(self.out/'source_training_mechanism.json',value)

    def epoch(self, model, ctx, rows, out, device):
        epoch = len(rows); expected = epoch*d.STEPS
        if (model is not self.model or self.pending_aux is not None or self.pending_update is not None
                or self.actual_steps != expected or self.late_calls != expected
                or self.callback.counts['calls'] != expected or model._feature_l_steps != expected):
            raise ValueError('Actual optimizer/L context/callback budget or pending observation differs')
        expected_calls = sum(d.active(step) for step in range(1,expected+1))
        if self.callback.counts['scheduled_calls'] != expected_calls: raise ValueError('Scheduled auxiliary budget differs')
        expected_u = max(0,epoch-self.c['label_epochs'])*56700
        if self.U_packets != expected_u: raise ValueError('Original full U epoch exposure differs')
        if epoch == d.EPOCHS:
            final = _sinc_snapshot(model); initial = self.initial_sinc
            change = final['filters']-initial['filters']
            self.final_sinc = dict(kernel_count=24,
                changed_kernels=int((change.flatten(1)!=0).any(1).sum()),
                kernel_change_norm=float(change.double().norm()), kernel_change_max=float(change.abs().max()),
                low_hz_change_max=float((final['low_hz']-initial['low_hz']).abs().max()),
                band_hz_change_max=float((final['band_hz']-initial['band_hz']).abs().max()),
                initial_artifact='source_sinc_initial.pt', final_artifact='source_sinc_final.pt')
            torch.save(final,self.out/'source_sinc_final.pt')
        self._write_report(epoch)

    @contextmanager
    def native_scope(self, native):
        self.native = native
        core, step = native.compute_core_losses, torch.optim.AdamW.step
        self.original_step = step
        def losses(*args,**kwargs):
            result = core(*args,**kwargs)
            # Native EMA deepcopy is earlier than the first main core call.
            self._bind()
            return result
        def actual_step(optimizer,*args,**kwargs):
            result = step(optimizer,*args,**kwargs)
            if optimizer is self.optimizer: self.actual_steps += 1
            pending = self.pending_update
            if pending is not None and optimizer is pending['optimizer']:
                expected = pending['shadow']['total_after']
                error = max(float((p.detach()-q).abs().max()) for p,q in zip(pending['params'],expected))
                scale = max(1.,max(float(p.detach().abs().max()) for p in pending['params']))
                tolerance = 1e-6*scale
                if error > tolerance: raise ValueError('Auxiliary shadow differs from actual once-only AdamW')
                pending['record']['actual_update'] = dict(actual_optimizer_steps=1,
                    actual_shadow_max_abs_error=error, actual_shadow_FP32_tolerance=tolerance,
                    bitwise_equality_claimed=False, main_base_includes_actual_U=True)
                self.probe_steps.append(pending['record']['step'])
                with (self.out/'actual_sharpness_optimizer_increment.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(pending['record'],allow_nan=False)+'\n')
                self.pending_update = None
            return result
        native.compute_core_losses, torch.optim.AdamW.step = losses, actual_step
        try: yield
        finally:
            native.compute_core_losses, torch.optim.AdamW.step = core, step
            if self.bound:
                delattr(self.model,'_source_labeled_feature_hook')
                delattr(self.model,'_source_optimizer_update_probe_hook')
            self.bound = False; self.pending_aux = self.pending_update = None; self.native = None
