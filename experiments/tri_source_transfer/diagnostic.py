"""Paired component removals on fixed source checkpoints and source L references."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from experiments.pamf_ssl_research.source_branch_gradients import _relation, _sum, component_closure
from experiments.pamf_ssl_research.source_optimizer_shadow import shadow_step
from experiments.pamf_ssl_research.u_gradprotect_objective import projected_gradients
from experiments.pamf_repl_weight.source_adamw_diagnostic import capture_guard_derivatives, difference

RUN = '20261116-diagnostic-tri-source-transfer-m6-s2026092701-r01'
SEED = 2026092701
SETTINGS = ('main_split', 'source_rx_2', 'source_rx_3', 'source_rx_4', 'source_rx_6', 'ten_class')
PARTS = ('pseudo_CE', 'pair_KL', 'shallow', 'joint', 'extrap_delta')
GROUPS = ('t2', 'cls_head', 'other')
ARMS = ('L_only', 'full_U', *(f'without_{p}' for p in PARTS), 'ordinary_PL_at_Tri')


def validate_config(c):
    if any(c.get(k) for k in ('truth', 'target_truth', 'target_inputs', 'views_root', 'tests', 'dataset_path', 'teacher_checkpoint')):
        raise ValueError('Only original source handles are allowed')
    if (c['model_seed'] != SEED or c['method_id'] != 'pamf_t1_e50' or c['ablation'] != 'F21'
            or c['satellite_concat'] or c['own_EMA'] or c['native_U_gate'] or c['posthoc_head_fit']
            or c['hard_pl']['weight'] != 6 or c['hard_pl']['threshold'] != .95
            or not c['hard_pl_no_uas'] or c['hard_pl_gradient_guard']['U_to_L_norm_cap'] != 1):
        raise ValueError('Original Tri recipe is required')
    row = c['paper_row']
    if row['structure'] != 'full' or row['signature_views'] != 3 or not row['permuted']:
        raise ValueError('Fixed full three-view permutation is required')


def choose_indices(metadata, *, allowed_rx, n, seed, classes=None, forbidden=()):
    allowed = np.isin(metadata['rx_i'], list(allowed_rx))
    allowed &= ~np.isin(np.arange(len(allowed)), list(forbidden))
    rng = np.random.default_rng(seed)
    if classes is None:
        candidates = np.flatnonzero(allowed)
        if len(candidates) < n:
            raise ValueError('Insufficient distinct source records')
        return rng.choice(candidates, n, replace=False)
    selected = []
    for label in range(classes):
        candidates = np.flatnonzero(allowed & (metadata['y'] == label))
        needed = n // classes + int(label < n % classes)
        if len(candidates) < needed:
            raise ValueError('Insufficient labeled source class records')
        selected.extend(rng.choice(candidates, needed, replace=False).tolist())
    rng.shuffle(selected)
    return np.array(selected)


def batch(dataset, indices):
    return torch.utils.data.default_collate([dataset[int(i)] for i in indices])


def check_partition(labeled, unlabeled, li, ui, ri, held, mode):
    lid, uid = labeled.metadata['ids'], unlabeled.metadata['ids']
    if set(lid[li]) & set(lid[ri]) or set(uid[ui]) & set(lid[ri]):
        raise ValueError('Reference/update physical records overlap')
    if not np.all(labeled.metadata['rx_i'][ri] == held):
        raise ValueError('Wrong reference receiver')
    if mode == 'rx_held_out' and (held in labeled.metadata['rx_i'][li] or held in unlabeled.metadata['rx_i'][ui]):
        raise ValueError('Held receiver entered the update')
    if mode not in ('rx_held_out', 'sample_disjoint_two_rx'):
        raise ValueError('Unregistered reference mode')


def decompose(o0, o1, m0, m1, r0, r1, y, accepted, physical, aux, weight):
    from experiments.pamf_ssl_research.hard_pl_objective import masked_pair
    o = masked_pair(o0, o1, y, accepted, 1.)
    m = masked_pair(m0[~physical], m1[~physical], y[~physical], accepted[~physical], 1.)
    a = masked_pair(m0[physical], m1[physical], y[physical], accepted[physical], 0.)
    r = masked_pair(r0, r1, y, accepted, 1.)
    scale = weight / 1.25
    return {'pseudo_CE': scale * (.5*o[1] + .25*m[1] + .25*a[1] + .25*r[1]),
            'pair_KL': scale * (.5*o[2] + .25*m[2] + .25*r[2]), 'shallow': scale*.1*aux}


@contextmanager
def component_capture():
    from experiments.pamf_ssl_research import hard_pl_objective as base
    from experiments.pamf_ssl_research import hard_pl_joint_objective as joint
    from experiments.pamf_ssl_research import hard_pl_receiver_extrapolation_objective as rx
    old = base.mechanism_loss, base.HardPLObjective.loss, joint.JointNoAsMixStyle.loss, rx.ReceiverExtrapolation.loss
    active = {}
    def base_loss(self, model, device):
        active['objective'] = self
        try:
            value = old[1](self, model, device)
            self._base_value = value
            return value
        finally:
            active.clear()
    def mechanism(*args):
        obj = active['objective']
        obj.components = decompose(*args, obj.settings['weight'])
        return old[0](*args)
    def joint_loss(self, model, device):
        value = old[2](self, model, device)
        self.components['joint'] = value - self._base_value
        self._joint_value = value
        return value
    def rx_loss(self, model, device):
        value = old[3](self, model, device)
        self.components['extrap_delta'] = value - self._joint_value
        return value
    base.mechanism_loss, base.HardPLObjective.loss = mechanism, base_loss
    joint.JointNoAsMixStyle.loss, rx.ReceiverExtrapolation.loss = joint_loss, rx_loss
    try:
        yield
    finally:
        base.mechanism_loss, base.HardPLObjective.loss, joint.JointNoAsMixStyle.loss, rx.ReceiverExtrapolation.loss = old


class Objective(GuardedReceiverExtrapolation):
    def unlabeled_student_forward(self, model, x, domain, pseudo, accepted, metadata):
        self.pseudo, self.accepted_mask = pseudo.detach().clone(), accepted.detach().clone()
        return super().unlabeled_student_forward(model, x, domain, pseudo, accepted, metadata)

    def guide(self, labeled, model):
        self.native_graph_ready = time.perf_counter()
        pending = self.pending_gradient_loss
        torch.testing.assert_close(sum(self.components.values()), pending)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        with capture_guard_derivatives() as vectors:
            value = super().guide(labeled, model)
        if len(vectors) != 2:
            raise ValueError('Exactly the native L/U gradients required')
        self.L, self.U = vectors
        self.native_grad_done = time.perf_counter()
        self.part_gradients = {k: tuple(None if g is None else g.detach().clone() for g in
            torch.autograd.grad(self.components[k], [p for _, p in named], retain_graph=True, allow_unused=True)) for k in PARTS}
        self.closure = component_closure(named, _sum(named, *self.part_gradients.values()), self.U)
        self.component_seconds = time.perf_counter() - self.native_grad_done
        self.loss_values = {k: float(v.detach()) for k, v in self.components.items()}
        return value


def source_folder(project, setting):
    if setting == 'main_split':
        return project/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F'/f'main_split-repl-shuffled_time-3view-s{SEED}'/'source'
    return project/'runs/20261027-phase1-pamf-paper-completion-m119-r03/F'/f'{setting}-factorial-ssl-full-permuted-s{SEED}'/'source'


def parent_config(folder, epoch):
    from experiments.pamf_branch_training.source import checked_e20_parent
    done = read(folder/'complete.json')
    c = deepcopy(done['config'])
    if (done['status'] != 'SOURCE_FROZEN' or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
            or done['epoch'] != 50):
        raise ValueError('Frozen source-only original model required')
    if 'paper_row' not in c:
        view = c['view_comparison']
        if view.get('view_count') != 3:
            raise ValueError('Main source is not the three-view model')
        c['paper_row'] = dict(structure='full', signature_views=3, permuted=True)
    validate_config(c)
    checked_e20_parent(dict(c, shared_prefix=str(folder/'segment20')))
    contract = read(c['source_contract'])
    selected = folder/f'segment{epoch}'
    segment = read(selected/'complete.json')
    sc = segment['config']
    if (segment['status'] != 'SOURCE_FROZEN' or segment['target_access'] or segment['U_truth_read'] or segment['head_fit_updates']
            or read(sc['source_contract']) != contract or sc['run_id'] != c['run_id'] or sc['row_id'] != c['row_id']
            or sc['model_seed'] != SEED):
        raise ValueError('Checkpoint provenance mismatch')
    cp = torch.load(selected/'final_ssdg.pth', map_location='cpu', weights_only=False)
    expected_epochs = 20 if epoch == 20 else 10
    if (cp['run_id'] != c['run_id'] or cp['candidate_id'] != c['row_id'] or cp['epoch'] != expected_epochs
            or cp['a1_ema_successful_updates'] != expected_epochs*222 or cp['args']['teacher_ckpt']
            or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher']):
        raise ValueError('Checkpoint payload provenance mismatch')
    if epoch == 30 and (segment['initialization'] != 'own_previous_segment'
            or segment['ancestors'] != [str(folder/'segment20')]
            or cp['args']['baseline_ckpt'] != str(folder/'segment20/final_ssdg.pth')):
        raise ValueError('E30 must inherit only its own matching E20')
    c.update(segment_epochs=10 if epoch == 20 else 20, epoch_offset=epoch, step_offset=epoch*222,
             parent_source=str(selected))
    return c, contract, cp


def build(c, cp):
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    with f_train.class_dimension(c), native_scope(f_train.original_config(SEED)) as native:
        args = f_train.arguments(c, 'cpu')
        model = t1_model.install(native.build_baseline_model(args, 'cpu'), c)
        base = native.build_aug_base_cfg(args)
        aug = native.make_augmentor(base)
        prior = SimpleNamespace(**cp['args'])
        prior.epochs, prior.label_epochs, prior.pseudo_epochs = 20, 13, 7
        native.configure_augmentor_for_epoch(aug, base, 13, prior)
    model.load_state_dict(cp['model'], strict=True)
    return model, args, aug


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        out = model(x)
        ce = F.cross_entropy(out, y, reduction='none')
    if not torch.isfinite(out).all() or not torch.isfinite(ce).all():
        raise ValueError('Nonfinite reference predictions')
    return dict(ce=float(ce.mean()), correct=int((out.argmax(1)==y).sum()), n=len(y),
                by_class={str(int(k)): float(ce[y==k].mean()) for k in y.unique()})


def snapshot(c, cp, datasets, held, mode, nonce):
    from experiments.pamf_paper_completion.model import installed as model_scope
    from experiments.cvs_phase1_repair.design import cosine_lr
    labeled, unlabeled = datasets
    classes = c['num_classes']
    allowed = c['source_rxs'] if mode == 'sample_disjoint_two_rx' else [r for r in c['source_rxs'] if r != held]
    ri = choose_indices(labeled.metadata, allowed_rx=[held], n=8*classes, seed=SEED+held, classes=classes)
    li = choose_indices(labeled.metadata, allowed_rx=allowed, n=128, seed=nonce, classes=classes, forbidden=ri)
    ui = choose_indices(unlabeled.metadata, allowed_rx=allowed, n=256, seed=nonce+1)
    check_partition(labeled, unlabeled, li, ui, ri, held, mode)
    lb, ub, rb = batch(labeled, li), batch(unlabeled, ui), batch(labeled, ri)
    if not bool((ub[1] == -1).all()):
        raise ValueError('U truth exposed')
    holder = {}
    torch.manual_seed(nonce)
    wall_start = time.perf_counter()
    with model_scope(c, holder, ssl_enabled=True, objective_factory=Objective, validate=validate_config), component_capture():
        model, args, augmentor = build(c, cp)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        params = [p for _, p in named]
        before = evaluate(model, rb[0], rb[1])
        start = time.perf_counter()
        model.eval()
        reference_loss = F.cross_entropy(model(rb[0]), rb[1])
        rg = tuple(None if v is None else v.detach() for v in torch.autograd.grad(reference_loss, params, allow_unused=True))
        ref_grad_seconds = time.perf_counter()-start
        model.load_state_dict(cp['model'], strict=True)
        model.train()
        model._cd_U_loader = [ub]
        model._cd_U_iter = iter([ub])
        x, y, domain, meta = lb
        start = time.perf_counter()
        ordinary = augmentor(x, labels=y, no_pa=not bool(args.aug_enable_pa_normal))
        if not torch.isfinite(ordinary).all():
            raise ValueError('Nonfinite original L augmentation')
        with model._phase1_labeled_feature_context():
            output = model(ordinary, return_aux=True, domain_labels=domain)
        ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
        correction, _ = model._source_labeled_feature_hook(model=model, labeled_output=output, labels=y,
            extra=(None, meta), domains=domain, optimizer_step=1, identity_loss=ce, labeled_count=128, labeled_inputs=x)
        obj = holder['objective']
        bridge_start = time.perf_counter()
        (ce+correction).backward()
        bridge_seconds = time.perf_counter()-bridge_start
        native_gradient_seconds = obj.native_grad_done-start+bridge_seconds
        components = obj.part_gradients
        full_grad, full_details = projected_gradients(named, obj.L, obj.U)
        for (_, p), g in zip(named, full_grad):
            if (p.grad is None) != (g is None):
                raise ValueError('Native backward connectivity mismatch')
            if g is not None:
                torch.testing.assert_close(p.grad, g, rtol=1e-5, atol=1e-7)
        state = obj.state_dict()
        if state['calls'] != 1 or state['seen'] != 256 or state['student_views'] != 1536 or state['U_truth_read']:
            raise ValueError('Exactly one complete blind U computation required')
        # Plain PL is a one-step objective comparator at the same Tri checkpoint,
        # not an independently trained PL model. It has no REPL guard or style.
        model.load_state_dict(cp['model'], strict=True)
        model.train()
        torch.manual_seed(nonce+20000)
        start = time.perf_counter()
        pl_logits = model.forward_identity_only(ub[0], domain_labels=ub[2])['tx_logits']
        pl = (F.cross_entropy(pl_logits, obj.pseudo, reduction='none')*obj.accepted_mask).mean()
        plg = tuple(None if v is None else v.detach() for v in torch.autograd.grad(pl, params, allow_unused=True))
        pl_seconds = time.perf_counter()-start
        gradients = {'L_only': obj.L, 'full_U': full_grad}
        for key in PARTS:
            remaining = difference(obj.U, components[key], named)
            gradients['without_'+key] = projected_gradients(named, obj.L, remaining)[0]
        gradients['ordinary_PL_at_Tri'] = _sum(named, obj.L, plg)
        model.load_state_dict(cp['model'], strict=True)
        options = dict(base_lr=float(cosine_lr(float(args.lr), 1, args.epochs, float(args.phase1_lr_min))),
                       weight_decay=float(args.weight_decay))
        arms = {}
        for arm in ARMS:
            model.load_state_dict(cp['model'], strict=True)
            start = time.perf_counter()
            descent, metadata = shadow_step(model, named, gradients[arm], **options)
            step_seconds = time.perf_counter()-start
            with torch.no_grad():
                for (_, p), d in zip(named, descent):
                    p.sub_(d)
            start = time.perf_counter()
            score = evaluate(model, rb[0], rb[1])
            arms[arm] = dict(reference=score, optimizer_seconds=step_seconds,
                             reference_forward_seconds=time.perf_counter()-start,
                             reference_first_order=sum(_relation(named, rg, descent, g)['dot'] for g in GROUPS),
                             optimizer=metadata)
        base_ce, full_ce = arms['L_only']['reference']['ce'], arms['full_U']['reference']['ce']
        for arm, value in arms.items():
            value['benefit_vs_L_only'] = base_ce-value['reference']['ce']
            value['benefit_of_included_component'] = value['reference']['ce']-full_ce if arm.startswith('without_') else None
        model.load_state_dict(cp['model'], strict=True)
        after = evaluate(model, rb[0], rb[1])
        if before != after:
            raise ValueError('Restored reference prediction differs')
        result = dict(held_rx=int(held), reference_mode=mode, reference=before, update_rxs=allowed,
            L_n=128, U_n=256, reference_n=len(ri), actual_U_accepted=state['accepted'],
            U_contrast_anchors=state['contrast_anchors'],
            U_extrapolation_eligible=state['U_receiver_extrapolation']['eligible_packet_views'],
            component_losses=obj.loss_values, component_closure=obj.closure, original_guard=full_details,
            component_vs_reference={k:{g:_relation(named, rg, v, g) for g in GROUPS} for k,v in components.items()},
            arms=arms, timings=dict(native_graph_and_gradient_seconds=native_gradient_seconds,
                component_gradient_seconds=obj.component_seconds, reference_gradient_seconds=ref_grad_seconds,
                plain_PL_gradient_seconds=pl_seconds, total_snapshot_seconds=time.perf_counter()-wall_start),
            physical_reference_update_disjoint=True, U_truth_reads=0, V_reads=0, target_reads=0,
            checkpoint_parameter_writes=0, persisted_optimizer_updates=0)
        del model, holder, obj, gradients, components, params, named, rg
    gc.collect()
    return result


def collect(runtime, output):
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    activate(runtime)
    numerical_policy()
    from experiments.cric_dura_comparison.f_data import SourceDataset
    output = Path(output)
    results = output/'observations.jsonl'
    if results.exists():
        raise FileExistsError('Immutable observations already exist')
    expected = 38
    completed = 0
    with results.open('x', encoding='utf-8') as stream:
        for setting in SETTINGS:
            folder = source_folder(Path(runtime['project']), setting)
            for epoch in (20, 30):
                c, contract, cp = parent_config(folder, epoch)
                datasets = SourceDataset(c['source_cache'], 'L_s', contract), SourceDataset(c['source_cache'], 'U_s', contract)
                rxs = sorted(c['source_rxs'])
                helds = sorted(set((rxs[0], rxs[len(rxs)//2], rxs[-1])))
                for mode in (('rx_held_out','sample_disjoint_two_rx') if setting == 'source_rx_2' else ('rx_held_out',)):
                    for held in helds:
                        print(json.dumps(dict(status='SNAPSHOT_STARTED', setting=setting, epoch=epoch, held_rx=held, mode=mode)), flush=True)
                        value = snapshot(c, cp, datasets, held, mode, SEED+100*held)
                        value.update(setting=setting, source_epoch=epoch, source_checkpoint=str(folder/f'segment{epoch}/final_ssdg.pth'),
                                     model_seed=SEED, source_counts=contract['counts'], source_rx_count=len(rxs))
                        stream.write(json.dumps(value, allow_nan=False)+'\n')
                        stream.flush()
                        completed += 1
                        print(json.dumps(dict(status='SNAPSHOT_COMPLETE', completed=completed, expected=expected,
                                              seconds=value['timings']['total_snapshot_seconds'])), flush=True)
                del datasets, cp
                gc.collect()
    if completed != expected:
        raise ValueError('Incomplete diagnostic matrix')
    done = dict(status='SOURCE_DIAGNOSTIC_COMPLETE', snapshots=completed, disposable_optimizer_steps=completed*len(ARMS),
                model_seed=SEED, source_epochs=[20,30], U_truth_reads=0, V_reads=0, target_reads=0,
                fresh_moments_at_native_segment_boundaries=True, original_models_modified=False)
    (output/'complete.json').write_text(json.dumps(done, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    collect(read(args.runtime), args.output)
