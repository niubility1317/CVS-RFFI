"""Bounded geometry training integration checks; synthetic IQ only, no parent I/O."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import tempfile

import torch
from torch.nn import functional as F

from . import design as d
from .actions import action_contract, load_actions
from .source import (SourceStream, forward_objective, objective, gradient_diagnostics,
                     parameter_groups, learning_rate, save_final_checkpoint, save_completion, SourceIdentityView)
from .receiver import fit_receiver, load_receiver
from .receiver_checks import synthetic_cache
from experiments.cvs_stage2_decouple import design as old_design
from experiments.cvs_stage2_decouple.actions import PhysicalActionNet
from experiments.cvs_stage2_decouple.deterministic_forward import deterministic_identity
from experiments.cvs_stage2_decouple.common import identity_logits


def configs():
    return {a:d.config(next(r for r in d.rows() if r['arm'] == a)) for a in d.ARMS}


def schedule_checks(bank):
    reference = None
    for arm in d.ARMS:
        stream = SourceStream(bank.payload['source_count'], bank, d.SEEDS[0])
        result, counts = [], dict(linear=0, temporal=0, relation=0)
        for step in range(1, 8881):
            indices, relation = stream.step(step)
            result.append(indices)
            if relation is not None:
                assert torch.equal(indices[-32:], relation['indices'])
                counts['relation'] += 1
            if step % 4 == 0:
                counts[d.config(next(r for r in d.rows() if r['arm'] == arm))['schedule']['action_order'][(step//4-1)%4]] += 1
        result = torch.stack(result)
        if reference is None:
            reference = result
        else:
            assert torch.equal(reference, result)
        assert counts == dict(linear=1110, temporal=1110, relation=1110)
    assert learning_rate(444, 3e-5) == 3e-5 and learning_rate(8880, 1e-4) == 1e-6
    return dict(all_8880_batches_identical_across_B0_B3=True, LLTT=True,
        action_batch_last32=True, linear=1110, temporal=1110, relation=1110, learning_rate_boundaries=True)


def objective_checks():
    cs = configs()
    logits = torch.randn(128, 6, requires_grad=True)
    y = torch.arange(128) % 6
    parent = torch.ones(128)*3
    clean = torch.randn(32, 6, requires_grad=True)
    augmented = clean.detach().clone().requires_grad_()
    total, w, p = objective(cs['B0'], logits, y, parent, 888)
    assert torch.equal(total, F.cross_entropy(logits, y))
    assert all(float(v.detach()) == 0 for k,v in w.items() if k != 'ce')
    _, w, p = objective(cs['B1'], logits, y, parent, 444,
        clean_logits=clean, augmented_logits=augmented)
    assert float(p['finite_lt'].detach()) == 0
    grad = torch.autograd.grad(p['finite_lt'], augmented, retain_graph=True)[0]
    assert not grad.any()
    assert torch.equal(w['keep'], .1*p['keep']) and torch.equal(w['finite_lt'], p['finite_lt'])
    _, w, p = objective(cs['B1'], logits, y, parent, 4,
        clean_logits=clean, augmented_logits=augmented)
    assert p['ramp'] == 4/444.
    for step in (1, 2, 3, 5, 7):
        _, w, _ = objective(cs['B3'], logits, y, parent, step)
        assert all(float(v.detach()) == 0 for k,v in w.items() if k != 'ce')
    zero_ce = (logits*0).sum()
    record = gradient_diagnostics(dict(ce=zero_ce, keep=zero_ce), dict(ce=zero_ce, keep=zero_ce),
                                  [dict(name='test', params=[logits])])
    assert record['raw_ratio_keep_test'] is None and record['weighted_ratio_keep_test'] is None
    assert not record['ratios_used_for_weighting']
    return dict(B0_exact_CE=True, weights_01_1_1=True, ramp444=True, zero_action_no_LT_gradient=True,
                no_offschedule_auxiliary=True, CE_tiny_ratio_NA=True, no_ratio_weight_adaptation=True)


def actual_graph(device, bank):
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    cs = configs()
    parent_c = cs['B3']['parent_contract']
    args = old_design.make_args(parent_c, device)
    args.num_domains, args.input_len = 5, 256
    torch.manual_seed(217)
    with numerical_context(d.FULL_FP32_POLICY), installed(parent_c) as native:
        model = native.build_baseline_model(args, torch.device(device))
    groups, _ = parameter_groups(model)
    model.train()
    with torch.no_grad():
        model.id_backbone.encoder.response_gain.fill_(.1)
    actions = load_actions(dict(format_version=2, seed=271, contract=action_contract(),
        networks={k:PhysicalActionNet(k).state_dict() for k in ('linear','temporal')}), device)
    actions_before = deepcopy(actions.networks.state_dict())
    frozen_before = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    domain_calls, identity_calls = [], []
    domain_hook = model.dom_backbone.register_forward_hook(lambda *a:domain_calls.append(1))
    identity_hook = model.id_backbone.register_forward_hook(lambda *a:identity_calls.append(1))
    x = torch.randn(128, 2, 256, device=device)
    y = torch.arange(128, device=device) % 6
    pm = torch.ones(128, device=device)*3
    sample = bank.sample_relations(torch.Generator().manual_seed(127))
    # Synthetic source tags correspond to fixture relationships; this is a graph
    # check, not an experiment or evidence about a lawful parent prediction.
    y[-32:] = bank.payload['source_y'][sample['indices']].to(device)
    global_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None
    total0, _, _, observed0 = forward_objective(cs['B0'], model, x, y, pm, 888, relation=sample, bank=bank)
    assert len(identity_calls) == 1 and observed0['clean_logits'] is None
    # Restore pre-forward randomness for the CE-alone comparison.
    torch.random.set_rng_state(global_before)
    if cuda_before is not None:
        torch.cuda.set_rng_state_all(cuda_before)
    baseline = identity_logits(model, x)
    assert torch.equal(observed0['logits'], baseline) and torch.equal(total0, F.cross_entropy(baseline, y))
    ce_after = torch.random.get_rng_state().clone()
    ce_cuda_after = torch.cuda.get_rng_state_all() if cuda_before is not None else None
    del total0, observed0, baseline
    torch.random.set_rng_state(global_before)
    if cuda_before is not None:
        torch.cuda.set_rng_state_all(cuda_before)
    identity_calls.clear()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    total, weighted, parts, observed = forward_objective(cs['B3'], model, x, y, pm, 888,
        actions=actions, relation=sample, bank=bank)
    assert len(identity_calls) == 3 and not domain_calls
    assert torch.equal(ce_after, torch.random.get_rng_state())
    if cuda_before is not None:
        assert all(torch.equal(a,b) for a,b in zip(ce_cuda_after,torch.cuda.get_rng_state_all()))
    assert observed['action_stats']['selected_positions'] == '96:128'
    assert parts['receiver_details']['variance_gradient_retained']
    rgrad = torch.autograd.grad(weighted['receiver'], groups[0]['params'], retain_graph=True, allow_unused=True)
    rnorm = sum(g.detach().square().sum() for g in rgrad if g is not None).sqrt()
    assert torch.isfinite(rnorm) and rnorm > 0
    gradients = gradient_diagnostics(weighted, {k:parts[k] for k in weighted}, groups)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    total.backward()
    missing = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for g in groups for p in g['params'])
    optimizer.step()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    assert all(torch.equal(p, dict(model.named_parameters())[n]) for n,p in frozen_before.items())
    assert all(torch.equal(p, actions.networks.state_dict()[n]) for n,p in actions_before.items())
    # Walk the real model's final diagnostic path through all eight views and
    # time-dependency probe. A tiny batch must reach the full-V guard, not fail
    # earlier on a missing geometry-config field or incompatible forward.
    from experiments.cvs_feature_disentangle.validation import evaluate_source_stress
    tiny = [(x[:5], y[:5], torch.zeros(5, dtype=torch.long),
             dict(rx_i=torch.tensor([1,3,4,6,8]), day_i=torch.zeros(5, dtype=torch.long)))]
    identity_calls.clear()
    stress_view = SourceIdentityView(model)
    probe_names = [n for n,_ in stress_view.id_backbone.named_modules() if n.split('.')[-1] == 't_proj']
    assert len(probe_names) == 1
    with tempfile.TemporaryDirectory(prefix='geometry-stress-interface-') as directory:
        try:
            evaluate_source_stress(stress_view, tiny, device, cs['B3'], directory)
        except ValueError as error:
            assert str(error) == 'Incomplete source V pressure validation', str(error)
        else:
            raise AssertionError('An incomplete V was incorrectly accepted')
    assert len(identity_calls) == 9 and not domain_calls
    domain_hook.remove()
    identity_hook.remove()
    return dict(actual_random_architecture=True, no_checkpoint_loaded=True, full_batch=128,
        deterministic_clean_batch=32, augmented_batch=32, identity_forward_count=3,
        B0_only_one_CE_forward=True, B0_matches_original_dropout_CE=True,
        mechanisms_leave_CE_RNG_stream_unchanged=True, full160_R_encoder_gradient=float(rnorm),
        actual_model_all_stress_views_reach_expected_full_V_guard=True,
        actual_stress_identity_forwards=9, stress_identity_probe_count=1,
        receiver_variance_differentiable=True, domain_calls=0, frozen_domain_and_actions_unchanged=True,
        missing_trainable_gradients=missing, loss=float(total.detach()),
        gradient_diagnostics=gradients, seconds=time.perf_counter()-started,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None)


class FastStressFixture(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.t_proj = torch.nn.Linear(2,160)
        self.head = torch.nn.Linear(160,6)
        self._stage2_baseline_args = dict(synthetic_fixture=True)
        self.to(device)

    def forward(self, x):
        return self.head(self.id_backbone.t_proj(x.mean(2)))

    def forward_identity_only(self, x):
        return dict(tx_logits=self(x))


class FullSyntheticV:
    def __iter__(self):
        generator = torch.Generator().manual_seed(151)
        for start in range(0,27000,300):
            ids = torch.arange(start,start+300)
            yield (torch.randn(300,2,256,generator=generator), ids%6, ids%5,
                   dict(rx_i=torch.tensor([1,3,4,6,8])[ids%5], day_i=ids%4))


def completion_checks(device):
    """Exercise real source validation, stress, serialization and readback code.

    Temporary synthetic artifacts are removed on return. This does not assert
    that any production model was trained or any physical source was evaluated.
    """
    from experiments.cvs_feature_disentangle.validation import evaluate_source_stress
    from experiments.cvs_feature_disentangle.source import source_validation
    c = configs()['B3']
    model = FastStressFixture(device).eval()
    loader = FullSyntheticV()
    with torch.no_grad():
        validation = source_validation(model, loader, device)
    optimizer = torch.optim.AdamW(model.parameters())
    with tempfile.TemporaryDirectory(prefix='geometry-finalization-') as directory:
        out = Path(directory)/'source'
        out.mkdir()
        checkpoint = save_final_checkpoint(c,model,optimizer,out)
        stress = evaluate_source_stress(SourceIdentityView(model),loader,device,c,out)
        assert len(stress['views']) == 8 and all(v['count'] == 27000 for v in stress['views'].values())
        result = dict(status='SOURCE_TRAINED',epoch=40,optimizer_steps=8880,logged_steps=8880,
                      config=c,checkpoint=str(checkpoint),checkpoint_selection='final_only',
                      target_access=False,synthetic_integration_fixture=True,final_source_metrics=validation)
        save_completion(c,out,result)
        ck = torch.load(checkpoint,map_location='cpu',weights_only=False)
        completion, frozen = d.read(out/'completion.json'), d.read(out.parent/'source_frozen.json')
        assert completion['config'] == frozen['config'] == ck['config'] == c
        assert frozen['checkpoint'] == str(checkpoint) and ck['run_id'] == d.RUN
        assert ck['candidate_id'] == c['row_id'] and ck['parent_checkpoint'] == c['parent_checkpoint']
        assert ck['epoch'] == completion['epoch'] == frozen['epoch'] == 40
        assert ck['optimizer_steps'] == completion['optimizer_steps'] == frozen['optimizer_steps'] == 8880
        assert completion['status'] == 'SOURCE_TRAINED' and frozen['status'] == 'SOURCE_FROZEN'
        assert all(not value['target_access'] for value in (completion,frozen,ck,stress))
    return dict(source_validation_completed=True, full_synthetic_V_count=27000, stress_views=8,
        time_dependency_probe=True, final_checkpoint_strict_reload=True,
        completion_and_frozen_readback=True, temporary_synthetic_artifacts_removed=True,
        production_training_or_source_result_claim=False)


def run(device):
    torch.set_num_threads(2)
    bank = load_receiver(fit_receiver(synthetic_cache(), 237))
    return dict(status='PASS', target_access=False, device=device, objective=objective_checks(),
                schedule=schedule_checks(bank), actual_graph=actual_graph(device, bank),
                completion=completion_checks(device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.device)
    if args.output:
        d.write(args.output, result)
    print(json.dumps(result), flush=True)
