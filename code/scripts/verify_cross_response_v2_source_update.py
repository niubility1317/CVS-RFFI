"""Actual CORE90 controlled source one-step mechanism audit on synthetic IQ.

This is NOT the full SSDG baseline objective or evidence of real-source benefit.
No checkpoint/target is read. Every A/B pair starts at the same state, uses the
same train batch and evaluates disjoint source recordings in four conditions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import torch
import torch.nn.functional as F

from post_stage_common import build_baseline_model
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.cross_response.legacy_losses import covariance_orth_loss
from cvsrffi.cross_response.predictor import SharedResponsePredictor
from cvsrffi.cross_response.readouts import ResponseReadouts
from cvsrffi.cross_response.replay_audit import (
    capture_rng_state, restore_rng_state, disposable_counterfactual,
    first_divergence, grouped_source_risk, snapshot_training_state,
)
from cvsrffi.cross_response.roles import make_roles
from cvsrffi.cross_response.statistics import WaveformStatistics, FixedSourceNormalizer
from cvsrffi.cross_response.training import parameter_roles, response_backward

SCENARIOS = ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')
SOURCE_RXS = (1, 3, 4, 6, 8)


def synthetic_batch(txs, receivers, k, *, seed, namespace, device):
    gen = torch.Generator().manual_seed(seed)
    time = torch.arange(256).float()/256
    rows, labels, domains, ids = [], [], [], []
    for tx in txs:
        for rx in receivers:
            for record in range(k):
                frequency = 5 + 13*tx + .2*rx + .03*tx*rx
                phase = .1*rx + .015*torch.randn((), generator=gen)
                angle = 2*math.pi*frequency*time+phase
                iq = torch.stack((angle.cos(), angle.sin()))
                iq += .12*torch.stack(((angle*1.3).cos(), (angle*1.3).sin()))
                iq += .03*torch.randn(2, 256, generator=gen)
                rows.append(iq)
                labels.append(tx)
                domains.append(SOURCE_RXS.index(rx))
                ids.append((namespace, tx, rx, record))
    return (torch.stack(rows).to(device), torch.tensor(labels, device=device),
            torch.tensor(domains, device=device), ids)


def paired_summary(values):
    x = torch.tensor(values, dtype=torch.float64)
    sd = float(x.std(unbiased=True)) if len(x) > 1 else None
    se = sd/math.sqrt(len(x)) if sd is not None else None
    return {'n_paired_batches': len(x), 'values': values, 'mean': float(x.mean()),
            'sample_sd': sd, 'standard_error': se,
            'range': [float(x.min()), float(x.max())],
            'uncertainty_unit': 'synthetic_paired_batch_not_training_seed',
            'confidence_interval': None,
            'confidence_interval_reason': 'small_fixed_synthetic_fixture_no_population_claim'}


def run(output, *, device='cuda:0', repeats=3, seed=392005):
    if repeats < 3:
        raise ValueError('at least three paired batches required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old_rng = capture_rng_state()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        args = json.loads((CODE/'configs/phase1_core90_cross_response_v1.json').read_text(encoding='utf-8'))['baseline_args']
        args.update(num_classes=6, num_domains=5, input_len=256,
                    use_mixstyle=False, from_scratch=True, baseline_ckpt='')
        model = build_baseline_model(SimpleNamespace(**args), torch.device(device))
        train_x, train_y, train_d, train_ids = synthetic_batch(range(4), SOURCE_RXS[:4], 2,
            seed=seed+1, namespace='warmup', device=device)
        model.eval()
        with torch.no_grad():
            out = model(train_x, y_tx=None, grl_lambda=0., return_aux=True, domain_labels=train_d)
        auxiliary = torch.nn.ModuleDict({'readouts': ResponseReadouts(out['z_id'].shape[1], out['z_dom'].shape[1], 16),
                                         'predictor': SharedResponsePredictor(16, 8, mode='bilinear', rank=4)}).to(device)
        statistics = WaveformStatistics('fft', bands=8, input_length=256).to(device)
        normalizer = FixedSourceNormalizer(8).to(device).fit(statistics(train_x), source_role='source_train')
        optimizer = torch.optim.AdamW(list(model.parameters())+list(auxiliary.parameters()), lr=2e-4, weight_decay=1e-4)
        # Controlled source-only warmup creates actual AdamW moments. It is not
        # inherited training or an externally loaded checkpoint.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(train_x, y_tx=None, grl_lambda=0., return_aux=True, domain_labels=train_d)
        (F.cross_entropy(out['tx_logits'], train_y)+.05*covariance_orth_loss(out['z_id'], out['z_dom'])).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report = {'scope': 'SYNTHETIC_CORE90_CONTROLLED_CE_ORTH_RESPONSE_NOT_FULL_BASELINE',
                  'real_source_benefit': 'UNPROVEN', 'target_read': False,
                  'initialization': 'scratch_plus_one_controlled_source_warmup',
                  'model': 'CORE90_M_lite_d_dual', 'device': device,
                  'precision': 'FP32_deterministic', 'seed': seed, 'repeats': repeats,
                  'train_block': 'P4_Q4_K2_32_records_controlled_fixture_not_formal_batch128',
                  'base_objective': 'full_six_class_CE + 0.05 * legacy_covariance_orth_loss',
                  'omitted_baseline_terms': ['domain_CE', 'GRL_adversarial', 'pseudo', 'EMA', 'Fishr', 'sat_consistency'],
                  'response': {'lambda': .1, 'cap': .1, 'role_policy': 'balanced_partitions_v2'},
                  'source_validation_autograd': False, 'audit_state_promoted': False,
                  'pairs': []}
        for repeat in range(repeats):
            train_x, train_y, train_d, train_ids = synthetic_batch(range(4), SOURCE_RXS[:4], 2,
                seed=seed+100+repeat, namespace=f'train_{repeat}', device=device)
            query_x, query_y, query_d, query_ids = synthetic_batch(range(6), SOURCE_RXS, 1,
                seed=seed+200+repeat, namespace=f'query_{repeat}', device=device)
            probe_x, probe_y, probe_d, probe_ids = synthetic_batch(range(4), SOURCE_RXS[:4], 2,
                seed=seed+300+repeat, namespace=f'probe_{repeat}', device=device)
            channel_generator = torch.Generator(device=device).manual_seed(seed+400+repeat)
            views = {'clean': query_x}
            for scenario in SCENARIOS[1:]:
                views[scenario], _ = apply_sat_channel_for_scenario(query_x, scenario,
                    SimpleNamespace(sat_fs_hz=25e6, sat_fc_hz=2.462e9), gen=channel_generator)
            role = make_roles(tuple(range(4)), tuple(range(4)), repeat, policy='balanced_partitions_v2')
            state = {'auxiliary': auxiliary, 'scaler': torch.amp.GradScaler('cuda', enabled=False),
                     'train_x': train_x, 'train_y': train_y, 'train_d': train_d,
                     'response_target': normalizer(statistics(train_x)).reshape(4,4,2,8).mean(2),
                     'query_views': views, 'query_y': query_y, 'query_d': query_d,
                     'probe_x': probe_x, 'probe_y': probe_y, 'probe_d': probe_d,
                     'role': role}

            def train_step(m, opt, s, include_response):
                m.train()
                opt.zero_grad(set_to_none=True)
                out = m(s['train_x'], y_tx=None, grl_lambda=0., return_aux=True, domain_labels=s['train_d'])
                ce = F.cross_entropy(out['tx_logits'], s['train_y'])
                orth = covariance_orth_loss(out['z_id'], out['z_dom'])
                zi, zd = out['z_id'].reshape(4,4,2,-1), out['z_dom'].reshape(4,4,2,-1)
                r = s['role']
                qt, qr, dt, dr = map(list, (r.query_tx, r.query_rx, r.donor_tx, r.donor_rx))
                t, d = s['auxiliary']['readouts'](zi, zd, qt, qr, dt, dr)
                prediction = s['auxiliary']['predictor'](t, d)
                response = (prediction-s['response_target'][qt][:,qr]).square().mean()
                roles = parameter_roles(m, s['auxiliary'], ('id_backbone.cls_head',))
                logs = response_backward(baseline_loss=ce+.05*orth, identity_loss=ce,
                    response_loss=response, decision_loss=ce*0, cross_loss=ce*0,
                    roles=roles, scaler=s['scaler'], lambda_resp=.1 if include_response else 0.,
                    lambda_dec=0., lambda_cross=0., joint_open=s['route']=='joint_tail', gradient_cap=.1)
                finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for group in opt.param_groups for p in group['params'])
                if not finite:
                    raise ValueError('nonfinite controlled update')
                opt.step()
                return dict(logs, CE=float(ce.detach()), orth=float(orth.detach()), response=float(response.detach()), optimizer_steps=1)

            def evaluate(m, s):
                if torch.is_grad_enabled():
                    raise AssertionError('source validation must not build an autograd graph')
                risk = {}
                for scene in SCENARIOS:
                    out = m(s['query_views'][scene], y_tx=None, grl_lambda=0., return_aux=True, domain_labels=s['query_d'])
                    if out['tx_logits'].requires_grad:
                        raise AssertionError('validation logits must be detached')
                    risk[scene] = grouped_source_risk(out['tx_logits'], s['query_y'],
                        [SOURCE_RXS[int(v)] for v in s['query_d'].tolist()], [scene]*len(s['query_y']))
                return risk

            def probe(m, s):
                m.eval()
                out = m(s['probe_x'], y_tx=None, grl_lambda=0., return_aux=True, domain_labels=s['probe_d'])
                orth = covariance_orth_loss(out['z_id'], out['z_dom'])
                ce = F.cross_entropy(out['tx_logits'], s['probe_y'])
                roles = parameter_roles(m, None, ('id_backbone.cls_head',))
                identity = roles['identity_front']+roles['identity_tail']
                gradients = torch.autograd.grad(orth, [p for _, p in identity], allow_unused=True, retain_graph=True)
                ce_gradients = torch.autograd.grad(ce, [p for _, p in identity], allow_unused=True)
                return {'orth_loss': orth.detach(), 'CE': ce.detach(),
                        'orth_identity_gradients': dict(zip([n for n, _ in identity], gradients)),
                        'CE_identity_gradients': dict(zip([n for n, _ in identity], ce_gradients))}

            for route in ('domain_only', 'joint_tail'):
                state['route'] = route
                before = snapshot_training_state(model, optimizer, extra_state=state)
                result = disposable_counterfactual(model, optimizer, train_step=train_step, evaluate=evaluate,
                    train_physical_ids=train_ids, eval_physical_ids=query_ids,
                    train_role='source_train', eval_role='source_validation', persistent_state=state, probe=probe)
                unchanged = first_divergence(before, snapshot_training_state(model, optimizer, extra_state=state))
                if unchanged is not None:
                    raise AssertionError(f'caller state changed: {unchanged}')
                a, b = result['base'], result['base_response']
                pa, pb = a['indirect_coupling_probe'], b['indirect_coupling_probe']
                gradient_deltas = {n: None if pa['orth_identity_gradients'][n] is None else
                    pb['orth_identity_gradients'][n]-pa['orth_identity_gradients'][n]
                    for n in pa['orth_identity_gradients']}
                delta_norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradient_deltas.values() if g is not None))
                evidence_path = output/f'pair_{repeat}_{route}.pt'
                # Save exact per-parameter updates and next-step probe gradients,
                # but never emit a resumable/promotable audit model checkpoint.
                torch.save({branch: {'parameter_delta': result[branch]['parameter_delta'],
                                     'probe': result[branch]['indirect_coupling_probe']}
                            for branch in ('initial','base','base_response')}, evidence_path)
                pair = {'repeat': repeat, 'route': route, 'caller_state_unchanged': True,
                        'physical_train_eval_disjoint': not bool(set(train_ids)&set(query_ids)),
                        'physical_query_probe_disjoint': not bool(set(query_ids)&set(probe_ids)),
                        'evidence': evidence_path.name,
                        'first_orth_gradient_difference': first_divergence(pa['orth_identity_gradients'], pb['orth_identity_gradients']),
                        'orth_identity_gradient_delta_norm': delta_norm,
                        'first_CE_gradient_difference': first_divergence(pa['CE_identity_gradients'], pb['CE_identity_gradients']),
                        'branches': {branch: {'risk': result[branch]['risk'], 'delta_norm': result[branch]['delta_norm'],
                                             'step_evidence': result[branch]['step_evidence'],
                                             'probe_orth_loss': float(result[branch]['indirect_coupling_probe']['orth_loss'])}
                                     for branch in ('initial','base','base_response')}}
                report['pairs'].append(pair)
                print(json.dumps({'repeat': repeat, 'route': route, 'orth_grad_delta': delta_norm,
                                  'clean_CE_B_minus_A': b['risk']['clean']['overall']['ce']-a['risk']['clean']['overall']['ce']}), flush=True)
                del result, before
        report['paired_uncertainty'] = {}
        for route in ('domain_only', 'joint_tail'):
            pairs = [p for p in report['pairs'] if p['route']==route]
            report['paired_uncertainty'][route] = {scene: paired_summary([
                p['branches']['base_response']['risk'][scene]['overall']['ce']-
                p['branches']['base']['risk'][scene]['overall']['ce'] for p in pairs]) for scene in SCENARIOS}
        report['status'] = 'VERIFIED_SYNTHETIC_MECHANISM_ONLY'
        path = output/'report.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        return path
    finally:
        restore_rng_state(old_rng)
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repeats', type=int, default=3)
    options = parser.parse_args()
    print(run(options.output, device=options.device, repeats=options.repeats))
