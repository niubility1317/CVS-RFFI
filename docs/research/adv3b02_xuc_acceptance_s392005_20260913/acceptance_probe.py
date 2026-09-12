"""Bounded CPU acceptance of existing kernels; NOT an XUC training entrypoint.

Uses synthetic IQ, never datasets, checkpoints, target inputs or remote jobs.
Run with the project's verified ssr-gpu Python. Child processes isolate snapshots.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('E:/type10-7')
X = ROOT / 'code/snapshots/a1_fast_v2_20260909_wt'
U = ROOT / 'code/snapshots/core90_cross_response_v2_20260911_wt'
GAME = ROOT / 'code/snapshots/core90_game_20260911_wt'
DESIGN = ROOT / 'automation_reports/CV-SincNet/adv3b02_xuc_fusion_design_s392005_20260913'
OUT = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_probe():
    sys.path.insert(0, str(U / 'code'))
    import torch
    import torch.nn.functional as F
    from model_dual_cvsincnet import build_dual_model
    from cvsrffi.cross_response.tensor_ops import normalized_identity_interaction_loss
    xmod = load_file('acceptance_x_kernel', X / 'code/cvsrffi/a1_ecrs_cross_rx.py')
    torch.set_num_threads(2)
    torch.manual_seed(392005)
    recipe = json.loads((DESIGN / 'core90_recipe_reference.json').read_text(encoding='utf-8'))['baseline_args']
    supported = set(__import__('inspect').signature(build_dual_model).parameters)
    kw = {k: v for k, v in recipe.items() if k in supported}
    kw.update(num_classes=6, num_domains=15, model_size='M', model_variant='lite_d',
              dataset='wisig', input_len=256, mixstyle_on=recipe['use_mixstyle'])
    model = build_dual_model(**kw).cpu().train()
    bn = [n for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    mix = [n for n, m in model.named_modules() if m.__class__.__name__ == 'MixStyle1D']
    tx = torch.arange(4).view(4, 1, 1).expand(4, 4, 2).reshape(-1)
    rx = torch.arange(4).view(1, 4, 1).expand(4, 4, 2).reshape(-1)
    day = torch.ones_like(tx)
    iq = torch.randn(32, 2, 256)
    # No grid runtime is present here: direct kernel composition only.
    out = model(iq, y_tx=tx, domain_labels=rx, return_aux=True)
    z = out['z_id']
    assert z.shape == (32, 160)
    lx, anchors = xmod.cross_rx_triplet_loss(z, tx, rx, day, torch.zeros_like(tx), margin=.2)
    lu, diag = normalized_identity_interaction_loss(z.reshape(4, 4, 2, -1), min_norm=1e-8)
    assert lu is not None and diag['available'] and anchors == 32
    gx = torch.autograd.grad(.05 * lx, z, retain_graph=True)[0]
    gu = torch.autograd.grad(.01 * lu, z, retain_graph=True)[0]
    assert torch.isfinite(gx).all() and torch.isfinite(gu).all()
    assert gx.norm() > 0 and gu.norm() > 0
    params = list(model.named_parameters())
    def routes(loss):
        gradients = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True)
        active = [n for (n, _), g in zip(params, gradients) if g is not None and bool(g.abs().sum() > 0)]
        return dict(active_parameter_tensors=len(active), active_names=active,
                    identity_active=any(n.startswith('id_backbone.') for n in active),
                    domain_active=any(n.startswith('dom_backbone.') for n in active),
                    adversary_active=any(n.startswith('adv_head.') for n in active))
    rx_route, ru_route = routes(.05 * lx), routes(.01 * lu)
    for route in (rx_route, ru_route):
        assert route['identity_active'] and not route['domain_active'] and not route['adversary_active']
    # A single CE+X+U optimizer update is not the complete CORE90 objective/EG.
    total = F.cross_entropy(out['tx_logits'], tx) + .05 * lx + .01 * lu
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    before = {n: p.detach().clone() for n, p in params}
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    assert torch.isfinite(norm)
    optimizer.step()
    changed = [n for n, p in params if not torch.equal(p.detach(), before[n])]
    assert changed and all(torch.isfinite(p).all() for _, p in params)
    # Unit geometry does not make feature collapse safe: X also has zero gradient
    # at exactly identical unit records despite its positive margin loss.
    collapsed = torch.ones(32, 160, requires_grad=True)
    cx, ca = xmod.cross_rx_triplet_loss(collapsed, tx, rx, day, torch.zeros_like(tx), margin=.2)
    cu, cd = normalized_identity_interaction_loss(collapsed.reshape(4, 4, 2, -1), min_norm=1e-8)
    cg = torch.autograd.grad(.05 * cx + .01 * cu, collapsed)[0]
    assert abs(float(cu.detach())) < 1e-10 and abs(float(cx.detach()) - .2) < 1e-5
    assert cg.norm() < 1e-7
    return dict(status='VERIFIED', scope='synthetic_real_model_kernel_composition_only',
                seed=392005, device='cpu', torch=torch.__version__, input_shape=list(iq.shape),
                model_kwargs=kw, identity_shape=list(z.shape), batchnorm_modules=bn, mixstyle_modules=mix,
                x=dict(loss=float(lx.detach()), valid_anchors=anchors, weighted_feature_gradient_norm=float(gx.norm()), **rx_route),
                u=dict(loss=float(lu.detach()), available=diag['available'], weighted_feature_gradient_norm=float(gu.norm()), **ru_route),
                weighted_gradient_cosine=float(F.cosine_similarity(gx.flatten(), gu.flatten(), dim=0)),
                composed_update=dict(loss=float(total.detach()), preclip_gradient_norm=float(norm), changed_parameter_tensors=len(changed)),
                collapse_counterexample=dict(x_loss=float(cx.detach()), u_loss=float(cu.detach()), weighted_gradient_norm=float(cg.norm()),
                                             conclusion='X+U alone does not prove anti-collapse'),
                full_core90_objective_tested=False, grid_forward_context_tested=False,
                cstar_tested=False, extragradient_transaction_tested=False, training_run_started=False)


def native_probe():
    sys.path[:0] = [str(X / 'code/scripts'), str(X / 'code')]
    from run_a1_fast_v2 import v2_matrix
    matrix = v2_matrix()
    native = {k.removeprefix('--'): v for k, v in matrix['core90_options'].items()}
    core = json.loads((DESIGN / 'core90_recipe_reference.json').read_text(encoding='utf-8'))['baseline_args']
    keys = ['epochs', 'label_epochs', 'pseudo_epochs', 'batch_size', 'unlabeled_batch_size', 'amp',
            'use_muse_ssdg', 'use_a1_r3', 'daot_efficiency_mode', 'lambda_u', 'lr', 'weight_decay',
            'use_fasttrust_rc4', 'rc4_use_anchor', 'from_scratch', 'baseline_ckpt', 'teacher_ckpt']
    return dict(status='VERIFIED', scope='read_only_native_factory_resolution',
                selected_options={k: dict(core90=core.get(k, '<absent>'), native=native.get(k, '<absent>')) for k in keys},
                native_options=native, rows=matrix['rows'], training_run_started=False)


def runtime_probe():
    sys.path[:0] = [str(U / 'tests'), str(U / 'code')]
    import tempfile
    import torch
    from test_cross_response_v2_integration import runtime
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='xuc_u_acceptance_') as folder:
        model, rt, ctx = runtime(Path(folder), 'Ux_normalized')
        config = rt.config
        assert config['identity_interaction_enabled'] and config['interaction_mode'] == 'normalized'
        assert not config['response_enabled'] and not config['decision_enabled']
        x, y, domain, meta = next(iter(ctx['train_loader']))
        plan = rt.begin_batch((domain, meta))
        with rt.forward_context(model, plan, len(x)):
            out = model(x)
        terms = rt.losses(model, out, y, plan)
        assert float(terms[0].detach()) == float(terms[1].detach()) == 0
        assert float(terms[2].detach()) > 0
        base = torch.nn.functional.cross_entropy(out['tx_logits'], y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        optimizer.zero_grad(set_to_none=True)
        rt.backward(model, base, base, terms, torch.amp.GradScaler('cuda', enabled=False))
        changed_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer.step()
        rt.commit(True)
        changed = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), changed_before[n])]
        assert changed and rt.counts['successful_steps'] == 1
        return dict(status='VERIFIED', scope='existing_Ux_normalized_runtime_synthetic_model_only',
                    response_enabled=config['response_enabled'], decision_enabled=config['decision_enabled'],
                    interaction_enabled=config['identity_interaction_enabled'], interaction_mode=config['interaction_mode'],
                    lambda_cross=config['lambda_cross'], cross_loss=float(terms[2].detach()),
                    successful_steps=rt.counts['successful_steps'], blocks=len(plan.blocks),
                    changed_parameters=changed, cstar_or_fusion_tested=False)


def run_all(selected=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    jobs = [
        ('x_unit_tests', X, ['-m', 'pytest', '-o', 'addopts=', '-q', 'tests/test_a1_ecrs_cross_rx.py',
                            '--junitxml=' + str(OUT / 'x_unit_tests.xml')]),
        ('u_normalized_tests', U, ['-m', 'pytest', '-o', 'addopts=', '-q', 'tests/test_cross_response_v2_geometry_gate.py',
                                '-k', 'unit_records or invalid_norm or normalized_candidate or reduced_precision',
                                '--junitxml=' + str(OUT / 'u_normalized_tests.xml')]),
        ('kernel_probe', U, [str(Path(__file__).resolve()), '--mode', 'kernel']),
        ('native_factory', X, [str(Path(__file__).resolve()), '--mode', 'native']),
        ('u_roles_tests', U, ['-m', 'pytest', '-o', 'addopts=', '-q', 'tests/test_cross_response_v2_roles.py',
                             '--junitxml=' + str(OUT / 'u_roles_tests.xml')]),
        ('u_runtime_probe', U, [str(Path(__file__).resolve()), '--mode', 'runtime']),
        ('game_solver_tests', GAME, ['-m', 'pytest', '-o', 'addopts=', '-q', 'code/tests/test_game_tracking_solvers.py',
                                    '-k', 'bilinear_exact or adamw_predictor or shared_parameters or bn_and_dropout or nonfinite_corrector or frozen_head or plain_mode or full_predictor',
                                    '--junitxml=' + str(OUT / 'game_solver_tests.xml')]),
    ]
    if selected:
        wanted = set(selected.split(','))
        assert wanted <= {j[0] for j in jobs}, wanted
        jobs = [j for j in jobs if j[0] in wanted]
    previous = json.loads((OUT / 'probe_results.json').read_text(encoding='utf-8')) if selected and (OUT / 'probe_results.json').exists() else {}
    results = [r for r in previous.get('jobs', []) if r['name'] not in {j[0] for j in jobs}]
    for name, cwd, argv in jobs:
        started = time.monotonic()
        process = subprocess.run([sys.executable, *argv], cwd=cwd, env=env, capture_output=True, text=True,
                                 encoding='utf-8', errors='strict', timeout=180)
        (OUT / (name + '.log')).write_text(process.stdout + process.stderr, encoding='utf-8', newline='\n')
        result = dict(name=name, returncode=process.returncode, elapsed_seconds=round(time.monotonic()-started, 3),
                      command=[sys.executable, *argv], cwd=str(cwd), log=name+'.log')
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = dict(status='VERIFIED' if all(r['returncode'] == 0 for r in results) else 'FAILED',
                   scope='existing_kernel_tests_and_synthetic_probes_only', interpreter=sys.executable,
                   conda_prefix=os.environ.get('CONDA_PREFIX'), jobs=results, remote_or_training_actions=False)
    (OUT / 'probe_results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'kernel', 'native', 'runtime'], default='all')
    parser.add_argument('--jobs', help='Only refresh these comma-separated job names; preserve other results')
    args = parser.parse_args()
    if args.mode == 'all':
        result = run_all(args.jobs)
    else:
        result = {'kernel': kernel_probe, 'native': native_probe, 'runtime': runtime_probe}[args.mode]()
        (OUT / (args.mode + '_probe.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, ensure_ascii=False))
    if result['status'] != 'VERIFIED':
        raise SystemExit(1)
