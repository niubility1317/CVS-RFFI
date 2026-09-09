"""Check actual model wiring, gradients, fresh-state round trips and source entry."""
import argparse
import json
from pathlib import Path
import sys
import time
from copy import deepcopy
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SSDG import train_ssdg as train
from run_a1_fast_v2 import v2_command
from run_a1_mechanism_screen import mechanism_matrix
from cvsrffi.a1_response_surface import response_pair_loss
from cvsrffi.a1_fisher_training import fisher_labeled_objective
from cvsrffi.a1_r3_objective import r3_pair_objective


def args_for(row):
    matrix = mechanism_matrix()
    args = train.build_arg_parser().parse_args(v2_command(matrix, project_root=Path('/unused'), run_root=Path('/new'), row=row)[3:])
    train._validate_daot_config(args)
    train._validate_a1_scratch_only(args)
    train._validate_a1_ecrs_config(args)
    return args


def build(args, device):
    torch.manual_seed(args.seed)
    cfg = train._apply_model_cli_args(train.merge_checkpoint_args({}, args, input_len=256, num_domains=15), args)
    return train.build_baseline_model(cfg, device)


def run_models(device, folder):
    torch.set_num_threads(2)
    checks = []
    for row in mechanism_matrix()['rows']:
        started = time.monotonic()
        args = args_for(row)
        model = build(args, device)
        x = torch.randn(12, 2, 256, device=device)
        y = torch.arange(12, device=device) % 6
        rx = torch.arange(12, device=device) // 6
        amp_checks = []
        for amp in ([False, True] if device.type == 'cuda' else [False]):
            model.train(); model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x, y_tx=y, return_aux=True, return_physical_gate_diag=args.a1_fisher_supervision)
                loss = torch.nn.functional.cross_entropy(out['tx_logits'], y)
                details = {}
                if args.a1_response_surface:
                    auxiliary = torch.nn.functional.cross_entropy(out['response_logits'], y)
                    pair, count = response_pair_loss(model.a1_response, x, x + .03 * torch.randn_like(x))
                    loss = loss + .15 * auxiliary + .03 * pair
                    details.update(response_pair_count=count, response_pair=float(pair.detach()))
                if args.a1_fisher_supervision:
                    auxiliary, telemetry = fisher_labeled_objective(model, out['aux_id'], y, len(y), 65)
                    loss = loss + auxiliary
                    details.update({k: float(v) for k, v in telemetry.items()})
                if args.use_a1_r3:
                    auxiliary, telemetry = r3_pair_objective(model=model, clean_iq=x, domains=rx,
                        physical_ids=[f'source:{i}' for i in range(len(x))], role='U_s', args=args,
                        epoch=65, batch_idx=1, optimizer_step=1, apply_sat_fn=train.apply_sat_channel_for_scenario)
                    loss = loss + auxiliary
                    details['r3_loss'] = float(auxiliary.detach())
            assert torch.isfinite(loss), row['id']
            loss.backward()
            grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            assert grads and all(torch.isfinite(g).all() for g in grads.values()), row['id']
            if args.a1_response_surface:
                assert any(g.norm() > 0 for n, g in grads.items() if n.startswith('a1_response.encoder.'))
                if args.a1_response_rho > 0:
                    assert model.a1_response.projection.weight.grad.norm() > 0
                assert any(g.norm() > 0 for n, g in grads.items() if n.startswith('id_backbone.'))
            if args.a1_fisher_supervision:
                assert any(g.norm() > 0 for n, g in grads.items() if '.branch_heads.' in n)
            amp_checks.append({'amp': amp, 'loss': float(loss.detach()), **details})
        model.eval()
        with torch.no_grad():
            expected = model(x, return_aux=True)['z_id']
            identity = model.forward_identity_only(x)['z_id']
            torch.testing.assert_close(identity, expected, rtol=2e-4, atol=2e-5)
        artifact = folder / (row['id'] + '.fresh.pth')
        torch.save({'model': model.state_dict(), 'args': vars(args), 'lineage': 'this_check_fresh_random'}, artifact)
        del model
        saved = torch.load(artifact, map_location=device, weights_only=False)
        from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
        reloaded, load_audit = build_exact_ssdg_model_from_checkpoint(saved, input_len=256, device=device)
        assert load_audit['checkpoint_load_strict']
        reloaded.eval()
        with torch.no_grad():
            torch.testing.assert_close(reloaded(x, return_aux=True)['z_id'], expected, rtol=2e-4, atol=2e-5)
        del saved, reloaded
        check = {'id': row['id'], 'status': 'PASS', 'seconds': time.monotonic()-started, 'checks': amp_checks}
        checks.append(check)
        print(json.dumps({'id': row['id'], 'status': 'PASS', 'seconds': round(check['seconds'], 2)}), flush=True)
    return checks


def check_late_daot(device):
    checks = []
    for name in ('B0_FIXED', 'D1_THREE_VIEW', 'D2_PHYSICAL_ORBIT', 'D3_TANGENT'):
        row = next(r for r in mechanism_matrix()['rows'] if r['id'] == name)
        args = args_for(row)
        # All rows use the actual matrix; increase only the diagnostic sample
        # ratio so the tiny fixture cannot randomly miss every tangent row.
        args.daot_tangent_sample_ratio = 1.
        model = build(args, device).train()
        teacher = deepcopy(model).eval().requires_grad_(False)
        x = torch.randn(12, 2, 256, device=device)
        y = torch.arange(12, device=device) % 6
        domains = torch.arange(12, device=device) % 15
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            clean = model(x, y_tx=y, return_aux=True)
            result = train._compute_daot_labeled_step(model=model, ema_model=teacher,
                student_clean=clean, x_clean=x, y_clean=y, d_clean=domains,
                args=args, epoch=65, batch_idx=1, apply_sat_fn=train.apply_sat_channel_for_scenario,
                prototype_matrix=None)
        assert torch.isfinite(result['loss']) and result['loss'].item() > 0
        if name == 'D3_TANGENT':
            assert result['components']['tangent'].item() > 0
        result['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        checks.append({'id': name, 'epoch': 65, 'components': {k: float(v.detach()) for k, v in result['components'].items()}})
        print(json.dumps({'late_daot': name, 'status': 'PASS'}), flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models-only', action='store_true')
    parser.add_argument('--entries-only', action='store_true')
    parser.add_argument('--late-only', action='store_true')
    args = parser.parse_args()
    folder = args.output.with_suffix('')
    folder.mkdir(parents=True, exist_ok=False)
    models = [] if args.entries_only or args.late_only else run_models(torch.device(args.device), folder)
    late = [] if args.entries_only else check_late_daot(torch.device(args.device))
    entries = []
    if not args.models_only and not args.late_only:
        from check_a1_ecrs_cross_rx import run_check
        for name in ('D3_TANGENT', 'F2_R3_IDENTITY', 'G1_FISHER_GATE', 'E2_RESPONSE_PAIR'):
            def factory(name=name):
                matrix = mechanism_matrix()
                matrix['rows'] = [r for r in matrix['rows'] if r['id'] == name]
                return matrix
            result = run_check(torch.device(args.device), folder / ('entry_' + name), matrix_factory=factory, expect_cross_rx=False)
            entries.append({'id': name, **result})
            print(json.dumps({'entry': name, 'status': result['status']}), flush=True)
    result = {'status': 'PASS', 'models': models, 'entries': entries, 'late_daot': late,
              'historical_checkpoint_loads': 0, 'target_inputs': 0,
              'boundary': 'synthetic activation and fresh checkpoint round trip; no performance claim'}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
