"""Synthetic checks for the L-only moment/MixStyle diagnostic; no data I/O."""
import argparse
import json
import math

import numpy as np
import torch

from .style_stats import SOURCE_DAYS, SOURCE_LABELS, SOURCE_RX, style_audit


def _fixture(device='cpu', offset=False):
    labels, rx, days, values = [], [], [], []
    wave = torch.tensor([-1., 1., -1., 1., -1., 1., -1., 1.], dtype=torch.float64)
    for day in SOURCE_DAYS:
        for tx in SOURCE_LABELS:
            for ri, receiver in enumerate(SOURCE_RX):
                for pi in range(4):
                    noise = (-.15, -.05, .05, .15)[pi]
                    shift = float(ri - 2) if offset else 0.
                    values.append(torch.stack([wave + noise + shift + tx*.1 + day*.2,
                                               2*wave + noise + shift + tx*.2 + day*.1]))
                    labels.append(tx); rx.append(receiver); days.append(day)
    return (torch.stack(values).to(device), torch.tensor(labels), torch.tensor(rx),
            torch.tensor(days), [f'synthetic-L-{i}' for i in range(len(values))])


def _reject(args, expected):
    try:
        style_audit(*args, seed=551)
    except ValueError:
        return expected
    raise AssertionError('Accepted illegal input: '+expected)


def run(device='cpu'):
    torch.set_num_threads(2)
    args = _fixture(device)
    snapshots = [v.clone() if isinstance(v, torch.Tensor) else list(v) for v in args]
    rng = torch.random.get_rng_state().clone()
    np_rng = np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None
    zero = style_audit(*args, seed=551)
    again = style_audit(*args, seed=551)
    assert zero == again
    for before, after in zip(snapshots, args):
        assert torch.equal(before, after) if isinstance(before, torch.Tensor) else before == after
    assert torch.equal(rng, torch.random.get_rng_state())
    after_rng = np.random.get_state()
    assert np_rng[0] == after_rng[0] and np.array_equal(np_rng[1], after_rng[1]) and np_rng[2:] == after_rng[2:]
    if cuda_rng is not None:
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert zero['summary']['rounds'] == 36 and zero['summary']['conditional_cells'] == 18
    assert 0 < zero['summary']['enabled_rounds'] < 36
    for cell in zero['conditional_cells']:
        assert cell['count'] == 20 and set(cell['rx_counts'].values()) == {4}
        mu = cell['moments']['mu']
        assert mu['between_rx_mse'] < 1e-28
        assert math.isclose(mu['within_rx_mse'], .0125, abs_tol=1e-14)
        assert mu['between_denominator'] == 10
        assert all(v['within_denominator'] == 8 for v in mu['receivers'])
        assert cell['moments']['log_sigma']['between_rx_mse'] < 1e-28
    shifted_args = _fixture(device, offset=True)
    shifted = style_audit(*shifted_args, seed=551)
    for cell in shifted['conditional_cells']:
        mu = cell['moments']['mu']
        assert math.isclose(mu['between_rx_mse'], 2., abs_tol=1e-14)
        assert math.isclose(mu['within_rx_mse'], .0125, abs_tol=1e-14)
        assert math.isclose(mu['between_over_within'], 160., abs_tol=1e-9)
        assert math.isclose(mu['receiver_mean_noise_mse_estimate'], .0125/3, abs_tol=1e-14)
    assert shifted['summary']['moments']['mu']['actual_delta_mse_mean'] > zero['summary']['moments']['mu']['actual_delta_mse_mean']
    for record in shifted['rounds']:
        index = record['sample_indices']
        assert len(index) == len(set(index)) == 60
        by, brx, bd = [v[index] for v in shifted_args[1:4]]
        assert len(set(bd.tolist())) == 1
        for tx in SOURCE_LABELS:
            for r in SOURCE_RX:
                assert int(((by == tx) & (brx == r)).sum()) == 2
        if record['enabled']:
            for i, j in enumerate(record['donor_batch_indices']):
                assert by[i] == by[j] and brx[i] != brx[j] and bd[i] == bd[j]
                assert record['physical_ids'][i] != record['physical_ids'][j]
        else:
            assert record['moments']['mu']['delta_mse'] == 0.
            assert record['feature']['relative_l2_mean'] == 0.
    # Scaling produces a separately known log-sigma RX signal.
    scale = shifted_args[0].clone()
    for ri, receiver in enumerate(SOURCE_RX):
        mask = shifted_args[2] == receiver
        scale[mask.to(scale.device)] *= 2.**ri
    scale_result = style_audit(scale, *shifted_args[1:], seed=551)
    assert scale_result['summary']['moments']['log_sigma']['between_rx_mse_mean'] > .9
    invalid = []
    bad = list(args); bad[1] = bad[1].float(); invalid.append(_reject(bad, 'floating labels'))
    bad = list(args); bad[2] = bad[2].clone(); bad[2][0] = 0; invalid.append(_reject(bad, 'target RX'))
    bad = list(args); bad[3] = bad[3].clone(); bad[3][0] = 0; invalid.append(_reject(bad, 'non-source day'))
    bad = list(args); bad[4] = list(bad[4]); bad[4][1] = bad[4][0]; invalid.append(_reject(bad, 'duplicate PID'))
    bad = list(args); bad[4] = list(bad[4]); bad[4][0] = ''; invalid.append(_reject(bad, 'empty PID'))
    bad = list(args); bad[1] = bad[1][:-1]; invalid.append(_reject(bad, 'metadata length'))
    bad = list(args); bad[0] = bad[0].clone(); bad[0][0, 0, 0] = float('nan'); invalid.append(_reject(bad, 'nonfinite maps'))
    bad = list(args); bad[0] = bad[0].to(torch.complex128); invalid.append(_reject(bad, 'complex maps'))
    keep = torch.arange(3, len(args[0]))
    bad = [v[keep.to(v.device)] if isinstance(v, torch.Tensor) else [v[i] for i in keep.tolist()] for v in args]
    invalid.append(_reject(bad, 'cell has fewer than two PID'))
    empty = [v[:0] for v in args]; invalid.append(_reject(empty, 'empty cache'))
    constant = style_audit(torch.zeros_like(args[0]), *args[1:], seed=551)
    assert constant['summary']['moments']['mu']['ratio_of_mean_between_to_mean_within'] is None
    native = args[0].float().requires_grad_()
    native_before = native.detach().clone()
    native_result = style_audit(native, *args[1:], seed=551)
    assert torch.equal(native.detach(), native_before) and native.requires_grad and native.grad is None
    assert native_result['input_dtype'] == 'torch.float32'
    assert native_result['summary']['rounds'] == 36
    json.dumps([zero, shifted, scale_result, constant, native_result], allow_nan=False)
    return dict(status='PASS', device=device, source_role='synthetic L_s', target_access=False,
                zero_RX_difference=True, known_RX_mu_offset_MSE=2., within_mu_MSE=.0125,
                known_scale_log_sigma_signal=True, mean_noise_denominator_verified=True,
                rounds=36, balanced_batch=60, fixed_seed_reproducible=True,
                donor_rules_verified=True, disabled_rounds_retained=True,
                input_unchanged=True, global_RNG_unchanged=True, JSON_finite=True,
                float32_cached_maps_supported=True, no_autograd_graph_or_gradient=True,
                illegal_inputs_rejected=invalid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2, allow_nan=False))
