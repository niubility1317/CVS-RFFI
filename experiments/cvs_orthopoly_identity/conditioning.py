"""Post-freeze coordinate diagnostic on the existing public RF cascade.

No formal source/target IQ, learned checkpoint, classifier labels, or model
updates are used. The least-squares fit is a shared coordinate inverse on
public synthetic IQ, not identity training or an experiment selection rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cvs_coupled_identity.model import delay
from experiments.cvs_orthopoly_identity.model import packet_basis, build
from experiments.cvs_reference_identity.physics import TX_ROWS, RX_ROWS, received


def original_coordinates(z, lag):
    power = z.square().sum(1)
    p = power / 4 if lag == 0 else (power + delay(power, 4)) / 8
    return torch.stack([delay(term, m) for m in range(4)
                        for term in (z, z * p[:, None], z * p.square()[:, None])], 2)


def inverse_transport(moments):
    """Return the exact per-packet, per-delay 3x3 inverse coordinate map.

    [v,v*p,v*p^2] = T(mu,var,a) @ [v,4*v*d,16*v*(d^2-a*d-var)].
    T is an IQ-dependent coordinate transform, not a recovered PA operator.
    """
    rows = []
    for item in moments:
        mu, var, a = (item[k][:, 0] for k in ('mean', 'variance', 'projection'))
        zero = torch.zeros_like(mu)
        one = torch.ones_like(mu)
        rows.append(torch.stack((torch.stack((one, zero, zero), -1),
                                 torch.stack((mu, one / 4, zero), -1),
                                 torch.stack((mu.square() + var, (2 * mu + a) / 4, one / 16), -1)), 1))
    return torch.stack(rows, 1)


def restore_coordinates(basis, transport):
    if basis.ndim != 4 or basis.shape[1:3] != (2, 12):
        raise ValueError('Expected twelve complex orthogonal coordinates')
    if transport.shape != (len(basis), 4, 3, 3):
        raise ValueError('Expected same-packet four inverse order maps')
    shaped = basis.reshape(len(basis), 2, 4, 3, basis.shape[-1])
    return torch.einsum('bmqr,bimrt->bimqt', transport, shaped).reshape_as(basis)


def relative_error(value, reference):
    return float((value - reference).norm() / reference.norm().clamp_min(1e-30))


def shared_inverse_fit(basis, original):
    """Fit each delay's single real map across all public packets and I/Q.

    Each packet contributes the same number of coordinates. No receiver or
    transmitter identifier chooses a map. Complex fixed convolutions have a
    larger function class; this probe does not bound a full CNN's expressivity.
    """
    errors = []
    maps = []
    for m in range(4):
        x = basis[:, :, 3*m:3*m+3].permute(0, 1, 3, 2).reshape(-1, 3)
        y = original[:, :, 3*m:3*m+3].permute(0, 1, 3, 2).reshape(-1, 3)
        coefficients = torch.linalg.lstsq(x, y, driver='gelsd').solution
        errors.append(relative_error(x @ coefficients, y))
        maps.append(coefficients.tolist())
    return dict(per_delay_relative_residual=errors, fitted_maps=maps,
                scope='One real order-mixing matrix per delay across public synthetic packets; not all convolutions or a trained classifier')


@torch.no_grad()
def public_conditioning_probe():
    torch.set_num_threads(2)
    z = torch.tensor(np.stack([received(t, r) for t in TX_ROWS for r in RX_ROWS]).tolist(), dtype=torch.float64)
    records = []
    for lag, variant in ((0, 'orthopoly_instant'), (4, 'orthopoly_memory4')):
        basis, moments = packet_basis(z, lag, return_moments=True)
        transport = inverse_transport(moments)
        original = original_coordinates(z, lag)
        restored = restore_coordinates(basis, transport)
        # Disposable source-family scratch convolution; no weight inheritance.
        with torch.random.fork_rng():
            torch.manual_seed(2026100207)
            model = build(variant).double().eval()
        conv = model.core.behavior[0].conv
        reference_output = conv(original)
        restored_output = conv(restored)
        fit = shared_inverse_fit(basis, original)
        per_delay = []
        for m, item in enumerate(moments):
            per_delay.append(dict(delay=m,
                mu_min=float(item['mean'].min()), mu_max=float(item['mean'].max()),
                variance_min=float(item['variance'].min()), variance_max=float(item['variance'].max()),
                projection_min=float(item['projection'].min()), projection_max=float(item['projection'].max()),
                inverse_matrix_range_max=float((transport[:, m].amax(0) - transport[:, m].amin(0)).abs().max())))
        records.append(dict(variant=variant, envelope_lag=lag, packets=len(z),
            raw_order1_max_error=float((basis[:, :, 0] - z).abs().max()),
            restored_coordinates_max_error=float((restored - original).abs().max()),
            restored_coordinates_relative_error=relative_error(restored, original),
            scratch_convolution_max_error=float((restored_output - reference_output).abs().max()),
            scratch_convolution_relative_error=relative_error(restored_output, reference_output),
            shared_order_inverse=fit, moment_ranges=per_delay))
    return dict(status='PUBLIC_COORDINATE_DIAGNOSTIC_COMPLETE', synthetic_only=True,
        numerical_policy='CPU float64; disposable scratch convolution; algebra diagnostic, not formal FP32 performance',
        formal_data_access=False, source_checkpoint_access=False, target_query_access=False,
        target_truth_access=False, classifier_labels_used=False, model_updated=False,
        source_selection_changed=False, records=records,
        conclusions=dict(raw_IQ_information_preserved=True,
            inverse_is_packet_dependent=True,
            exact_inverse_restores_original_input_function=True,
            exact_inverse_alone_is_a_new_performance_improvement=False,
            cause_of_source_regression_proven=False,
            full_classifier_expressivity_bound=False,
            hardware_coefficients_recovered=False),
        interpretation='A hard packet basis substitution changes the coefficient chart seen by fixed shared filters. Its exact inverse restores the original lift and cannot alone improve the function. This checks a design limitation, not identity performance or TX/RX separation.')


def main(root):
    path = root / 'automation_reports/CV-SincNet/20261002-phase1-cvs-orthopoly-identity-manysig-m8-r01/evidence/public_conditioning_transport.json'
    result = public_conditioning_probe()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise FileExistsError('Preserve the existing public coordinate diagnostic')
    path.write_text(text, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
