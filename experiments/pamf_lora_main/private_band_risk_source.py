"""Test a received-only estimate of incremental private-spectrum waveform risk."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .channel_blind import corruption_score
from .channel_blind_source import partition, setup
from .complex_vector_source import publish
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import coordinates
from .offband_noise_source import draw_noise, replay
from .private_transport_bands import named_band_masks
from .private_transport_source import proposed_reads
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .variance_bound import spectral_contrast, variance_interval
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-private-band-risk-source-cpu-r02'
SEED = 2026114117
PER_RX = 30
PROBES = 4
EPS = .025
BANDS = ('1_to_8', '9_to_32', '33_to_64')
VIEWS_HELD = (VIEWS[3], VIEWS[-1])
ISOLATION = dict(target_reads=0, TX_reads=0, F_reads=0, V_reads=0, training_updates=0)


def fft(iq):
    return torch.fft.fft(iq, norm='ortho')


def band_inner(left, right, weight, masks):
    return (((left.conj() * right).real * weight).sum(1)) @ masks.T


def stein_trace(edit_at, spectrum, variance, weight, masks, generator):
    """Covariance-weighted central differences in fixed received coordinates."""
    total = spectrum.real.new_zeros((len(spectrum), len(masks)))
    for _ in range(PROBES):
        axes = torch.randint(2, (2, *spectrum.shape), device=spectrum.device, generator=generator)
        i, q = (2 * axes - 1).to(spectrum.real.dtype)
        probe = torch.complex(i, q) * (variance * .5).sqrt()
        derivative = (edit_at(spectrum + EPS * probe) - edit_at(spectrum - EPS * probe)) / (2 * EPS)
        total += band_inner(probe, derivative, weight, masks)
    return total / PROBES


def minimizing_dose(linear, quadratic):
    return (-linear / quadratic.clamp_min(1e-12)).clamp(0, 1)


def private_projection(spectrum):
    return spectrum - spectrum.mean(1, keepdim=True)


@torch.no_grad()
def received_readout(iq, tracker, model, threshold):
    y, rot, q, rms = coordinates(tracker, iq)
    weight = rms.square()
    spectrum = fft(y)
    active = corruption_score(y) > threshold
    masks = named_band_masks(y).to(y.real.dtype)
    midpoint = spectrum.clone()
    edit = torch.zeros_like(spectrum)
    trace = y.real.new_zeros((len(y), len(masks)))
    generator = torch.Generator(device=y.device).manual_seed(SEED)
    for idx in active.nonzero().flatten().split(12):
        if len(idx) == 0:
            continue
        rank, deltas = proposed_reads(y[idx], q[idx], model)
        midpoint[idx] = fft(rank + deltas[0])
        edit[idx] = fft(deltas[2] - deltas[0])
        _, _, outer = model.propose(y[idx], q[idx])
        contrast, weights = spectral_contrast(spectrum[idx], q[idx])
        coarse_noise, _ = variance_interval(outer, contrast, weights, model.selectors)
        variance = coarse_noise @ model.bands

        def edit_at(perturbed):
            _, alternatives = proposed_reads(torch.fft.ifft(perturbed, norm='ortho'), q[idx], model)
            return fft(alternatives[2] - alternatives[0])

        trace[idx] = stein_trace(edit_at, spectrum[idx], variance, weight[idx], masks, generator)
    quadratic = band_inner(edit, edit, weight, masks)
    uncorrected = band_inner(midpoint - spectrum, edit, weight, masks)
    linear = uncorrected + trace
    if not all(torch.isfinite(x).all() for x in (edit, linear, quadratic)):
        raise ValueError('Nonfinite private spectrum risk computation')
    return dict(active=active, basis=rot*rms, spectrum=spectrum, midpoint=midpoint, edit=edit,
                masks=masks, weight=weight, trace=trace, linear=linear, uncorrected=uncorrected,
                quadratic=quadratic, dose=minimizing_dose(linear,quadratic))


def linear_correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    scale = a.norm() * b.norm()
    return float(a.dot(b) / scale) if scale > 0 else None


def summarize(readout, reference, groups):
    """Clean or isolated noise-free reference is first used on this side."""
    weight, mask = readout['weight'], readout['masks']
    mid, edit = readout['midpoint'], readout['edit']
    exact_linear = band_inner(mid-reference, edit, weight, mask)
    quadratic = readout['quadratic']
    true_delta = 2 * exact_linear + quadratic
    estimated_delta = 2 * readout['linear'] + quadratic
    base_error = band_inner(mid-reference, mid-reference, weight, mask)
    private = private_projection(reference)
    power = band_inner(private, private, weight, mask)
    valid = readout['active'][:, None] & (quadratic > 1e-10) & (power > 1e-10)
    partition = {'pooled': torch.ones(len(mid), dtype=torch.bool, device=mid.device)}
    partition.update({name: tensor(groups == name, mid.device) for name in SOURCE})
    doses = {'midpoint':torch.zeros_like(exact_linear), 'fixed_step_half':torch.full_like(exact_linear,.5),
             'full':torch.ones_like(exact_linear), 'received_risk':readout['dose'],
             'observed_residual_only':minimizing_dose(readout['uncorrected'],quadratic),
             'isolated_minimizer':minimizing_dose(exact_linear,quadratic)}
    rows = {}
    for scope, available in partition.items():
        bands = rows[scope] = {}
        for i, name in enumerate(BANDS):
            take = available & valid[:, i]
            if not take.any():
                bands[name] = dict(n=0)
                continue
            observed_risk = estimated_delta[take,i]
            actual_risk = true_delta[take,i]
            err = base_error[take,i]
            data = dict(n=int(take.sum()), full_actual_improvement=float((actual_risk < 0).float().mean()),
                        full_estimated_improvement=float((observed_risk < 0).float().mean()),
                        full_sign_agreement=float(((observed_risk < 0) == (actual_risk < 0)).float().mean()),
                        full_risk_correlation=linear_correlation(observed_risk/err.clamp_min(1e-10),
                                                                 actual_risk/err.clamp_min(1e-10)))
            results = data['readouts'] = {}
            for key, dose in doses.items():
                selected = dose[take,i]
                paired_error = err + 2 * selected * exact_linear[take,i] + selected.square() * quadratic[take,i]
                changed = private_projection(mid[take] + selected[:, None, None]*edit[take])
                selected_mask = mask[i:i+1]
                true_private = private[take]
                actual_weight = weight[take]
                align = band_inner(changed,true_private,actual_weight,selected_mask)[:,0]
                shifted = band_inner(changed,1j*true_private,actual_weight,selected_mask)[:,0]
                truth_energy = power[take,i]
                gain = torch.complex(align, shifted)/truth_energy
                results[key] = dict(median_dose=float(selected.median()),
                                    relative_mse=float(paired_error.sum()/err.sum().clamp_min(1e-12)),
                                    median_real_gain=float(gain.real.median()),
                                    median_abs_phase_rad=float(gain.angle().abs().median()))
            bands[name] = data
    return dict(n=len(mid), active=int(readout['active'].sum()), strata=rows)


@torch.no_grad()
def smoke(rt):
    (_, calibration, _), contract = setup(rt,'cpu',smoke=True)
    root = Path(rt['run_root'])
    model, threshold = parent_model(root,contract,'cpu')
    tracker = PacketRestorer('consensus').eval()
    iq = real_iq(colored_observation(complex_iq(calibration[:2]),torch.Generator().manual_seed(SEED)))
    tested = received_readout(iq,tracker,model,-1.)
    bypass = received_readout(calibration[:2],tracker,model,float('inf'))
    if (int(tested['active'].sum()) != len(iq) or tested['edit'].abs().amax() == 0
            or bypass['dose'].count_nonzero() or not torch.equal(bypass['midpoint'],bypass['spectrum'])):
        raise ValueError('Actual source checkpoint risk or clean bypass failed')
    clean = fft(complex_iq(calibration[:2])/tested['basis'])
    grouped = summarize(tested,clean,np.repeat(SOURCE[0],len(iq)))['strata']
    if grouped['pooled'] != grouped[SOURCE[0]] or not any(row['n'] for row in grouped['pooled'].values()):
        raise ValueError('Actual source checkpoint risk summary failed')
    write(root/'smoke.json',dict(status='PASS',tracked=len(iq),threshold=threshold,**ISOLATION))
    print('LORA_PRIVATE_RISK_CHECKPOINT_PASS',flush=True)


@torch.no_grad()
def probe(rt):
    start_time = perf_counter()
    source = Path(rt['source_root'])
    root = Path(rt['run_root'])
    contract = source_contract(source)
    (_, _, held), verified = setup(rt,'cpu')
    if verified != contract:
        raise ValueError('Source U capture and replay contracts differ')
    raw, rx, ids, _ = source_role(source,contract,'U_s')
    _, _, indices = partition(rx)
    selected = np.concatenate([chunk[:PER_RX] for chunk in np.split(indices,len(SOURCE))])
    clean = torch.cat([chunk[:PER_RX] for chunk in held.chunk(len(SOURCE))])
    groups = rx[selected]
    model, threshold = parent_model(root,contract,'cpu')
    tracker = PacketRestorer('consensus').eval()
    generator = torch.Generator().manual_seed(SEED + 1)

    def evaluate(capture, reference_iq, *, subtract_noise=False):
        visible = received_readout(capture,tracker,model,threshold)
        reference = complex_iq(reference_iq)/visible['basis']
        if subtract_noise:
            reference = visible['spectrum']-fft(reference)
        else:
            reference = fft(reference)
        return summarize(visible,reference,groups)

    rows = {'clean':evaluate(clean,clean)}
    white, _, _ = draw_noise(clean,'white',generator)
    colored = real_iq(colored_observation(complex_iq(clean),generator))
    for name, iq in (('generic_white',white),('generic_colored',colored)):
        rows[name] = evaluate(iq,clean)
        print('LORA_PRIVATE_RISK_READY',name,rows[name]['active'],flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            rows[route] = {}
            for view in VIEWS_HELD:
                jobs = ((route,view,raw[i],rx[i],ids[i]) for i in selected)
                iq, epsilon = (tensor(np.stack(parts),'cpu')
                               for parts in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                rows[route][view] = evaluate(iq,epsilon,subtract_noise=True)
                print('LORA_PRIVATE_RISK_READY',route,view,rows[route][view]['active'],flush=True)
    write(root/'source-u-private-risk.json', dict(status='SOURCE_DIAGNOSTIC_COMPLETE', n=len(clean),
          role='held_U_s', noise_reference_used_after_readout=True, probes=PROBES, epsilon=EPS,
          threshold=threshold, worker_seconds=perf_counter()-start_time, rows=rows, **ISOLATION))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    try:
        for phase in ('smoke','probe'):
            if start(rt,phase,'source',module=rt['runner']).wait():
                raise RuntimeError(f'Private-band risk {phase} failed')
        write(root/'state.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE'))
    except Exception as exc:
        write(root/'state.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    sub = commands.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        sub.add_argument('--'+name,type=Path,required=True)
    for phase in ('dispatch','smoke','probe'):
        sub = commands.add_parser(phase)
        sub.add_argument('runtime',type=Path)
        if phase != 'dispatch':
            sub.add_argument('row',choices=('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        return publish(**args,run=RUN,runner=__spec__.name)
    rt = read(args['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    activate(rt)
    (smoke if action == 'smoke' else probe)(rt)


if __name__ == '__main__':
    main()
