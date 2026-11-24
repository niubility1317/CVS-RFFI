"""Measure paired clean waveform fidelity after WiSig source predictions freeze."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_unrolled_restore import wisig_anchored_window_source as runner
from experiments.pamf_unrolled_restore import wisig_preamble_dura_source as preamble
from experiments.pamf_unrolled_restore import wisig_repeat_drift_source as study
from experiments.pamf_unrolled_restore import wisig_r69_stage_isolation as stage_study
from experiments.pamf_unrolled_restore.wisig_repeat_drift import STF_BEGIN, STF_END
from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex
from tools.wisig_anchored_window_audit import FIELDS, timing_reference, waveform_metrics


MODES = ('direct','r69','polar_gated','drift_timed','drift_engineering',
         'drift_gated','drift_phase_only','drift_no_observed')
REGION_FIELDS = ('complex_nmse','amplitude_nmse','phase_cosine')
PREAMBLE_MODES = ('direct', 'r69', 'preamble_timed', *(f'preamble_{name}' for name in preamble.READOUTS))
ACTION_FIELDS = ('predict_residual_power','identity_residual_power','estimated_variance')


def fidelity(waves, clean, mask):
    reference_amplitude, amplitude = clean.abs(), waves.abs()
    weights = mask * reference_amplitude.square()
    energy = weights.sum(-1).clamp_min(1e-8)
    cosine = (waves * clean.conj()).real / (amplitude * reference_amplitude).clamp_min(1e-8)
    return torch.stack((((waves-clean).abs().square()*mask).sum(-1) / energy,
                        ((amplitude-reference_amplitude).square()*mask).sum(-1) / energy,
                        (cosine.clamp(-1.,1.)*weights).sum(-1) / energy), dim=-1)


def decompose(waves, reference, delta, time, frequency):
    original = time + delta[:,None]
    available = (original>=0) & (original<len(time))
    stf = (original>=STF_BEGIN) & (original<STF_END)
    # All proposals use the same nominal coordinate and overlap window.
    return dict(audit=torch.stack([waveform_metrics(wave,reference,available,frequency) for wave in waves]),
                observed_stf=fidelity(waves,reference,stf),
                other_overlap=fidelity(waves,reference,available & ~stf))


def single_packet_latency(model, gate, sample, gate_features=None):
    timings = []
    for step in range(28):
        torch.cuda.synchronize()
        start = perf_counter()
        _, active = runner.guarded(model,gate,sample,gate_features=gate_features)
        torch.cuda.synchronize()
        if step>=8:
            timings.append(1000.*(perf_counter()-start))
    return dict(active=bool(active.item()),median_ms=float(np.median(timings)))


def action_error(receiver, sample, clean):
    stages,action,variance,support,level,*_ = receiver.stages(receiver.analyze(sample))
    desired = clean/level
    observed = stages[0]
    model_error = (observed-action.apply(desired)).abs().square()
    identity_error = (observed-desired).abs().square()
    denom = support.sum(-1).clamp_min(1.)
    return torch.stack(tuple((field*support).sum(-1)/denom for field in (model_error,identity_error,variance)),dim=-1)


def preamble_readouts(model, gate, received, clean):
    analysis = model.analyze(received)
    stages, action, variance, level, shift, _ = model.stages(analysis)
    _, observed, _, support, *_ = analysis
    margin, noise = preamble.gate_features(model, analysis)
    active = (margin > gate[0]) | (noise > gate[1])
    restored = tuple(value * level for value in stages)
    guarded = torch.where(active[:, None], restored[-1], received)
    reference = clean / level
    estimate_error = (observed - action.apply(reference)).abs().square()
    identity_error = (observed - reference).abs().square()
    total = support.sum(-1).clamp_min(1)
    residuals = torch.stack(tuple((field * support).sum(-1) / total
                                  for field in (estimate_error, identity_error, variance)), dim=-1)
    return (observed * level, *restored, guarded), shift, active, residuals


@torch.no_grad()
def measure(runtime):
    root = Path(runtime['run_root'])
    marker = read(root/'predictions/complete.json')
    scores = read(root/'source-scores.json')
    stages = runtime['run_id']==stage_study.RUN
    observed_solver = runtime['run_id']=='20261124-phase1-wisig-observed-solve-source-m4-r01'
    preamble_dura = runtime['run_id'] == preamble.RUN
    if observed_solver:
        from experiments.pamf_unrolled_restore import wisig_observed_solve_source as solver
        modes, prediction_modes = ('direct','r69',*solver.MODES[8:]),solver.MODES
    elif preamble_dura:
        modes, prediction_modes = PREAMBLE_MODES, preamble.MODES
    else:
        modes, prediction_modes = (stage_study.MODES,stage_study.MODES) if stages else (MODES,study.MODES)
    if (not (stages or observed_solver or preamble_dura or runtime['run_id']==study.RUN)
            or read(root/'state.json')['status']!='SOURCE_SCORED'
            or marker['status']!='FROZEN' or marker['target_reads'] or marker['source_truth_reads']
            or marker['modes']!=list(prediction_modes) or not scores['truth_last'] or scores['target_reads']):
        raise ValueError('Expected the already-scored dual-route WiSig source predictions')
    activate(runtime)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache,_,_,_,ids,rx,_ = runner.core.baseline.handles(runtime)
    r69 = runner.core.baseline.original_r69(Path(runtime['parent_root']))
    if stages:
        stage_readouts = stage_study.StageReadout(r69.pipeline)
    elif observed_solver:
        (receiver,gate), = runner.load_models(runtime,arms=('coupled',),build=solver.build)
    elif preamble_dura:
        (receiver, gate), = runner.load_models(runtime, arms=('preamble',), build=preamble.build)
    else:
        models = list(runner.load_models(runtime,arms=study.ARMS,build=study.build))
    time = torch.arange(256,device='cuda:0')
    offsets = torch.arange(-64,65,device='cuda:0')
    positions = time[None] + offsets[:,None]
    frequency = torch.fft.fftfreq(256,device='cuda:0').abs()
    groups = {str(key):tensor(np.flatnonzero(rx==key),'cuda:0').long() for key in np.unique(rx)}
    groups['all'] = torch.arange(len(ids),device='cuda:0')
    routes, latency = {}, {}
    for route, path in runner.ROUTES.items():
        folder = cache/path
        clean = np.load(folder/'clean.npy',mmap_mode='r',allow_pickle=False)
        routes[route] = {}
        for view in runner.core.baseline.VIEWS:
            with np.load(root/f'predictions/{route}_{view}.npz',allow_pickle=False) as predictions:
                if not np.array_equal(predictions['ids'],ids):
                    raise ValueError('Physical source row order changed')
            raw = np.load(folder/f'{view}.npy',mmap_mode='r',allow_pickle=False)
            if raw.shape!=clean.shape or raw.shape!=(len(ids),2,256):
                raise ValueError('Paired source IQ has unexpected shape')
            results, shift_errors, operator_errors = [], [], []
            activity, reference_shifts, reference_confidence = [], [], []
            for offset in range(0,len(ids),48):
                rows = slice(offset,offset+48)
                iq = tensor(raw[rows],'cuda:0')
                received = as_complex(iq)
                reference = as_complex(tensor(clean[rows],'cuda:0'))
                delta, coherence = timing_reference(received,reference,positions,offsets)
                if stages:
                    waves = torch.stack((received,*(as_complex(value) for value in stage_readouts(iq))))
                elif observed_solver:
                    restored, = r69(iq)
                    options,active,shift = solver.proposals(receiver,gate,received)
                    if 'active_source' not in latency and active.any():
                        latency['active_source'] = single_packet_latency(receiver,gate,received[active][:1],solver.gate_features)
                    waves = torch.stack((received,as_complex(restored),*options))
                    shift_errors.append((shift-delta).abs())
                    operator_errors.append(action_error(receiver,received,reference))
                elif preamble_dura:
                    restored, = r69(iq)
                    options, shift, active, errors = preamble_readouts(receiver, gate, received, reference)
                    waves = torch.stack((received, as_complex(restored), *options))
                    shift_errors.append((shift-delta).abs())
                    operator_errors.append(errors)
                    activity.append(active)
                    reference_shifts.append(delta)
                    reference_confidence.append(coherence)
                else:
                    restored, = r69(iq)
                    variants = [study.proposals(model,gate,received) for model,gate in models]
                    polar, (drift,active,shift) = variants
                    if 'active_source' not in latency and active.any():
                        latency['active_source'] = single_packet_latency(*models[-1],received[active][:1])
                    waves = torch.stack((received,as_complex(restored),polar[0][-1],drift[0],drift[1],*drift[-3:]))
                    shift_errors.append((shift-delta).abs())
                results.append(decompose(waves,reference,delta,time,frequency))
            merged = {region:torch.cat([batch[region] for batch in results],dim=1) for region in results[0]}
            routes[route][view] = {}
            errors = torch.cat(shift_errors) if shift_errors else None
            action_errors = torch.cat(operator_errors) if operator_errors else None
            if preamble_dura:
                active = torch.cat(activity)
                reference_shift = torch.cat(reference_shifts)
                coherence = torch.cat(reference_confidence)
            for name,samples in groups.items():
                group = dict(count=len(samples),median={region:matrix[:,samples].median(dim=1).values.tolist()
                                                        for region,matrix in merged.items()})
                if errors is not None:
                    group.update(shift_error_median_samples=int(errors[samples].median()),
                                 colocated_within_two=int((errors[samples]<=2).sum()))
                if action_errors is not None:
                    group['action_median'] = action_errors[samples].median(0).values.tolist()
                if preamble_dura:
                    missing = samples[reference_shift[samples] != 0]
                    colocated = samples[errors[samples] <= 2]
                    group.update(active=int(active[samples].sum()),
                                 reference_squared_coherence_median=float(coherence[samples].median()),
                                 paired_missing_edge_packets=len(missing),
                                 missing_edge_nmse_median=merged['audit'][:, missing, -1].median(1).values.tolist()
                                 if len(missing) else None,
                                 action_median_within_two=action_errors[colocated].median(0).values.tolist()
                                 if len(colocated) else None)
                routes[route][view][name] = group
    return dict(status='VERIFIED',target_reads=0,paired_clean_diagnostic_only=True,
                rows=len(ids),modes=modes,audit_fields=FIELDS,region_fields=REGION_FIELDS,routes=routes,
                b1=latency,device=torch.cuda.get_device_name(),
                action_fields=ACTION_FIELDS if observed_solver or preamble_dura else ())


if __name__ == '__main__':
    print(json.dumps(measure(read(Path(sys.argv[1]))),allow_nan=False))
