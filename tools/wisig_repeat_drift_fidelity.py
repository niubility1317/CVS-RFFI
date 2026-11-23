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
from experiments.pamf_unrolled_restore import wisig_repeat_drift_source as study
from experiments.pamf_unrolled_restore import wisig_r69_stage_isolation as stage_study
from experiments.pamf_unrolled_restore.wisig_repeat_drift import STF_BEGIN, STF_END
from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex
from tools.wisig_anchored_window_audit import FIELDS, timing_reference, waveform_metrics


MODES = ('direct','r69','polar_gated','drift_timed','drift_engineering',
         'drift_gated','drift_phase_only','drift_no_observed')
REGION_FIELDS = ('complex_nmse','amplitude_nmse','phase_cosine')


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


def single_packet_latency(model, gate, sample):
    timings = []
    for step in range(28):
        torch.cuda.synchronize()
        start = perf_counter()
        _, active = runner.guarded(model,gate,sample)
        torch.cuda.synchronize()
        if step>=8:
            timings.append(1000.*(perf_counter()-start))
    return dict(active=bool(active.item()),median_ms=float(np.median(timings)))


@torch.no_grad()
def measure(runtime):
    root = Path(runtime['run_root'])
    marker = read(root/'predictions/complete.json')
    scores = read(root/'source-scores.json')
    stages = runtime['run_id']==stage_study.RUN
    modes, prediction_modes = (stage_study.MODES,stage_study.MODES) if stages else (MODES,study.MODES)
    if (runtime['run_id'] not in (study.RUN,stage_study.RUN) or read(root/'state.json')['status']!='SOURCE_SCORED'
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
            results, shift_errors = [], []
            for offset in range(0,len(ids),48):
                rows = slice(offset,offset+48)
                iq = tensor(raw[rows],'cuda:0')
                received = as_complex(iq)
                reference = as_complex(tensor(clean[rows],'cuda:0'))
                delta,_ = timing_reference(received,reference,positions,offsets)
                if stages:
                    waves = torch.stack((received,*(as_complex(value) for value in stage_readouts(iq))))
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
            for name,samples in groups.items():
                group = dict(count=len(samples),median={region:matrix[:,samples].median(dim=1).values.tolist()
                                                        for region,matrix in merged.items()})
                if errors is not None:
                    group.update(shift_error_median_samples=int(errors[samples].median()),
                                 colocated_within_two=int((errors[samples]<=2).sum()))
                routes[route][view][name] = group
    return dict(status='VERIFIED',target_reads=0,paired_clean_diagnostic_only=True,
                rows=len(ids),modes=modes,audit_fields=FIELDS,region_fields=REGION_FIELDS,routes=routes,
                b1=latency,device=torch.cuda.get_device_name())


if __name__ == '__main__':
    print(json.dumps(measure(read(Path(sys.argv[1]))),allow_nan=False))
