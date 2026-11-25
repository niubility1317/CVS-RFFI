"""Score the receive-frame echo and mirror treatment with frozen LoRa controls."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.channel_blind_source import panel, partition, receive_only
from experiments.pamf_lora_main.complex_vector_source import load_reader, score
from experiments.pamf_lora_main.dura_apertures import ApertureTrackers
from experiments.pamf_lora_main.dura_dual_aperture_evaluate import bind_source_controls
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.local_wiener_source import freeze, main
from experiments.pamf_lora_main.receive_frame_action import receive_fit, restore_pair
from experiments.pamf_lora_main.relative_echo import shift


RUN = '20261125-phase1-lora-receive-frame-action-source-m2-r01'
DEVICE = 'cuda:0'
MODES = ('direct', 'dura_half', 'expanded_half', 'tracked_half', 'receive_half',
         'tracked_then_expanded', 'receive_then_expanded')
COMPARISONS = (('receive_half', 'tracked_half'), ('receive_half', 'direct'),
               ('receive_half', 'dura_half'), ('receive_then_expanded', 'tracked_then_expanded'),
               ('receive_then_expanded', 'expanded_half'), ('receive_then_expanded', 'dura_half'))
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0, trace_reads=0)


def source(rt):
    root = Path(rt['source_root'])
    contract = source_contract(root)
    data, rx, _, _ = source_role(root, contract, 'U_s')
    _, cal, held = partition(rx, fit=4000)
    return data, contract, cal, held


@torch.no_grad()
def calibrate(rt):
    started = perf_counter()
    data, _, cal, _ = source(rt)
    trackers = ApertureTrackers().to(DEVICE).eval()
    selected = tensor(data[cal].view('float32'), DEVICE).view(torch.complex64)
    peaks = []
    for tracker, count in ((trackers.narrow, 256), (trackers.wide, 384)):
        scores = []
        for batch in selected.split(32):
            wave = tracker.aperture(batch)[..., :count]
            if count == 384:
                wave = wave[:, 1:7]
            y, phase, *_ = coordinates(tracker, real_iq(wave))
            _, evidence, *_ = receive_fit(y, phase)
            scores.append(evidence)
        scores = torch.cat(scores)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite clean source receive-frame evidence')
        peaks.append(float(scores.max()))
    write(Path(rt['run_root'])/'gate.json', dict(status='SOURCE_FROZEN', rows=len(cal), peaks=peaks,
          threshold=1.05*max(peaks), optimizer_updates=0, parameters=0, seconds=perf_counter()-started, **ISOLATION))


def gate_value(root):
    record = read(root/'gate.json')
    if (record['status'] != 'SOURCE_FROZEN' or record['rows'] != 2500 or len(record['peaks']) != 2
            or any(record[key] != 0 for key in ISOLATION)
            or record['threshold'] != 1.05*max(record['peaks'])
            or not 0. < record['threshold'] < 1.):
        raise ValueError('Expected the common two-rate source U-only evidence threshold')
    return record['threshold']


def render(iq, baseline, tracker, expanded, threshold):
    a, b, active = restore_pair(iq, tracker, threshold)
    original, _ = baseline(iq)
    expanded_out = expanded(iq)
    first = expanded_out.clone()
    second = expanded_out.clone()
    if active.any():
        first[active] = expanded(a[active])
        second[active] = expanded(b[active])
    return torch.stack((iq, original, expanded_out, a, b, first, second)), int(active.sum())


@torch.no_grad()
def smoke(rt, threshold):
    data, contract, cal, held = source(rt)
    root = Path(rt['run_root'])
    reader = load_reader(root, contract, DEVICE)
    binding, expanded = bind_source_controls(rt, 'full', contract, DEVICE)
    baseline, tracker = binding[-2:]
    capture = tensor(data[np.concatenate((cal[:2], held[:10]))].view('float32'), DEVICE).view(torch.complex64)
    native = ApertureTrackers().to(DEVICE).eval().narrow
    clean = real_iq(native.aperture(capture[:2]))
    raw = capture[2:].unflatten(-1, (8, 1024))
    sign = (1.-2.*(torch.arange(8, device=DEVICE)//2%2)).view(1, 8, 1)
    affected = raw + .21*sign*raw.conj() + .095j*sign*shift(raw, 8)
    disturbed = real_iq(native.aperture(affected.flatten(-2)))
    a, b, clean_active = restore_pair(clean, tracker, threshold)
    views, changed = render(disturbed, baseline, tracker, expanded, threshold)
    logits = reader(views.flatten(0, 1))[0]
    if (clean_active.any() or not torch.equal(a, clean) or not torch.equal(b, clean)
            or not changed or not torch.isfinite(logits).all()):
        raise ValueError('A genuine frozen-F no-target smoke must exercise both gates and receive frames')
    durations = {}
    for name, example in (('clean', clean[:1]), ('generic_disturbed', disturbed[:1])):
        values = []
        for _ in range(18):
            torch.cuda.synchronize()
            start = perf_counter()
            _, _, active = restore_pair(example, tracker, threshold)
            torch.cuda.synchronize()
            values.append(1e3*(perf_counter()-start))
        durations[name] = dict(median_ms=float(np.median(values[5:])), active=bool(active[0]))
    write(root/'smoke.json', dict(status='PASS', target_reads=0, V_reads=0, active_in_smoke=changed,
          clean_active=0, modes=list(MODES), analytic_paired_b1=durations, device=torch.cuda.get_device_name()))


def evaluate(rt, route, threshold):
    contract = source_contract(Path(rt['source_root']))
    binding, expanded = bind_source_controls(rt, route, contract, DEVICE)

    def paths(iq, _unused, baseline, tracker, _models):
        return render(iq, baseline, tracker, expanded, threshold)

    score(rt, route, panel=panel, modes=MODES, render=paths, binding=binding, frozen_models=(),
          require_clean_bypass=False, replay_fn=receive_only, finalize=freeze)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, modes=MODES, comparisons=COMPARISONS,
         fit=calibrate, load_gate=gate_value, check=smoke, evaluate=evaluate)
