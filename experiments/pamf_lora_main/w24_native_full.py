"""Frozen PAMF W24 comparison on native-rate Full practical LoRa packets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath
import os
import subprocess
import time
import zipfile

import numpy as np
from scipy.signal import resample_poly
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.data import CLASSES, SOURCE, TARGET, VIEWS, WINDOW_START
from experiments.pamf_lora_main.full_packet_data import LIMIT
from experiments.pamf_lora_main.packet_data import CHIRPS, PACKET_LEN
from experiments.pamf_lora_main.packet_physics import CONTEXT, channel, chirp_samples, hardware


RUN = '20261125-phase1-lora-w24-native-full-s2026092702-r02'
W24_RUN = '20261125-phase1-lora-pamf-tri-clean-nativewide98-m2-r01'
W24_COMMIT = 'e6dde7d40f2cc3dd7b21d83daa4bcbabac52d3ae'
METHOD = 'phase_native24'
SEED = 2026092702
MODES = ('direct', 'coherent_half_center', 'complex_response_half_center')
DATA_RUN = '20261122-phase1-lora-full-aligned-data-r01'
BATCH = 64
STRIDE = 1024
EDGE = 24


def native_full_view(packet, scene, receiver, physical_index, namespace):
    """Process every recorded sample, with zeros only beyond the capture."""
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed

    if scene == 'clean':
        return packet
    if scene not in VIEWS or packet.shape != (PACKET_LEN,):
        raise ValueError('Expected a registered Full scene and one native LoRa packet')
    normalized, _ = normalize_input(packet)
    cfg = channel(scene)
    delay = cfg.fractional_delay_half_length
    seed = stable_seed(2026112207, namespace, scene, receiver, str(physical_index))
    stream = ChannelStream(cfg, seed, hardware(scene, f'{namespace}/{receiver}'))
    received, meta = stream.process(np.pad(normalized, (0, delay)),
                                    preceding_context=np.zeros(CONTEXT, dtype=normalized.dtype))
    if (meta['boundary'] != 'provided_context' or meta['channel_equalization_applied']
            or meta['numerical_filter_latency_samples'] != delay):
        raise ValueError('Full practical alignment or receive-side settings differ')
    packet = received[delay:delay + PACKET_LEN]
    if packet.shape != (PACKET_LEN,) or not np.isfinite(packet).all():
        raise ValueError('Incomplete native Full practical observation')
    return packet.astype(np.complex64, copy=False)


def project_delta(packet, delta, scale=1.):
    """Write back only the six registered DURA chirp interiors."""
    left, right = EDGE - WINDOW_START, STRIDE - EDGE - WINDOW_START
    corrected = packet.copy()
    active = np.any(delta != 0, axis=(1, 2, 3))
    if active.any():
        wave = delta[active, :, 0] + 1j * delta[active, :, 1]
        scales = np.broadcast_to(scale, (len(packet),))[active]
        restored = resample_poly(wave, 125, 32, axis=-1) * scales[:, None, None]
        edit = restored[..., left:right]
        # Ramp inside the protected interval to avoid a discontinuity at the splice.
        ramp = np.sin(np.pi * np.arange(EDGE) / (2 * EDGE)) ** 2
        edit[..., :EDGE] *= ramp
        edit[..., -EDGE:] *= ramp[::-1]
        for pos, chirp in enumerate(CHIRPS):
            start = chirp * STRIDE
            corrected[active, start + EDGE:start + STRIDE - EDGE] += edit[:, pos]
    return corrected


def load_models(rt, device):
    from experiments.pamf_lora_main.full_packet_views import source_contract
    from experiments.pamf_lora_main.full_dura_detail import (
        COMPLEX_RESPONSE_RUN, load_detail, phase_baseline_models,
    )
    from experiments.pamf_lora_main.tri_clean95 import NATIVE_WIDE_ROWS, clean_model

    source = Path(rt['source_root'])
    contract = source_contract(source)
    parent = Path(rt['w24_run_root'])
    folder = parent / 'F' / METHOD
    frozen = read(folder / 'complete.json')
    fit = frozen['frozen']
    if (read(parent / 'selection.json')['names'] != [METHOD]
            or frozen['status'] != 'SOURCE_FROZEN'
            or (frozen['method'], frozen['seed'], fit['epoch'], fit['correct'], fit['n'])
            != (METHOD, SEED, 108, 11_781, 12_000)
            or frozen['target_reads'] != 0 or frozen['satellite_batches'] != 0):
        raise ValueError('W24 original source-frozen checkpoint provenance differs')
    recognizer = clean_model(dict(scratch=True), parent, contract, NATIVE_WIDE_ROWS[METHOD], device)
    recognizer.load_state_dict(torch.load(folder / 'best.pt', map_location=device, weights_only=True),
                               strict=True)
    response_root = parent.parent / COMPLEX_RESPONSE_RUN
    base, detail = phase_baseline_models(response_root, contract, SEED, device)
    response = load_detail(response_root, contract, SEED, device)
    return recognizer.eval().requires_grad_(False), base, detail, response


@torch.inference_mode()
def recover(packets, restorer, device):
    from experiments.pamf_lora_main.full_dura_detail import extra_correction, half_baseline, scale_detail

    base, detail, response = restorer
    chirps = np.stack([chirp_samples(packet) for packet in packets])
    scale = np.sqrt(np.mean(np.abs(chirps) ** 2, axis=(1, 2)))
    if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ValueError('Empty or nonfinite native receive packet')
    channels = np.stack((chirps.real, chirps.imag), axis=2) / scale[:, None, None, None]
    iq = tensor(channels.astype(np.float32, copy=False), device)
    half, active = half_baseline(base, detail, iq)
    chosen, restored = extra_correction(response, half, torch.zeros_like(active))
    corrected = half.clone()
    corrected[chosen] = scale_detail(half[chosen], restored, .5)
    return (packets, *(project_delta(packets, np.from_dlpack((row - iq).cpu().contiguous()), scale)
                       for row in (half, corrected)))


@torch.inference_mode()
def classify(readout, packets, device):
    from experiments.pamf_lora_main.clean_aperture import ALL_CHIRPS, clean_iq

    iq = clean_iq(packets, 384, ALL_CHIRPS, native_edges=True, edge_points=EDGE)
    if iq.shape != (len(packets), 8, 2, 432):
        raise ValueError('Expected the registered W24 aperture')
    logits = readout(tensor(iq, device))[0]
    if logits.shape != (len(packets), len(CLASSES)) or not torch.isfinite(logits).all():
        raise ValueError('Nonfinite or incomplete frozen W24 prediction')
    return np.asarray(logits.argmax(1).tolist(), dtype=np.int16)


def smoke(rt, models, device):
    from experiments.pamf_lora_main.full_packet_views import native, source_contract

    source = Path(rt['source_root'])
    count = source_contract(source)['counts']['U_s']
    clean = native(source, 'U_s', count)[:4]
    received = np.stack([native_full_view(x, VIEWS[-1], SOURCE[0], i, 'w24-smoke') for i, x in enumerate(clean)])
    for packets in (clean, received):
        for restored in recover(packets, models[1:], device):
            original = packets.reshape(-1, 8, STRIDE)
            changed = restored.reshape(original.shape)
            if (not np.array_equal(original[..., :EDGE], changed[..., :EDGE])
                    or not np.array_equal(original[..., -EDGE:], changed[..., -EDGE:])):
                raise ValueError('DURA projection modified a W24 native boundary')
            classify(models[0], restored, device)
    write(Path(rt['run_root']) / 'smoke.json', dict(
        status='PASS', W24_commit=W24_COMMIT, source_packets=4, target_reads=0, target_truth_reads=0,
        aperture=[8, 2, 432], processing_route='full_native'))


def channel_job(scene, receiver, item):
    index, packet = item
    return native_full_view(packet, scene, receiver, index, 'target')


def receipt():
    return dict(status='PREDICTIONS_FROZEN', W24_commit=W24_COMMIT, seed=SEED,
                modes=list(MODES), receivers=list(TARGET), views=list(VIEWS), per_receiver=LIMIT,
                processing_route='full_native', target_truth_read=False)


def predict(rt, models, device):
    from experiments.pamf_lora_main.full_packet_views import native, target_contract

    target = Path(rt['native_target_root'])
    target_contract(target)
    output = Path(rt['run_root']) / 'predictions'
    output.mkdir()
    activity = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for rx in TARGET:
            raw = native(target, rx, LIMIT)
            for scene in VIEWS:
                if scene == 'clean':
                    packets = raw
                else:
                    jobs = pool.map(partial(channel_job, scene, rx), enumerate(raw), chunksize=16)
                    packets = np.stack(list(jobs))
                predictions = {mode: [] for mode in MODES}
                active = dict.fromkeys(MODES[1:], 0)
                for i in range(0, LIMIT, BATCH):
                    original = packets[i:i + BATCH]
                    waves = recover(original, models[1:], device)
                    direct = classify(models[0], original, device)
                    predictions['direct'].append(direct)
                    for mode, wave in zip(MODES[1:], waves[1:], strict=True):
                        use = np.any(wave != original, axis=1)
                        labels = direct.copy()
                        if use.any():
                            labels[use] = classify(models[0], wave[use], device)
                        active[mode] += int(use.sum())
                        predictions[mode].append(labels)
                for mode, rows in predictions.items():
                    np.save(output / f'{rx}__{scene}__{mode}.npy', np.concatenate(rows))
                activity[f'{rx}/{scene}'] = active
                print('W24_FULL_PREDICTED', rx, scene, active, flush=True)
    write(output / 'activity.json', activity)
    write(output / 'complete.json', receipt())


def validate_predictions(folder):
    if read(folder / 'complete.json') != receipt():
        raise ValueError('Missing or incompatible W24 frozen matrix')
    result = {}
    activity = read(folder / 'activity.json')
    if set(activity) != {f'{rx}/{view}' for rx in TARGET for view in VIEWS}:
        raise ValueError('Missing W24 activation records')
    for rx in TARGET:
        for view in VIEWS:
            expected = activity[f'{rx}/{view}']
            if set(expected) != set(MODES[1:]) or any(type(n) is not int or not 0 <= n <= LIMIT for n in expected.values()):
                raise ValueError('Invalid W24 activity counts')
            for mode in MODES:
                data = np.load(folder / f'{rx}__{view}__{mode}.npy', allow_pickle=False)
                if data.shape != (LIMIT,) or data.dtype != np.int16 or data.min() < 0 or data.max() >= len(CLASSES):
                    raise ValueError('Invalid W24 labels in frozen matrix')
                if mode != 'direct' and np.count_nonzero(data != result[rx, view, 'direct']) > expected[mode]:
                    raise ValueError('Changed labels exceed changed IQ count')
                result[rx, view, mode] = data
    return result, activity


def score(predictions, dataset, output):
    from experiments.pamf_lora_main.data import label_ids, packet_indices, recording
    from experiments.pamf_lora_main.scoring import macro_f1

    pred, activity = validate_predictions(predictions)
    output.mkdir(parents=True, exist_ok=False)
    truth = {}
    with zipfile.ZipFile(dataset / 'multiple_receiver_test.zip') as archive:
        for rx_index, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as capture:
                truth[rx] = label_ids(capture, LIMIT)[packet_indices(LIMIT, rx_index)]
    rows = {}
    for mode in MODES:
        rows[mode] = {}
        for view in VIEWS:
            y = np.concatenate([truth[rx] for rx in TARGET])
            p = np.concatenate([pred[rx, view, mode] for rx in TARGET])
            direct = np.concatenate([pred[rx, view, 'direct'] for rx in TARGET]) == y
            correct = p == y
            rows[mode][view] = dict(n=len(y), correct=int(correct.sum()), accuracy=float(correct.mean()),
                                     macro_f1=macro_f1(y, p), repairs=int((correct & ~direct).sum()),
                                     harms=int((~correct & direct).sum()),
                                     by_rx={rx: int((pred[rx, view, mode] == truth[rx]).sum()) for rx in TARGET})
    total = {mode: sum(rows[mode][v]['correct'] for v in VIEWS if v != 'clean') for mode in MODES}
    write(output / 'scores.json', dict(freeze=receipt(), truth_last=True, views=rows,
                                       six_correct=total, six_n=LIMIT * len(TARGET) * (len(VIEWS) - 1),
                                       activity=activity))


def dispatch(rt, runtime):
    from experiments.pamf_lora_main.runner import available

    root = Path(rt['run_root'])
    root.mkdir()
    write(root / 'state.json', dict(status='WAIT_GPU'))
    deadline = time.monotonic() + 3600
    while not (gpus := available(rt['allowed_gpus'], max_compute_processes=rt['max_compute_processes'])):
        if time.monotonic() >= deadline:
            write(root / 'state.json', dict(status='FAILED', reason='GPU capacity wait expired'), exclusive=False)
            raise RuntimeError('No GPU capacity within the registered wait window')
        time.sleep(30)
    gpu = next(i for i in rt['allowed_gpus'] if i in gpus)
    w24 = Path(rt['w24_release_root'])
    if read(w24 / 'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('W24 frozen source release code differs')
    # Execute W24 with the exact source tree that produced its checkpoint.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(w24),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    command = [rt['python'], '-u', str(Path(__file__).resolve()), 'worker', str(runtime)]
    with (root / 'predict.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(command, cwd=w24, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(root / 'state.json', dict(status='RUNNING', pid=proc.pid, gpu=gpu), exclusive=False)
        result = proc.wait()
    if result != 0 or read(root / 'predictions' / 'complete.json') != receipt():
        write(root / 'state.json', dict(status='FAILED', pid=proc.pid, gpu=gpu, exit_code=result), exclusive=False)
        raise RuntimeError('W24 Full worker failed; preserve the original output root')
    write(root / 'state.json', dict(status='PREDICTIONS_FROZEN', gpu=gpu), exclusive=False)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'worker'):
        commands.add_parser(action).add_argument('runtime', type=Path)
    publication = commands.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        publication.add_argument('--' + key, type=Path, required=True)
    scoring = commands.add_parser('score')
    for key in ('predictions', 'dataset', 'output'):
        scoring.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        context = read(args.context)
        rt = {key: context[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        native = project / 'releases' / f'{DATA_RUN}-release' / 'data'
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(project / 'releases' / f'{RUN}-release'),
                  run_root=str(project / 'runs' / RUN), source_root=str(native / 'source'),
                  native_target_root=str(native / 'target'), w24_run_root=str(project / 'runs' / W24_RUN),
                  w24_release_root=str(project / 'releases' / f'{W24_RUN}-release'),
                  allowed_gpus=[3, 0, 5, 7, 2, 6, 1, 4], max_compute_processes=8, cpu_workers=12)
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'score':
        score(args.predictions, args.dataset, args.output)
    elif args.action == 'dispatch':
        dispatch(read(args.runtime), args.runtime)
    else:
        from experiments.pamf_lora_main.runner import activate
        runtime = read(args.runtime)
        activate(runtime)
        models = load_models(runtime, 'cuda:0')
        smoke(runtime, models, 'cuda:0')
        predict(runtime, models, 'cuda:0')


if __name__ == '__main__':
    main()
