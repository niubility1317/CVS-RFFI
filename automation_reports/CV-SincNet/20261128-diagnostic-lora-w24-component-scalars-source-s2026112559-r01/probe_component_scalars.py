"""Source-only projection limits of the frozen four-component LoRa direction head."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


RUN = '20261128-diagnostic-lora-w24-component-scalars-source-s2026112559-r01'
MODES = ('direct', 'trained_grouped', 'real_gain', 'phase_only', 'complex_gain', 'exact_center')
CONTRASTS = tuple((mode, 'trained_grouped') for mode in MODES[2:]) + (
    ('complex_gain', 'real_gain'), ('exact_center', 'complex_gain'))


def scalar_edits(learned, target, phase):
    from experiments.pamf_lora_main.w24_component_directions import components

    predicted, frame = components(learned, phase)
    wanted, _ = components(target, phase)
    energy = predicted.abs().square().sum((-2, -1), keepdim=True)
    projection = (predicted.conj() * wanted).sum((-2, -1), keepdim=True)
    coefficient = torch.where(energy > 1e-24, projection / energy.clamp_min(1e-24), 0.)
    magnitude = coefficient.abs()
    unit = torch.where(magnitude > 1e-12, coefficient / magnitude.clamp_min(1e-12), 1.)
    coefficients = (coefficient.real.clamp_min(0.), unit, coefficient)
    active = target.abs().amax((-2, -1), keepdim=True) > 1e-12
    projected = tuple(torch.fft.ifft((predicted * c).sum(1), norm='ortho') * frame for c in coefficients)
    return tuple(torch.where(active, edit, learned) for edit in (*projected, target))


def load_formal(runtime, config):
    original = Path(config['w24_release_root'])
    os.chdir(original)
    sys.path.insert(0, str(original))
    from experiments import pamf_lora_main

    directory = runtime.parent / 'experiments' / 'pamf_lora_main'
    pamf_lora_main.__path__.append(str(directory))
    loader = spec_from_file_location('experiments.pamf_lora_main.w24_formal', directory / 'w24_formal.py')
    formal = module_from_spec(loader)
    loader.loader.exec_module(formal)
    return formal


@torch.no_grad()
def render(formal, classifier, parent, tracker, head, raw, signal):
    from experiments.pamf_lora_main.w24_peer_spectral_direction_source import (
        CENTER, center_rms, complex_tensor, coordinates, matched_component_edit, model_iq, native_aperture, tensor,
    )

    def aperture_iq(value):
        return tensor(model_iq(value, center_rms(value)), 'cpu')

    wave = complex_tensor(raw)
    restored = torch.lerp(wave, parent.restore_native(wave), .5)
    received, half = (native_aperture(np.from_dlpack(value.contiguous())) for value in (wave, restored))
    direct = aperture_iq(received)
    _, phase, _, _ = coordinates(tracker, direct[..., :CENTER])
    center = received[..., :CENTER]
    base = complex_tensor(half[..., :CENTER] - center)
    cross = head(wave, restored - wave)[1]
    proposal = native_aperture(np.from_dlpack((wave + cross).contiguous()))[..., :CENTER]
    learned = matched_component_edit(base, complex_tensor(proposal - center), phase)
    target = complex_tensor(native_aperture(signal)[..., :CENTER] - center)
    views = [direct]
    for edit in (learned, *scalar_edits(learned, target, phase)):
        adjusted = half.copy()
        adjusted[..., :CENTER] = center + np.from_dlpack(edit.contiguous())
        views.append(aperture_iq(adjusted))
    return np.stack([formal.classify_aperture(classifier, iq) for iq in views])


def predict(formal, config, output):
    from experiments.pamf_lora_main.w24_edit_budget_audit import full_components
    from experiments.pamf_lora_main.w24_factor_audit import panel

    if not formal.component_directions_ready(config):
        raise ValueError('Expected the registered frozen four-component directions')
    torch.set_num_threads(2)
    (classifier,), (parent, tracker, _, head), _, _, _, proof = formal.bindings(config, (formal.FULL,), 'cpu')
    classify = partial(render, formal, classifier, parent, tracker, head)
    source = Path(config['source_root'])
    contract = formal.source_contract(source)
    legal = formal.native(source, 'U_s', contract['counts']['U_s'])[0]
    for scene in ('clean', formal.VIEWS[-1]):
        received, signal = full_components(scene, (0, formal.SOURCE[0], legal), verify=True)
        labels = classify(received[None], signal[None])
        if labels.shape != (len(MODES), 1) or (scene == 'clean' and not np.all(labels[1:] == labels[1])):
            raise ValueError('Frozen source U smoke failed')
    formal.write(output/'smoke.json', dict(status='PASS', role='U_s', target_reads=0,
                 source_truth_reads=0, **proof))
    print('COMPONENT_SCALAR_SOURCE_SMOKE PASS', flush=True)

    rows, rx, ids = panel(source, contract)
    reference = Path(config['candidate_root'])
    stamp = formal.read(reference/'source-predictions.json')
    if (stamp['status'], stamp['physical_ids'], stamp['views'], stamp['target_reads']) != (
            'PREDICTIONS_FROZEN', ids, list(formal.VIEWS), 0):
        raise ValueError('Expected the frozen source panel of the direction run')
    shared = [stamp['modes'].index(mode) for mode in MODES[:2]]
    data = formal.native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), formal.SOURCE[int(r)], iq) for row, r, iq in zip(rows, rx, data, strict=True)]
    predictions = {}
    with (ProcessPoolExecutor(max_workers=2) as pool,
          np.load(reference/'predictions.npz', allow_pickle=False) as archived):
        for scene in formal.VIEWS:
            stream = pool.map(partial(full_components, scene), jobs, chunksize=6)
            parts = []
            for start in range(0, len(rows), 16):
                y, signal = (np.stack(values) for values in zip(
                    *(next(stream) for _ in rows[start:start+16]), strict=True))
                parts.append(classify(y, signal))
            predictions[scene] = np.concatenate(parts, axis=1)
            if not np.array_equal(predictions[scene][:2], archived[scene][shared]):
                raise ValueError(f'Frozen paired source baseline changed in {scene}')
            print('COMPONENT_SCALAR_SOURCE_PREDICTED', scene, len(rows), flush=True)
    with (output/'predictions.npz').open('xb') as file:
        np.savez(file, **predictions)
    formal.write(output/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', views=list(formal.VIEWS),
                 modes=list(MODES), physical_ids=ids, source_truth_reads=0, target_reads=0,
                 waveform=dict(isolated_noise_truth=True, source_parent=reference.name,
                               preserve_native_edges=True, replay_parity=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('predict', 'score'))
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    config = json.loads(runtime.read_text(encoding='utf-8'))
    formal = load_formal(runtime, config)
    if (config['run_id'] != formal.RUN_COMPONENT_DIRECTIONS_CPU or
            config['third_prior'] != formal.COMPONENT_DIRECTIONS_KIND):
        raise ValueError('Expected the pinned four-component Full CPU runtime')
    config.pop('native_target_root')
    output = Path(config['project']) / 'runs' / RUN
    if args.phase == 'predict':
        output.mkdir()
        predict(formal, config, output)
    else:
        from experiments.pamf_lora_main.w24_factor_audit import score_source

        score_source(runtime, modes=MODES, output=output, contrasts=CONTRASTS)


if __name__ == '__main__':
    main()
