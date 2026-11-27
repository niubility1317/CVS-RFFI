"""Measure batch-one CPU inference from source U_s IQ using the frozen Full release."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns, sleep

import numpy as np
import torch


def load_formal(runtime, rt):
    original = Path(rt['w24_release_root'])
    os.chdir(original)
    sys.path.insert(0, str(original))
    from experiments import pamf_lora_main

    directory = runtime.parent / 'experiments' / 'pamf_lora_main'
    pamf_lora_main.__path__.append(str(directory))
    loader = spec_from_file_location('experiments.pamf_lora_main.w24_formal', directory / 'w24_formal.py')
    formal = module_from_spec(loader)
    loader.loader.exec_module(formal)
    return formal


def wait_for_smoke(root):
    state_file, marker = root / 'state.json', root / 'smoke.json'
    print('Waiting for the formal source-only smoke', file=sys.stderr, flush=True)
    while True:
        try:
            if marker.is_file():
                proof = json.loads(marker.read_text(encoding='utf-8'))
                if proof['status'] != 'PASS' or proof['target_reads'] != 0:
                    raise ValueError('Expected a passed source-only smoke for this freeze')
                return
            state = json.loads(state_file.read_text(encoding='utf-8'))
            if state['status'] == 'FAILED':
                raise RuntimeError('The formal controller failed before the source-only smoke')
        except json.JSONDecodeError:
            # The formal controller writes these small JSON files in place.
            sleep(2)
            continue
        sleep(30)


@torch.no_grad()
def measure(formal, heads, models, inputs, arm_names=('peer_grouped', 'trained_grouped')):
    from experiments.pamf_lora_main.w24_peer_spectral_direction_source import (
        CENTER, center_rms, complex_tensor, coordinates, matched_component_edit, model_iq, native_aperture, tensor,
    )

    (classifier,) = heads
    parent, tracker, peer, trained = models

    def direct(packet):
        return formal.classify_aperture(classifier, formal.received_aperture(packet, 'cpu'))[0]

    def one_head(head, packet):
        # The formal matrix shares these operations across two heads. Time them for each standalone route.
        wave = complex_tensor(packet)
        half_wave = torch.lerp(wave, parent.restore_native(wave), .5)
        observed, half = (native_aperture(np.from_dlpack(value.contiguous())) for value in (wave, half_wave))
        original = tensor(model_iq(observed, center_rms(observed)), 'cpu')
        _, phase, _, _ = coordinates(tracker, original[..., :CENTER])
        center = observed[..., :CENTER]
        cross = head(wave, half_wave - wave)[1]
        proposed = native_aperture(np.from_dlpack((wave + cross).contiguous()))[..., :CENTER] - center
        edit = matched_component_edit(complex_tensor(half[..., :CENTER] - center), complex_tensor(proposed), phase)
        adjusted = half.copy()
        adjusted[..., :CENTER] = center + np.from_dlpack(edit.contiguous())
        return formal.classify_aperture(classifier, tensor(model_iq(adjusted, center_rms(adjusted)), 'cpu'))[0]

    paths = {'direct': direct, arm_names[0]: lambda packet: one_head(peer, packet),
             arm_names[1]: lambda packet: one_head(trained, packet),
             'formal_three_readouts': lambda packet: formal.component_direction_labels(heads, models, packet)[0][0, :, 0]}
    names = tuple(paths)
    measurement = {}
    for scene, packets in inputs.items():
        timings = {name: [] for name in names}
        for i, raw in enumerate(packets):
            packet = np.array(raw, copy=True)[None]
            predictions = {}
            for offset in range(len(names)):
                name = names[(i + offset) % len(names)]
                start = perf_counter_ns()
                predictions[name] = paths[name](packet)
                elapsed = (perf_counter_ns() - start) / 1e6
                if i >= 8:
                    timings[name].append(elapsed)
            if not np.array_equal([predictions[name] for name in names[:-1]], predictions[names[-1]]):
                raise ValueError('Timed batch-one paths disagreed with the official decision')
        measurement[scene] = {name: dict(n=len(ms), median_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95)))
                              for name, ms in timings.items()}
    return measurement


def profile(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    formal = load_formal(runtime, rt)
    matrices = {formal.RUN_COMPONENT_DIRECTIONS_CPU: (formal.COMPONENT_DIRECTIONS_KIND,
                                                     ('peer_grouped', 'trained_grouped'))}
    if hasattr(formal, 'RUN_ONLINE_DIRECTIONS_CPU'):
        matrices[formal.RUN_ONLINE_DIRECTIONS_CPU] = (formal.ONLINE_DIRECTIONS_KIND,
                                                    ('fixed_noise_grouped', 'online_noise_grouped'))
    kind, names = matrices.get(rt.get('run_id'), (None, None))
    if kind is None or rt.get('third_prior') != kind or rt.get('ungated') is not True:
        raise ValueError('Expected the registered frozen component-directions CPU runtime')
    wait_for_smoke(Path(rt['run_root']))
    rt.pop('native_target_root')
    torch.set_num_threads(2)
    heads, models, _, _, _, provenance = formal.bindings(rt, (formal.FULL,), 'cpu')
    from experiments.pamf_lora_main.w24_native_full import native_full_view

    source = Path(rt['source_root'])
    clean = formal.native(source, 'U_s', formal.source_contract(source)['counts']['U_s'])[:32]
    scene = formal.VIEWS[-1]
    low = [native_full_view(packet, scene, formal.SOURCE[0], i, 'source-U-component-latency')
           for i, packet in enumerate(clean)]
    result = measure(formal, heads, models, {'clean': clean, scene: low}, names)
    return dict(status='MEASURED', source_role='U_s', target_reads=0, label_reads=0, formal_decision_parity=True,
                runtime_commit=rt['code_commit'], torch=torch.__version__, threads=torch.get_num_threads(),
                initialization_included=False, simulation_included=False, measurement=result, **provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    print(json.dumps(profile(parser.parse_args().runtime.resolve()), indent=2))
