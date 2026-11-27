"""Time the frozen Full Practical readouts on source-only N607 IQ."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
import torch

@torch.no_grad()
def profile(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    original = Path(rt['w24_release_root'])
    os.chdir(original)
    sys.path.insert(0, str(original))
    from experiments import pamf_lora_main

    # Follow the archived prediction worker's source import order.
    directory = runtime.resolve().parent/'experiments'/'pamf_lora_main'
    pamf_lora_main.__path__.append(str(directory))
    loader = spec_from_file_location('experiments.pamf_lora_main.w24_formal', directory/'w24_formal.py')
    formal = module_from_spec(loader)
    loader.loader.exec_module(formal)
    from experiments.pamf_lora_main.w24_native_full import native_full_view
    from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_KIND

    if not formal.cpu_practical(rt) or rt.get('third_prior') != PRACTICAL_KIND or rt.get('ungated') is not True:
        raise ValueError('The frozen two-arm Full Practical CPU runtime is required')
    rt.pop('native_target_root')
    heads, arms, tracker, _, _, provenance = formal.bindings(rt, (formal.FULL,), 'cpu')
    (classifier,) = heads
    source = Path(rt['source_root'])
    bank = formal.native(source, 'U_s', formal.source_contract(source)['counts']['U_s'])
    clean = bank[:32]
    low = [native_full_view(packet, formal.VIEWS[-1], formal.SOURCE[0], i, 'source-U-practical-latency')
           for i, packet in enumerate(clean)]
    inputs = [(np.array(raw, copy=True)[None], noisy[None]) for raw, noisy in zip(clean, low, strict=True)]
    names = ('direct', *(f'{arm}_half' for arm in provenance['parameters']))
    if formal.evaluation_modes(formal.FULL, True, third_prior=PRACTICAL_KIND)[-2:] != names[1:]:
        raise ValueError('Frozen fit-arm and prediction identifiers must match')
    paths = (None, *arms)
    timings = {scene: {name: [] for name in names} for scene in ('clean', formal.VIEWS[-1])}
    for i, packet_pair in enumerate(inputs):
        for scene, packet in zip(timings, packet_pair, strict=True):
            predicted = np.empty(len(paths), dtype=np.int16)
            for offset in range(len(paths)):
                j = (i + offset) % len(paths)
                start = perf_counter_ns()
                aperture = formal.received_aperture(packet, 'cpu')
                if paths[j] is not None:
                    aperture = formal.native_aperture(paths[j], packet, aperture, .5)
                predicted[j] = formal.classify_aperture(classifier, aperture)[0]
                elapsed = (perf_counter_ns() - start) / 1e6
                if i >= 8:
                    timings[scene][names[j]].append(elapsed)
            expected, _ = formal.prior_labels(heads, arms, tracker, packet, 'cpu')
            if not np.array_equal(predicted, expected[0, :, 0]):
                raise ValueError('Timed B1 paths changed the frozen formal decisions')
    summary = {scene: {name: dict(n=len(ms), median_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95)))
                       for name, ms in by_arm.items()} for scene, by_arm in timings.items()}
    return dict(status='MEASURED', source_role='U_s', target_reads=0, label_reads=0, formal_decision_parity=True,
                runtime_commit=rt['code_commit'], torch=torch.__version__, threads=torch.get_num_threads(),
                initialization_included=False, simulation_included=False, measurement=summary, **provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    print(json.dumps(profile(parser.parse_args().runtime.resolve()), indent=2))
