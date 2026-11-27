"""Measure individual and paired post-peer readouts on legal source IQ."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns, sleep

import numpy as np
import torch


def load_formal(runtime, config):
    original = Path(config['w24_release_root'])
    os.chdir(original)
    sys.path.insert(0, str(original))
    from experiments import pamf_lora_main

    directory = runtime.parent/'experiments'/'pamf_lora_main'
    pamf_lora_main.__path__.append(str(directory))
    loader = spec_from_file_location('experiments.pamf_lora_main.w24_formal', directory/'w24_formal.py')
    formal = module_from_spec(loader)
    loader.loader.exec_module(formal)
    return formal


def wait_smoke(root):
    print('Waiting for source-only smoke', file=sys.stderr, flush=True)
    while True:
        try:
            marker = root/'smoke.json'
            if marker.is_file():
                ready = json.loads(marker.read_text(encoding='utf-8'))
                if ready['status'] != 'PASS' or ready['target_reads'] != 0:
                    raise ValueError('Expected the real frozen source-only smoke')
                return
            state = json.loads((root/'state.json').read_text(encoding='utf-8'))
            if state['status'] == 'FAILED':
                raise RuntimeError('Post-peer formal release failed before source-only smoke')
        except json.JSONDecodeError:
            # The formal producer can be writing either short JSON record.
            sleep(2)
            continue
        sleep(30)


@torch.no_grad()
def profile(runtime):
    config = json.loads(runtime.read_text(encoding='utf-8'))
    formal = load_formal(runtime, config)
    if config['run_id'] != formal.RUN_POST_PEER_CPU or config['third_prior'] != formal.POST_PEER_KIND:
        raise ValueError('Expected the immutable post-peer CPU release')
    wait_smoke(Path(config['run_root']))
    config.pop('native_target_root')
    torch.set_num_threads(2)
    heads, bundle, _, _, _, proof = formal.bindings(config, (formal.FULL,), 'cpu')
    from experiments.pamf_lora_main.w24_native_full import native_full_view
    from experiments.pamf_lora_main.w24_post_peer_refinement_source import ARMS, receive

    source = Path(config['source_root'])
    clean = formal.native(source, 'U_s', formal.source_contract(source)['counts']['U_s'])[:32]
    scene = formal.VIEWS[-1]
    low = [native_full_view(packet, scene, formal.SOURCE[0], i, 'source-U-post-peer-latency')
           for i, packet in enumerate(clean)]
    (head,) = heads
    inputs, models = bundle

    def standalone(model, packet):
        view = receive(inputs, (model,), packet)[-1]
        return formal.classify_aperture(head, view)[0]

    paths = (
        lambda packet: formal.classify_aperture(head, formal.received_aperture(packet, 'cpu'))[0],
        lambda packet: standalone(models[0], packet),
        lambda packet: standalone(models[1], packet),
        lambda packet: formal.post_peer_labels(heads, bundle, packet)[0][0, :, 0],
    )
    names = ('direct', *ARMS, 'formal_three_readouts')
    measurements = {}
    for name, packets in (('clean', clean), (scene, low)):
        times = [[] for _ in paths]
        for i, raw in enumerate(packets):
            packet = np.array(raw, copy=True)[None]
            decisions = [None] * len(paths)
            for offset in range(len(paths)):
                j = (i+offset) % len(paths)
                start = perf_counter_ns()
                decisions[j] = paths[j](packet)
                elapsed = (perf_counter_ns()-start) / 1e6
                if i >= 8:
                    times[j].append(elapsed)
            if not np.array_equal(decisions[:3], decisions[3]):
                raise ValueError('A timed source decision differs from the full formal entry')
        measurements[name] = {arm:dict(n=len(x), median_ms=float(np.median(x)), p95_ms=float(np.percentile(x, 95)))
                              for arm, x in zip(names, times, strict=True)}
    return dict(status='MEASURED', source_role='U_s', target_reads=0, label_reads=0, formal_decision_parity=True,
                runtime_commit=config['code_commit'], torch=torch.__version__, threads=torch.get_num_threads(),
                initialization_included=False, simulation_included=False, measurement=measurements, **proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    print(json.dumps(profile(parser.parse_args().runtime.resolve()), indent=2))
