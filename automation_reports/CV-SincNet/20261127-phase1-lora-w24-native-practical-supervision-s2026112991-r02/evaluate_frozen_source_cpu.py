"""Complete the registered Practical r02 source V readouts on CPU."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from itertools import repeat
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


MODES = ('direct', 'practical_matched_dose_half', 'practical_exact_additive_half')
OUTPUT = '20261128-diagnostic-lora-w24-practical-r02-frozen-source-s2026112559-r01'
REFERENCE = '20261127-diagnostic-lora-w24-practical-direction-factor-s2026112559-r01'


def load_formal(runtime, rt):
    original = Path(rt['w24_release_root'])
    os.chdir(original)
    sys.path.insert(0, str(original))
    from experiments import pamf_lora_main

    directory = runtime.parent / 'experiments' / 'pamf_lora_main'
    pamf_lora_main.__path__.append(str(directory))
    loader = spec_from_file_location('experiments.pamf_lora_main.w24_formal', directory / 'w24_formal.py')
    module = module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def predict(formal, rt, output):
    from experiments.pamf_lora_main.w24_edit_budget_audit import WAVEFORM_NAMESPACE
    from experiments.pamf_lora_main.w24_factor_audit import panel
    from experiments.pamf_lora_main.w24_native_full import native_full_view
    from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_ARMS, PRACTICAL_RUN

    if (Path(rt['candidate_root']) != output.parent/PRACTICAL_RUN or
            formal.read(Path(rt['candidate_root'])/'state.json') != dict(
                status='SOURCE_FROZEN', arms=list(PRACTICAL_ARMS), steps=24000)):
        raise ValueError('Both Practical r02 source checkpoints must already be frozen')
    torch.set_num_threads(2)
    heads, arms, tracker, _, _, provenance = formal.bindings(rt, (formal.FULL,), 'cpu')
    evaluate = partial(formal.prior_labels, heads, arms, tracker, device='cpu')
    source = Path(rt['source_root'])
    contract = formal.source_contract(source)
    one = formal.native(source, 'U_s', contract['counts']['U_s'])[0]
    for scene in ('clean', formal.VIEWS[-1]):
        packet = native_full_view(one, scene, formal.SOURCE[0], 0, WAVEFORM_NAMESPACE)[None]
        labels, active = evaluate(packet)
        if labels.shape != (1, len(MODES), 1) or active != 1:
            raise ValueError('Failed the real-checkpoint no-target smoke')
    formal.write(output/'smoke.json', dict(status='PASS', source_role='U_s', target_reads=0,
                                        source_truth_reads=0, **provenance))
    print('PRACTICAL_R02_SOURCE_SMOKE PASS', flush=True)

    rows, rx, ids = panel(source, contract)
    ref = output.parent/REFERENCE
    prior = formal.read(ref/'source-predictions.json')
    if (prior['status'], prior['physical_ids'], prior['views'], prior['target_reads']) != (
            'PREDICTIONS_FROZEN', ids, list(formal.VIEWS), 0):
        raise ValueError('Expected the same fixed source V panel as the exact-additive diagnostic')
    shared = [prior['modes'].index(m) for m in ('direct', 'frozen_half')]
    data = formal.native(source, 'V', contract['counts']['V'])[rows]
    receivers = [formal.SOURCE[int(receiver)] for receiver in rx]
    predictions = {}
    with (ProcessPoolExecutor(max_workers=2) as pool,
          np.load(ref/'predictions.npz', allow_pickle=False) as archived):
        for scene in formal.VIEWS:
            stream = pool.map(native_full_view, data, repeat(scene), receivers, rows.tolist(),
                              repeat(WAVEFORM_NAMESPACE), chunksize=6)
            batches = []
            for i in range(0, len(rows), 16):
                y = np.stack([next(stream) for _ in rows[i:i+16]])
                labels, active = evaluate(y)
                if active != len(y):
                    raise ValueError(f'Incomplete source predictions for {scene}')
                batches.append(labels[0])
            predictions[scene] = np.concatenate(batches, axis=1)
            if not np.array_equal(predictions[scene][[0, 2]], archived[scene][shared]):
                raise ValueError(f'Shared frozen predictions changed for {scene}')
            print('PRACTICAL_R02_SOURCE_PREDICTED', scene, len(rows), flush=True)

    with (output/'predictions.npz').open('xb') as file:
        np.savez(file, **predictions)
    formal.write(output/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', physical_ids=ids,
                 modes=list(MODES), views=list(formal.VIEWS), target_reads=0, source_truth_reads=0,
                 waveform=dict(received_iq_only=True, matched_prior=REFERENCE, run=PRACTICAL_RUN,
                               inference_commit=rt['code_commit'], shared_decision_parity=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('predict', 'score'))
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    formal = load_formal(runtime, rt)
    if not formal.cpu_practical(rt):
        raise ValueError('Expected the frozen Practical r02 Full CPU prediction release')
    rt.pop('native_target_root')
    output = Path(rt['project'])/'runs'/OUTPUT
    if args.phase == 'predict':
        output.mkdir()
        predict(formal, rt, output)
    else:
        from experiments.pamf_lora_main.w24_factor_audit import score_source

        score_source(runtime, modes=MODES, output=output, contrasts=((MODES[1], MODES[2]),))


if __name__ == '__main__':
    main()
