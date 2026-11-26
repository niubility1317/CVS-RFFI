"""Measure source-only instrumented noise correlation in the registered Full aperture."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


RUN = '20261126-phase1-lora-w24-noise-color-s2026112991-r01'
SCENES = ('practical_high', 'practical_low_urban')
NAMESPACE = 'source-U-noise-color-diagnostic'


def audit(runtime):
    rt = json.loads(runtime.read_text(encoding='utf-8'))
    if rt['run_id'] != RUN or rt['third_prior'] != 'noise-color':
        raise ValueError('Expected the registered source noise-color run')
    sys.path.insert(0, rt['w24_release_root'])
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(rt['release_root']) / 'experiments/pamf_lora_main'))
    from experiments.pamf_lora_main.full_packet_views import native, source_contract
    from experiments.pamf_lora_main.w24_aperture_noise_audit import native_aperture
    from experiments.pamf_lora_main.w24_factor_audit import trace_components
    from experiments.pamf_lora_main.w24_native_full import SOURCE, native_full_view

    source = Path(rt['source_root'])
    contract = source_contract(source)
    physical_ids = contract['role_ids']['U_s']
    iq = native(source, 'U_s', contract['counts']['U_s'])
    rows = [(receiver, [i for i, key in enumerate(physical_ids)
                        if key.startswith(receiver + '/')][-8:]) for receiver in SOURCE]
    if any(len(ids) != 8 for _, ids in rows):
        raise ValueError('Expected eight fixed source U_s packets per registered receiver')
    summaries = {}
    for scene in SCENES:
        fields = []
        for receiver, ids in rows:
            for index in ids:
                received, signal, noise = trace_components(iq[index], scene, receiver, index, NAMESPACE)
                if not fields and not np.array_equal(
                        received, native_full_view(iq[index], scene, receiver, index, NAMESPACE)):
                    raise ValueError('Trace does not match the registered source Full forward')
                fields.append(noise.astype(np.complex64))
        center = native_aperture(np.stack(fields))[..., :384]
        x0, x1 = center[..., :-1], center[..., 1:]
        norm = np.sqrt((np.abs(x0)**2).mean((1, 2)) * (np.abs(x1)**2).mean((1, 2)))
        lag = np.abs((x1 * x0.conj()).mean((1, 2))) / np.maximum(norm, 1e-10)
        summaries[scene] = dict(n=len(fields), lag_abs_q10_q50_q90=np.quantile(lag, [.1, .5, .9]).tolist())
    return dict(status='SOURCE_DIAGNOSTIC', namespace=NAMESPACE, by_scene=summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().runtime)))
