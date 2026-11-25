"""Compare WiSig missing-suffix readouts on distinct source U captures."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_unrolled_restore.wisig_differential_timing import observations
from experiments.pamf_unrolled_restore.wisig_ltf_suffix import FILTERS, MODES, readouts
from experiments.pamf_unrolled_restore.wisig_smooth_preamble import TIME
from experiments.pamf_unrolled_restore.wisig_uncertain_source import source_iq
from tools.wisig_repeat_prefix_probe import COUNT, FIELDS, SHIFTS, cases, source_threshold, summarize


def probe(rt):
    started = perf_counter()
    threshold = source_threshold(rt)
    wave, available, calibration = source_iq(Path(rt['source_root']))
    held = available[4*COUNT:5*COUNT]
    if len(held) != COUNT or np.intersect1d(held, calibration).size:
        raise ValueError('Expected the independent source U suffix cohort')
    ground = wave[held]
    clean = ground[:, 0] + 1j*ground[:, 1]
    results = {}
    for kind, complete in cases(clean).items():
        for amount in SHIFTS:
            incoming = complete[:, np.abs(TIME-amount)]
            raw, info, response, fits = observations(np.stack((incoming.real, incoming.imag), 1))
            delta, margin = fits['differential']
            selected = np.where(margin > threshold, delta, 0)
            outputs, gate = readouts(raw, selected, info.coarse_rad_per_sample, response)
            coordinates = TIME+selected[:, None]
            inside = (coordinates >= 0) & (coordinates < len(TIME))
            original = np.take_along_axis(raw, coordinates.clip(0, len(TIME)-1), axis=1)
            suffix = coordinates >= len(TIME)
            for output in outputs.values():
                if not np.array_equal(output[inside], original[inside]) or not np.array_equal(output[~suffix], outputs['public'][~suffix]):
                    raise ValueError('The suffix readout changed an observed point, prefix or bypass')
            exact = selected == amount
            summary = dict(exactly_timed=int(exact.sum()), moved=int((selected != 0).sum()),
                           held_accepted={k: int((flag & exact).sum()) for k, flag in gate.items()})
            if exact.any():
                used = {k: v[exact, -amount:] for k, v in outputs.items()}
                summary.update(hypothetical_received=summarize(used, complete[exact, -amount:], MODES),
                               source_clean=summarize(used, clean[exact, -amount:], MODES))
            results[f'{kind}_{amount}'] = summary
    return dict(status='VERIFIED', role='U_s', held=len(held), held_offset=4*COUNT, threshold=threshold,
                modes=MODES, filters=FILTERS, fields=FIELDS, seconds=perf_counter()-started,
                integrity='all observed overlap, bypasses and prefixes are exact',
                source_V_reads=0, F_reads=0, TX_reads=0, target_reads=0, checkpoint_reads=0,
                optimizer_updates=0, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    print(json.dumps(probe(read(parser.parse_args().runtime)), allow_nan=False))
