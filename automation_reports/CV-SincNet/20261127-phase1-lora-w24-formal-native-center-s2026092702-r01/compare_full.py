"""Recount scored Full decisions against the existing best and across TX groups."""

import argparse
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'formal', 'outer', 'dataset', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from experiments.cric_dura_comparison.io import read, write
    from experiments.pamf_lora_main.w24_formal import (
        FULL, LIMIT, NATIVE_CENTER_KIND, OUTER_HALF, SEED, TARGET, VIEWS, evaluation_modes,
    )
    from experiments.pamf_lora_main.w24_formal_score import (
        label_ids, load_frozen, packet_indices, paired, readout, recording,
    )

    old, new = (read(root/'scores/scores.json') for root in (args.outer, args.formal))
    if any(d['status'] != 'ANALYZED' or not d['truth_last'] or d['seeds'] != [SEED]
           or d['n_physical'] != 30000 or d['n_six'] != 180000 for d in (old, new)):
        raise ValueError('Both runs need complete independent truth-last scoring')
    new_modes = evaluation_modes(FULL, True, third_prior=NATIVE_CENTER_KIND)
    old_modes = evaluation_modes(FULL, True, outer_objective=True)
    prior = load_frozen(args.outer/'predictions', outer_objective=True)[0][FULL]
    pred = load_frozen(args.formal/'predictions', third_prior=NATIVE_CENTER_KIND)[0][FULL]
    if not np.array_equal(pred[:, :, :, :4], prior[:, :, :, :4]):
        raise ValueError('Four official Full reference decisions must match exactly')
    with (tempfile.TemporaryDirectory(dir=args.formal) as temp,
          zipfile.ZipFile(args.dataset/'multiple_receiver_test.zip') as archive):
        labels = []
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', Path(temp)) as capture:
                labels.append(label_ids(capture, LIMIT)[packet_indices(LIMIT, i)])
    truth = np.asarray(labels)[:, None, None, :]
    current = {m: pred[:, :, :, j] == truth for j, m in enumerate(new_modes)}
    outer = prior[:, :, :, old_modes.index(OUTER_HALF)] == truth
    for mode, hits in current.items():
        if readout(hits, (SEED,)) != new['readouts'][FULL][mode]:
            raise ValueError(f'Independent formal score failed recount for {mode}')
    if readout(outer, (SEED,)) != old['readouts'][FULL][OUTER_HALF]:
        raise ValueError('Archived outer comparison failed recount')
    current[OUTER_HALF] = outer
    by_tx = {}
    for tx in range(10):
        select = truth == tx
        by_tx[str(tx)] = dict(
            n=int(select.sum()),
            counts={m: dict(zip(VIEWS, (hits & select).sum((0, 2, 3)).tolist(), strict=True))
                    for m, hits in current.items()},
        )
    contrasts = {}
    for m in new_modes[-2:]:
        for baseline in (OUTER_HALF, 'direct', 'complex_response_half'):
            delta = current[m].astype(np.int16) - current[baseline]
            contrasts[f'{m}_vs_{baseline}'] = paired(delta, (SEED,))['pooled']
    write(args.output, dict(status='ANALYZED', truth_last=True, exact_reference_parity=True,
                           by_tx=by_tx, paired=contrasts))


if __name__ == '__main__':
    main()
