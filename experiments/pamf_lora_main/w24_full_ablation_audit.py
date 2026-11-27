"""Recount all registered ablations after independent Full truth-last scoring."""

import argparse
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .w24_formal import (
    FULL, LIMIT, NATIVE_CENTER_KIND, NATIVE_DIRECTION_KINDS, SEED, TARGET, VIEWS,
    evaluation_modes,
)
from .w24_formal_score import (
    full_target_acceptance, label_ids, load_frozen, packet_indices, paired, readout,
    recording, transmitter_groups,
)
from .w24_third_prior import PRACTICAL_KIND


PARENT_MODE = 'practical_exact_additive_half'
BASELINES = ('direct', 'complex_response_half', PARENT_MODE)


def summarize(hits, candidates, truth):
    results = {}
    for i, candidate in enumerate(candidates):
        baselines = (*BASELINES, *candidates[:i])
        candidate_hits = hits[candidate]
        counts = readout(candidate_hits, (SEED,))['pooled']
        block = np.stack([hits[name] for name in (*baselines, candidate)], axis=3)
        groups = transmitter_groups(block, truth, (*baselines, candidate), (SEED,))
        results[candidate] = dict(
            scores=counts,
            acceptance=full_target_acceptance(counts),
            paired={name: paired(candidate_hits.astype(np.int16) - hits[name], (SEED,))['pooled']
                    for name in baselines},
            by_tx={tx: dict(n_physical=group['n_physical'],
                            scores=group['readouts'][candidate]['pooled'],
                            paired={name: group['paired'][name]['pooled'] for name in baselines})
                   for tx, group in groups.items()},
        )
    return results


def completed_score(root):
    score = read(root/'scores/scores.json')
    n_packets = len(TARGET)*LIMIT
    if (score['status'] != 'ANALYZED' or score['truth_last'] is not True
            or score['seeds'] != [SEED] or score['n_physical'] != n_packets
            or score['n_six'] != n_packets*(len(VIEWS)-1)):
        raise ValueError(f'Expected complete independent Full score in {root}')
    return score['readouts'][FULL]


def audit(formal, practical, kind, dataset, output):
    scores, practical_scores = completed_score(formal), completed_score(practical)
    shared = evaluation_modes(FULL, True)
    modes = evaluation_modes(FULL, True, third_prior=kind)
    prior_modes = evaluation_modes(FULL, True, third_prior=PRACTICAL_KIND)
    pred = load_frozen(formal/'predictions', third_prior=kind)[0][FULL]
    prior = load_frozen(practical/'predictions', third_prior=PRACTICAL_KIND)[0][FULL]
    n_shared = len(shared)
    if (modes[:n_shared] != shared or prior_modes[:n_shared] != shared
            or not np.array_equal(pred[:, :, :, :n_shared], prior[:, :, :, :n_shared])):
        raise ValueError('Registered Full archive references disagree across runs')

    with (tempfile.TemporaryDirectory() as temp,
          zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive):
        labels = []
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', Path(temp)) as capture:
                labels.append(label_ids(capture, LIMIT)[packet_indices(LIMIT, i)])
    truth = np.asarray(labels)
    by_packet = truth[:, None, None, :]
    hits = {mode: pred[:, :, :, i] == by_packet for i, mode in enumerate(modes)}
    hits[PARENT_MODE] = prior[:, :, :, prior_modes.index(PARENT_MODE)] == by_packet
    for mode, correctness in hits.items():
        expected = practical_scores if mode == PARENT_MODE else scores
        if readout(correctness, (SEED,)) != expected[mode]:
            raise ValueError(f'Independent Full score failed recount for {mode}')

    write(output, dict(status='ANALYZED', truth_last=True, seed=SEED, n_physical=truth.size,
                       shared_archive_readouts=list(shared),
                       candidates=summarize(hits, modes[n_shared:], truth)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('formal', 'practical', 'dataset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--kind', choices=(*NATIVE_DIRECTION_KINDS, NATIVE_CENTER_KIND), required=True)
    audit(**vars(parser.parse_args()))
