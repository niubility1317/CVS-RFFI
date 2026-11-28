"""Compare two already-frozen target pilots on matching packets after truth is available."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from experiments.pamf_lora_main import shared_target_screen as physical
from experiments.pamf_lora_main import shared_phasetrack_target as phase


def compare(old_root, new_root, dataset, truth_path, output):
    old = physical.frozen_pair(old_root)[dataset]
    new = phase.frozen_pair(new_root)[dataset]
    if not np.array_equal(old[0], new[0]):
        raise ValueError('Target packets differ')
    ids = old[0]
    for view in physical.VIEWS:
        if not np.array_equal(old[1][view][:-1], new[1][view][:-1]):
            raise ValueError('Fixed same-packet controls differ')
    if dataset == 'lora':
        from experiments.pamf_lora_main.data import label_ids, packet_indices, recording
        scratch = output.parent/'paired_truth_scratch'
        scratch.mkdir(exist_ok=False)
        with zipfile.ZipFile(truth_path/'multiple_receiver_test.zip') as archive:
            with recording(archive, f'{physical.LORA_RX}_test.h5', scratch) as capture:
                labels = label_ids(capture, 2000)[packet_indices(2000, 0)][ids]
        mask = np.ones(len(ids), dtype=bool)
    else:
        with truth_path.open(encoding='utf-8') as stream:
            truth = json.load(stream)
        rows = [truth[str(x)] for x in ids]
        mask = np.asarray([str(row['receiver']) == physical.WISIG_RX_LABEL for row in rows])
        labels = np.asarray([row['label'] for row in rows], dtype=np.int64)[mask]
    if not len(labels):
        raise ValueError('No fixed receiver samples')
    rng = np.random.default_rng(physical.SEED)
    draws = rng.integers(0, len(labels), (2000, len(labels)))
    results = {}
    for view in physical.VIEWS:
        a, b = old[1][view][-1, mask] == labels, new[1][view][-1, mask] == labels
        delta = b.astype(np.int8)-a.astype(np.int8)
        results[view] = dict(n=len(labels), physical_correct=int(a.sum()), phase_correct=int(b.sum()),
            repaired=int((~a & b).sum()), harmed=int((a & ~b).sum()), net=int(delta.sum()),
            ci95=np.quantile(delta[draws].sum(1), (.025, .975)).tolist())
    with output.open('x', encoding='utf-8') as stream:
        json.dump(dict(status='PAIRED_TARGET_DEVELOPMENT_COMPARISON', truth_last_current_comparison=True,
            exposure='same target labels and physicalsolve result were previously exposed',
            dataset=dataset, reference='physicalsolve', candidate='phasetrack', results=results), stream,
            ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('old_root', type=Path)
    p.add_argument('new_root', type=Path)
    p.add_argument('dataset', choices=('lora', 'wisig'))
    p.add_argument('truth_path', type=Path)
    p.add_argument('output', type=Path)
    compare(**vars(p.parse_args()))
