"""Decompose frozen DURA errors across three simulated views of source IQ."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.train_source import SOURCE_EPOCHS, load_source_training


def decompose(outputs, clean):
    """Per-record three-view NMSE = squared mean error + error dispersion."""
    if outputs.ndim != 4 or outputs.shape[1:] != (3, 2, 256) or clean.shape != (len(outputs), 2, 256):
        raise ValueError('Three matched output views and clean IQ required')
    if not torch.isfinite(outputs).all() or not torch.isfinite(clean).all():
        raise FloatingPointError('Finite source IQ required')
    error = outputs-clean[:, None]
    center = error.mean(1)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    total = error.square().mean((1, 2, 3))/energy
    mean_error = center.square().mean((1, 2))/energy
    dispersion = (error-center[:, None]).square().mean((1, 2, 3))/energy
    return torch.stack((total, mean_error, dispersion), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, _, _, _, _ = source_inputs(args.root)
    _, fresh, clean, _ = load_source_training(args.root, ids)
    inputs, scenes = [], []
    for epoch in SOURCE_EPOCHS:
        folder = fresh/'epochs'/f'{epoch:02d}'
        meta = core.read(folder/'complete.json')
        if (meta.get('count') != core.COUNT or
                meta.get('namespace') != f'learned-action/train/epoch{epoch:02d}' or
                any(meta.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
            raise ValueError('Approved source-only simulated views required')
        values = np.load(folder/'received.npy', mmap_mode='r', allow_pickle=False)
        with np.load(folder/'groups.npz', allow_pickle=False) as groups:
            scene = groups['scene'].copy()
        if values.shape != clean.shape or scene.shape != (core.COUNT,):
            raise ValueError('Matching source IQ and scene records required')
        inputs.append(values)
        scenes.append(scene[ids])
    dura = frozen_dura(args.root, 'cpu')
    clean_iq = core.tensor(clean[ids], 'cpu')
    outputs = []
    started = time.monotonic()
    with torch.no_grad():
        for epoch, values in zip(SOURCE_EPOCHS, inputs):
            chunks = []
            for start in range(0, len(ids), 8):
                x = core.tensor(values[ids[start:start+8]], 'cpu')
                restored = dura(x)
                if restored.shape != x.shape or not torch.isfinite(restored).all():
                    raise ValueError('Finite frozen source DURA outputs required')
                if not outputs and start == 0:
                    args.out.mkdir(parents=True)
                    print('CROSSVIEW_SOURCE_SMOKE_PASS', flush=True)
                chunks.append(restored)
            outputs.append(torch.cat(chunks))
            print('CROSSVIEW_SOURCE_EPOCH_COMPLETE', epoch, len(ids), flush=True)
    result = decompose(torch.stack(outputs, dim=1), clean_iq)
    torch.testing.assert_close(result[:, 0], result[:, 1]+result[:, 2], rtol=1e-5, atol=1e-7)
    clean_count = np.stack(scenes, axis=1).__eq__('clean').sum(axis=1)
    summaries = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for count in ('all', 0, 1, 2, 3):
            mask = np.isin(rx, receivers)
            if count != 'all':
                mask &= clean_count == count
            if not mask.any():
                continue
            cell = result[torch.tensor(mask.tolist(), dtype=torch.bool)]
            summaries[group+'/'+str(count)] = dict(count=len(cell), total_nmse=float(cell[:, 0].mean()),
                finite_three_view_mean_error_nmse=float(cell[:, 1].mean()),
                finite_three_view_dispersion_nmse=float(cell[:, 2].mean()))
    scene_counts = {view: int(sum(np.count_nonzero(scene == view) for scene in scenes)) for view in core.VIEWS}
    core.write(args.out/'result.json', dict(status='CROSSVIEW_SOURCE_DIAGNOSTIC_COMPLETE', commit=args.commit,
        records=len(ids), views_per_record=3, source_epochs=list(SOURCE_EPOCHS), groups=summaries,
        scene_counts=scene_counts, elapsed_seconds=time.monotonic()-started, TX_labels_read=0,
        F_weights_read=0, F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Finite three-view source decomposition, not an irreducible-bias estimate; averaging requires unavailable repeated views and is not deployed or an identity metric.'))
    print('CROSSVIEW_SOURCE_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
