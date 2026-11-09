"""Source-only post-freeze clean-pseudoclass agreement; never trains G."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs, SOURCE_PILOT
from experiments.phase1_dura_orthogonal.train_source import extract_context, load_source_training
from experiments.phase1_dura_orthogonal.source_frozen_agreement import summarize, classes
from experiments.phase1_cumulant_context.ground import load_fixed as load_context
from experiments.phase1_cumulant_rank.ground import load_fixed as load_rank
from experiments.phase1_blind_polyphase.ground import load_fixed as load_blind

MODELS = (
    ('cumulant', '20261109-phase1-dura-cumulant-context-source-s2026092701-r01', '5ed56dd1decb913bb0733339462e08cc22a57831', load_context),
    ('rank', '20261109-phase1-dura-cumulant-rank-source-s2026092701-r01', '1e83c8f23c837cf6ff68c218d2934cad4403411b', load_rank),
    ('blind_polyphase', '20261109-phase1-blind-polyphase-source-s2026092701-r01', 'cd0687eed0fa69db343bb6000b486143698fe797', load_blind),
)

@torch.no_grad()
def run(root, out, commit):
    if out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Unique output and CPU-only execution required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(root)
    _, _, _, fit = load_source_training(root, ids, fit_size=32768)
    if len(ids) != 672 or np.intersect1d(ids, fit).size:
        raise ValueError('Pilot IDs disjoint from three fixed source fits required')
    with np.load(root/'runs'/SOURCE_PILOT/'source_predictions.npz', allow_pickle=False) as z:
        if not np.array_equal(z['indices'], ids):
            raise ValueError('Identical previously fixed source pilot required')
        saved = z['values'][:, [1, 3, 5]].astype(np.int64)
    models = {}
    dura = reader = None
    for name, source, source_commit, loader in MODELS:
        model, anchor, classifier = loader(root, source, source_commit, 'cpu')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen recovery model required')
        models[name] = model
        if dura is None:
            dura, reader = anchor, classifier
    if any(p.requires_grad for p in dura.parameters()) or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Frozen DURA and reader required')
    chunks = []
    started = time.monotonic()
    for start in range(0, len(ids), 8):
        selected = ids[start:start+8]
        x = core.tensor(received[selected], 'cpu')
        y = core.tensor(clean[selected], 'cpu')
        anchor, direction, variance = extract_context(dura, x)
        columns = [classes(reader, y), classes(reader, x), classes(reader, anchor)]
        columns.append(classes(reader, models['cumulant'](x, anchor, direction, variance)))
        columns.append(classes(reader, models['rank'](x, anchor, direction, variance)))
        columns.append(classes(reader, models['blind_polyphase'](x)))
        block = np.from_dlpack(torch.stack(columns, 1).cpu().contiguous()).copy()
        if not np.array_equal(block[:, :3], saved[start:start+len(selected)]):
            raise ValueError('Frozen clean/raw/DURA reference disagreement')
        if start == 0:
            out.mkdir(parents=True)
            print('FROZEN_MODELS_SOURCE_SMOKE_PASS', len(selected), flush=True)
        chunks.append(block)
        if (start+len(selected)) % 168 == 0:
            print('FROZEN_MODELS_SOURCE_PROGRESS', start+len(selected), flush=True)
    values = np.concatenate(chunks, 0)
    names = ('raw', 'dura', 'cumulant', 'rank', 'blind_polyphase')
    predictions = {name: values[:, i+1] for i, name in enumerate(names)}
    with (out/'source_predictions.npz').open('xb') as f:
        np.savez(f, ids=ids, rx=rx, scene=scene, clean_proxy=values[:, 0], **predictions)
    groups = summarize(values[:, 0], predictions, rx, scene, views=core.VIEWS)
    paired = {}
    mask = scene != 'clean'
    baseline = predictions['dura'][mask] == values[mask, 0]
    for name in ('cumulant', 'rank', 'blind_polyphase'):
        match = predictions[name][mask] == values[mask, 0]
        paired[name] = dict(gained=int(np.count_nonzero(~baseline & match)),
                            lost=int(np.count_nonzero(baseline & ~match)))
    core.write(out/'result.json', dict(status='SOURCE_FROZEN_MODELS_PROXY_COMPLETE', commit=commit,
        records=len(ids), groups=groups, paired_practical=paired,
        candidate_commits={name: sha for name, _, sha, _ in MODELS},
        metric='agreement_with_frozen_clean_pseudoclass_not_accuracy',
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, U_s_truth_reads=0, V_reads=0,
        target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Explored source; frozen clean prediction is a pseudo-class, not TX truth; no target claims.'))
    print('FROZEN_MODELS_SOURCE_COMPLETE', len(ids), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    run(a.root, a.out, a.commit)

if __name__ == '__main__':
    main()
