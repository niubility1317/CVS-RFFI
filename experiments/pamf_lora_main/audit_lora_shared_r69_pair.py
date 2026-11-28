"""Truth-last same-packet comparison of fixed shared gain and LoRa-trained R69."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
CANDIDATE = ROOT / 'runs/20261128-phase1-shared-r17-realgain-s2026112893-r01/lora'
CONTROL = ROOT / 'runs/20261128-diagnostic-lora-r69-full-w24-s2026112895-r01'
GAIN = ROOT / 'runs/20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01'
VIEWS = ('clean', 'practical_low_urban')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


for folder in (CANDIDATE, CONTROL):
    mark = read(folder / 'screen-frozen.json')
    if (mark['status'], mark['views'], mark['count'], mark['source_truth_reads'], mark['target_reads']) != (
            'PREDICTIONS_FROZEN', list(VIEWS), 60, 0, 0):
        raise ValueError('Both complete prediction matrices must be fixed before source truth')
with np.load(CANDIDATE / 'screen.npz', allow_pickle=False) as archive:
    ids = archive['ids'].copy()
    candidate = {view: archive[view].copy() for view in VIEWS}
with np.load(CONTROL / 'screen.npz', allow_pickle=False) as archive:
    other_ids = archive['ids'].copy()
    control = {view: archive[view].copy() for view in VIEWS}
if not np.array_equal(ids, other_ids) or len(set(ids.tolist())) != 60:
    raise ValueError('Physical packet IDs differ')
for view in VIEWS:
    if candidate[view].shape != (4, 60) or control[view].shape != (3, 60):
        raise ValueError('Prediction matrix is incomplete')
    if not np.array_equal(candidate[view][0], control[view][0]):
        raise ValueError('Same received packet and reader required')
old = read(GAIN / 'frequency_inphase_native/fit-inputs.json')
with np.load(Path(old['source_root']) / 'V_metadata.npz', allow_pickle=False) as metadata:
    truth_map = dict(zip(metadata['ids'].tolist(), metadata['y'].tolist(), strict=True))
labels = np.asarray([truth_map[x] for x in ids.tolist()])
result = {}
for view in VIEWS:
    a, b = candidate[view][-1] == labels, control[view][-1] == labels
    delta = a.astype(np.int16) - b.astype(np.int16)
    rng = np.random.default_rng(2026112900)
    draws = rng.integers(0, len(delta), (2000, len(delta)))
    result[view] = dict(n=60, shared_correct=int(a.sum()), lora_r69_correct=int(b.sum()),
                        repaired=int((a & ~b).sum()), harmed=int((~a & b).sum()), net=int(delta.sum()),
                        ci95=np.quantile(delta[draws].sum(1), (.025, .975)).tolist())
print(json.dumps(dict(status='SOURCE_PAIRED_AUDIT', truth_last=True, same_ids=True,
                      direct_predictions_match=True, views=result,
                      control_identity='LoRa-trained R69 architecture adaptation; not original WiSig weights',
                      exposure='previously exposed source V, target_reads=0'), ensure_ascii=False))
