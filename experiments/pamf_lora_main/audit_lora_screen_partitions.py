"""Read-only post-screen analysis of prediction agreement and source TX."""
import argparse
import json
from pathlib import Path
import numpy as np

project = Path('/home/szu2070436088/2510044040/CV-SincNet')
cases = {
    '20261128-phase1-lora-r17-realgain-e104-s2026112861-r01': 'fusion',
    '20261128-phase1-lora-joint-r17-realgain-s2026112873-r01': 'joint_fusion',
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', choices=tuple(cases))
selected = parser.parse_args().run
run = project / 'runs' / selected
parent = project / 'runs/20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01'
def read(p):
    with p.open(encoding='utf-8') as stream:
        return json.load(stream)
frozen = read(run / 'screen-frozen.json')
modes = ('direct', 'lora_native_r17', 'real_gain', cases[selected])
assert frozen['status'] == 'PREDICTIONS_FROZEN' and frozen['modes'] == list(modes)
assert frozen['source_truth_reads'] == 0 and frozen['target_reads'] == 0 and frozen['count'] == 300
with np.load(run / 'screen.npz', allow_pickle=False) as data:
    ids, predictions = data['ids'].copy(), data['practical_low_urban'].copy()
assert predictions.shape == (4, 300) and len(set(ids.tolist())) == 300
config = read(parent / 'frequency_inphase_native/fit-inputs.json')
with np.load(Path(config['source_root']) / 'V_metadata.npz', allow_pickle=False) as meta:
    lookup = dict(zip(meta['ids'].tolist(), meta['y'].tolist(), strict=True))
truth = np.asarray([lookup[x] for x in ids.tolist()])
direct, r17, real, fusion = predictions
def summary(mask):
    correct = predictions[:, mask] == truth[mask]
    good, changed = correct[2], correct[3]
    return dict(n=int(mask.sum()), correct=correct.sum(axis=1).tolist(),
                repaired=int((changed & ~good).sum()), harmed=int((~changed & good).sum()),
                net=int(changed.sum() - good.sum()))
groups = {'r17_agrees_real': summary(r17 == real), 'r17_differs_real': summary(r17 != real),
          'direct_agrees_real': summary(direct == real), 'direct_differs_real': summary(direct != real)}
by_tx = {str(int(k)): summary(truth == k) for k in np.unique(truth)}
print(json.dumps(dict(status='READ_ONLY_POST_SCREEN', run=run.name,
    scenario='practical_low_urban', receiver='b200_1', n=len(ids),
    class_ids_are_original_W24_indices=True, modes=list(modes),
    prediction_agreement=groups, by_tx=by_tx), ensure_ascii=False))
