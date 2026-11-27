import json
from pathlib import Path
import sys

import numpy as np


project = Path.cwd()
run = sys.argv[1]
release = project / 'releases' / f'{run}-release'
runtime = json.loads((release/'runtime.json').read_bytes())
root = Path(runtime['run_root']) / 'full_native'
frozen = json.loads((root/'source-predictions.json').read_bytes())
published = json.loads((root/'source-scores.json').read_bytes())
assert frozen['status'] == 'PREDICTIONS_FROZEN' and published['status'] == 'ANALYZED'
assert not (frozen['target_reads'] or frozen['source_truth_reads'] or published['target_reads'])
assert published['source_truth_last'] and len(frozen['physical_ids']) == 300
views, modes = frozen['views'], frozen['modes']
with np.load(root/'predictions.npz', allow_pickle=False) as archive:
    predictions = np.stack([archive[v] for v in views])
with np.load(Path(runtime['source_root'])/'V_metadata.npz', allow_pickle=False) as metadata:
    lookup = {key:i for i,key in enumerate(metadata['ids'].tolist())}
    rows = [lookup[key] for key in frozen['physical_ids']]
    rx, y = metadata['rx_i'][rows], metadata['y'][rows]
hits = predictions == y


def cell(delta):
    return dict(repairs=int((delta > 0).sum()), harms=int((delta < 0).sum()), net=int(delta.sum()))


paired = {}
grouped = {}
for contrast, summary in published['paired_contrasts'].items():
    a, b = contrast.split('_vs_')
    delta = hits[:, modes.index(a)].astype(np.int16) - hits[:, modes.index(b)].astype(np.int16)
    paired[contrast] = dict(**{v:cell(s) for v,s in zip(views, delta, strict=True)}, six=cell(delta[1:]))
    assert paired[contrast]['six']['net'] == summary['net_six']
    assert paired[contrast][views[-1]]['net'] == summary['net_low']
    grouped[contrast] = {axis:[
        dict(group=int(key), n=int((group==key).sum()), six=cell(delta[1:,group==key]),
             low=cell(delta[-1,group==key]), clean=cell(delta[0,group==key]))
        for key in np.unique(group)
    ] for axis,group in (('rx',rx),('tx',y))}
print(json.dumps(dict(paired=paired, groups=grouped)))
