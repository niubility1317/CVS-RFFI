"""Read-only physical coverage metadata; no model or target performance access."""
import json
import pickle
from pathlib import Path

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig')
summaries = {}
for name in ('ManySig', 'ManyTx'):
    with (ROOT / (name + '.pkl')).open('rb') as f:
        d = pickle.load(f)
    tx = [str(x) for x in d.get('tx_list', d.get('node_list', []))]
    rx = [str(x) for x in d['rx_list']]
    days = [str(x) for x in d['capture_date_list']]
    eq = list(d['equalized_list']).index(1)
    counts = {r: {t: [0 if d['data'][i][j][k][eq] is None else len(d['data'][i][j][k][eq])
                        for k in range(len(days))] for i, t in enumerate(tx)} for j, r in enumerate(rx)}
    summaries[name] = dict(tx=tx, rx=rx, days=days, counts=counts)
    del d
common_rx = sorted(set(summaries['ManySig']['rx']) & set(summaries['ManyTx']['rx']))
common_tx = sorted(set(summaries['ManySig']['tx']) & set(summaries['ManyTx']['tx']))
print(json.dumps(dict(datasets=summaries, common_rx=common_rx, common_tx=common_tx)))
