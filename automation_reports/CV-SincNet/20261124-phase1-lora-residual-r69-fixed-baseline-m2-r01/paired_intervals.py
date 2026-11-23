
from pathlib import Path
import json
import tempfile
import zipfile

import numpy as np

from experiments.pamf_lora_main.data import TARGET, VIEWS, label_ids, packet_indices, recording
from experiments.pamf_lora_main.full_packet_data import LIMIT
from experiments.pamf_lora_main.residual_dura_score import prediction

root = Path('E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/residual_fixed_r69_baseline_v1')
archive = Path('F:/FULL wisig/Radio Frequency Fingerprint LoRa Dataset With Multiple Receivers')/'multiple_receiver_test.zip'
seeds = (2026092701, 2026092702, 2026092703)
dirs = {seed: (root.parent/'dura_residual_gate_readout' if seed==2026092702 else root/str(seed))/'predictions'
        for seed in seeds}
summary = json.loads((root/'three-seed-summary.json').read_text(encoding='utf-8'))
if summary['status'] != 'SCORED_COMPLETE' or not summary['truth_last']:
    raise ValueError('The frozen residual control must be scored before paired uncertainty')
n_draws = 1600
rng = np.random.default_rng(2026112420)
bootstrap = np.zeros((3, n_draws), dtype=int)
nets = np.zeros(3, dtype=int)
with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(prefix='lora-baseline-interval-',
                                                                   dir=root) as scratch:
    for index, rx in enumerate(TARGET):
        with recording(source, f'{rx}_test.h5', Path(scratch)) as h5:
            truth = label_ids(h5, LIMIT)[packet_indices(LIMIT, index)]
        delta = np.zeros((len(VIEWS), LIMIT), dtype=np.int16)
        for folder in dirs.values():
            for i, view in enumerate(VIEWS):
                recovered = prediction(folder/f'{rx}__{view}__dura_gated_half.npy') == truth
                direct = prediction(folder/f'{rx}__{view}__direct.npy') == truth
                delta[i] += recovered.astype(np.int16)-direct
        by_packet = np.stack((delta[0], delta[1:].sum(0), delta[-1]))
        nets += by_packet.sum(1)
        draws = rng.integers(LIMIT, size=(n_draws, LIMIT))
        for k, values in enumerate(by_packet):
            bootstrap[k] += values[draws].sum(1)
total = summary['pooled']
keys = ('clean_correct','six_correct','low_correct')
assert nets.tolist()==[total['dura_gated_half'][key]-total['direct'][key] for key in keys]
result = dict(bootstrap_seed=2026112420, bootstrap_draws=n_draws,
              pairing='Stratified by fixed receiver; keep each physical packet together across seeds and views',
              compares='frozen residual R69 gated half vs matched residual direct',
              metrics={name:dict(net=int(nets[k]), interval=np.quantile(bootstrap[k],(.025,.975)).tolist())
                       for k,name in enumerate(('clean','six','low_urban'))})
with (root/'paired-intervals.json').open('x', encoding='utf-8') as out:
    json.dump(result,out,indent=2)
print(json.dumps(result),flush=True)
