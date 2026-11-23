"""Read frozen decisions and source-U head behavior without touching recognition or TX truth."""
import json
from functools import partial
from pathlib import Path
import sys

root = Path(sys.argv[1])
run = root.name
release = root.parent.parent / 'releases' / (run + '-release')
sys.path.insert(0, str(release))

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_role, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.repeat_covariance import RepeatCovariance, repeated_noise

torch.set_num_threads(2)
assert read(root / 'state.json')['status'] == 'SOURCE_SCORED'
cfg = read(release / 'runtime.json')
source = Path(cfg['source_root'])
contract = source_contract(source)
_, rx, _, _ = source_role(source, contract, 'U_s')
_, _, held = partition(rx)
sample = held[::10]
data = np.load(root.parent / DATA_RUN / 'source_views' / 'U_s.npy', mmap_mode='r', allow_pickle=False)
clean = complex_iq(tensor(data[0, sample], 'cpu'))
generator = torch.Generator().manual_seed(2026112417)
disturbed = real_iq(repeated_noise(clean, generator))
observed, _, quality, _ = coordinates(PacketRestorer('consensus').eval(), disturbed)

def energy(v):
    return (v-v.mean(-1, keepdim=True)).abs().square().mean(dim=(1, 2))

metrics = {}
outputs = {}
with torch.no_grad():
    for correlated in (False, True):
        name = 'covariance' if correlated else 'diagonal'
        model = RepeatCovariance(correlated).eval()
        saved = torch.load(root / f'{name}-model.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(saved['weights'], strict=True)
        features = {}

        def capture(tag, _module, _args, value):
            features[tag] = value.clone()

        for stage in ('coarse','fine'):
            getattr(model, stage).register_forward_hook(partial(capture, stage))
        coarse, final = model(observed, quality)
        outputs[name] = [coarse, final]
        grid = torch.tensor([.10,.50,.90])
        metrics[name] = {
            'coarse_logit_adjustment': features['coarse'].flatten().quantile(grid).tolist(),
            'fine_logit_adjustment': features['fine'].flatten().quantile(grid).tolist(),
            'noncarrier_fraction_coarse': (energy(coarse)/energy(observed)).quantile(grid).tolist(),
            'noncarrier_fraction_final': (energy(final)/energy(observed)).quantile(grid).tolist(),
        }

predictions = {}
for route in ('full','residual'):
    d = read(root / f'{route}-predictions.json')
    x = np.asarray(list(d['rows'].values()))
    predictions[route] = {f'{arm}_vs_{control}': int((x[:, d['modes'].index(arm)] !=
                                  x[:, d['modes'].index(control)]).sum())
                          for arm, control in [('covariance_coarse','diagonal_coarse'),
                                               ('covariance','diagonal'),
                                               ('covariance_half','diagonal_half'),
                                               ('covariance','covariance_coarse')]}
distance = (outputs['covariance'][-1] - outputs['diagonal'][-1]).abs().square().mean(dim=(1,2))
result = dict(status='VERIFIED', held_U=len(sample), seed=2026112417, target_reads=0, TX_reads=0,
              metrics=metrics, frozen_decision_differences=predictions,
              relative_final_iq_distance=(distance/energy(observed)).quantile(grid).tolist())
print(json.dumps(result,indent=2))
