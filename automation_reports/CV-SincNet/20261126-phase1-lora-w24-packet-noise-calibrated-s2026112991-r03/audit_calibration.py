"""Read-only calibration audit on 100 disjoint LoRa U_s source packets."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from experiments import pamf_lora_main

release = Path(os.environ['EXPERIMENT_RELEASE'])
pamf_lora_main.__path__.append(str(release/'experiments/pamf_lora_main'))
from experiments.pamf_lora_main import w24_native_innovation as native_module
from experiments.pamf_lora_main import w24_packet_noise as model_module
from experiments.pamf_lora_main.full_packet_views import source_role
from experiments.pamf_lora_main.w24_factor_audit import trace_components
from experiments.pamf_lora_main.w24_native_full import SOURCE, native_full_view

rt = native_module.read(release/'runtime.json')
source, root = Path(rt['source_root']), Path(rt['run_root'])
assert native_module.read(root/'fit-complete.json')['status'] == 'SOURCE_FROZEN'
tracker, _ = native_module.source_tools(rt, 'cpu')
contract = native_module.source_contract(source)
raw, rx, _, _ = source_role(source, contract, 'U_s')
fit_ids = np.random.default_rng(native_module.SEED).choice(
    len(raw), native_module.FIT_PACKETS+native_module.HOLD_PACKETS, replace=False)
rng = np.random.default_rng(2026112701)
rows = np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(rx==r),fit_ids),20,replace=False) for r in SOURCE])
taps = native_module.high_pass_taps('cpu',torch.float32)
states = torch.load(root/'native-innovation.pt',map_location='cpu',weights_only=True)
candidates = []
for state in states:
    model = model_module.CalibratedRestorer().eval()
    model.load_state_dict(state)
    candidates.append(model)

results = {}
namespace = 'w24-native-innovation-noise-isolation-v1'
with torch.inference_mode():
    for scene in ('practical_high','practical_mid_urban','practical_low_urban'):
        rows_db = []
        for i in range(0,len(rows),10):
            items = [(raw[j],scene,rx[j],int(j),namespace) for j in rows[i:i+10]]
            observed,_,noise = zip(*(trace_components(*x) for x in items),strict=True)
            if i == 0:
                assert np.array_equal(observed[0].astype(np.complex64),native_full_view(*items[0]))
            iq = native_module.packet_tensor(np.stack(observed).astype(np.complex64),'cpu')
            inputs,frame,_,feature,_ = model_module.packet_observation(tracker,iq,taps)
            oracle = tracker.aperture(native_module.packet_tensor(
                np.stack(noise).astype(np.complex64),'cpu'))[...,:native_module.CENTER]/frame
            truth = (oracle.abs().square().mean((-1,-2))+1e-5).log()
            observed_power = [8*x[:,0,3,0] for x in feature]
            estimates = [m(*inputs,x)[-1] for m,x in zip(candidates,feature,strict=True)]
            rows_db.append(torch.stack((truth,*observed_power,*estimates),dim=-1))
        joined = np.asarray(torch.cat(rows_db).tolist())
        delta_db = 10/np.log(10)*(joined[:,1:]-joined[:,:1])
        results[scene] = {
            'spearman':[float(spearmanr(joined[:,0],joined[:,j]).statistic) for j in range(1,5)],
            'bias_db_median':np.median(delta_db,axis=0).tolist(),
            'absolute_db_error_median':np.median(abs(delta_db),axis=0).tolist(),
        }
print(json.dumps({'status':'ANALYZED','ordered_estimates':['aperture_raw','native_raw','aperture_calibrated','native_calibrated'],
                  'scope':'separate source U_s added-noise isolation, not the fitted total-disturbance target',
                  'source_packets':len(rows),'fit_pool_overlap':0,'target_reads':0,'source_truth_reads':0,'scenes':results},indent=2))
