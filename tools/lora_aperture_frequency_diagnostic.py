"""Audit receive-only LoRa frequency rules on scored pairs with isolated noise traces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.aperture_tracking import ApertureTracking, carrier_from_frame, center, wrap
from experiments.pamf_lora_main.aperture_tracking_source import MODES, RUN
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
from experiments.pamf_lora_main.data import FS_HZ, SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.local_wiener_source import ROUTES, read_frozen, threshold
from experiments.pamf_lora_main.runner import activate


HZ = FS_HZ/(2*torch.pi)
KEYS = ('legacy_hz','split_hz','adaptive_hz','legacy_relative_hz','split_relative_hz','adaptive_relative_hz',
        'noiseless_split_drift_hz','noiseless_relative_spread_hz','half_uncertainty_hz','shrinkage')


def errors(reference, estimates):
    return tuple(wrap(value-reference).abs()*HZ for value in estimates)


@torch.no_grad()
def measurements(iq, epsilon, tracker, aperture, gate):
    observed, frame, _, _ = coordinates(tracker,iq)
    active = corruption_score(observed)>gate
    chirp = tracker.reference.chirp
    legacy = carrier_from_frame(frame,chirp)
    split, adaptive, uncertainty, strength = aperture.frequencies(iq,chirp,legacy)
    ideal_iq = iq-epsilon
    ideal_frame = tracker.track(ideal_iq)[1]
    ideal = carrier_from_frame(ideal_frame,chirp)
    ideal_split = aperture.frequencies(ideal_iq,chirp,ideal)[0]
    rates = (legacy,split,adaptive)
    relative = tuple(wrap(f-center(f)) for f in rates)
    ideal_relative = wrap(ideal-center(ideal))
    values = torch.stack((*errors(ideal,rates),*errors(ideal_relative,relative),
                          errors(ideal,(ideal_split,))[0],ideal_relative.abs()*HZ,
                          uncertainty*HZ,strength),dim=-1)
    return np.asarray(active.tolist(),dtype=bool),np.asarray(values.tolist(),dtype=np.float32)


def summarize(enabled, values, receivers):
    result = {}
    for receiver in ('all',*SOURCE):
        chosen = np.ones(len(enabled),dtype=bool) if receiver=='all' else receivers==receiver
        counts = {'packets':int(chosen.sum()),'active_packets':int((chosen & enabled).sum())}
        for key, mask in (('all',chosen),('active',chosen & enabled)):
            if mask.any():
                v = values[mask].reshape(-1,len(KEYS))
                q = np.quantile(v,(.5,.9),axis=0).T.tolist()
                counts[key] = dict(zip(KEYS,q,strict=True))
        result[receiver] = counts
    return result


def diagnose(rt, route, device):
    root, source = Path(rt['run_root']),Path(rt['source_root'])
    if rt.get('run_id')!=RUN or read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('Expected the completed dual-route aperture source score')
    frozen = {label:read_frozen(root,label,MODES) for label in ROUTES}
    raw, rx, ids, _ = source_role(source,source_contract(source),'V')
    chosen = panel(rx)
    if any(record[0]!=ids[chosen].tolist() for record in frozen.values()):
        raise ValueError('Diagnostic rows must agree with both frozen source predictions')
    rows = list(zip(raw[chosen],rx[chosen],ids[chosen],strict=True))
    tracker = PacketRestorer('consensus').to(device).eval()
    aperture = ApertureTracking().to(device).eval()
    gate = threshold(root)
    full = np.load(root.parent/DATA_RUN/'source_views'/'V.npy',mmap_mode='r',allow_pickle=False)
    records = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, view in enumerate(VIEWS):
            iq, noise = (np.stack(v) for v in zip(*pool.map(partial(replay,route,view),rows,chunksize=8),strict=True))
            if route=='full' and not np.array_equal(iq,full[i,chosen]):
                raise ValueError('Frequency diagnostic replay changed the registered full view')
            if route=='residual':
                for j in range(0,len(rows),len(rows)//len(SOURCE)):
                    if not np.array_equal(iq[j],receive_only(route,view,rows[j])[0]):
                        raise ValueError('Frequency diagnostic replay changed residual IQ')
            parts = [measurements(a,b,tracker,aperture,gate) for a,b in zip(tensor(iq,device).split(32),
                        tensor(noise,device).split(32),strict=True)]
            enabled, metrics = (np.concatenate(value) for value in zip(*parts,strict=True))
            if (enabled.sum()!=frozen[route][2][view]['active'] or not np.isfinite(metrics).all()):
                raise ValueError('Diagnostic gate or estimates differ from expectations')
            records[view] = summarize(enabled,metrics,rx[chosen])
    return dict(route=route,n=len(rows),target_reads=0,quantiles=[.5,.9],scenes=records)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    parser.add_argument('route',choices=ROUTES)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    runtime = read(args.runtime)
    activate(runtime)
    print(json.dumps(diagnose(runtime,args.route,args.device),allow_nan=False))
