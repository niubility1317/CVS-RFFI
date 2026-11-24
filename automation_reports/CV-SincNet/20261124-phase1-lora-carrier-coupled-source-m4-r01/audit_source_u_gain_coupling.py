"""Audit LoRa carrier scale and private complex detail on separate source U."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.carrier_coupled import observed_frame, packet_frame
from experiments.pamf_lora_main.carrier_coupled_source import RECIPES
from experiments.pamf_lora_main.channel_blind import corruption_score, zero_carrier
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.mirror_refine_source import load_model
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.private_band_risk_source import band_inner, fft, private_projection
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate


def carrier_log(wave):
    level = wave.mean(-1,keepdim=True).abs().clamp_min(1e-5).log()
    return level-level.mean(1,keepdim=True)


def detail_ratio(wave):
    return zero_carrier(wave)/wave.mean(-1,keepdim=True).abs().clamp_min(.05)


def fidelity(value, expected, weight, masks):
    change = private_projection(fft(value))
    reference = private_projection(fft(expected))
    energy = band_inner(reference,reference,weight,masks)
    error = band_inner(change-reference,change-reference,weight,masks)
    cross = torch.complex(band_inner(change,reference,weight,masks),
                          band_inner(change,1j*reference,weight,masks))
    measures = []
    for band in range(energy.shape[1]):
        valid = energy[:,band]>1e-12
        gain = cross[valid,band]/energy[valid,band]
        measures.append(dict(relative_error=float(error[:,band].sum()/energy[:,band].sum()),
                             gain=float(gain.real.median()),phase_rad=float(gain.angle().abs().median())))
    return measures


@torch.no_grad()
def main():
    started=perf_counter()
    torch.set_num_threads(2)
    rt=read(Path('runtime.json'))
    activate(rt)
    root,source=Path(rt['run_root']),Path(rt['source_root'])
    contract=source_contract(source)
    raw,rx,ids,_=source_role(source,contract,'U_s')
    train,cal,hold=partition(rx)
    picked=np.concatenate([v[:30] for v in np.split(hold,len(SOURCE))])
    if np.intersect1d(picked,np.concatenate((train,cal))).size:
        raise ValueError('Expected independent held source U')
    models={name:load_model(root,recipe,contract,'cpu') for name,recipe in RECIPES.items()}
    threshold=models['coupled_gain'][1]
    if models['independent_gain'][1]!=threshold:
        raise ValueError('Matched arms must have the same clean threshold')
    tracker=PacketRestorer('consensus').eval()
    clean_iq=tensor(np.stack([chirp_windows(raw[i]) for i in picked]))
    separate_clean=observed_frame(tracker,clean_iq)[0]
    results={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            results[route]={}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs=((route,view,raw[i],rx[i],ids[i]) for i in picked)
                records=zip(*pool.map(replay,jobs,chunksize=6),strict=True)
                received,noise=(tensor(np.stack(v)) for v in records)
                local=coordinates(tracker,received)
                active=corruption_score(local[0])>threshold
                y,rotation,q,rms=packet_frame(local)
                base=rotation*rms
                outputs={'direct':y}
                gains={}
                for arm,(model,_) in models.items():
                    coarse,fine=model(y,q,rotation)
                    raw_gain,fine_gain=model.gains(y,q)
                    gains[arm]=dict(mean_first=float(raw_gain[active].mean()),
                                    mean_fine=float(fine_gain[active].mean()),
                                    fine_log_abs_median=float(fine_gain[active].log().abs().median()))
                    outputs[arm+'_coarse']=torch.where(active[:,None,None],coarse,y)
                    outputs[arm+'_half']=torch.where(active[:,None,None],torch.lerp(y,fine,.5),y)
                noiseless=complex_iq(received-noise)/base
                mask=named_band_masks(y).float()
                aligned_clean=complex_iq(clean_iq)/base
                weight=rms.square()
                reference=carrier_log(separate_clean)*(20/np.log(10))
                pieces={}
                for key,wave in outputs.items():
                    error=(carrier_log(wave)*(20/np.log(10))-reference)
                    pieces[key]=dict(carrier_relative_db_rmse=float(error[active].square().mean().sqrt()),
                      carrier_relative_amp_mae=float(error[active].abs().mean()),
                      packet_rms_ratio_median=float((wave[active].abs().square().mean((-1,-2))
                                                     /y[active].abs().square().mean((-1,-2))).sqrt().median()),
                      packet_rms_ratio_p10=float((wave[active].abs().square().mean((-1,-2))
                                                  /y[active].abs().square().mean((-1,-2))).sqrt().quantile(.1)),
                      private_clean=fidelity(wave,aligned_clean,weight,mask),
                      private_noiseless_relative=fidelity(detail_ratio(wave),detail_ratio(noiseless),weight,mask))
                results[route][view]=dict(active=int(active.sum()),gains=gains,readouts=pieces)
                print('LORA_GAIN_COUPLING_U',route,view,int(active.sum()),flush=True)
    print(json.dumps(dict(status='COMPLETE',n=len(picked),seconds=perf_counter()-started,
                          reference_after_frozen_output=True,target_reads=0,V_reads=0,TX_reads=0,F_reads=0,
                          rows=results),allow_nan=False),flush=True)


if __name__=='__main__':
    main()
