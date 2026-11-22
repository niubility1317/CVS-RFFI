"""Test how well the DURA real-linear action can express aligned LoRa full observations."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.full_packet_views import source_role
from experiments.pamf_lora_main.full_supervised import handles
from experiments.pamf_lora_main.packet_physics import CONTEXT, NATIVE_RATE, channel, chirp_samples, chirp_windows, hardware
from experiments.pamf_lora_main.runner import activate


FIT_PROBES=4
TOTAL_PROBES=6
PER_RECEIVER=4
CENTER=slice(32,224)


def bandlimited_probes(n,seed):
    rng=np.random.default_rng(seed)
    f=np.fft.fftfreq(n,d=1/NATIVE_RATE)
    keep=np.abs(f)<=62_500
    samples=rng.normal(size=(TOTAL_PROBES,n))+1j*rng.normal(size=(TOTAL_PROBES,n))
    samples=np.fft.ifft(samples*keep,axis=-1)
    return samples/np.sqrt(np.mean(np.abs(samples)**2,axis=-1,keepdims=True))


def fit_action(x,y,mirror):
    # Fit separately at each time point; the DURA FIR is fixed to the identity.
    probes=np.moveaxis(x,0,-1)
    response=np.moveaxis(y,0,-1)
    design=np.stack((probes,probes.conj()),axis=-1) if mirror else probes[...,None]
    return (np.linalg.pinv(design,rcond=1e-6)@response[...,None])[...,0]


def apply_action(x,coefficients):
    output=x*coefficients[...,0]
    if coefficients.shape[-1]==2:
        output+=x.conj()*coefficients[...,1]
    return output


def chirp_nmse(x,y):
    power=np.mean(np.abs(y)**2,axis=-1)
    return np.mean(np.abs(x-y)**2,axis=-1)/np.maximum(power,1e-8)


def learned_errors(model,expected,inputs,outputs,clean_input,clean_output,observed_scale,
                   observed_phase):
    import torch
    from experiments.cric_dura_comparison.io import tensor

    def apply(operator,values):
        a=tensor(np.stack((values.real,values.imag),axis=1).astype(np.float32))
        estimate=operator(torch.complex(a[:,0],a[:,1]))
        return np.asarray(estimate.tolist())

    clean_scale=np.sqrt(np.mean(np.abs(clean_input)**2))
    if observed_phase:
        y=expected[:,0]+1j*expected[:,1]
        cross=(y*(clean_input/clean_scale).conj()).sum(-1,keepdims=True)
        phase=np.exp(1j*np.where(np.abs(cross)>1e-8,np.angle(cross),0.))
        inputs=inputs*phase[None,...]
        clean_input=clean_input*phase
    with torch.no_grad():
        _,operator,_=model.estimate(tensor(expected))
        predicted=np.stack([apply(operator,v/clean_scale) for v in inputs])
        clean_hat=apply(operator,clean_input/clean_scale)
    truth=outputs/observed_scale
    fitted=predicted[:FIT_PROBES]
    cross=(truth[:FIT_PROBES]*fitted.conj()).sum((0,2))
    energy=np.square(np.abs(fitted)).sum((0,2))
    gain=np.divide(cross,energy,out=np.ones_like(cross),where=energy>1e-8)
    phase=np.exp(1j*np.angle(gain))
    held=truth[FIT_PROBES:]
    out={}
    for key,factor in (('learned_action',np.ones_like(gain)),('learned_after_phase',phase),
                       ('learned_after_gain',gain)):
        test=predicted[FIT_PROBES:]*factor[None,:,None]
        restored=clean_hat*factor[:,None]
        out[key]=dict(held_probe_nmse=chirp_nmse(test,held).reshape(-1).tolist(),
                      held_probe_center_nmse=chirp_nmse(test[...,CENTER],held[...,CENTER]).reshape(-1).tolist(),
                      actual_packet_nmse=chirp_nmse(restored,clean_output/observed_scale).tolist())
    return out


def source_probes(z,rx,physical_id,scene,expected):
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed
    cfg=channel(scene)
    seed=stable_seed(2026112207,'source-V',scene,rx,str(physical_id))
    receiver=hardware(scene,f'source-V/{rx}')

    def observe(w,quality=None):
        return ChannelStream(cfg,seed,receiver).process(
            w[CONTEXT:],preceding_context=w[:CONTEXT],return_traces=True,quality_snr_db=quality)

    original,_=normalize_input(z)
    cached,meta,ref=observe(original)
    offset=meta['numerical_filter_latency_samples']-CONTEXT
    if not np.array_equal(chirp_windows(cached,offset=offset),expected):
        raise ValueError('Replay differs from the registered full practical observation')
    clean_input=chirp_samples(original)
    clean_output=chirp_samples(ref['pre_agc_signal'],offset=offset)
    probes=bandlimited_probes(len(original),stable_seed(2026112227,str(physical_id),'action-probe'))
    x=np.stack([chirp_samples(w) for w in probes])
    outputs=[]
    for w in probes:
        _,_,p=observe(w,meta['quality_snr_db'])
        if not np.array_equal(p['pre_agc_noise'],ref['pre_agc_noise']):
            raise ValueError('Full comparator changed the frozen receiver/noise trajectory')
        outputs.append(chirp_samples(p['pre_agc_signal'],offset=offset))
    y=np.stack(outputs)
    noise=chirp_samples(ref['pre_agc_noise'],offset=offset)
    return x,y,clean_input,clean_output,noise


def diagnose_packet(z,rx,physical_id,scene,expected,model=None,observed_phase=False):
    x,y,clean_input,clean_output,noise=source_probes(z,rx,physical_id,scene,expected)
    rows={}
    for name,mirror in (('main_only',False),('main_and_image',True)):
        coef=fit_action(x[:FIT_PROBES],y[:FIT_PROBES],mirror)
        test=apply_action(x[FIT_PROBES:],coef)
        truth=y[FIT_PROBES:]
        clean_hat=apply_action(clean_input,coef)
        stats=dict(held_probe_nmse=chirp_nmse(test,truth).reshape(-1).tolist(),
                   held_probe_center_nmse=chirp_nmse(test[...,CENTER],truth[...,CENTER]).reshape(-1).tolist(),
                   actual_packet_nmse=chirp_nmse(clean_hat,clean_output).tolist())
        rows[name]=stats
    if model is not None:
        observed_scale=np.sqrt(np.mean(np.abs(clean_output+noise)**2))
        rows.update(learned_errors(model,expected,x,y,clean_input,clean_output,observed_scale,
                                   observed_phase))
    if any(not np.isfinite(v).all() for stats in rows.values() for v in stats.values()):
        raise FloatingPointError('Nonfinite matched action probe')
    return rows


def distribution(values):
    a=np.asarray(values)
    median,p95=np.quantile(a,[.5,.95])
    return dict(n=len(a),mean=float(a.mean()),median=float(median),p95=float(p95))


def frozen_action(rt,c):
    from experiments.cric_dura_comparison.g_lineage import load
    root=Path(rt['run_root'])/'G'/'lineage'/'action'
    proof=read(root/'complete.json')
    if (proof['status']!='SOURCE_FROZEN' or proof['stage']!='action' or proof['updates']!=4940
            or any(proof[k] for k in ('F_reads','TX_labels_read','target_reads'))):
        raise ValueError('Require the frozen source-only action model')
    model,item=load(root/'final.pt',c,'LU')
    if item['stage']!='action' or item['model_seed']!=2026092702:
        raise ValueError('Unexpected registered source action model')
    return model.eval().requires_grad_(False)


def audit(rt,read_frozen_action=False):
    source,cache,c=handles(rt)
    model=frozen_action(rt,c) if read_frozen_action else None
    observed_phase=False
    if read_frozen_action:
        config=read(Path(rt['run_root'])/'G'/'lineage'/'config.json')
        observed_phase=bool(config.get('observed_phase',False))
    raw,rx,ids,_=source_role(source,c,'V')
    v=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    rng=np.random.default_rng(2026112227)
    chosen={r:rng.choice(np.flatnonzero(rx==r),PER_RECEIVER,replace=False) for r in SOURCE}
    views={}
    for j,scene in enumerate(VIEWS[1:],1):
        records={}
        for receiver,indices in chosen.items():
            by_mode={}
            for i in indices:
                row=diagnose_packet(raw[i],receiver,ids[i],scene,v[j,i],model,observed_phase)
                for mode,stats in row.items():
                    samples=by_mode.setdefault(mode,{k:[] for k in stats})
                    for k,values in stats.items():
                        samples[k].extend(values)
            records[receiver]={m:{k:distribution(v) for k,v in a.items()} for m,a in by_mode.items()}
        views[scene]=records
    return dict(status='SOURCE_ACTION_EXPRESSIVITY_COMPLETE',source_packets=len(SOURCE)*PER_RECEIVER,
                full_cache_replay_verified=True,fit_probes=FIT_PROBES,held_probes=TOTAL_PROBES-FIT_PROBES,
                measured_true_waveform_unused_for_fit=True,compares_frozen_action=read_frozen_action,
                observed_phase=observed_phase,F_reads=0,TX_reads=0,target_reads=0,views=views)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source_config',type=Path)
    parser.add_argument('--frozen-action',action='store_true')
    args=parser.parse_args()
    rt=read(args.source_config)
    if set(rt)&{'truth','target_views','target_truth','F_checkpoint'}:
        raise ValueError('Expected source-only worker configuration')
    activate(rt)
    print(json.dumps(audit(rt,args.frozen_action),allow_nan=False),flush=True)
