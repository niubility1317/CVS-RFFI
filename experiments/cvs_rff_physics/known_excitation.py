"""Source-only correspondence diagnostic for the standard 20 MHz Wi-Fi L-STF.

Per-packet nuisance estimation is diagnostic, never an identity classifier or
TX hardware parameter estimator. No packet-level arrays leave this process.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]
import numpy as np

TONES = np.array([-24,-20,-16,-12,-8,-4,4,8,12,16,20,24])
SIGNS = np.array([1,-1,1,-1,-1,1,-1,-1,1,1,1,1])
AUTHOR_COMMIT = 'ad0598e4a874f4b8e1f391a1e0323e80df2b34ff'
REFERENCE = ('https://raw.githubusercontent.com/bastibl/gr-ieee802-11/'
             + AUTHOR_COMMIT + '/examples/wifi_phy_hier.grc')
WINDOWS = ((0,80),(80,160),(160,240))
DELAYS = np.arange(320, dtype=np.float64)/16


def template(samples, delay=0.):
    """Analytic 20 MHz coefficients sampled at 25 Msps, unit mean power.

    Tone spacing is 312.5 kHz: exp(j*2*pi*k*(n-delay)/80).
    This is not resampling of the author's 10 MHz demonstration waveform.
    """
    n = np.asarray(samples, dtype=np.float64)
    s = np.sum(SIGNS*(1+1j)*np.exp(2j*np.pi*(n[...,None]-delay)*TONES/80), axis=-1)
    return s/np.sqrt(24)


def packet_statistics(iq):
    a = np.asarray(iq, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (2,256) or not np.isfinite(a).all():
        raise ValueError('Expected finite (B,2,256) received IQ')
    z = a[:,0]+1j*a[:,1]
    reference = np.stack([template(np.arange(20), d) for d in DELAYS])
    # s*(n)=-j*s(n-10); conjugation cannot be identified separately from
    # time/phase for this excitation. Do not report a selected I/Q convention.
    bank = reference
    bank_norm = np.sum(abs(bank)**2, axis=1)
    values = {}
    for lo,hi in WINDOWS:
        w = z[:,lo:hi]
        cross = np.sum(w[:,20:]*w[:,:-20].conjugate(), axis=1)
        denom = np.sqrt(np.sum(abs(w[:,20:])**2,axis=1)*np.sum(abs(w[:,:-20])**2,axis=1))
        omega = np.angle(cross)/20
        corrected = w*np.exp(-1j*omega[:,None]*np.arange(80))
        mean = corrected.reshape(-1,4,20).mean(axis=1)
        energy = np.sum(abs(mean)**2,axis=1)
        score = abs(mean @ bank.conjugate().T)**2 / np.maximum(energy[:,None]*bank_norm,1e-24)
        best = np.argmax(score,axis=1)
        spectrum = abs(np.fft.fft(mean,axis=1))**2
        occupied = np.mod(TONES//4,20)
        total = spectrum.sum(1)
        prefix = f'w{lo}_{hi}_'
        values[prefix+'lag20_coherence'] = abs(cross)/np.maximum(denom,1e-24)
        values[prefix+'coherent_cycle_fraction'] = 4*energy/np.maximum(np.sum(abs(corrected)**2,axis=1),1e-24)
        values[prefix+'template_correlation'] = np.sqrt(np.clip(score[np.arange(len(w)),best],0,1))
        values[prefix+'occupied_tone_fraction'] = spectrum[:,occupied].sum(1)/np.maximum(total,1e-24)
        values[prefix+'dc_fraction'] = spectrum[:,0]/np.maximum(total,1e-24)
        values[prefix+'wrapped_offset_hz'] = omega*25_000_000/(2*np.pi)
        values[prefix+'best_delay_samples'] = DELAYS[best]
    if any(not np.isfinite(v).all() for v in values.values()):
        raise FloatingPointError('Nonfinite correspondence statistic')
    return values


def validate_config(c):
    from experiments.cvs_rff_physics.source_preamble import validate_config as old_validate
    old_validate(dict(c,method='rff_source_preamble'))
    if c.get('method')!='rff_known_excitation' or c.get('reference')!=REFERENCE:
        raise ValueError('Fixed reference/config mismatch')
    return c


def run(c):
    import torch
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    from experiments.cvs_identity_ce.source import source_args,write
    validate_config(c)
    out = Path(c['output_root']); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    split = build_contract_split(source_args(c))
    if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:
        raise ValueError('Source roles/empty target changed')
    actual=json.loads((out/'source_contract.json').read_text())
    expected=json.loads(Path(c['source_contract']).read_text())
    if any(actual.get(k)!=v for k,v in expected.items()):
        raise ValueError('Actual physical role contract mismatch')
    resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,torch_version=torch.__version__,
        commit=(ROOT/'release_commit.txt').read_text().strip(),device='cpu',source_counts=split.split_info['counts'],
        target_constructed=False,U_s_use='unused',checkpoint_access=False,training=False,
        source_roles_exact_match=True,per_packet_fit='wrapped CFO, fractional delay grid, complex gain',
        iq_convention_identified=False,conjugation_ambiguity='s*(n)=-j*s(n-10) at 25 Msps',
        cross_packet_fit=False,identity_fit=False,hardware_parameter_fit=False,input_modified=False,
        diagnostic_windows=WINDOWS,delay_step_samples=1/16,cfo_alias_period_hz=1_250_000,
        sample_rate_hz=25_000_000,nominal_wifi_bandwidth_hz=20_000_000)
    write(out/'resolved_config.json',resolved)
    print('RESOLVED_CONFIG '+json.dumps(resolved),flush=True)
    started=time.perf_counter();groups={};counts={};all_values={}
    for role,dataset in [('L_s',split.train),('V',split.val)]:
        count=0
        loader=make_cvs_loader(dataset,batch_size=128,shuffle=False,num_workers=0,device=torch.device('cpu'),drop_last=False)
        role_values={}
        for batch in loader:
            values=packet_statistics(batch['iq'].tolist())
            y,rx,day=(np.asarray(batch[k].tolist()) for k in ('label','receiver','day'))
            if not set(rx.tolist())<=set(c['source_receivers']) or not set(day.tolist())<=set(c['source_days']):
                raise ValueError('Unexpected source receiver/day')
            for name,v in values.items(): role_values.setdefault(name,[]).append(v)
            for ty in np.unique(y):
                for r in np.unique(rx):
                    for d in np.unique(day):
                        mask=(y==ty)&(rx==r)&(day==d);n=int(mask.sum())
                        if not n:continue
                        g=groups.setdefault((role,int(ty),int(r),int(d)),dict(count=0,sums={},squares={}))
                        g['count']+=n
                        for name,v in values.items():
                            g['sums'][name]=g['sums'].get(name,0.)+float(v[mask].sum())
                            g['squares'][name]=g['squares'].get(name,0.)+float((v[mask]**2).sum())
            count+=len(y)
        counts[role]=count
        all_values[role]={k:np.concatenate(v) for k,v in role_values.items()}
        print('ROLE '+json.dumps(dict(role=role,count=count,elapsed_seconds=time.perf_counter()-started)),flush=True)
    if counts!={'L_s':6300,'V':27000}:raise ValueError('Incomplete source diagnostic')
    records=[]
    for (role,tx,rx,day),g in sorted(groups.items()):
        n=g['count'];means={k:v/n for k,v in g['sums'].items()}
        records.append(dict(role=role,tx_index=tx,receiver=rx,day=day,count=n,means=means,
            population_sd={k:float(np.sqrt(max(0,g['squares'][k]/n-v*v))) for k,v in means.items()}))
    summaries={role:{k:dict(mean=float(v.mean()),population_sd=float(v.std()),
        quantiles=dict(zip(['p01','p10','p50','p90','p99'],np.quantile(v,[.01,.1,.5,.9,.99]).tolist())))
        for k,v in values.items()} for role,values in all_values.items()}
    result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',counts=counts,source_only=True,target_access=False,
        checkpoint_access=False,training=False,identity_fit=False,hardware_parameter_fit=False,
        per_packet_nuisance_estimation=True,summaries=summaries,groups=records,
        scope='Known excitation correspondence only; residual includes RX/channel/equalizer/noise, not identified TX.',
        elapsed_seconds=time.perf_counter()-started)
    write(out/'excitation_statistics.json',result)
    write(out/'completion.json',dict(status=result['status'],counts=counts,groups=len(records),target_access=False,
        elapsed_seconds=result['elapsed_seconds']))
    print('COMPLETE '+json.dumps(dict(counts=counts,groups=len(records),elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text(encoding='utf-8')))
