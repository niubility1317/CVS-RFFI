"""Full L_s/V preamble diagnostics; no U_s retrieval, target, model, or fit."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'code')]
import numpy as np
import torch

WINDOWS = ((0,80),(80,160),(160,240),(0,256))
LAGS = (1,5,20)


def packet_statistics(iq):
    """iq=(B,2,256); no cross-packet estimator or fitted nuisance state."""
    a = np.asarray(iq,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(2,256) or not np.isfinite(a).all():
        raise ValueError('Expected finite256 received I/Q')
    z=a[:,0]+1j*a[:,1]
    p=abs(z)**2
    eps=1e-6
    values={}
    amp=np.sqrt(p)
    values['amplitude_cv']=amp.std(1)/(amp.mean(1)+eps)
    values['normalized_peak_power']=p.max(1)/(p.mean(1)+eps)
    for lo,hi in WINDOWS:
        x,y=z[:,lo+20:hi],z[:,lo:hi-20]
        cross=np.sum(x*y.conjugate(),axis=1)
        normal=np.sqrt(np.sum(abs(x)**2,axis=1)*np.sum(abs(y)**2,axis=1)+eps)
        coherence=cross/normal
        prefix=f'w{lo}_{hi}_lag20'
        values[prefix+'_coherence']=abs(coherence)
        values[prefix+'_real']=coherence.real
        values[prefix+'_imag']=coherence.imag
        # Wrapped relative TX-RX CFO proxy only; no calibration or correction.
        values[prefix+'_wrapped_offset_hz']=np.angle(coherence)*25_000_000/(2*np.pi*20)
    scan=[]
    for d in range(1,33):
        x,y=z[:,40+d:160],z[:,40:160-d]
        scan.append(abs(np.sum(x*y.conjugate(),axis=1))/np.sqrt(np.sum(abs(x)**2,axis=1)*np.sum(abs(y)**2,axis=1)+eps))
    for d in LAGS:
        u=z[:,d:]*z[:,:-d].conjugate()/np.sqrt((p[:,d:]+eps)*(p[:,:-d]+eps))
        closure=u[:,d:]*u[:,:-d].conjugate()
        values[f'lag{d}_increment_real']=u.mean(1).real
        values[f'lag{d}_increment_imag']=u.mean(1).imag
        values[f'lag{d}_closure_real']=closure.mean(1).real
        values[f'lag{d}_closure_imag']=closure.mean(1).imag
        values[f'lag{d}_closure_modulus']=abs(closure).mean(1)
    if any(not np.isfinite(v).all() for v in values.values()):
        raise FloatingPointError('Nonfinite diagnostic')
    return values,np.stack(scan,axis=1),p


def validate_config(c):
    for key in ('checkpoint','resume','teacher','target_inputs','target_truth','p1_truth','p1_capsule'):
        if c.get(key):raise ValueError('Source-only diagnostic rejects '+key)
    expected=dict(method='rff_source_preamble',split_seed=392005,roles=['L_s','V'],
                  source_receivers=[1,3,4,6,8],source_days=[1,2,3],equalized=1,
                  out_len=256,rms_normalize=True,augmentation=False,model=False)
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Fixedsource diagnostic contract mismatch')
    return c


def run(c):
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    from experiments.cvs_identity_ce.source import source_args,write
    validate_config(c)
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    split=build_contract_split(source_args(c))
    if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:
        raise ValueError('Source roles or emptytarget invariant changed')
    commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'LOCAL'
    resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=commit,
                  torch_version=torch.__version__,device='cpu',source_counts=split.split_info['counts'],
                  target_constructed=False,U_s_use='unused',checkpoint_access=False,fit=False,
                  statistic_windows=WINDOWS,closure_lags=LAGS,lag_scan=[1,32],sample_rate_hz=25_000_000,
                  wrapped_offset_note='Modulo1.25MHz relative-offset proxy;not TX oscillator recovery or input correction.')
    write(out/'resolved_config.json',resolved);print('RESOLVED_CONFIG '+json.dumps(resolved),flush=True)
    groups={};started=time.perf_counter();counts={};profiles={}
    for role,dataset in [('L_s',split.train),('V',split.val)]:
        loader=make_cvs_loader(dataset,batch_size=256,shuffle=False,num_workers=0,device=torch.device('cpu'),drop_last=False)
        count=0
        for batch in loader:
            # Explicit copy avoids the verified Torch2.1/NumPy2 bridge mismatch.
            values,scan,power=packet_statistics(batch['iq'].tolist())
            y,rx,day=(np.asarray(batch[k].tolist()) for k in ('label','receiver','day'))
            if not set(rx.tolist())<=set(c['source_receivers']) or not set(day.tolist())<=set(c['source_days']):
                raise ValueError('Unexpected source receiver/day')
            for ty in np.unique(y):
                for r in np.unique(rx):
                    for d in np.unique(day):
                        mask=(y==ty)&(rx==r)&(day==d);n=int(mask.sum())
                        if not n:continue
                        key=(role,int(ty),int(r),int(d))
                        g=groups.setdefault(key,dict(count=0,sums={},squares={},lag_scan_sum=np.zeros(32),power_sum=np.zeros(256)))
                        g['count']+=n;g['lag_scan_sum']+=scan[mask].sum(0);g['power_sum']+=power[mask].sum(0)
                        for name,v in values.items():
                            g['sums'][name]=g['sums'].get(name,0.)+float(v[mask].sum())
                            g['squares'][name]=g['squares'].get(name,0.)+float((v[mask]**2).sum())
            g=profiles.setdefault(role,dict(count=0,lag_scan_sum=np.zeros(32),power_sum=np.zeros(256),sums={}))
            g['count']+=len(y);g['lag_scan_sum']+=scan.sum(0);g['power_sum']+=power.sum(0)
            for name,v in values.items():g['sums'][name]=g['sums'].get(name,0.)+float(v.sum())
            count+=len(y)
        counts[role]=count
        print('ROLE '+json.dumps(dict(role=role,count=count,elapsed_seconds=time.perf_counter()-started)),flush=True)
    if counts!={'L_s':6300,'V':27000}:raise ValueError('Incompletefullsource diagnostic')
    records=[]
    for (role,tx,rx,day),g in sorted(groups.items()):
        n=g['count'];means={k:v/n for k,v in g['sums'].items()}
        sd={k:float(np.sqrt(max(0.,g['squares'][k]/n-v*v))) for k,v in means.items()}
        records.append(dict(role=role,tx_index=tx,receiver=rx,day=day,count=n,means=means,population_sd=sd,
                            lag_scan_mean=(g['lag_scan_sum']/n).tolist(),mean_power_profile=(g['power_sum']/n).tolist()))
    summaries={}
    for role,g in profiles.items():
        scan=g['lag_scan_sum']/g['count']
        summaries[role]=dict(count=g['count'],means={k:v/g['count'] for k,v in g['sums'].items()},
                             lag_scan_mean=scan.tolist(),lag_scan_best_lag=int(np.argmax(scan)+1),
                             mean_power_profile=(g['power_sum']/g['count']).tolist())
    result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',source_roles=counts,source_only=True,target_access=False,
                checkpoint_access=False,fit=False,training=False,input_modified=False,
                scope='ReceivedequalizedIQ repetition/observable statistics only;no calibrated hardware parameters.',
                summaries=summaries,groups=records,elapsed_seconds=time.perf_counter()-started)
    write(out/'preamble_statistics.json',result)
    write(out/'completion.json',dict(status=result['status'],counts=counts,groups=len(records),target_access=False,
                                    elapsed_seconds=result['elapsed_seconds']))
    print('COMPLETE '+json.dumps(dict(counts=counts,groups=len(records),elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text(encoding='utf-8')))
