"""CPU audit of existing complete IQ predictions; no model is loaded."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
RUN='20261031-analysis-DURA-recoverability-s2026092701-r01'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
CASES={
 'DURA_v1':'20261021-phase1-restoration-stage2-graph-s2026092701-r01/stage2_residual_graph/target_confirmation/native_map_prior/prediction',
 'noVar_E20':'20261030-phase1-DURA-direct-noVarAux-s2026092701-r01/scratch_joint/target_prediction',
 'RXcanonical_E20':'20261031-phase1-DURA-source-RXcanonical-s2026092701-r01/scratch_joint/target_prediction',
 'raw':'20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09/prediction'}


def measure(clean,prediction,correct):
    values={k:[] for k in ('nmse','phase','scalar','prefix','tail','tail_energy')}
    for offset in range(0,len(clean),1024):
        x=np.asarray(clean[offset:offset+1024],dtype=np.float64);y=np.asarray(prediction[offset:offset+1024],dtype=np.float64)
        xc=x[:,0]+1j*x[:,1];yc=y[:,0]+1j*y[:,1];energy=(abs(xc)**2).sum(1)
        if np.any(energy<=1e-12):raise ValueError('Nonzero clean energy required')
        error=abs(yc-xc)**2;cross=(yc.conj()*xc).sum(1)
        phase=np.exp(1j*np.angle(cross));scalar=cross/np.maximum((abs(yc)**2).sum(1),1e-12)
        for k,v in dict(nmse=error.sum(1)/energy,phase=(abs(yc*phase[:,None]-xc)**2).sum(1)/energy,
            scalar=(abs(yc*scalar[:,None]-xc)**2).sum(1)/energy,prefix=error[:,:232].sum(1)/energy,
            tail=error[:,232:].sum(1)/energy,tail_energy=(abs(xc[:,232:])**2).sum(1)/energy).items():values[k].append(v)
    a={k:np.concatenate(v) for k,v in values.items()};n=a['nmse'];m=float(n.mean())
    if not np.allclose(n,a['prefix']+a['tail'],atol=1e-12,rtol=1e-12):raise ValueError('Partitions disagree')
    return dict(mean_packet_NMSE=m,energy_fit_score_percent=100*(1-m),
        packet_quantiles={str(q):float(np.quantile(n,q)) for q in (.1,.5,.9,.95,.99)},
        packet_fraction_NMSE_at_most={str(v):float((n<=v).mean()) for v in (.01,.03,.05,.1)},
        phase_aligned_oracle_NMSE=float(a['phase'].mean()),complex_scalar_oracle_NMSE=float(a['scalar'].mean()),
        phase_oracle_error_removed_fraction=float((m-a['phase'].mean())/max(m,1e-30)),
        complex_scalar_oracle_error_removed_fraction=float((m-a['scalar'].mean())/max(m,1e-30)),
        prefix232_error_contribution=float(a['prefix'].mean()),tail24_error_contribution=float(a['tail'].mean()),
        tail24_error_fraction=float(a['tail'].mean()/max(m,1e-30)),tail24_clean_energy_fraction=float(a['tail_energy'].mean()),
        mean_NMSE_on_correct=float(n[correct].mean()) if correct.any() else None,
        mean_NMSE_on_wrong=float(n[~correct].mean()) if (~correct).any() else None,
        accuracy=float(correct.mean()*100),correct=int(correct.sum()),count=len(n))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU truth-side analysis only')
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    root=args.root;output=root/'runs'/RUN;output.mkdir();started=time.perf_counter()
    formal=root/'runs/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference'
    ids=np.load(formal/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,):raise ValueError('Original target population required')
    arrays={}
    for name,relative in CASES.items():
        folder=root/'runs'/relative;done=json.loads((folder/'complete.json').read_text())
        expected=dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(VIEWS),TX_truth_reads=0,state_updates=0,
            F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',F_seed=2026092701,original_default_head=True)
        if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Existing prediction metadata mismatch '+name)
        if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Prediction ID mismatch')
        arrays[name]={}
        for view in VIEWS:
            stem='restored' if name=='DURA_v1' else 'identity' if name=='raw' else None
            ip=formal/(view+'.npy') if name=='raw' else folder/(view+('.restored.iq.npy' if name=='DURA_v1' else '.iq.npy'))
            lp=folder/(view+('.'+stem+'.logits.npy' if stem else '.logits.npy'))
            iq=np.load(ip,mmap_mode='r',allow_pickle=False);logits=np.load(lp,mmap_mode='r',allow_pickle=False)
            if iq.shape!=(168000,2,256) or logits.shape!=(168000,6) or iq.dtype!=np.float32 or logits.dtype!=np.float32:
                raise ValueError('Existing array shape/dtype mismatch')
            if not np.isfinite(iq).all() or not np.isfinite(logits).all():raise ValueError('Existing array nonfinite')
            arrays[name][view]=(iq,logits)
    truth=json.loads((root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text())
    labels=np.asarray([truth[str(pid)]['label'] for pid in ids]);clean=np.load(formal/'clean.npy',mmap_mode='r',allow_pickle=False)
    results={}
    for name,views in arrays.items():
        results[name]={}
        for view,(iq,logits) in views.items():
            value=measure(clean,iq,logits.argmax(1)==labels);results[name][view]=value
            print(json.dumps(dict(event='EXISTING_IQ_AUDITED',case=name,view=view,**value)),flush=True)
    rawclean=results['raw']['clean']['accuracy']
    for views in results.values():
        for value in views.values():
            value['recognition_retained_vs_rawclean_percent']=100*value['accuracy']/rawclean
            value['gap_to_rawclean_pp']=rawclean-value['accuracy']
    payload=dict(status='ANALYZED',code_commit=args.commit,count=168000,views=list(VIEWS),cases=results,raw_clean_accuracy=rawclean,
        all_existing_predictions_verified_before_truth=True,new_G_F_forwards=0,optimizer_updates=0,target_training=False,target_feedback=False,
        seconds=time.perf_counter()-started,interpretation='energy fit=100*(1-mean packet NMSE),not recovered information;clean-dependent oracle alignment is not deployed or a universal upper bound')
    with (output/'results.json').open('x',encoding='utf-8') as stream:json.dump(payload,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(event='RECOVERABILITY_AUDIT_COMPLETE',seconds=payload['seconds'])),flush=True)


if __name__=='__main__':main()
