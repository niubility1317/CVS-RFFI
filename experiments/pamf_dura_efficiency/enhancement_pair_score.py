"""Independent, read-only truth-side arithmetic for the frozen E30 readouts."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

RUN='20261031-phase1-DURA-IQenhance-E30-s2026092701-r01'
MODEL_COMMIT='489f339757b7b3fb9214b3341d780ec98bb14e89'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def statistics(labels,prediction,previous):
    matrix=np.bincount(labels*6+prediction,minlength=36).reshape(6,6)
    diagonal=matrix.diagonal().astype(np.float64)
    precision=diagonal/np.maximum(matrix.sum(0),1);recall=diagonal/np.maximum(matrix.sum(1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    right=prediction==labels;old=previous==labels
    return dict(count=len(labels),accuracy=float(right.mean()*100),macro_f1=float(f1.mean()*100),
        correct=int(right.sum()),repaired=int((right&~old).sum()),damaged=int((~right&old).sum()),
        net_repaired=int(right.sum()-old.sum()),predictions_changed=int((prediction!=previous).sum()))


def correction_metrics(physical,enhanced):
    ratios=[];energies=[];orthogonality=[]
    for offset in range(0,len(physical),1024):
        b=np.asarray(physical[offset:offset+1024],dtype=np.float64)
        delta=np.asarray(enhanced[offset:offset+1024],dtype=np.float64)-b
        energy=np.sum(b*b,axis=(1,2));change=np.sum(delta*delta,axis=(1,2))
        if np.any(energy<=1e-12):raise ValueError('Nondegenerate physical IQ required for relative bound')
        relative=change/energy;ratios.append(np.sqrt(relative));energies.append(relative)
        z=b[:,0]+1j*b[:,1];d=delta[:,0]+1j*delta[:,1]
        meaningful=relative>1e-6
        orthogonality.extend((abs((z.conj()*d).sum(1))[meaningful]/
            np.sqrt(energy[meaningful]*change[meaningful])).tolist())
    r=np.concatenate(ratios);e=np.concatenate(energies)
    if r.max()>.05001:raise ValueError('Actual IQ correction exceeded registered five-percent RMS bound')
    orthmax=max(orthogonality,default=0.)
    if orthmax>.003:raise ValueError('Actual energetic correction has an unexpected global gain/phase component')
    return dict(mean_relative_RMS=float(r.mean()),median_relative_RMS=float(np.median(r)),
        p95_relative_RMS=float(np.quantile(r,.95)),maximum_relative_RMS=float(r.max()),
        mean_relative_energy=float(e.mean()),maximum_relative_energy=float(e.max()),
        energetic_corrections_checked=len(orthogonality),maximum_energetic_complex_orthogonality=orthmax)


def read_json(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--analysis-commit',required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
    root=args.root;run=root/'runs'/RUN
    if read_json(run/'complete.json')['status']!='ANALYZED':raise ValueError('Official full row must close before this independent audit')
    formal=root/'runs/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference'
    ids=np.load(formal/'physical_ids.npy',allow_pickle=False);arrays={};summaries={}
    if ids.shape!=(168000,):raise ValueError('Original full physical population required')
    for arm in ('scratch_joint','physical'):
        row=run/arm;folder=row/'target_prediction';done=read_json(folder/'complete.json')
        expected=dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(VIEWS),role='target',arm=arm,
            F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',F_seed=2026092701,original_default_head=True,
            TX_truth_reads=0,state_updates=0,head_fit_updates=0,code_commit=MODEL_COMMIT)
        if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Both full registered readouts required')
        if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Readout ID mismatch')
        arrays[arm]={}
        for view in VIEWS:
            iq=np.load(folder/(view+'.iq.npy'),mmap_mode='r',allow_pickle=False)
            logits=np.load(folder/(view+'.logits.npy'),mmap_mode='r',allow_pickle=False)
            if (iq.shape!=(168000,2,256) or logits.shape!=(168000,6) or iq.dtype!=np.float32
                or logits.dtype!=np.float32 or not np.isfinite(iq).all() or not np.isfinite(logits).all()):
                raise ValueError('Incomplete registered IQ/logits')
            arrays[arm][view]=(iq,logits.argmax(1))
        summaries[arm]=read_json(row/'target_summary.json')
    baselines={}
    for name,relative,stem in (
        ('raw','20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09/prediction','identity'),
        ('DURA','20261021-phase1-restoration-stage2-graph-s2026092701-r01/stage2_residual_graph/target_confirmation/native_map_prior/prediction','restored')):
        folder=root/'runs'/relative;done=read_json(folder/'complete.json')
        expected=dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(VIEWS),TX_truth_reads=0,state_updates=0,
            F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',F_seed=2026092701,original_default_head=True)
        if any(done.get(k)!=v for k,v in expected.items()):
            raise ValueError('Complete frozen matching baselines required')
        if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Baseline ID mismatch')
        baselines[name]={}
        for view in VIEWS:
            logits=np.load(folder/(view+'.'+stem+'.logits.npy'),mmap_mode='r',allow_pickle=False)
            if logits.shape!=(168000,6) or not np.isfinite(logits).all():raise ValueError('Incomplete baseline logits')
            baselines[name][view]=logits.argmax(1)
    # All 14 new predictions and all baseline views checked before truth/clean.
    truth=read_json(root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    labels=np.asarray([truth[str(pid)]['label'] for pid in ids],dtype=np.int64)
    rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids])
    day=np.asarray([str(truth[str(pid)].get('day','unavailable')) for pid in ids])
    clean=np.load(formal/'clean.npy',mmap_mode='r',allow_pickle=False)
    overall={};records=[];verified_records=0
    dimensions={'overall':np.full(len(ids),'all'),'TX':labels.astype(str),'RX':rx,'day':day}
    rawclean=float((baselines['raw']['clean']==labels).mean()*100)
    for view in VIEWS:
        physical,pp=arrays['physical'][view];enhanced,ep=arrays['scratch_joint'][view]
        row=dict(physical=statistics(labels,pp,baselines['raw'][view]),
            enhanced=statistics(labels,ep,baselines['raw'][view]),
            enhanced_vs_physical=statistics(labels,ep,pp),correction=correction_metrics(physical,enhanced))
        for arm,name in (('physical','physical'),('scratch_joint','enhanced')):
            current=row[name];pred=arrays[arm][view][1];official=summaries[arm]['overall'][view]
            for k,v in current.items():
                if k!='predictions_changed' and abs(v-official[k])>1e-9:raise ValueError('Official aggregate disagrees '+arm+'/'+view+'/'+k)
            current['raw_clean_gap_pp']=rawclean-current['accuracy']
            current['delta_DURA_pp']=current['accuracy']-100*float((baselines['DURA'][view]==labels).mean())
            errors=[]
            for offset in range(0,len(ids),1024):
                x=np.asarray(clean[offset:offset+1024],dtype=np.float64)
                iq=np.asarray(arrays[arm][view][0][offset:offset+1024],dtype=np.float64)
                errors.append(np.sum((iq-x)**2,axis=(1,2))/np.sum(x*x,axis=(1,2)))
            current['paired_iq_NMSE']=float(np.concatenate(errors).mean())
            if abs(current['paired_iq_NMSE']-official['paired_iq_nmse'])>2e-8:raise ValueError('Official IQ NMSE disagrees')
        overall[view]=row
        for dimension,groups in dimensions.items():
            for value in np.unique(groups):
                mask=groups==value
                records.append(dict(view=view,dimension=dimension,value=str(value),**statistics(labels[mask],ep[mask],pp[mask])))
    for arm in ('physical','scratch_joint'):
        for record in summaries[arm]['records']:
            view=record['view'];mask=dimensions[record['dimension']]==record['value']
            current=statistics(labels[mask],arrays[arm][view][1][mask],baselines['raw'][view][mask])
            for key,value in current.items():
                if key not in ('correct','predictions_changed') and abs(value-record[key])>1e-9:raise ValueError('Official group disagrees')
            verified_records+=1
    print(json.dumps(dict(status='INDEPENDENTLY_RECOMPUTED',model_run=RUN,model_commit=MODEL_COMMIT,
        analysis_commit=args.analysis_commit,raw_clean_accuracy=rawclean,overall=overall,
        enhanced_vs_physical_records=records,official_records_verified=verified_records,
        count=168000,all_fourteen_predictions_before_truth=True,new_G_F_forwards=0,target_feedback=False)))


if __name__=='__main__':main()
