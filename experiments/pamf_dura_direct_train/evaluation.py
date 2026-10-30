"""Frozen prediction and independent truth-side scoring, never imported by train."""
import gc
import json
import os
import time

import numpy as np
import torch

from .runner import VIEWS, F_RUN, read_json, write_json, load_model, parent_model
from experiments.pamf_dura_diagnostics.runner import tensor, numpy_values


def reference(p,role):
    if role=='source':
        from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
        folder=p['cache']/'validation'
        ids,source,index=source_reference(folder,p['source_cache'])
        return folder,ids,source,index
    if role!='target':raise ValueError('Registered source/target evaluation role required')
    manifest=read_json(p['formal']/'manifest.json')
    expected=dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Original formal received handles required')
    ids=np.load(p['formal']/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,):raise ValueError('Original formal population required')
    return p['formal'],ids,None,None


def frozen(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    model=load_model(p['output']/args.arm/'final.pt','cuda:0',arm=args.arm,commit=args.commit)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    return model,reader,InferenceEngine(model,reader,reuse=True,graph=True,native=False)


@torch.no_grad()
def predict(args,p,role):
    done=read_json(p['output']/args.arm/'training_complete.json')
    if (done['status'],done['epoch'],done['updates'],done['training_F_access'],done['target_reads']) != ('SOURCE_TRAINED',20,4940,False,0):
        raise ValueError('Complete frozen legal source training before prediction required')
    folder,ids,_,_=reference(p,role)
    model,reader,engine=frozen(args,p)
    output=p['output']/args.arm/(role+'_prediction');output.mkdir()
    started=time.perf_counter();timings={}
    for view in VIEWS:
        received=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if received.shape!=(len(ids),2,256) or received.dtype!=np.float32:raise ValueError('Original FP32 received view required')
        iq=np.lib.format.open_memmap(output/(view+'.iq.npy'),mode='w+',dtype=np.float32,shape=received.shape)
        logits=np.lib.format.open_memmap(output/(view+'.logits.npy'),mode='w+',dtype=np.float32,shape=(len(ids),6))
        raw=None
        if role=='source':raw=np.lib.format.open_memmap(output/(view+'.raw_logits.npy'),mode='w+',dtype=np.float32,shape=(len(ids),6))
        start=time.perf_counter()
        for offset in range(0,len(ids),128):
            end=min(offset+128,len(ids));x=tensor(received[offset:end],'cuda:0')
            restored,variance,features,scores=engine(x)
            if restored.shape!=x.shape or scores.shape!=(len(x),6):raise ValueError('Invalid frozen pipeline output')
            iq[offset:end]=numpy_values(restored);logits[offset:end]=numpy_values(scores)
            if raw is not None:raw[offset:end]=numpy_values(reader(x))
        iq.flush();logits.flush()
        if raw is not None:raw.flush()
        timings[view]=time.perf_counter()-start
        print(json.dumps(dict(event='FROZEN_VIEW_COMPLETE',arm=args.arm,role=role,view=view,count=len(ids))),flush=True)
    np.save(output/'physical_ids.npy',ids,allow_pickle=False)
    write_json(output/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),
        role=role,arm=args.arm,F_run=F_RUN,F_seed=2026092701,original_default_head=True,
        TX_truth_reads=0,state_updates=0,head_fit_updates=0,code_commit=args.commit,
        seconds=time.perf_counter()-started,per_view_seconds=timings,setup=engine.setup()))


def complete_arrays(folder,ids,*,arm,role,commit):
    complete=read_json(folder/'complete.json')
    expected=dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),role=role,arm=arm,
        F_run=F_RUN,F_seed=2026092701,original_default_head=True,TX_truth_reads=0,
        state_updates=0,head_fit_updates=0,code_commit=commit)
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('All fixed views must complete before truth')
    if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Prediction physical IDs differ')
    arrays={}
    for view in VIEWS:
        fields={}
        expected_fields={'iq':(len(ids),2,256),'logits':(len(ids),6)}
        if role=='source':expected_fields['raw_logits']=(len(ids),6)
        for key,shape in expected_fields.items():
            array=np.load(folder/(view+'.'+key+'.npy'),mmap_mode='r',allow_pickle=False)
            if array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all():
                raise ValueError('Incomplete/nonfinite prediction: truth remains closed')
            fields[key]=array
        arrays[view]=fields
    return arrays


def old_predictions(folder,ids,readout):
    complete=read_json(folder/'complete.json')
    expected=dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),TX_truth_reads=0,
        state_updates=0,F_run=F_RUN,F_seed=2026092701,original_default_head=True)
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Matching frozen sameF50 baseline required')
    if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Baseline physical IDs differ')
    output={}
    for view in VIEWS:
        logits=np.load(folder/(view+'.'+readout+'.logits.npy'),mmap_mode='r',allow_pickle=False)
        if logits.shape!=(len(ids),6) or not np.isfinite(logits).all():raise ValueError('Complete baseline predictions required')
        output[view]=logits
    return output


def score(args,p,role):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU truth-side scorer required')
    folder,ids,source,index=reference(p,role)
    output=p['output']/args.arm
    # Validate every preregistered new view and baseline BEFORE truth is opened.
    predictions=complete_arrays(output/(role+'_prediction'),ids,arm=args.arm,role=role,commit=args.commit)
    dura=None
    if role=='source':raw={view:predictions[view]['raw_logits'] for view in VIEWS}
    else:
        raw=old_predictions(p['raw'],ids,'identity')
        dura=old_predictions(p['dura'],ids,'restored')
    if role=='source':
        labels=np.load(p['source_cache']/'V/labels.npy',allow_pickle=False)[index]
        rx,day=source.rx[index],source.day[index]
    else:
        truth=read_json(p['truth'])
        labels=np.asarray([truth[str(pid)]['label'] for pid in ids],dtype=np.int64)
        rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids])
        day=np.asarray([str(truth[str(pid)].get('day','unavailable')) for pid in ids])
    from experiments.pamf_restoration.frozen import classification_statistics
    clean=np.load(folder/'clean.npy',mmap_mode='r',allow_pickle=False)
    raw_clean_accuracy=float((raw['clean'].argmax(1)==labels).mean()*100)
    records=[];overall={}
    for view in VIEWS:
        values=predictions[view]
        decision=values['logits'].argmax(1);before=raw[view].argmax(1)
        metrics=classification_statistics(labels,decision,before)
        metrics['correct']=int((labels==decision).sum())
        metrics['raw_clean_gap_pp']=raw_clean_accuracy-metrics['accuracy']
        if dura is not None:
            previous=dura[view].argmax(1)
            metrics['vs_DURA']=classification_statistics(labels,decision,previous)
            metrics['vs_DURA']['delta_pp']=metrics['accuracy']-float((labels==previous).mean()*100)
        overall[view]=metrics
        for dimension,groups in [('overall',np.asarray(['all']*len(ids))),('TX',labels),('RX',rx),('day',day)]:
            for group in np.unique(groups):
                mask=groups==group
                records.append(dict(view=view,dimension=dimension,value=str(group),
                    **classification_statistics(labels[mask],decision[mask],before[mask])))
        # IQ scoring is also truth-side; paired clean is never a forward input.
        error=[]
        for offset in range(0,len(ids),2048):
            x=clean[offset:offset+2048];y=values['iq'][offset:offset+2048]
            error.append(((y-x)**2).mean((1,2))/np.maximum((x*x).mean((1,2)),1e-4))
        overall[view]['paired_iq_nmse']=float(np.concatenate(error).mean())
    write_json(output/(role+'_summary.json'),dict(status='ANALYZED',role=role,count=len(ids),
        overall=overall,records=records,raw_clean_accuracy=raw_clean_accuracy,
        six_scene_gap3pp=all(overall[view]['raw_clean_gap_pp']<=3 for view in VIEWS[1:]),
        clean_damage_at_most1pp=overall['clean']['raw_clean_gap_pp']<=1,
        lowurban70=overall['practical_low_urban']['accuracy']>=70,
        all_predictions_before_truth=True,target_feedback=False,original_default_head=True,
        F_run=F_RUN,F_seed=2026092701))


@torch.no_grad()
def profile(args,p):
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    baseline,_=parent_model(p,'cuda:0');baseline.eval().requires_grad_(False)
    student=load_model(p['output']/args.arm/'final.pt','cuda:0',arm=args.arm,commit=args.commit)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    values=tensor(np.load(p['cache']/'validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:128],'cuda:0')
    torch.cuda.reset_peak_memory_stats()
    records={}
    for scope in ('G','G_plus_F'):
        engines={name:InferenceEngine(model,reader if scope=='G_plus_F' else None,reuse=True,graph=True,native=False)
                 for name,model in dict(DURA=baseline,student=student).items()}
        for batch in (1,128):
            times={name:[] for name in engines}
            for repeat in range(3):
                order=list(engines) if repeat%2==0 else list(reversed(engines))
                for name in order:
                    engine=engines[name]
                    for _ in range(3):engine(values[:batch])
                    torch.cuda.synchronize()
                    for _ in range(16):
                        started=time.perf_counter();result=engine(values[:batch]);torch.cuda.synchronize()
                        times[name].append((time.perf_counter()-started)*1000)
            records[scope+'_B'+str(batch)]={name:dict(median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),
                throughput_samples_s=float(batch*1000/np.mean(v)),iterations=48) for name,v in times.items()}
        del engines
        gc.collect();torch.cuda.empty_cache()
    write_json(p['output']/args.arm/'matched_profile.json',dict(status='PROFILE_COMPLETE',cases=records,
        parameters=266951,parameter_bytes=266951*4,checkpoint_bytes=(p['output']/args.arm/'final.pt').stat().st_size,
        ConvLinear_MAC=25068072,prior_calls=2,CG_updates=4,FFT_IFFT=27,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],
        timing_scope='Samecard alternating3rounds;FP32/TF32off;preloadedGPUinput;GorG+originalFdefaulthead;copy/ownedoutput/validity included;capture/loader/transfer/queueexcluded',
        memory_scope='Shared benchmark resident models and graphs;not standalone deployment VRAM'))
