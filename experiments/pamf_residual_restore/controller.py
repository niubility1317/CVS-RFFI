"""One owner: source cache, frozen scale, matched refinement, frozen selection."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from . import design as d
from .models import NAMES
from .runtime import ROOT,parent_metadata
from experiments.pamf_reception_restore.data import prepare_cache
from experiments.pamf_restoration.runner import array_tensor,code_commit,read_json
from experiments.pamf_restoration.data import SCENES,IQSource,write_json
from experiments.pamf_restoration.dispatcher import available_cards,capacity,progress


def waveform_choice(rows):
    order=('coarse',*NAMES)
    if len(rows)!=len(order) or {r['candidate'] for r in rows}!=set(order):
        raise ValueError('Coarse plus all three fixed source correctors required')
    for row in rows:
        if set(row)!={'candidate','clean_nmse','mean_residual_nmse'}:
            raise ValueError('No TX/F/recognition fields allowed in waveform selection')
        if any(not np.isfinite(row[key]) or row[key]<0 for key in ('clean_nmse','mean_residual_nmse')):
            raise ValueError('Finite nonnegative source waveform errors required')
    legal=[r for r in rows if r['clean_nmse']<=.02]
    key=(lambda r:(r['mean_residual_nmse'],order.index(r['candidate']))) if legal else (
         lambda r:(r['clean_nmse'],r['mean_residual_nmse'],order.index(r['candidate'])))
    return min(legal or rows,key=key)['candidate']


def source_selection(output,parent):
    parent_metadata(parent)
    baseline=read_json(parent/d.COARSE_NAME/'metrics.json')
    base_clean=baseline['views']['clean']['classification']['AsF']['accuracy']
    base_low=baseline['views']['practical_low_urban']['classification']['AsF']['accuracy']
    base_mean=float(np.mean([baseline['views'][s]['summary']['restored']['iq_nmse'] for s in SCENES]))
    waveform=[dict(candidate='coarse',clean_nmse=baseline['views']['clean']['summary']['restored']['iq_nmse'],mean_residual_nmse=base_mean)]
    identity=[]
    for name in NAMES:
        item=read_json(output/name/'metrics.json')
        clean=item['views']['clean'];low=item['views']['practical_low_urban']['classification']['AsF']
        mean=float(np.mean([item['views'][s]['summary']['restored']['iq_nmse'] for s in SCENES]))
        error=clean['summary']['restored']['iq_nmse'];gain=low['accuracy']-base_low
        drop=base_clean-clean['classification']['AsF']['accuracy']
        stochastic=low['stochastic_diagnostic']['any_class_switch_percent'] if name!='residual_deterministic' else 0.
        if name!='residual_deterministic' and (low['stochastic_diagnostic']['count']!=128 or low['stochastic_diagnostic']['draws']!=3):
            raise ValueError('Preregistered source stochastic diagnostic coverage missing')
        qualified=gain>=.5 and drop<=.2 and error<=.02 and mean<=base_mean and stochastic<=5.
        waveform.append(dict(candidate=name,clean_nmse=error,mean_residual_nmse=mean))
        identity.append(dict(candidate=name,qualified=bool(qualified),source_lowurban_accuracy=low['accuracy'],
            gain_over_coarse_pp=gain,clean_drop_over_coarse_pp=drop,mean_nmse=mean,coarse_mean_nmse=base_mean,
            stochastic_switch_percent=stochastic))
    pure=waveform_choice(waveform)
    eligible=[r for r in identity if r['qualified']]
    assisted=max(eligible,key=lambda r:r['source_lowurban_accuracy'])['candidate'] if eligible else None
    targets=[]
    for name in (pure,assisted):
        if name is not None and name!='coarse' and name not in targets:targets.append(name)
    result=dict(status='SOURCE_SELECTION_FROZEN',waveform_selected=pure,identity_assisted_selected=assisted,
        target_candidates=targets,coarse_reference=d.PARENT_RUN+'/'+d.COARSE_NAME,
        waveform_input=waveform,identity_diagnostic=identity,source_qualified=bool(eligible),
        training_TX_labels_read=0,waveform_selection_TX_F_access=False,identity_selection_uses_source_V_TX_labels=True,
        target_reads_before_selection=0,no_target_feedback=True,
        parent_prediction_reuse_only=pure=='coarse',policy=d.configuration()['thresholds'])
    write_json(output/'source_decision.json',result)
    return result


def child(args,mode,name,gpu):
    command=[sys.executable,'-u','-m','experiments.pamf_residual_restore.runner',mode,
        '--cache',str(args.output/'cache'),'--output',str(args.output),
        '--parent-restoration',str(args.parent_restoration),'--source-cache',str(args.source_cache),
        '--frozen-bundle',str(args.frozen_bundle),'--expected-commit',args.expected_commit]
    if mode!='calibrate':command.extend(['--candidate',name])
    if mode=='confirm':command.extend(['--target-reference',str(args.target_reference),'--target-truth',str(args.target_truth)])
    log=args.output/'logs'/(name+'-'+mode+'.log')
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                     MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return process,dict(mode=mode,candidate=name,pid=process.pid,gpu=gpu,cwd=str(ROOT),argv=command,log=str(log),started=time.time())


def run_queue(args,names,mode,launches):
    import fcntl
    pending=list(names);active={};completed=[];failed=[]
    shared=args.output.parent/'receiver_residual_capacity4.lock'
    while pending or active:
        for name,(process,record) in list(active.items()):
            if process.poll() is None:continue
            required=(args.output/'residual_scale.json') if mode=='calibrate' else (
                args.output/name/'batch1_profile.json' if mode=='worker' else args.output/'target_confirmation'/name/'complete.json')
            if process.returncode or not required.is_file():failed.append(dict(candidate=name,pid=process.pid,exit_code=process.returncode))
            else:completed.append(name)
            del active[name]
        if pending and not failed:
            with shared.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                reserved={key:value[1] for key,value in active.items()}
                slots=available_cards(capacity(),d.GPUS,reserved)
                if slots:
                    gpu,name=slots[0],pending.pop(0);process,record=child(args,mode,name,gpu)
                    active[name]=(process,record);launches.append(record)
                    progress(args.output/'worker_launches.json',dict(rows=launches))
                    deadline=time.monotonic()+45
                    while process.poll() is None and time.monotonic()<deadline:
                        if process.pid in capacity()[gpu]['pids']:break
                        time.sleep(.5)
                fcntl.flock(lock,fcntl.LOCK_UN)
        progress(args.output/'queue_state.json',dict(phase=mode,active=[v[1] for v in active.values()],
            pending=pending,completed=completed,failed=failed,updated=time.time()))
        if failed and not active:break
        if pending or active:time.sleep(2)
    if failed:raise RuntimeError('Technical row failure; healthy active rows preserved; unstarted rows held: '+str(failed))
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-cache','parent-cache','parent-restoration','output','frozen-bundle','target-reference','target-truth'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    args=parser.parse_args()
    for key in ('source_cache','parent_cache','parent_restoration','output','frozen_bundle','target_reference','target_truth'):
        setattr(args,key,getattr(args,key).resolve())
    if args.output.exists():raise FileExistsError('Immutable residual output exists; reconcile before launching')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned residual method differs')
    args.output.mkdir(parents=True);(args.output/'logs').mkdir()
    os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(2)
    write_json(args.output/'resolved_config.json',d.configuration())
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,
        commit=args.expected_commit,controller='CPU only',owner='codex/root/residual-refinement',started=time.time()))
    try:
        metadata=parent_metadata(args.parent_restoration)
        parent_cache=read_json(args.parent_restoration/'cache/complete.json')
        if Path(parent_cache['parent_source_cache']).resolve()!=args.parent_cache:
            raise ValueError('Fixed parent source waveform cache differs')
        from experiments.pamf_restoration.frozen import load_readers
        readers=load_readers(args.frozen_bundle,args.source_cache,'cpu')
        source=IQSource(args.source_cache,'L_s');value=array_tensor(source.iq[:2],'cpu')
        with torch.no_grad():
            if any(not torch.isfinite(model(value)).all() for model in readers.values()):raise ValueError('Frozen F source smoke failed')
        write_json(args.output/'reference_smoke.json',dict(status='PASS',source_samples=2,TX_labels_read=0,target_reads=0,parent=metadata))
        del readers,source,value
        prepare_cache(args.parent_cache,args.output/'cache',epochs=d.EPOCHS,workers=16,
            namespace_prefix=d.NAMESPACE,validation_full_reference=args.parent_restoration/'cache/validation_full')
        launches=[];run_queue(args,['scale'],'calibrate',launches)
        source_rows=run_queue(args,NAMES,'worker',launches)
        decision=source_selection(args.output,args.parent_restoration)
        target_rows=run_queue(args,decision['target_candidates'],'confirm',launches)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=source_rows,target_rows=target_rows,
            source_qualified=decision['source_qualified'],training_TX_labels_read=0,target_feedback=False,
            parent_target_predictions_reused=decision['waveform_selected']=='coarse'))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
            traceback=traceback.format_exc(),automatic_retry=False,artifacts_preserved=True))
        raise


if __name__=='__main__':main()
