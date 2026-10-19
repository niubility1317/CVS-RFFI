"""One N607 owner, three cards, same-row automatic prediction and truth-last."""
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
from .data import prepare_cache
from .runner import ROOT,array_tensor,code_commit,read_json
from experiments.pamf_restoration.data import SCENES,IQSource,write_json
from experiments.pamf_restoration.dispatcher import available_cards,capacity,progress


def waveform_choice(rows):
    if len(rows)!=len(NAMES) or {r['candidate'] for r in rows}!=set(NAMES):
        raise ValueError('All six fixed source candidates required')
    for row in rows:
        if set(row)!={'candidate','clean_nmse','mean_residual_nmse'}:
            raise ValueError('TX/F/recognition fields forbidden in waveform selector')
        if any(not np.isfinite(row[k]) or row[k]<0 for k in ('clean_nmse','mean_residual_nmse')):
            raise ValueError('Finite waveform metrics required')
    legal=[r for r in rows if r['clean_nmse']<=.02]
    key=(lambda r:(r['mean_residual_nmse'],NAMES.index(r['candidate']))) if legal else (
         lambda r:(r['clean_nmse'],r['mean_residual_nmse'],NAMES.index(r['candidate'])))
    return min(legal or rows,key=key)['candidate']


def source_selection(output):
    waveform,identity=[],[]
    for name in NAMES:
        item=read_json(output/name/'metrics.json')
        clean=item['views']['clean'];low=item['views']['practical_low_urban']['classification']['AsF']
        mean=float(np.mean([item['views'][s]['summary']['restored']['iq_nmse'] for s in SCENES]))
        baseline=float(np.mean([item['views'][s]['summary']['identity']['iq_nmse'] for s in SCENES]))
        clean_error=clean['summary']['restored']['iq_nmse']
        waveform.append(dict(candidate=name,clean_nmse=clean_error,mean_residual_nmse=mean))
        clean_class=clean['classification']['AsF']
        gain=low['accuracy']-low['baseline_accuracy'];drop=clean_class['baseline_accuracy']-clean_class['accuracy']
        qualified=gain>=.5 and drop<=.2 and clean_error<=.02 and mean<=baseline
        identity.append(dict(candidate=name,qualified=bool(qualified),source_lowurban_accuracy=low['accuracy'],
            source_gain_pp=gain,clean_drop_pp=drop,mean_nmse=mean,baseline_nmse=baseline))
    pure=waveform_choice(waveform)
    eligible=[r for r in identity if r['qualified']]
    assisted=max(eligible,key=lambda r:r['source_lowurban_accuracy'])['candidate'] if eligible else None
    targets=[pure]+([assisted] if assisted is not None and assisted!=pure else [])
    decision=dict(status='SOURCE_SELECTION_FROZEN',waveform_selected=pure,identity_assisted_selected=assisted,
        target_candidates=targets,waveform_input=waveform,identity_diagnostic=identity,
        training_TX_labels_read=0,waveform_selection_TX_F_access=False,identity_selection_uses_source_V_TX_labels=True,
        target_reads_before_selection=0,no_target_feedback=True,source_qualified=bool(eligible),
        policy=d.configuration()['source_selection'])
    write_json(output/'source_decision.json',decision)
    return decision


def child(args,mode,name,gpu):
    command=[sys.executable,'-u','-m','experiments.pamf_reception_restore.runner',mode,
        '--candidate',name,'--cache',str(args.output/'cache'),'--output',str(args.output),
        '--source-cache',str(args.source_cache),'--frozen-bundle',str(args.frozen_bundle),
        '--expected-commit',args.expected_commit]
    if mode=='confirm':command.extend(['--target-reference',str(args.target_reference),'--target-truth',str(args.target_truth)])
    log=args.output/'logs'/(name+('-target' if mode=='confirm' else '')+'.log')
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                     MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(mode=mode,candidate=name,pid=process.pid,gpu=gpu,cwd=str(ROOT),argv=command,log=str(log),started=time.time())
    return process,record


def run_queue(args,names,mode,launches):
    import fcntl
    pending=list(names);active={};completed=[];failed=[]
    shared=args.output.parent/'receiver_residual_capacity4.lock'
    while pending or active:
        for name,(process,record) in list(active.items()):
            if process.poll() is None:continue
            required=(args.output/name/'batch1_profile.json') if mode=='worker' else (
                       args.output/'target_confirmation'/name/'complete.json')
            if process.returncode or not required.is_file():failed.append(dict(candidate=name,pid=process.pid,exit_code=process.returncode))
            else:completed.append(name)
            del active[name]
        if pending and not failed:
            with shared.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                reserved={k:v[1] for k,v in active.items()}
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
    if failed:raise RuntimeError('Technical row failure; healthy rows preserved; unstarted rows not submitted: '+str(failed))
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-cache','parent-cache','output','frozen-bundle','target-reference','target-truth'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    args=parser.parse_args()
    for key in ('source_cache','parent_cache','output','frozen_bundle','target_reference','target_truth'):
        setattr(args,key,getattr(args,key).resolve())
    if args.output.exists():raise FileExistsError('Immutable output exists; reconcile before any launch')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned code differs')
    args.output.mkdir(parents=True);(args.output/'logs').mkdir()
    os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(2)
    write_json(args.output/'resolved_config.json',d.configuration())
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,
        commit=args.expected_commit,controller='CPU only',owner='codex/root/reception-restoration',started=time.time()))
    try:
        from experiments.pamf_restoration.frozen import load_readers
        readers=load_readers(args.frozen_bundle,args.source_cache,'cpu')
        source=IQSource(args.source_cache,'L_s');value=array_tensor(source.iq[:2],'cpu')
        with torch.no_grad():
            if any(not torch.isfinite(model(value)).all() for model in readers.values()):raise ValueError('Existing F no-query checkpoint smoke failed')
        write_json(args.output/'reference_smoke.json',dict(status='PASS',source_samples=2,TX_labels_read=0,target_reads=0))
        del readers,source,value
        prepare_cache(args.parent_cache,args.output/'cache',epochs=d.EPOCHS,
                      workers=d.configuration()['data_generation_workers'])
        launches=[];source_rows=run_queue(args,NAMES,'worker',launches)
        decision=source_selection(args.output)
        target_rows=run_queue(args,decision['target_candidates'],'confirm',launches)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=source_rows,target_rows=target_rows,
            source_qualified=decision['source_qualified'],training_TX_labels_read=0,target_feedback=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
            traceback=traceback.format_exc(),automatic_retry=False,artifacts_preserved=True))
        raise


if __name__=='__main__':main()
