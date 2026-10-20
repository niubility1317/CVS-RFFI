import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from . import design as d
from .runtime import ROOT,sources
from .analysis import analyze
from experiments.pamf_analytic_identity_probe.analysis import representations
from experiments.pamf_analytic_unrolled_restore.descriptors import AnalyticViews
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.runner import array_tensor,tensor_array,code_commit,read_json
from experiments.pamf_restoration.data import IQSource

PATH_ARGS=('runs_root','parent_action','source_cache','output')


def command(args,mode):
    result=[sys.executable,'-u','-m','experiments.pamf_physics_identity_probe.runner',mode]
    for key in (*PATH_ARGS,'expected_commit'):result+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return result


@torch.no_grad()
def smoke(args):
    calibration=read_json(args.runs_root/d.REFERENCE/'calibration.json')
    x=IQSource(args.source_cache,'L_s').iq[:2]
    transform=AnalyticViews().eval()
    if list(transform.parameters()):raise ValueError('No trained parameter access in analytic source smoke')
    raw={k:tensor_array(v) for k,v in transform(array_tensor(x,'cpu')).items()}
    values=representations(raw,calibration['moments'])
    if set(values)!=set(d.GROUPS) or any(v.shape!=(2,d.GROUPS[k]) or not np.isfinite(v).all() for k,v in values.items()):
        raise ValueError('Actual source analytic no-query smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',actual_source_Ls_IQ=2,model_checkpoint_loads=0,
        F_access=0,TX_labels_read=0,V_reads=0,target_reads=0,state_updates=0))


@torch.no_grad()
def extract(args):
    ids,_,_,existing,arrays,calibration=sources(args,with_arrays=True)
    result={key.replace('__restored__','__r17_reference__'):value for key,value in existing.items()}
    transform=AnalyticViews().eval()
    if list(transform.parameters()):raise ValueError('Parameter-free analytic transform required')
    started=time.perf_counter()
    for name in d.NEW_ROWS:
        for view in d.VIEWS:
            parts={g:[] for g in d.GROUPS}
            for begin in range(0,len(ids),128):
                raw={g:tensor_array(v) for g,v in transform(array_tensor(arrays[(view,name)][begin:begin+128],'cpu')).items()}
                for group,value in representations(raw,calibration['moments']).items():parts[group].append(value)
            for group in parts:result[view+'__'+name+'__'+group]=np.concatenate(parts[group])
            print(json.dumps(dict(event='compact_ANALYTIC_VIEW_COMPLETE',candidate=name,view=view,count=len(ids),F_forward_calls=0)),flush=True)
    with (args.output/'analytic_features.npz').open('xb') as stream:np.savez(stream,**result)
    np.save(args.output/'physical_ids.npy',ids,allow_pickle=False)
    write_json(args.output/'calibration_reference.json',dict(reference_run=d.REFERENCE,reference_commit=d.REFERENCE_COMMIT,
        count=calibration['count'],L_count=calibration['L_count'],U_count=calibration['U_count'],new_calibration_samples=0,
        new_parameters_fitted=0,V_fit_updates=0,source_labels_read=0,F_access=False))
    write_json(args.output/'feature_complete.json',dict(status='FEATURES_COMPLETE',count=1800,views=list(d.VIEWS),groups=d.GROUPS,
        readouts=list(d.READOUTS),TX_label_reads=0,F_weights_read=0,F_embeddings_logits_read=0,extra_restorer_predictions=0,
        cells=len(result),reused_cells=56,new_cells=56,new_calibration_samples=0,analytic_transform_samples=25200,seconds=time.perf_counter()-started))


def load_features(args,ids):
    header=read_json(args.output/'feature_complete.json')
    required=dict(status='FEATURES_COMPLETE',count=1800,views=list(d.VIEWS),groups=d.GROUPS,readouts=list(d.READOUTS),cells=112,
        TX_label_reads=0,F_weights_read=0,F_embeddings_logits_read=0,extra_restorer_predictions=0,new_calibration_samples=0)
    if any(header.get(k)!=v for k,v in required.items()) or not np.array_equal(np.load(args.output/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('Every matched source descriptor cell required before source labels')
    expected={v+'__'+r+'__'+g for v in d.VIEWS for r in d.READOUTS for g in d.GROUPS};result={}
    with np.load(args.output/'analytic_features.npz',allow_pickle=False) as archive:
        if set(archive.files)!=expected:raise ValueError('All112sourceanalyticcells required before labels')
        for key in expected:
            value=archive[key];view,row,group=key.split('__')
            if value.shape!=(1800,d.GROUPS[group]) or not np.isfinite(value).all():raise ValueError('Complete finite analytic features required')
            result[(view,row,group)]=value
    return result


def score(args,*,label_loader=np.load):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU truth-side source diagnostics required')
    ids,source,index,_,_,_=sources(args,with_arrays=False);features=load_features(args,ids)
    labels=label_loader(args.source_cache/'V/labels.npy',allow_pickle=False)[index]
    result=analyze(features,ids,labels,source.rx[index],source.day[index])
    value=dict(result,status='ANALYZED',run=d.RUN,role='source_V',count=1800,views=list(d.VIEWS),groups=d.GROUPS,candidates=list(d.CANDIDATES),
        source_labels_after_all_features=True,source_V_fit_updates=0,new_neural_training_updates=0,head_fit_updates=0,
        extra_restorer_predictions=0,F_weights_read=0,F_embeddings_logits_read=0,target_reads=0,new_calibration_samples=0,
        runtime_decisions_changed=False,parent_training_or_decisions_changed=False,configuration=d.configuration(args.expected_commit))
    write_json(args.output/'summary.json',value)
    print(json.dumps(dict(event='compact_ANALYTIC_IDENTITY_SCORED',records=len(result['records']),comparisons=len(result['comparisons']),target_reads=0)),flush=True)


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable source-only CPU root required')
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/compact-analytic-identity',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration(args.expected_commit))
    try:
        parent=args.runs_root/d.PARENT
        if read_json(parent/'launch.json').get('commit')!=d.PARENT_COMMIT:
            raise ValueError('Pinned original F-free compact source run required before waiting')
        smoke(args)
        while not all((parent/name/'source_complete.json').exists() for name in d.NEW_ROWS):
            if (parent/'failure.json').exists() or any((parent/name/'failure.json').exists() for name in d.NEW_ROWS):
                raise RuntimeError('Parent technical failure;preserve diagnostic output,noautomaticretry')
            print(json.dumps(dict(event='WAIT_FIXED_SOURCE_ROWS',parent=d.PARENT,target_reads=0,source_score_reads=0)),flush=True)
            time.sleep(30)
        sources(args,with_arrays=False)
        import fcntl
        with (parent/'score.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            stages(args)
        write_json(args.output/'complete.json',dict(status='ANALYZED',new_neural_training_updates=0,head_fit_updates=0,
            extra_restorer_predictions=0,F_weights_read=0,F_embeddings_logits_read=0,target_reads=0,new_calibration_samples=0,
            parent_training_or_decisions_changed=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def stages(args):
        for mode in ('extract','score'):
            with (args.output/'logs'/(mode+'.log')).open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command(args,mode),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                write_json(args.output/(mode+'_launch.json'),dict(pid=process.pid,cwd=str(ROOT),argv=command(args,mode),started=time.time()))
                if process.wait():raise RuntimeError('CPU '+mode+' failed;preserve outputs,noautomaticretry')


def main():
    parser=argparse.ArgumentParser(description='Fixed F-independent matched source diagnostic for both final compact rows and originalR17;no targets')
    parser.add_argument('mode',choices=('run','extract','score'))
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned source-only diagnostic run/code required')
    torch.set_num_threads(2)
    if args.mode=='run':run(args)
    elif args.mode=='extract':extract(args)
    else:score(args)


if __name__=='__main__':main()
