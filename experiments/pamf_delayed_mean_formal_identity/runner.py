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
from .runtime import ROOT,validate_parent,inputs,calibration,cached_reference
from .analysis import analyze
from experiments.pamf_analytic_identity_probe.analysis import representations
from experiments.pamf_analytic_unrolled_restore.descriptors import AnalyticViews
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,tensor_array,code_commit,read_json

PATH_ARGS=('runs_root','source_cache','formal_reference','target_truth','output')


def command(args,mode):
    result=[sys.executable,'-u','-m','experiments.pamf_delayed_mean_formal_identity.runner',mode]
    for key in (*PATH_ARGS,'expected_commit'):result.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    return result


def extractor():
    value=AnalyticViews().eval()
    if list(value.parameters()):raise ValueError('Only fixed parameter-free analytic transforms allowed')
    return value


@torch.no_grad()
def smoke(args):
    fixed=calibration(args)
    source=read_json(args.source_cache/'manifest.json')
    if source.get('status')!='READY' or source.get('native_role_ids')!='EXACT_MATCH':raise ValueError('Existing original source handles required')
    x=IQSource(args.source_cache,'L_s').iq[:2];transform=extractor()
    raw={key:tensor_array(value) for key,value in transform(array_tensor(x,'cpu')).items()}
    values=representations(raw,fixed['moments'])
    if any(values[key].shape!=(2,size) or not np.isfinite(values[key]).all() for key,size in d.GROUPS.items()):
        raise ValueError('Actual source-only analytic no-query smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',source_Ls_IQ=2,model_checkpoint_loads=0,F_access=0,
        TX_labels_read=0,V_reads=0,target_reads=0,parameter_updates=0))


@torch.no_grad()
def extract(args):
    ids,arrays,fixed=inputs(args,with_arrays=True);transform=extractor();started=time.perf_counter();cells=0
    folder=args.output/'features';folder.mkdir()
    for view in d.VIEWS:
        for name in d.READOUTS:
            if name in d.CACHED_READOUTS:
                cached=cached_reference(args,ids)
                for group,size in d.GROUPS.items():
                    source=cached/'features'/(view+'__'+d.CACHED_READOUTS[name]+'__'+group+'.npy')
                    value=np.load(source,mmap_mode='r',allow_pickle=False)
                    if value.shape!=(d.COUNT,size) or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Complete fixed cached analytic cell required')
                    (folder/(view+'__'+name+'__'+group+'.npy')).symlink_to(source)
                cells+=len(d.GROUPS)
                print(json.dumps(dict(event='FORMAL_ANALYTIC_REUSED_READOUT',view=view,readout=name,count=d.COUNT,F_forward_calls=0)),flush=True)
                continue
            destinations={}
            for group,size in d.GROUPS.items():
                path=folder/(view+'__'+name+'__'+group+'.npy')
                if path.exists():raise FileExistsError('Preserve existing feature output')
                destinations[group]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(d.COUNT,size))
            for begin in range(0,d.COUNT,256):
                x=array_tensor(arrays[(view,name)][begin:begin+256],'cpu')
                raw={key:tensor_array(value) for key,value in transform(x).items()}
                values=representations(raw,fixed['moments'])
                for group,value in values.items():destinations[group][begin:begin+len(value)]=value
            for value in destinations.values():value.flush()
            del destinations;cells+=len(d.GROUPS)
            print(json.dumps(dict(event='FORMAL_ANALYTIC_IQ_COMPLETE',view=view,readout=name,count=d.COUNT,F_forward_calls=0)),flush=True)
    np.save(args.output/'physical_ids.npy',ids,allow_pickle=False)
    write_json(args.output/'calibration_reference.json',dict(run=d.REFERENCE,commit=d.REFERENCE_COMMIT,
        count=8192,L_count=fixed['L_count'],U_count=fixed['U_count'],new_calibration_samples=0,new_parameters_fitted=0))
    write_json(args.output/'feature_complete.json',dict(status='FEATURES_COMPLETE',count=d.COUNT,views=list(d.VIEWS),
        groups=d.GROUPS,readouts=list(d.READOUTS),cells=cells,TX_labels_read=0,F_access=0,model_checkpoint_loads=0,
        extra_model_predictions=0,new_calibration_samples=0,reused_feature_cells=d.CACHED_CELLS,new_feature_cells=d.NEW_CELLS,analytic_transform_samples=d.COUNT*len(d.VIEWS)*len(d.NEW_ROWS),
        seconds=time.perf_counter()-started))


def load_features(args,ids):
    marker=read_json(args.output/'feature_complete.json')
    expected=dict(status='FEATURES_COMPLETE',count=d.COUNT,views=list(d.VIEWS),groups=d.GROUPS,readouts=list(d.READOUTS),
        cells=d.TOTAL_CELLS,TX_labels_read=0,F_access=0,model_checkpoint_loads=0,extra_model_predictions=0,new_calibration_samples=0,
        reused_feature_cells=d.CACHED_CELLS,new_feature_cells=d.NEW_CELLS,analytic_transform_samples=d.COUNT*len(d.VIEWS)*len(d.NEW_ROWS))
    if any(marker.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(args.output/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('Every fixed analytic cell must precede target labels')
    keys={(v,r,g) for v in d.VIEWS for r in d.READOUTS for g in d.GROUPS}
    paths={p.stem:p for p in (args.output/'features').glob('*.npy')}
    if set(paths)!={'__'.join(key) for key in keys}:raise ValueError('All registered feature cells required')
    result={}
    for key in keys:
        value=np.load(paths['__'.join(key)],mmap_mode='r',allow_pickle=False)
        if value.shape!=(d.COUNT,d.GROUPS[key[2]]) or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError('Complete finite matched formal features required before labels')
        result[key]=value
    return result


def score(args,*,truth_loader=read_json):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU-only truth-side analysis required')
    ids,_,_=inputs(args,with_arrays=False);features=load_features(args,ids)
    # First target truth open:parent scores and every280featurecell are fixed above.
    truth=truth_loader(args.target_truth)
    labels=np.asarray([truth[str(pid)]['label'] for pid in ids],dtype=np.int64)
    rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids]);day=np.asarray([str(truth[str(pid)]['day']) for pid in ids])
    if len(set(rx))!=7 or len(set(day))!=4:raise ValueError('Unchanged original formal receiver/day population required')
    result=analyze(features,ids,labels,rx,day)
    if len(result['records'])!=d.EXPECTED_RECORDS or len(result['query_label_negative_controls'])!=d.EXPECTED_CONTROLS or len(result['comparisons'])!=d.EXPECTED_COMPARISONS:
        raise ValueError('Complete fixed formal grouping and controls required')
    write_json(args.output/'summary.json',dict(result,status='ANALYZED',role='postscore_formal_diagnostic',run=d.RUN,
        count=d.COUNT,views=list(d.VIEWS),groups=d.GROUPS,candidates=list(d.CANDIDATES),
        all_features_before_labels=True,target_truth_reads=1,target_truth_for_evaluation_only=True,
        F_access=0,F_weights_read=0,F_embeddings_logits_read=0,model_checkpoint_loads=0,extra_model_predictions=0,
        neural_training_updates=0,head_fit_updates=0,new_calibration_samples=0,parent_training_or_decisions_changed=False,
        configuration=d.configuration(args.expected_commit)))
    print(json.dumps(dict(event='FORMAL_ANALYTIC_SCORED',groups=d.EXPECTED_RECORDS,controls=d.EXPECTED_CONTROLS,comparisons=d.EXPECTED_COMPARISONS,F_access=0)),flush=True)


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable CPU diagnostic root required')
    validate_parent(args);args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/compression-formal-analytic-identity',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration(args.expected_commit))
    try:
        smoke(args)
        while True:
            waiting=[]
            for run,plan in d.PARENTS.items():
                parent=args.runs_root/run
                if (parent/'failure.json').is_file() or any((parent/name/'failure.json').is_file() for name in plan['rows']):
                    raise RuntimeError('Parent technical failure;preserve diagnostic,no automatic retry')
                if not (parent/'complete.json').is_file():waiting.append(run)
            if not waiting:break
            print(json.dumps(dict(event='WAIT_PARENT_FULL_FORMAL_SCORES',parents=waiting,target_IQ_reads=0,target_truth_reads=0)),flush=True)
            time.sleep(30)
        inputs(args,with_arrays=False)
        import fcntl
        with (args.runs_root/d.SCORING_RUN/'score.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for mode in ('extract','score'):
                argv=command(args,mode)
                with (args.output/'logs'/(mode+'.log')).open('x',encoding='utf-8') as stream:
                    process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT)
                    write_json(args.output/(mode+'_launch.json'),dict(pid=process.pid,cwd=str(ROOT),argv=argv,started=time.time()))
                    if process.wait():raise RuntimeError('CPU '+mode+' failed;preserve artifacts,no automatic retry')
        write_json(args.output/'complete.json',dict(status='ANALYZED',parent_training_or_decisions_changed=False,
            extra_model_predictions=0,head_fit_updates=0,F_access=0,new_calibration_samples=0,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),
            outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='F-independent formal TX association after all fold/delayed/mean full-formal readouts finish scoring')
    parser.add_argument('mode',choices=('run','extract','score'))
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if code_commit()!=args.expected_commit:raise ValueError('Pinned diagnostic code required')
    torch.set_num_threads(2)
    if args.mode=='run':run(args)
    elif args.mode=='extract':extract(args)
    else:score(args)


if __name__=='__main__':main()
