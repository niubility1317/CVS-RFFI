"""Fixed final30 source-V validation for the already registered covariance family."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from . import conditioned_covariance_shared_trial as nonnegative
from . import signed_covariance_parallel_trial as signed
from . import coupled_covariance_trial as coupled
from .cache import check_F0_state,load_F0,tensor
from .conditioned_covariance import scratch_conditioned_covariance
from .signed_covariance import scratch_signed_covariance
from .coupled_covariance import scratch_coupled_covariance
from .cost_probe import write
from .covariance_guidance_source import METRICS,numpy_metrics,paired_cells,source_metrics
from .main_split_eval import valid_identity
from .phase_mixture_source import read_trace
from .source_action_factor_probe import references
from .source_inverse import R07,SCENES
from .source_pair import read,validate_source

RUN='20261107-diagnostic-Restore-CovarianceFamily-final30-sourceV-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_family_source_validation'
METHODS=('nonnegative','signed','coupled')
PAIRS=(('nonnegative','signed'),('signed','coupled'),('nonnegative','coupled'))
SPECS={
    'nonnegative':(nonnegative,'ebb279ff3ad430e6a8242686d816976339444588',scratch_conditioned_covariance),
    'signed':(signed,'96143e689001ad94c6885bc2d032a1c897a097a6',scratch_signed_covariance),
    'coupled':(coupled,'6d3dda3c88765c38e1813ecb733948234bb63190',scratch_coupled_covariance),
}
COUNT=480


def validate_parent(method,done,checkpoint):
    module,commit,_=SPECS[method]
    if (done['status'],done['code_commit'],done['epoch'],done['updates'])!=('SOURCE_TRAINED',commit,30,7410):
        raise ValueError('Only the registered source final30 checkpoint is allowed')
    forbidden=('G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates')
    if any(done[k]!=0 for k in forbidden) or (done['source_Ls_exposures'],done['source_Us_exposures'])!=(189000,1701000):
        raise ValueError('Original source-only scratch training required')
    if (set(checkpoint)!={'config','code_commit','epoch','updates','state'} or checkpoint['config']!=module.configuration(module.ARM)
            or (checkpoint['code_commit'],checkpoint['epoch'],checkpoint['updates'])!=(commit,30,7410)):
        raise ValueError('Exact original configuration and final30 required')
    valid_identity(done['F0_identity'])


def load_models(root):
    models={};identities=[]
    for method in METHODS:
        module,_,factory=SPECS[method]
        folder=root/'runs'/module.RUN/'rows'/module.ARM/'training'
        done=read(folder/'complete.json');checkpoint=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        validate_parent(method,done,checkpoint)
        model=factory(2026092701);expected=model.state_dict()
        if set(checkpoint['state'])!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=torch.float32 or not torch.isfinite(v).all() for k,v in checkpoint['state'].items()):
            raise ValueError('Exact finite FP32 model state required')
        model.load_state_dict(checkpoint['state'],strict=True)
        models[method]=model.eval().requires_grad_(False);identities.append(done['F0_identity'])
    if any(identity!=identities[0] for identity in identities):raise ValueError('One original frozen F0 required')
    return models,identities[0]


def predict(args,out):
    started=time.perf_counter()
    # All G factories/configurations resolve before the original F0 bootstrap.
    models,parent_identity=load_models(args.root)
    reader,identity=load_F0(validate_source(read(args.inputs)),'cpu');valid_identity(identity)
    if identity!=parent_identity:raise ValueError('Original F0 checkpoint identity mismatch')
    models_with_reader=[*models.values(),reader]
    snapshots=[{k:v.detach().clone() for k,v in model.state_dict().items()} for model in models_with_reader]
    cache=args.root/'runs'/R07/'cache'
    source=np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(source)!=2:raise ValueError('Two original L_s smoke rows required')
    with torch.inference_mode():
        sample=read_trace(cache/'epochs/01',source,63000)['received']
        for model in models.values():
            output=model(sample);logits=reader.classify_features(reader.features(output))
            if output.shape!=(2,2,256) or logits.shape!=(2,6) or not torch.isfinite(output).all() or not torch.isfinite(logits).all():
                raise ValueError('Actual source G/F0 smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_L_s_rows=2,all_three_actual_final30_G=True,actual_original_F0=True,
        TX_labels_read=0,V_rows=0,target_inputs=0,optimizer_updates=0))
    print('COVARIANCE_FAMILY_SOURCE_SMOKE_PASS',flush=True)
    ref,take=references(args.root,cache,identity)
    if len(take)!=COUNT or len(set(ref['ids'].tolist()))!=COUNT:raise ValueError('Original fixed V480 panel required')
    clean,scale=tensor(ref['clean'],'cpu'),tensor(ref['scale'],'cpu')
    # Same CPU backend for all candidates and their clean reference, no fitted head.
    with torch.inference_mode():
        features=reader.features(clean);clean_logits=reader.classify_features(features)
    with (out/'reference.npz').open('xb') as file:
        np.savez(file,ids=ref['ids'],rx=ref['rx'],day=ref['day'],source_index=take,clean=ref['clean'],scale=ref['scale'],
            features=np.asarray(features.tolist()),logits=np.asarray(clean_logits.tolist()))
    with torch.inference_mode():
        for scene in SCENES:
            trace=read_trace(cache/'validation_actions'/scene,take,1800)
            records={'ids':ref['ids']}
            for method,model in models.items():
                values={k:[] for k in (*METRICS,'iq','features','logits','relative_solve_residual')}
                for start in range(0,COUNT,16):
                    ix=slice(start,start+16)
                    output,aux=model(trace['received'][ix],return_aux=True)
                    z=reader.features(output);logits=reader.classify_features(z)
                    metrics=source_metrics(output,clean[ix],z,logits,features[ix],clean_logits[ix],scale)
                    metrics.update(iq=output,features=z,logits=logits,relative_solve_residual=aux['relative_solve_residual'])
                    if aux['relative_solve_residual'].max()>1e-8 or not all(torch.isfinite(v).all() for v in metrics.values()):
                        raise ValueError('Nonfinite frozen V output or invalid posterior solve')
                    for key,value in metrics.items():values[key].extend(value.tolist())
                records.update({method+'__'+k:np.asarray(v) for k,v in values.items()})
                print('COVARIANCE_FAMILY_SOURCE_CELL_COMPLETE',method,scene,COUNT,flush=True)
            with (out/(scene+'.npz')).open('xb') as file:np.savez(file,**records)
    for model,before in zip(models_with_reader,snapshots):check_F0_state(model,before)
    write(out/'predictions_complete.json',dict(status='SOURCE_FAMILY_V_PREDICTIONS_COMPLETE',code_commit=args.commit,
        F0_identity=identity,methods=list(METHODS),views=list(SCENES),source_V_records=COUNT,cells=21,
        source_G_rows=10080,F0_output_rows=10080,F0_clean_rows=COUNT,source_smoke_G_rows=6,source_smoke_F0_rows=6,
        parents={name:dict(run=spec[0].RUN,commit=spec[1],epoch=30,updates=7410) for name,spec in SPECS.items()},
        TX_labels_read=0,V_training_rows=0,OE_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        checkpoint_selection=False,seconds=time.perf_counter()-started))


def fixed_predictions(out,commit):
    done=read(out/'predictions_complete.json')
    expected=dict(status='SOURCE_FAMILY_V_PREDICTIONS_COMPLETE',code_commit=commit,methods=list(METHODS),views=list(SCENES),
        source_V_records=COUNT,cells=21,TX_labels_read=0,target_inputs=0,optimizer_updates=0,model_state_updates=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('All fixed source predictions required before truth')
    valid_identity(done['F0_identity'])
    with np.load(out/'reference.npz',allow_pickle=False) as file:ref={k:file[k].copy() for k in file.files}
    if ref['ids'].shape!=(COUNT,) or len(set(ref['ids'].tolist()))!=COUNT:raise ValueError('Fixed V480 unique IDs required')
    data={};error=0.
    for scene in SCENES:
        with np.load(out/(scene+'.npz'),allow_pickle=False) as file:a={k:file[k].copy() for k in file.files}
        if not np.array_equal(a['ids'],ref['ids']):raise ValueError('Fixed source prediction order differs')
        for method in METHODS:
            for key,shape in [('iq',(COUNT,2,256)),('features',(COUNT,160)),('logits',(COUNT,6))]:
                value=a[method+'__'+key]
                if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Complete finite source outputs required')
            calculated=numpy_metrics(a[method+'__iq'],ref['clean'],a[method+'__features'],a[method+'__logits'],ref['features'],ref['logits'],ref['scale'])
            for key,value in calculated.items():
                actual=a[method+'__'+key]
                if actual.shape!=(COUNT,) or not np.isfinite(actual).all() or not np.allclose(actual,value,rtol=2e-5,atol=2e-5):
                    raise ValueError('Independent source metric recomputation failed')
                error=max(error,float(np.max(np.abs(actual-value))));a[method+'__'+key]=value
        data[scene]=a
    return done,ref,data,error


def score(args,out):
    import fcntl
    done,ref,data,error=fixed_predictions(out,args.commit)
    # All 21 cells are present and recomputed before the independent truth read.
    lock=(args.root/'runs/20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01/score.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    c=validate_source(read(args.inputs));contract=read(Path(c['source_contract']))
    with np.load(Path(c['source_cache'])/'V_metadata.npz',allow_pickle=False) as file:
        ids=file['ids'].copy()
        if ids.tolist()!=contract['role_ids']['V'] or len(ids)!=27000:raise ValueError('Original source V metadata required')
        lookup={str(v):i for i,v in enumerate(ids)}
        if any(str(v) not in lookup for v in ref['ids']):raise ValueError('Only original source V truth allowed')
        index=np.asarray([lookup[str(v)] for v in ref['ids']])
        if not np.array_equal(file['rx_i'][index],ref['rx']) or not np.array_equal(file['day_i'][index],ref['day']):
            raise ValueError('Original V RX/day binding differs')
        labels=file['y'][index].copy()
    if labels.shape!=(COUNT,) or labels.dtype.kind not in 'iu' or (labels<0).any() or (labels>=6).any():raise ValueError('Original six-class source truth required')
    records=[];pairs=[];clean_correct=ref['logits'].argmax(1)==labels
    masks=[('overall','ALL',np.ones(COUNT,bool))]
    for name,values in [('RX',ref['rx']),('day',ref['day']),('TX',labels)]:masks.extend((name,str(v),values==v) for v in np.unique(values))
    for scene,a in data.items():
        values={method:{key:a[method+'__'+key] for key in METRICS} for method in METHODS}
        for dim,stratum,mask in masks:
            n=int(mask.sum())
            for method in METHODS:
                correct=values[method]['prediction']==labels
                records.append(dict(scene=scene,method=method,dimension=dim,stratum=stratum,count=n,correct=int((correct&mask).sum()),
                    raw_clean_correct=int((clean_correct&mask).sum()),metrics={k:float(values[method][k][mask].mean()) for k in METRICS if k!='prediction'}))
            for old,new in PAIRS:
                before=values[old]['prediction']==labels;after=values[new]['prediction']==labels
                pairs.append(dict(scene=scene,old=old,new=new,dimension=dim,stratum=stratum,count=n,
                    repairs=int((~before&after&mask).sum()),damage=int((before&~after&mask).sum()),
                    feature_TX_cells=paired_cells(values[old],values[new],labels,mask)))
    write(out/'results.json',dict(status='SOURCE_FAMILY_V_ANALYZED',code_commit=args.commit,predictions=done,
        records=records,pairs=pairs,metric_recomputation_max_error=error,source_V_truth_reads=COUNT,
        target_inputs=0,OE_reads=0,optimizer_updates=0,model_state_updates=0,V_training_rows=0,checkpoint_selection=False,
        source_V_historically_exposed=True,Main_performance_claim=False,goal_complete=False))
    print('COVARIANCE_FAMILY_SOURCE_SCORE_COMPLETE',flush=True)


def controller(args,out):
    out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(PID=os.getpid(),CWD=os.getcwd(),code_commit=args.commit,CUDA_empty=True,CPU_threads=2))
    for mode in ('predict','score'):
        command=[sys.executable,'-u','-m',MODULE,mode,'--root',str(args.root),'--inputs',str(args.inputs),'--commit',args.commit]
        with (out/'logs'/(mode+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(out/('submitted-'+mode+'.json'),dict(PID=child.pid,command=command,CWD=os.getcwd()))
            rc=child.wait()
        write(out/('terminal-'+mode+'.json'),dict(PID=child.pid,exit_code=rc))
        if rc:raise RuntimeError('Source family stage failed; preserve, no automatic retry')
    if read(out/'results.json')['status']!='SOURCE_FAMILY_V_ANALYZED':raise ValueError('Complete source truth-last result required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',code_commit=args.commit,model_state_updates=0,target_inputs=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('run','predict','score'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CPU-only diagnostic and actual execution commit required')
    os.nice(max(0,10-os.nice(0)));torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/'runs'/RUN
    {'run':controller,'predict':predict,'score':score}[args.mode](args,out)


if __name__=='__main__':main()
