import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from . import design as d
from experiments.pamf_r46_compact_refine.runtime import validate_parent_root
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_restoration.runner import code_commit,read_json
from experiments.pamf_restoration.data import write_json

ROOT=Path(__file__).resolve().parents[2]


def rotate(value,turns):
    if type(turns) is not int or turns not in d.TURNS:
        raise ValueError('Registered integer quarter turn required')
    if value.ndim!=3 or value.shape[1:]!=(2,256) or not value.is_floating_point():
        raise ValueError('Floating native IQ[B,2,256] required')
    if turns==0:return value
    if turns==1:return torch.stack((-value[:,1],value[:,0]),1)
    if turns==2:return -value
    return torch.stack((value[:,1],-value[:,0]),1)


def select_rows(roles):
    if roles.shape!=(63000,) or set(roles.tolist())!={'L_s','U_s'} or sum(roles=='L_s')!=6300:
        raise ValueError('Original6300Ls/56700Us source roles required;no V/TX inputs')
    rng=np.random.default_rng(d.SEED)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(roles==role),d.PER_ROLE,replace=False) for role in ('L_s','U_s')]))


def validate_paths(parent,action,source,output):
    if Path(parent).name!=d.PARENT or Path(action).name!=d.ACTION or Path(source).name!='source_cache' or Path(output).name!=d.RUN:
        raise ValueError('Only fixed source parent/cache and new registered output are accepted before IO')


def compare(rotated_output,base_output,clean,base_clean,turns):
    if any(x.shape!=clean.shape for x in (rotated_output,base_output,base_clean)):
        raise ValueError('Matched IQ samples required')
    restored=rotate(rotated_output,(-turns)%4)
    energy=clean.square().mean((1,2)).clamp_min(1e-4)
    return dict(equivariance_nmse=(restored-base_output).square().mean((1,2))/energy,
        paired_clean_nmse=(restored-clean).square().mean((1,2))/energy,
        unrotated_clean_nmse=(base_output-base_clean).square().mean((1,2))/energy)


def statistics(value):
    value=value.double()
    return dict(mean=float(value.mean()),median=float(value.median()),p95=float(torch.quantile(value,.95)),maximum=float(value.max()))


@torch.inference_mode()
def execute(args):
    validate_paths(args.parent,args.action,args.source,args.output)
    if args.output.exists():raise FileExistsError('Existing immutable run must be reconciled,never overwritten')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit empty CUDA devices required for CPU-only probe')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned method commit required')
    manifest=read_json(args.source/'manifest.json')
    if manifest['status']!='READY' or manifest['native_role_ids']!='EXACT_MATCH':raise ValueError('Reuse existing READY/EXACT_MATCH source data')
    validate_parent_root(args.parent)
    if read_json(args.action/'launch.json')['commit']!=d.ACTION_COMMIT:raise ValueError('Fixed original action-cache method required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False);started=time.time()
    write_json(args.output/'launch.json',dict(commit=args.expected_commit,PID=os.getpid(),cwd=str(ROOT),started=started,configuration=d.configuration()))
    try:
        model,payload=load_checkpoint(args.parent/d.PARENT_ROW/d.KIND/'final.pt','cpu')
        if (payload['arm'],payload['epoch'],payload['updates'])!=(d.PARENT_ROW,20,4940) or sum(p.numel() for p in model.parameters())!=350521:
            raise ValueError('Only the human-specified source-final350521 parent is allowed')
        model.eval().requires_grad_(False)
        state={key:value.clone() for key,value in model.state_dict().items()}
        cache=args.action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False)
        indices=select_rows(roles)
        smoke_indices=np.flatnonzero(roles=='L_s')[:2]
        received=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
        smoke=model(array_tensor(received[smoke_indices],'cpu'))
        if smoke.shape!=(2,2,256) or not torch.isfinite(smoke).all():raise ValueError('Actual2Ls full-chain CPU smoke failed')
        write_json(args.output/'smoke.json',dict(status='PASS',source_Ls_IQ=2,checkpoint='R46sourcefinal350521',F_access=0,TX_labels_read=0,V_reads=0,target_reads=0,parameter_updates=0))
        print(json.dumps(dict(event='SOURCE_NOQUERY_SMOKE_PASS',source_Ls=2)),flush=True)
        clean_array=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
        clean=array_tensor(clean_array[indices],'cpu')
        if clean.shape!=(128,2,256) or not torch.isfinite(clean).all():raise ValueError('Matched source IQ128 required')
        groups={'overall':torch.ones(128,dtype=torch.bool),**{r:torch.tensor((roles[indices]==r).tolist(),dtype=torch.bool) for r in ('L_s','U_s')}}
        records=[];measurements=[];means=[]
        for view in d.VIEWS:
            values=clean if view=='clean' else array_tensor(np.load(cache/('epochs/'+view[-2:]+'/received.npy'),mmap_mode='r',allow_pickle=False)[indices],'cpu')
            baseline=model.stage_outputs(values)[0]
            if len(baseline)!=3:raise ValueError('Three complete prior/CG stages required')
            for turns in d.TURNS:
                outputs=baseline if turns==0 else model.stage_outputs(rotate(values,turns))[0]
                for stage,(output,base) in enumerate(zip(outputs,baseline)):
                    metrics=compare(output,base,clean,clean,turns)
                    if any(not torch.isfinite(v).all() for v in metrics.values()):raise ValueError('Nonfinite source diagnostic')
                    if turns==0 and not torch.equal(metrics['equivariance_nmse'],torch.zeros(128)):
                        raise ValueError('Zero-rotation reference must be exactly zero')
                    measurements.append(dict(view=view,turns=turns,stage=stage,metrics={k:v.tolist() for k,v in metrics.items()}))
                    for role,mask in groups.items():
                        records.append(dict(view=view,turns=turns,stage=stage,role=role,count=int(mask.sum()),
                            metrics={key:statistics(value[mask]) for key,value in metrics.items()}))
                    if view!='clean' and turns and stage==2:means.append(float(metrics['equivariance_nmse'].mean()))
                print(json.dumps(dict(event='SOURCE_PHASE_COMPLETE',view=view,turns=turns,count=128)),flush=True)
        if any(not torch.equal(state[key],value) for key,value in model.state_dict().items()):raise ValueError('Frozen parent state changed')
        assert len(records)==108 and len(measurements)==36 and len(means)==6
        mean=sum(means)/6
        write_json(args.output/'per_example_metrics.json',dict(indices=indices.tolist(),roles=roles[indices].tolist(),records=measurements))
        summary=dict(status='ANALYZED',configuration=d.configuration(),method_commit=args.expected_commit,records=records,
            mean_received_nonzero_final_equivariance_nmse=mean,threshold=d.THRESHOLD,source_nontrivial_phase_dependence=mean>d.THRESHOLD,
            hypothesis_is_TX_or_target_proof=False,checkpoint_unchanged=True,source_IQ_unique=128,full_probe_model_inputs=1536,
            F_access=0,TX_labels_read=0,V_reads=0,target_reads=0,truth_reads=0,parameter_updates=0,seconds=time.time()-started,goal_complete=False)
        write_json(args.output/'summary.json',summary)
        write_json(args.output/'complete.json',dict(status='ANALYZED',records=108,source_only=True,method_commit=args.expected_commit,goal_complete=False))
        print(json.dumps(dict(event='SOURCE_PHASE_ANALYZED',mean_equivariance_NMSE=mean,source_threshold_exceeded=mean>d.THRESHOLD)),flush=True)
    except BaseException:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',traceback=traceback.format_exc(),goal_complete=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','action','source','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    execute(parser.parse_args())


if __name__=='__main__':main()
