"""N607 release and independent clean-only score for LoRa PAMF-Tri."""
import argparse
from pathlib import Path, PurePosixPath
import shlex
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .collect import receive, wait_for_predictions
from .data import CLASSES, TARGET, label_ids, packet_indices, recording
from .full_packet_data import LIMIT
from .full_packet_publish import DATA_RUN
from .publish import release, release_commit
from .scoring import macro_f1
from .tri_clean95 import (
    ANCHOR_ROWS, ANCHOR_RUN, BUDGET_RUN, CHIRP_READ_ROWS, CHIRP_READ_RUN, COVERAGE_ROWS, COVERAGE_RUN, DRIFT_RUN, EDGE_ROWS, EDGE_RUN, NOISE_RUN, QUICK_RUN, ROWS, RUN, SCRATCH_ROWS, SCRATCH_RUN,
    PHASE_JOINT_ROWS, PHASE_JOINT_RUN, PHASE_ROWS, PHASE_RUN, PHYSICS_ROWS, PHYSICS_RUN, ROBUST_ROWS, ROBUST_RUN, SEED,
    NATIVE_BAND_ROWS, NATIVE_BAND_RUN, NATIVE_RAW48_ROWS, NATIVE_RAW48_RUN, NATIVE_RAW_ROWS, NATIVE_RAW_RUN, NATIVE_JITTER_ROWS, NATIVE_JITTER_RUN, NATIVE_REPL32_ROWS, NATIVE_REPL32_RUN, NATIVE_RX_ROWS, NATIVE_RX_RUN, NATIVE_LAG_ROWS, NATIVE_LAG_RUN, NATIVE_WIDE_ROWS, NATIVE_WIDE_RUN, NATIVE_BOUNDARY_ROWS, NATIVE_BOUNDARY_RUN, GROUP_CAP_ROWS, GROUP_CAP_RUN, RTL_GATE_ROWS, RTL_GATE_RUN, TAIL_AVERAGE_ROWS, TAIL_AVERAGE_RUN, GROUP_FUSION_ROWS, GROUP_FUSION_RUN, PAIR_MARGIN_ROWS, PAIR_MARGIN_RUN, RANK_WEIGHT_ROWS, RANK_WEIGHT_RUN, RTL_ROWS, RTL_RUN, TIMEFREQ_RUN, TIMING_ROWS, TIMING_RUN, TRANSFER_RUN,
)


QUALIFIED_ROWS={ROBUST_RUN:ROBUST_ROWS,PHASE_RUN:PHASE_ROWS,ANCHOR_RUN:ANCHOR_ROWS,
                TIMING_RUN:TIMING_ROWS,PHASE_JOINT_RUN:PHASE_JOINT_ROWS,RTL_RUN:RTL_ROWS,
                PHYSICS_RUN:PHYSICS_ROWS,COVERAGE_RUN:COVERAGE_ROWS,CHIRP_READ_RUN:CHIRP_READ_ROWS,
                PAIR_MARGIN_RUN:PAIR_MARGIN_ROWS,RANK_WEIGHT_RUN:RANK_WEIGHT_ROWS,
                GROUP_FUSION_RUN:GROUP_FUSION_ROWS,TAIL_AVERAGE_RUN:TAIL_AVERAGE_ROWS,RTL_GATE_RUN:RTL_GATE_ROWS,GROUP_CAP_RUN:GROUP_CAP_ROWS,
                NATIVE_BOUNDARY_RUN:NATIVE_BOUNDARY_ROWS,NATIVE_WIDE_RUN:NATIVE_WIDE_ROWS,NATIVE_LAG_RUN:NATIVE_LAG_ROWS,
                NATIVE_RX_RUN:NATIVE_RX_ROWS,NATIVE_REPL32_RUN:NATIVE_REPL32_ROWS,
                NATIVE_JITTER_RUN:NATIVE_JITTER_ROWS,NATIVE_RAW_RUN:NATIVE_RAW_ROWS,
                NATIVE_RAW48_RUN:NATIVE_RAW48_ROWS,NATIVE_BAND_RUN:NATIVE_BAND_ROWS}


def publish(repo, output, context, ssh_config, quick=False, transfer=False, scratch=False, timefreq=False,
            noise=False,drift=False,budget=False,edge=False,robust=False,headphase=False,
            anchor=False,timing=False,phasejoint=False,rtl=False,physics=False,coverage=False,chirpread=False,pairmargin=False,rankweight=False,groupfusion=False,tailaverage=False,rtlgate=False,groupcap=False,nativeboundary=False,nativewide=False,nativelag=False,nativerx=False,nativerepl32=False,nativejitter=False,nativeraw=False,nativeraw48=False,nativeband=False):
    ctx=read(context)
    rt={k:ctx[k] for k in ('project','python','frozen_sources')}
    project=PurePosixPath(rt['project'])
    parent=project/'runs'/DATA_RUN
    native=project/'releases'/f'{DATA_RUN}-release'/'data'
    studies=((nativeband,NATIVE_BAND_RUN),(nativeraw48,NATIVE_RAW48_RUN),(nativeraw,NATIVE_RAW_RUN),(nativejitter,NATIVE_JITTER_RUN),(nativerepl32,NATIVE_REPL32_RUN),(nativerx,NATIVE_RX_RUN),(nativelag,NATIVE_LAG_RUN),(nativewide,NATIVE_WIDE_RUN),(nativeboundary,NATIVE_BOUNDARY_RUN),(groupcap,GROUP_CAP_RUN),(rtlgate,RTL_GATE_RUN),(tailaverage,TAIL_AVERAGE_RUN),(groupfusion,GROUP_FUSION_RUN),(rankweight,RANK_WEIGHT_RUN),(pairmargin,PAIR_MARGIN_RUN),(chirpread,CHIRP_READ_RUN),(coverage,COVERAGE_RUN),(physics,PHYSICS_RUN),(rtl,RTL_RUN),(phasejoint,PHASE_JOINT_RUN),(timing,TIMING_RUN),(anchor,ANCHOR_RUN),(headphase,PHASE_RUN),
             (robust,ROBUST_RUN),(edge,EDGE_RUN),(budget,BUDGET_RUN),
             (drift,DRIFT_RUN),(noise,NOISE_RUN),(timefreq,TIMEFREQ_RUN),(scratch,SCRATCH_RUN),
             (transfer,TRANSFER_RUN),(quick,QUICK_RUN))
    name=next((name for enabled,name in studies if enabled),RUN)
    rt.update(run_id=name,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.tri_clean95',
              release_root=str(project/'releases'/f'{name}-release'),run_root=str(project/'runs'/name),
              parent_root=str(parent),source_root=str(native/'source'),
              native_target_root=str(native/'target'),target_views=str(native/'target'),
              allowed_gpus=[3,5,6,7,0,1,2,4],max_workers=6,max_compute_processes=12)
    if scratch or timefreq or noise or drift or budget:
        rt.update(scratch=True,fixed_clean_cache=str(project/'runs'/RUN/'clean_cache'),
                  max_workers=4)
    if timefreq:
        del rt['native_target_root'],rt['target_views']
        rt.update(timefreq=True,allowed_gpus=[0,5,4,6,2,1,7],max_workers=2,min_free_mib=10000)
    if noise:
        del rt['native_target_root'],rt['target_views']
        rt.update(noise_study=True,allowed_gpus=[2,4,1,7],max_workers=2,min_free_mib=10000)
    if drift:
        del rt['native_target_root'],rt['target_views'],rt['fixed_clean_cache']
        rt.update(drift_study=True,allowed_gpus=[2,4,1],max_workers=2,min_free_mib=9000)
    if budget:
        del rt['native_target_root'],rt['target_views']
        rt.update(budget_study=True,allowed_gpus=[5,0,6,4],max_workers=2,min_free_mib=9500)
    if edge:
        rt.update(scratch=True,edge_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[6,3,5,0,4],max_workers=2,min_free_mib=10000)
    if robust:
        rt.update(scratch=True,robust_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[3,6,5,7,0],max_workers=4,min_free_mib=10000)
    if headphase:
        rt.update(scratch=True,phase_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[3,6,5,7],max_workers=2,min_free_mib=12500)
    if anchor:
        rt.update(scratch=True,anchor_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[5,6],max_workers=2,min_free_mib=11000)
    if timing:
        rt.update(scratch=True,timing_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[3,7],max_workers=1,min_free_mib=12500)
    if phasejoint:
        rt.update(scratch=True,phase_joint_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[2,7,4],max_workers=1,min_free_mib=12000)
    if rtl:
        rt.update(scratch=True,rtl_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[4,7,2,5],max_workers=1,min_free_mib=11000)
    if physics:
        rt.update(scratch=True,physics_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[6,5,3,2,7],max_workers=1,min_free_mib=12000)
    if coverage:
        rt.update(scratch=True,coverage_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[5,3,2,7,6],max_workers=1,min_free_mib=12000)
    if chirpread:
        rt.update(scratch=True,chirpread_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[3,6,2,5,7],max_workers=1,min_free_mib=12000)
    if pairmargin:
        rt.update(scratch=True,pairmargin_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[6,5,7,2,3],max_workers=1,min_free_mib=12000)
    if rankweight:
        rt.update(scratch=True,rankweight_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[5,7,2,3,6],max_workers=1,min_free_mib=12000)
    if groupfusion:
        rt.update(scratch=True,groupfusion_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[2,7,5,6,3],max_workers=2,min_free_mib=12000)
    if tailaverage:
        rt.update(scratch=True,tailaverage_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[5,6,3,7,2],max_workers=1,min_free_mib=12000)
    if rtlgate:
        rt.update(scratch=True,rtlgate_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[6,3,5,7,2],max_workers=1,min_free_mib=12000)
    if groupcap:
        rt.update(scratch=True,groupcap_study=True,fixed_clean_cache=str(project/'runs'/DRIFT_RUN/'clean_cache'),
                  allowed_gpus=[2,5,7,3,6],max_workers=1,min_free_mib=12000)
    if nativeboundary:
        rt.update(scratch=True,nativeboundary_study=True,allowed_gpus=[7,3,5,2,6],
                  max_workers=1,min_free_mib=12000)
    if nativeband:
        rt.update(scratch=True,native_band_study=True,fixed_clean_cache=str(project/'runs'/NATIVE_WIDE_RUN/'clean_cache'),
                  allowed_gpus=[2,3,7,6],max_workers=1,min_free_mib=12000)
    if nativeraw48:
        rt.update(scratch=True,native_raw48_study=True,fixed_clean_cache=str(project/'runs'/NATIVE_WIDE_RUN/'clean_cache'),
                  allowed_gpus=[3,7,6,2],max_workers=1,min_free_mib=12000)
    if nativeraw:
        rt.update(scratch=True,native_raw_study=True,fixed_clean_cache=str(project/'runs'/NATIVE_WIDE_RUN/'clean_cache'),
                  allowed_gpus=[2,3,7,6],max_workers=1,min_free_mib=12000)
    if nativejitter:
        rt.update(scratch=True,native_jitter_study=True,fixed_clean_cache=str(project/'runs'/NATIVE_WIDE_RUN/'clean_cache'),
                  allowed_gpus=[3,7,6,2],max_workers=1,min_free_mib=12000)
    if nativerepl32:
        rt.update(scratch=True,native_repl32_study=True,fixed_clean_cache=str(project/'runs'/NATIVE_WIDE_RUN/'clean_cache'),
                  allowed_gpus=[2,7,6,3],max_workers=1,min_free_mib=12000)
    if nativerx:
        rt.update(scratch=True,native_rx_study=True,fixed_clean_cache=str(project/'runs'/NATIVE_WIDE_RUN/'clean_cache'),
                  allowed_gpus=[3,2,7,6],max_workers=1,min_free_mib=12000)
    if nativelag:
        rt.update(scratch=True,native_lag_study=True,allowed_gpus=[7,6,3,2],
                  max_workers=2,min_free_mib=12000)
    if nativewide:
        rt.update(scratch=True,nativewide_study=True,allowed_gpus=[3,2,7,5,6],
                  max_workers=2,min_free_mib=12000)
    if quick or transfer:
        del rt['native_target_root'],rt['target_views']
        rt.update(quick=quick,transfer=transfer,fixed_clean_cache=str(project/'runs'/RUN/'clean_cache'))
        if transfer:
            rt.update(allowed_gpus=[3,6,5,7,0,2,4,1],max_workers=4,min_free_mib=12288)
        else:
            rt.update(allowed_gpus=[0,2,4,1],max_workers=4,min_free_mib=8000)
    release(repo,output,rt,ssh_config)


def frozen_predictions(output,names,*,qualified_rows=None):
    accepted=(len(names)==1 and names[0] in qualified_rows) if qualified_rows is not None else (
        bool(names) and names[0]=='control' and len(set(names))==len(names)
        and set(names)<=set(ROWS)|set(SCRATCH_ROWS)|set(EDGE_ROWS))
    if not accepted:
        raise ValueError('Invalid clean-only source selection')
    result={}
    for name in names:
        folder=output/'predictions'/name
        if read(folder/'complete.json')!=dict(method=name,seed=SEED,status='PREDICTIONS_FROZEN',
                target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,views=['clean']):
            raise ValueError('Incomplete clean-only prediction declaration')
        chunks=[]
        for rx in TARGET:
            pred=np.load(folder/f'{rx}__clean__direct.npy',allow_pickle=False)
            if pred.shape!=(LIMIT,) or pred.dtype.kind not in 'iu' or (pred<0).any() or (pred>=len(CLASSES)).any():
                raise ValueError(f'Incomplete clean prediction: {name}/{rx}')
            chunks.append(pred)
        result[name]=np.concatenate(chunks)
    return result


def score(submit, ssh_config, dataset, output):
    run_root=read(submit)['run_root']
    qualified_rows=QUALIFIED_ROWS.get(PurePosixPath(run_root).name)
    qualified=qualified_rows is not None
    root=shlex.quote(run_root)
    conn=['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    state=wait_for_predictions(conn,root,allow_source_only=qualified)
    if state['status']=='SOURCE_DIAGNOSTIC_COMPLETE':
        print('LoRa source selection below 98%; clean test stayed closed',flush=True)
        return
    output.mkdir(parents=True,exist_ok=False)
    receive(conn,root,output,main=False)
    preds=frozen_predictions(output,state['names'],qualified_rows=qualified_rows)
    truth=[]
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        for i,rx in enumerate(TARGET):
            with recording(archive,f'{rx}_test.h5',output) as h5:
                truth.append(label_ids(h5,LIMIT)[packet_indices(LIMIT,i)])
    truth=np.concatenate(truth)
    result={}
    base=preds['control']==truth if not qualified else None
    for name,p in preds.items():
        ok=p==truth
        result[name]=dict(n=len(ok),correct=int(ok.sum()),accuracy=float(ok.mean()),
            macro_f1=macro_f1(truth,p),repairs=int((ok&~base).sum()) if base is not None else None,
            harms=int((base&~ok).sum()) if base is not None else None,
            by_rx=dict(zip(TARGET,ok.reshape(len(TARGET),LIMIT).mean(1).tolist(),strict=True)),
            by_tx={str(tx):float(ok[truth==i].mean()) for i,tx in enumerate(CLASSES)})
    write(output/'scores.json',dict(status='SCORED_COMPLETE',views=['clean'],truth_last=True,seed=SEED,results=result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    actions=parser.add_subparsers(dest='kind',required=True)
    for command, fields in [('publish',('repo','output','context','ssh-config')),
                            ('score',('submit','output','dataset','ssh-config'))]:
        p=actions.add_parser(command)
        for k in fields:p.add_argument('--'+k,type=Path,required=True)
        if command=='publish':
            study=p.add_mutually_exclusive_group()
            study.add_argument('--quick',action='store_true')
            study.add_argument('--transfer',action='store_true')
            study.add_argument('--scratch',action='store_true')
            study.add_argument('--timefreq',action='store_true')
            study.add_argument('--noise',action='store_true')
            study.add_argument('--drift',action='store_true')
            study.add_argument('--budget',action='store_true')
            study.add_argument('--edge',action='store_true')
            study.add_argument('--robust',action='store_true')
            study.add_argument('--headphase',action='store_true')
            study.add_argument('--anchor',action='store_true')
            study.add_argument('--timing',action='store_true')
            study.add_argument('--phasejoint',action='store_true')
            study.add_argument('--rtl',action='store_true')
            study.add_argument('--physics',action='store_true')
            study.add_argument('--coverage',action='store_true')
            study.add_argument('--chirpread',action='store_true')
            study.add_argument('--pairmargin',action='store_true')
            study.add_argument('--rankweight',action='store_true')
            study.add_argument('--groupfusion',action='store_true')
            study.add_argument('--tailaverage',action='store_true')
            study.add_argument('--rtlgate',action='store_true')
            study.add_argument('--groupcap',action='store_true')
            study.add_argument('--nativeboundary',action='store_true')
            study.add_argument('--nativewide',action='store_true')
            study.add_argument('--nativelag',action='store_true')
            study.add_argument('--nativerx',action='store_true')
            study.add_argument('--nativerepl32',action='store_true')
            study.add_argument('--nativejitter',action='store_true')
            study.add_argument('--nativeraw',action='store_true')
            study.add_argument('--nativeraw48',action='store_true')
            study.add_argument('--nativeband',action='store_true')
    args=vars(parser.parse_args())
    kind=args.pop('kind')
    {'publish':publish,'score':score}[kind](**args)
