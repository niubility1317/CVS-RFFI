import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from .reference import prepare_reference,COUNT
from experiments.pamf_reception_restore.runner import load_checkpoint
from experiments.pamf_restoration.runner import code_commit,read_json,array_tensor,predict_row
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.dispatcher import capacity

ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-restoration-formal-frozen-coarse-s2026092701-r04'
PARENT='20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
PARENT_COMMIT='7efdba92932a86bf4b0492a565d35377564a26da'
CANDIDATE='tf_unet_mixed'
RESIDUAL_PARENT='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
RESIDUAL_COMMIT='b01b534f52daf9cae9e940ed6ec6f9243bccbdbe'
RESIDUAL_RUN='20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05'
RESIDUAL_CANDIDATE='residual_deterministic'


def source_parent(parent):
    decision=read_json(parent/'source_decision.json');launch=read_json(parent/'launch.json')
    if (parent.name!=PARENT or launch['commit']!=PARENT_COMMIT or decision['status']!='SOURCE_SELECTION_FROZEN' or
        decision['waveform_selected']!=CANDIDATE or decision['waveform_selection_TX_F_access'] or
        decision['training_TX_labels_read'] or not decision['no_target_feedback']):
        raise ValueError('Fixed source-waveform-selected R02 parent required')
    record=next(row for row in decision['identity_diagnostic'] if row['candidate']==CANDIDATE)
    if not record['qualified']:raise ValueError('Source qualification required for full confirmation')
    return dict(parent_run=PARENT,parent_commit=PARENT_COMMIT,candidate=CANDIDATE,
                source_waveform_selected=True,source_identity_qualified=True,new_target_selection=False)


def source_residual_parent(parent):
    decision=read_json(parent/'source_decision.json');launch=read_json(parent/'launch.json')
    if (parent.name!=RESIDUAL_PARENT or launch['commit']!=RESIDUAL_COMMIT or
        decision['status']!='SOURCE_SELECTION_FROZEN' or decision['waveform_selected']!=RESIDUAL_CANDIDATE or
        decision['identity_assisted_selected']!=RESIDUAL_CANDIDATE or
        decision['target_candidates']!=[RESIDUAL_CANDIDATE] or not decision['source_qualified'] or
        decision['waveform_selection_TX_F_access'] or decision['training_TX_labels_read']!=0 or
        decision['target_reads_before_selection']!=0 or not decision['no_target_feedback']):
        raise ValueError('Fixed source-selected qualified R03 deterministic corrector required')
    record=next(row for row in decision['identity_diagnostic'] if row['candidate']==RESIDUAL_CANDIDATE)
    training=read_json(parent/RESIDUAL_CANDIDATE/'training_complete.json')
    if (not record['qualified'] or training['epochs']!=20 or training['updates']!=4940 or
        training['coarse_updates']!=0 or training['TX_labels_read']!=0 or training['target_reads']!=0 or
        training['L_exposures']!=126000 or training['U_exposures']!=1134000):
        raise ValueError('R03 source qualification/training contract differs')
    return dict(parent_run=RESIDUAL_PARENT,parent_commit=RESIDUAL_COMMIT,candidate=RESIDUAL_CANDIDATE,
                source_waveform_selected=True,source_identity_qualified=True,new_target_selection=False,
                cumulative_restorer_epochs=40,coarse_updates_during_refinement=0)


def plan_settings(plan):
    if plan=='coarse_R04':return RUN,CANDIDATE,4,source_parent,load_checkpoint
    if plan=='residual_R05':
        from experiments.pamf_residual_restore.runtime import load_checkpoint as residual_loader
        return RESIDUAL_RUN,RESIDUAL_CANDIDATE,0,source_residual_parent,residual_loader
    raise ValueError('Only preregistered frozen confirmation plans are allowed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',choices=('coarse_R04','residual_R05'),default='coarse_R04')
    for key in ('formal-views','parent-restoration','source-cache','frozen-bundle','target-truth','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    args=parser.parse_args()
    for key in ('formal_views','parent_restoration','source_cache','frozen_bundle','target_truth','output'):
        setattr(args,key,getattr(args,key).resolve())
    if args.output.exists():raise FileExistsError('Immutable formal confirmation exists; reconcile')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned formal adapter code differs')
    run,candidate,gpu,parent_check,checkpoint_loader=plan_settings(args.plan)
    if args.output.name!=run:raise ValueError('Registered output run name differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('This run owns only its registered physical GPU')
    metadata=parent_check(args.parent_restoration)
    args.output.mkdir();row=args.output/candidate;row.mkdir()
    (row/'final.pt').symlink_to((args.parent_restoration/candidate/'final.pt').resolve())
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,
        commit=args.expected_commit,owner='codex/root/formal-restoration-confirmation',GPU=gpu,started=time.time()))
    write_json(args.output/'resolved_config.json',dict(metadata,count=COUNT,source_training_updates=0,
        adapter_updates=0,F_updates=0,classifier_fit_updates=0,physical_GPU=gpu,registered_plan=args.plan,
        target_access='All original fixed formal inputs, seven predictions then independent truth-last',
        feature_reference_cap=512,source_R03_feedback=False,all_owner_cap=2,min_free_MB=12000))
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from experiments.pamf_restoration.frozen import load_readers
        import fcntl
        shared=args.output.parent/('receiver_residual_capacity'+str(gpu)+'.lock')
        while True:
            with shared.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                state=capacity()[gpu]
                if len(state['pids'])<2 and state['free_mb']>=12000:
                    restorer,payload=checkpoint_loader(row/'final.pt',candidate,'cuda:0')
                    if payload['epoch']!=20 or payload['updates']!=4940:raise ValueError('Fixed final restorer required')
                    readers=load_readers(args.frozen_bundle,args.source_cache,'cuda:0')
                    source=IQSource(args.source_cache,'L_s');x=array_tensor(source.iq[:2],'cuda:0')
                    with torch.no_grad():
                        restored=restorer(x)
                        if not torch.isfinite(restored).all() or any(not torch.isfinite(m(restored)).all() for m in readers.values()):
                            raise ValueError('Real source composite checkpoint smoke failed')
                    del restorer,readers,source,x,restored;torch.cuda.empty_cache()
                    break
            time.sleep(2)
        write_json(args.output/'reference_smoke.json',dict(status='PASS',source_samples=2,
            trained_source_checkpoint=True,parent_run=metadata['parent_run'],original_default_heads=True,TX_labels_read=0,target_reads=0))
        reference=prepare_reference(args.formal_views,args.output/'reference')
        predict_row(candidate,reference,row,'cuda:0',args.frozen_bundle,args.source_cache,checkpoint_loader=checkpoint_loader)
        command=[sys.executable,'-X','utf8','-m','experiments.pamf_restoration.runner','score',
            '--reference',str(reference),'--row-output',str(row),'--target-truth',str(args.target_truth),
            '--feature-reference-cap','512']
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),check=True)
        write_json(args.output/'complete.json',dict(status='ANALYZED',candidate=candidate,count=COUNT,
            source_selection_reused=True,training_updates=0,adaptation_updates=0,
            original_formal_IQ_reused=True,truth_last=True,feature_reference_cap=512,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
            traceback=traceback.format_exc(),automatic_retry=False,artifacts_preserved=True))
        raise


if __name__=='__main__':main()
