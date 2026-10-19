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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('formal-views','parent-restoration','source-cache','frozen-bundle','target-truth','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    args=parser.parse_args()
    for key in ('formal_views','parent_restoration','source_cache','frozen_bundle','target_truth','output'):
        setattr(args,key,getattr(args,key).resolve())
    if args.output.exists():raise FileExistsError('Immutable formal confirmation exists; reconcile')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned formal adapter code differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='4':raise ValueError('This run owns only physical GPU4')
    metadata=source_parent(args.parent_restoration)
    args.output.mkdir();row=args.output/CANDIDATE;row.mkdir()
    (row/'final.pt').symlink_to((args.parent_restoration/CANDIDATE/'final.pt').resolve())
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,
        commit=args.expected_commit,owner='codex/root/formal-restoration-confirmation',GPU=4,started=time.time()))
    write_json(args.output/'resolved_config.json',dict(metadata,count=COUNT,source_training_updates=0,
        adapter_updates=0,F_updates=0,classifier_fit_updates=0,physical_GPU=4,
        target_access='All original fixed formal inputs, seven predictions then independent truth-last',
        feature_reference_cap=512,source_R03_feedback=False,all_owner_cap=2,min_free_MB=12000))
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from experiments.pamf_restoration.frozen import load_readers
        import fcntl
        shared=args.output.parent/'receiver_residual_capacity4.lock'
        while True:
            with shared.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                state=capacity()[4]
                if len(state['pids'])<2 and state['free_mb']>=12000:
                    restorer,payload=load_checkpoint(row/'final.pt',CANDIDATE,'cuda:0')
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
            trained_R02_checkpoint=True,original_default_heads=True,TX_labels_read=0,target_reads=0))
        reference=prepare_reference(args.formal_views,args.output/'reference')
        predict_row(CANDIDATE,reference,row,'cuda:0',args.frozen_bundle,args.source_cache,checkpoint_loader=load_checkpoint)
        command=[sys.executable,'-X','utf8','-m','experiments.pamf_restoration.runner','score',
            '--reference',str(reference),'--row-output',str(row),'--target-truth',str(args.target_truth),
            '--feature-reference-cap','512']
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),check=True)
        write_json(args.output/'complete.json',dict(status='ANALYZED',candidate=CANDIDATE,count=COUNT,
            source_selection_reused=True,training_updates=0,adaptation_updates=0,
            original_formal_IQ_reused=True,truth_last=True,feature_reference_cap=512,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
            traceback=traceback.format_exc(),automatic_retry=False,artifacts_preserved=True))
        raise


if __name__=='__main__':main()
