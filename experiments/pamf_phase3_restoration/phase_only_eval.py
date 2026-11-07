"""Full Main comparison of a fixed phase-only inference path on frozen BF."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from . import covariance_trial as parent
from . import multiview_trial as base
from .cache import check_F0_state, load_F0, tensor
from .coordinate_map_detail_trial import ARM as C4_ARM, RUN as C4_RUN
from .cost_probe import write
from .covariance_parameter_source import G_COMMIT, load_model
from .phase_ramp_posterior import fixed_grid, ramp_posteriors
from .source_pair import read, validate_source
from .variance_trial import recovery_gpu_choice

RUN = '20261107-evaluation-Restore-CovariancePhaseOnly-main-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.phase_only_eval'
ARM = 'TF_CovariancePhaseOnly'
STREAMS = ('raw','DURA_v1','original_bf','candidate')


class FrozenPhaseOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        phase,drift,log_prior = fixed_grid('cpu')
        self.register_buffer('phase',phase)
        self.register_buffer('drift',drift[4:5])
        self.register_buffer('log_prior',log_prior[4:5])

    def forward(self, received):
        estimate = self.model.estimate(received)
        output,_ = ramp_posteriors(estimate['action_matrix'],received.flatten(1).double(),
            self.model.prior_mean.double(),self.model.prior_covariance.double(),
            estimate['variance'].double(),self.phase,self.drift,self.log_prior)
        return output[:,1].float().reshape_as(received)


def frozen_controls(root, task, ids, identity):
    folder = root/'runs'/parent.RUN/'rows'/parent.ARM/'prediction'/task
    done = read(folder/'complete.json')
    expected = dict(status='PREDICTIONS_COMPLETE',code_commit=G_COMMIT,count=len(ids),truth_read=False,
        query_fit=False,model_state_updates=0,F0_identity=identity,task=task,run_id=parent.RUN,arm=parent.ARM,
        readouts=['raw','DURA_v1','candidate'],views=list(base.main_eval.VIEWS))
    if any(done.get(key)!=value for key,value in expected.items()):
        raise ValueError('Exact frozen original BF Main controls required')
    with np.load(folder/'predictions.npz',allow_pickle=False) as f:
        if not np.array_equal(f['ids'],ids):
            raise ValueError('Exact frozen Main physical order required')
        return {new+'__'+view:f[old+'__'+view].copy()
                for new,old in [('raw','raw'),('DURA_v1','DURA_v1'),('original_bf','candidate')]
                for view in base.main_eval.VIEWS}


def predict(args,out):
    c = validate_source(read(args.inputs))
    old,roles = base.source(args)
    model = load_model(args.root,'cuda:0')
    # The shared frozen-BF loader installs its own configuration for checkpoint verification.
    install()
    G = FrozenPhaseOnly(model).to('cuda:0').eval().requires_grad_(False)
    F,identity = load_F0(c,'cuda:0')
    base.main_eval.valid_identity(identity)
    snapshots = [{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (G,F)]
    source2 = np.flatnonzero(roles=='L_s')[:2]
    if len(source2)!=2:
        raise ValueError('Two original source L_s records required')
    original = np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:
        x = tensor(original[source2],'cuda:0')
    finally:
        original._mmap.close()
    with torch.inference_mode():
        restored = G(x)
        estimate = model.estimate(x)
        full,_ = ramp_posteriors(estimate['action_matrix'],x.flatten(1).double(),
            model.prior_mean.double(),model.prior_covariance.double(),
            estimate['variance'].double(),*fixed_grid('cuda:0'))
        torch.testing.assert_close(restored,full[:,1].float().reshape_as(x),rtol=2e-6,atol=2e-7)
        logits = F.classify_features(F.features(restored))
        if not torch.isfinite(restored).all() or logits.shape!=(2,6) or not torch.isfinite(logits).all():
            raise ValueError('Actual frozen BF, phase-only and original F0 smoke failed')
    out.mkdir()
    write(out/'smoke.json',dict(status='PASS',source_Ls_rows=2,phase_only_matches_pilot=True,actual_original_F0=True,
        actual_frozen_BF=True,TX_labels_read=0,target_inputs=0,model_updates=0,new_G_training_updates=0))
    print('PHASE_ONLY_MAIN_REAL_SOURCE_SMOKE_PASS',args.task,flush=True)
    from experiments.cric_dura_comparison.evaluation import handle
    official = base.main_eval.official(args.root)
    manifest,ids,positions = handle(official['tests'][args.task]['views_root'],official['classes'])
    if len(ids)!=base.main_eval.COUNTS[args.task]:
        raise ValueError('Full original Main population required')
    values = frozen_controls(args.root,args.task,ids,identity)
    for view in base.main_eval.VIEWS:
        path = manifest['clean_ref'] if view=='clean' else str(Path(manifest['observations_root'])/(view+'.npy'))
        a = np.load(path,mmap_mode='r',allow_pickle=False)
        decisions = []
        try:
            if a.ndim!=3 or a.shape[1:]!=(2,256) or a.dtype!=np.float32:
                raise ValueError('Original Main FP32 IQ required')
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    take = slice(start,start+128) if positions is None else positions[start:start+128]
                    iq = G(tensor(a[take],'cuda:0'))
                    logits = F.classify_features(F.features(iq))
                    if not torch.isfinite(iq).all() or not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite frozen phase-only IQ/logits')
                    decisions.extend(logits.argmax(1).cpu().tolist())
        finally:
            a._mmap.close()
        values['candidate__'+view] = np.asarray(decisions,np.int64)
        print('PHASE_ONLY_MAIN_VIEW_COMPLETE',args.task,view,len(ids),flush=True)
    for m,snapshot in zip((G,F),snapshots):
        check_F0_state(m,snapshot)
    with (out/'predictions.npz').open('xb') as f:
        np.savez(f,ids=ids,**values)
    write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',run_id=RUN,arm=ARM,task=args.task,count=len(ids),
        code_commit=args.commit,F0_identity=identity,readouts=list(STREAMS),views=list(base.main_eval.VIEWS),
        baseline_reused_from=parent.RUN,truth_read=False,query_fit=False,model_state_updates=0,new_G_training_updates=0,
        frozen_parent_training_epochs=30,frozen_parent_training_updates=7410,phase_grid_points=256,drift_points=1))


def c4_gpu_finished(root):
    folder = root/'runs'/C4_RUN
    for stage in ('train','cross_rx','cross_both'):
        path = folder/f'terminal-{C4_ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code']!=0 or stage=='cross_both'):
            return True
    return False


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CUDA-empty evaluation controller required')
    if read(args.root/'runs'/parent.RUN/'complete.json')['status']!='ARTIFACTS_COMPLETE':
        raise ValueError('Original frozen BF Main comparison must be complete')
    out = args.root/'runs'/RUN
    out.mkdir()
    (out/'logs').mkdir()
    row = out/'rows'/ARM
    (row/'prediction').mkdir(parents=True)
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit,
        new_G_training_updates=0,wait_for_last_gpu_stage_of=C4_RUN))
    if not c4_gpu_finished(args.root):
        print('WAITING_FOR_C4_LAST_GPU_STAGE',flush=True)
    while not c4_gpu_finished(args.root):
        time.sleep(5)
    for stage in ('cross_rx','cross_both','score'):
        admission = None
        if stage=='score':
            command = base.command(args,'score',ARM)
        else:
            choices = recovery_gpu_choice()
            while not choices:
                time.sleep(5)
                choices = recovery_gpu_choice()
            admission = choices[0]
            command = base.command(args,'predict',ARM,stage)
        with (out/'logs'/(ARM+'-'+stage+'.log')).open('xb') as log:
            process = subprocess.Popen(command,cwd=Path.cwd(),
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/('submitted-'+ARM+'-'+stage+'.json'),
            dict(pid=process.pid,command=command,CWD=os.getcwd(),admission=admission))
        code = process.wait()
        write(out/('terminal-'+ARM+'-'+stage+'.json'),dict(pid=process.pid,exit_code=code))
        if code:
            raise RuntimeError('Frozen phase-only evaluation failed; preserve artifacts')
    if read(row/'results.json')['status']!='ANALYZED':
        raise ValueError('All Main views and populations must be scored')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',all_Main_two_populations_seven_scored=True,
        new_G_training_updates=0,default_DURA_changed=False,goal_complete=False))


def install():
    base.RUN,base.MODULE,base.ARMS,base.STREAMS = RUN,MODULE,(ARM,),STREAMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['run','predict','score'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--arm',choices=[ARM])
    parser.add_argument('--task',choices=tuple(base.main_eval.COUNTS))
    args = parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('Actual Git code/CWD required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    install()
    if args.mode=='run':
        controller(args)
        return
    if args.arm!=ARM:
        parser.error('--arm required')
    row = args.root/'runs'/RUN/'rows'/ARM
    if args.mode=='predict':
        if not args.task:
            parser.error('--task required')
        predict(args,row/'prediction'/args.task)
    else:
        base.score(args,row/'score')


if __name__=='__main__':
    main()
