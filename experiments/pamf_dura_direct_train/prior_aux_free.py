"""Single-stage E30 ablation of only the pre-solve prior IQ auxiliary loss."""
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
from . import no_variance
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
base=no_variance.base
RUN='20261031-phase1-DURA-noPriorAux-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.prior_aux_free'
PREDECESSOR_MODULE='experiments.pamf_dura_direct_train.coarse_hint'
PREDECESSOR_COMMIT='64b8d453e4ead7b8242163bef67dc3db964d5a1f'
PREDECESSOR_RELEASE='20261031-DURA-coarse-IQhint-E30-r01'


def learning_rate(update):
    if not 0<=update<7410:raise ValueError('Fixed E30 budget required')
    if update<247:return .001*(update+1)/247
    return .00002+(.001-.00002)*.5*(1+math.cos(math.pi*(update-247)/(7410-247-1)))


def objective(*args,**kwargs):
    _,terms=source_objective(*args,**kwargs)
    # Both prior networks and all four CG updates remain in the final-IQ graph.
    # The pre-solve IQ error is observed, but is not an optimization target.
    return terms['iq_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse'],terms


def is_predecessor(tokens,cwd,root):
    try:
        index=tokens.index('-m')
        if tokens[index+1]!=PREDECESSOR_MODULE or tokens[index+2] not in ('worker','score'):return False
        if tokens[tokens.index('--commit')+1]!=PREDECESSOR_COMMIT:return False
        if tokens[tokens.index('--root')+1]!=str(root):return False
    except (ValueError,IndexError):return False
    return Path(cwd).resolve()==(root/'releases'/PREDECESSOR_RELEASE).resolve()


def active_predecessor(root):
    result=[]
    rows=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
    for row in rows:
        if PREDECESSOR_MODULE not in row or PREDECESSOR_COMMIT not in row:continue
        pid=int(row.split(None,1)[0]);proc=Path('/proc')/str(pid)
        try:
            tokens=[v.decode() for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v]
            cwd=os.readlink(proc/'cwd')
        except FileNotFoundError:continue
        if is_predecessor(tokens,cwd,root):result.append(pid)
    return result


def wait_evaluation(args,p):
    # The older healthy run has a per-run scorer lock. Wait for its actual
    # worker/scorer handles to exit, preserving the goal's one-CPU-scorer cap.
    started=time.perf_counter();last=-1.
    while True:
        active=active_predecessor(args.root)
        if not active:break
        if time.perf_counter()-started-last>=30:
            print(json.dumps(dict(event='WAIT_PREDECESSOR_EVALUATION',owned_predecessor_PIDs=active)),flush=True)
            last=time.perf_counter()-started
        time.sleep(5)
    base.write_json(p['output']/args.arm/'evaluation_queue.json',dict(status='ADMITTED',seconds=time.perf_counter()-started,
        predecessor_module=PREDECESSOR_MODULE,predecessor_commit=PREDECESSOR_COMMIT,
        actual_worker_and_score_handles_absent=True,marker_only_admission=False))


def configure():
    no_variance.configure();previous_configuration=base.configuration;previous_command=base.command;previous_train=base.train
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_no_prior_IQ_aux',epochs=30,updates=7410,new_cumulative_epochs=30,
            L_exposures=189000,U_exposures=1701000,corruption_epochs=20,prior_aux_weight=0.,
            objective='finalIQNMSE+.05multiscaleSTFT+.05sourceactionprobeNMSE;priorIQ/variance error diagnostics only',
            prior_networks_retained=2,CG_updates_retained=4,teacher_model=None,training_F_access=False,
            source_freeze='fixedfinalE30/7410,noearlystop/source-target-ranking',
            source_rationale='Fixedsource stagecomparison:betterdefault haspre-solveIQNMSE~.175 vsnoVar~.026;latentprior neednot itself be clean',
            allowed_GPUs=[6],max_total_visible_GPU_processes=3,host_numpy_conversion='DLPack_owned_FP32',
            evaluation_resource_queue=dict(predecessor_module=PREDECESSOR_MODULE,predecessor_commit=PREDECESSOR_COMMIT,
                wait_until_actual_worker_and_scorer_exit=True,reason='single CPU scorer per goal;no change to healthy predecessor'),
            optimization_control='Fixedsamebudget E30physicalreadout;wholeoldprocessF-guided andcostdisclosed;not an independentF-freerun')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==no_variance.MODULE else v for v in previous_command(args,mode,arm,role)]
    def train(args,p):
        previous_train(args,p)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        wait_evaluation(args,p)
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.learning_rate=learning_rate
    base.direct_objective=objective;base.train=train


if __name__=='__main__':configure();base.main()
