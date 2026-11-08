"""Continue original gradient-coupled scratch training from resource-stopped E15."""
import sys
from pathlib import Path

from . import coupled_gradient_trial as original
from . import coupled_resume_support as resume
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .fixed_prediction_score_recovery import original_scorer
from .variance_trial import recovery_gpu_choice

RUN='20261108-phase1-Restore-GradientCoupled-E30-s2026092701-r02'
MODULE='experiments.pamf_phase3_restoration.coupled_gradient_resume_trial'
ARM=original.ARM
PARENT_COMMIT='54094bbbdba953575533164be665b74763ac1a18'
PARENT_EPOCH=15
PARENT_CONFIG=original.configuration(ARM)


def configuration(arm):
    if arm!=ARM:raise ValueError('Original coupled arm required')
    cfg=dict(PARENT_CONFIG)
    cfg.update(run_id=RUN,technical_recovery_parent=original.RUN,technical_recovery_commit=PARENT_COMMIT,
        technical_recovery_epoch=PARENT_EPOCH,resource_policy='GPU5 only, admit at most two existing processes and >=12GiB free, total cap3.',
        scientific_change='Exact original coupled objective and source schedule; technical E15 continuation only.')
    return cfg


def factory(arm):
    configuration(arm);return original.factory(arm)


def train(args,out):
    parent=resume.parent_training(args.root,original,ARM,PARENT_EPOCH)
    recovery=dict(run=original.RUN,training=parent,epoch=PARENT_EPOCH,commit=PARENT_COMMIT,config=PARENT_CONFIG)
    training.train(args,out,build=factory,config_for=configuration,training_readout=original.readout,
        smoke_gradient_check=original.smoke_gradient,physical_detail_coupled=True,technical_resume=recovery)


def admitted_gpus():
    return [g for g in recovery_gpu_choice(allowed_gpus=(5,),maximum_processes=3) if g['free_memory_MB']>=12288]


def controller(args):return resume.controller(args,RUN,ARM,admitted_gpus,3)


def install():
    trial.RUN,trial.MODULE,trial.ARMS=RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train,trial.controller=configuration,factory,train,controller
    if len(sys.argv)>1 and sys.argv[1]=='score':
        if '--root' not in sys.argv:raise ValueError('Root required for original scorer')
        original_scorer(Path(sys.argv[sys.argv.index('--root')+1]))


if __name__=='__main__':install();trial.main()
