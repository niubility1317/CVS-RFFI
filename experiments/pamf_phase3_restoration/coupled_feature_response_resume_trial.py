"""Continue original coupled feature-response training from resource-stopped E02."""
import sys
from pathlib import Path

from . import coupled_feature_response_trial as original
from . import coupled_resume_support as resume
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .fixed_prediction_score_recovery import original_scorer
from .variance_trial import recovery_gpu_choice

RUN='20261108-phase1-Restore-CoupledFeatureResponse-E30-s2026092701-r02'
MODULE='experiments.pamf_phase3_restoration.coupled_feature_response_resume_trial'
ARM=original.ARM
PARENT_COMMIT='902e3b9bd9d9de5c480b10ea2578f21c3671c82a'
PARENT_EPOCH=2
PARENT_CONFIG=original.configuration(ARM)


def configuration(arm):
    if arm!=ARM:raise ValueError('Original coupled response arm required')
    cfg=dict(PARENT_CONFIG)
    cfg.update(run_id=RUN,technical_recovery_parent=original.RUN,technical_recovery_commit=PARENT_COMMIT,
        technical_recovery_epoch=PARENT_EPOCH,resource_policy='GPU6 only, user-approved max5 total compute processes; >=12GiB free.',
        scientific_change='Exact original coupled IQ/F0 response objective and source schedule; technical E02 continuation only.')
    return cfg


def factory(arm):
    configuration(arm);return original.factory(arm)


def train(args,out):
    parent=resume.parent_training(args.root,original,ARM,PARENT_EPOCH)
    recovery=dict(run=original.RUN,training=parent,epoch=PARENT_EPOCH,commit=PARENT_COMMIT,config=PARENT_CONFIG)
    training.train(args,out,build=factory,config_for=configuration,training_readout=original.response_readout,
        smoke_gradient_check=original.smoke_gradient,training_trace=True,training_posterior_multiplier=2,
        training_F0_multiplier=3,source_perturbed_F0_multiplier=1,physical_detail_coupled=True,technical_resume=recovery)


def admitted_gpus():
    return [g for g in recovery_gpu_choice(allowed_gpus=(6,),maximum_processes=5) if g['free_memory_MB']>=12288]


def controller(args):return resume.controller(args,RUN,ARM,admitted_gpus,5)


def install():
    trial.RUN,trial.MODULE,trial.ARMS=RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train,trial.controller=configuration,factory,train,controller
    if len(sys.argv)>1 and sys.argv[1]=='score':
        if '--root' not in sys.argv:raise ValueError('Root required for original scorer')
        original_scorer(Path(sys.argv[sys.argv.index('--root')+1]))


if __name__=='__main__':install();trial.main()
