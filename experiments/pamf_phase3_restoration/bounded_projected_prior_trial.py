"""Scratch30 projected-prior candidate with an input-relative correction limit."""
from functools import partial

from . import covariance_detail_trial as training
from . import multiview_trial as trial
from . import projected_prior_detail_trial as original
from .bounded_projected_prior import BoundedProjectedPriorDetail,MAX_RELATIVE_CORRECTION
from .projected_prior_detail import shrinkage_training_readout
from .variance_trial import recovery_gpu_choice

RUN='20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.bounded_projected_prior_trial'
ARM='TF_BoundedProjected'


def configuration(arm):
    if arm!=ARM:raise ValueError('Only the fixed bounded projected prior candidate is allowed')
    cfg=original.configuration(original.ARM)
    cfg.update(run_id=RUN,arm=ARM,
        learned_detail='Original conditional prior detail and posterior projection; limit correction L2 to twice current received IQ L2.',
        scientific_change='Add fixed input-relative L2 limit after prior projection during both source training and deployment; all G weights initialized from scratch.',
        max_relative_correction=MAX_RELATIVE_CORRECTION,matched_control=original.RUN+'/'+original.ARM,
        evidence='20261108-diagnostic-Restore-ProjectedSpikeTrace-sourceLU-s2026092701-r01; source-only internal activation spike.',
        original_parent_weights_loaded=False)
    return cfg


def factory(arm):
    return BoundedProjectedPriorDetail(configuration(arm)['model_seed'])


def training_readout(model,reader,received,clean,features,scale,objective,*,source_indices=None,epoch=0):
    restored,aux,loss,terms=shrinkage_training_readout(model,reader,received,clean,features,scale,objective,
        source_indices=source_indices,epoch=epoch)
    energy=clean.square().mean((1,2)).clamp_min(1e-4)
    terms['raw_projected_update_nmse']=aux['unbounded_projected_update'].square().mean((1,2))/energy
    terms['correction_scale']=aux['correction_scale']
    terms['correction_limited_fraction']=(aux['correction_scale']<1.).to(loss.dtype)
    return restored,aux,loss,terms


def train(args,out):
    training.train(args,out,build=factory,config_for=configuration,training_readout=training_readout)


def install():
    trial.RUN,trial.MODULE,trial.ARMS=RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train=configuration,factory,train
    trial.gpu_choice=partial(recovery_gpu_choice,allowed_gpus=(2,3,4),maximum_processes=3)


if __name__=='__main__':install();trial.main()
