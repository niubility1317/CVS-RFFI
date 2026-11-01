"""One source-only scratch30 ablation: full feature gradients versus VarF cap."""
import argparse
import copy
from pathlib import Path
import torch
from . import multiview_trial as trial
from .variance_trial import configuration as variance_configuration,load_scale,recovery_gpu_choice

RUN='20261101-phase1-Restore-MSAT-FullFGrad-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.full_feature_trial'
ARM='MSAT_FullFGrad'
TEMPLATE=variance_configuration('MSAT_VarF')


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed full-feature-gradient candidate allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,objective='IQNMSE+.05STFT+.1(F0cos+source-variance-standardized-featureMSE)/no-perrecord-IQ-gradient-cap',
        per_record_feature_input_gradient_cap=False,global_parameter_gradient_clip=1.,
        matched_control='20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r03/MSAT_VarF',
        scientific_change='Remove only per-record feature input-gradient cap;keep objective weights/globalclip/model/data/budget')
    return c


def full_feature_guide(reader,restored,clean,target_features,scale):
    if target_features.shape!=(len(restored),160) or scale.shape!=(160,):
        raise ValueError('Matched source fused160 features and source scale required')
    actual=reader.features(restored);target=target_features.detach()
    cosine=1-torch.nn.functional.cosine_similarity(actual,target,dim=1,eps=1e-8)
    standardized=((actual-target).square()/scale).mean(1)
    return .1*(cosine+standardized),dict(feature_error=cosine,feature_standardized_error=standardized,
        weight=torch.ones_like(cosine))


def install(scale):
    by_device={}
    def guide(reader,restored,clean,target):
        if restored.device not in by_device:by_device[restored.device]=scale.to(restored.device)
        return full_feature_guide(reader,restored,clean,target,by_device[restored.device])
    trial.RUN=RUN;trial.MODULE=MODULE;trial.ARMS=(ARM,)
    trial.configuration=configuration;trial.feature_guide=guide;trial.gpu_choice=recovery_gpu_choice


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);args,_=p.parse_known_args()
    install(load_scale(args.root));trial.main()
