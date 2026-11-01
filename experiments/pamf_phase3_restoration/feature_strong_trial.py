"""One matched scratch30 test of unit-weight continuous source feature regression."""
import argparse
import copy
from pathlib import Path
from . import feature_conditioned_trial as parent
from . import multiview_trial as trial
from .variance_trial import load_scale

RUN='20261101-phase1-Restore-FeatureStrong-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.feature_strong_trial'
ARM='MSAT_FeatureStrong'
TEMPLATE=parent.configuration(parent.ARM)
BASE_LATENT=parent.latent_feature_loss


def configuration(arm):
    if arm!=ARM:raise ValueError('Only the fixed unit-weight source feature candidate is allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,latent_feature_weight=1.,
        objective='IQNMSE+.05STFT+.1(F0outputcos+sourcevarMSE)+1.0(predicted-continuous-feature-cos+sourcevarMSE)',
        matched_control='20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01/MSAT_FeatureCond',
        scientific_change='Only continuous latent regression weight .1 to1;all architecture/output guidance/unitpower/source order/optimization/budget matched')
    return c


def strong_latent_feature_loss(predicted,target,scale):
    _,terms=BASE_LATENT(predicted,target,scale)
    return terms['latent_cosine_error']+terms['latent_standardized_error'],terms


def install(scale):
    parent.RUN=RUN;parent.MODULE=MODULE;parent.ARM=ARM
    parent.configuration=configuration;parent.latent_feature_loss=strong_latent_feature_loss
    parent.install(scale)


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);a,_=p.parse_known_args()
    install(load_scale(a.root));trial.main()
