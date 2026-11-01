"""Scratch30 matched test of a source-clean amplitude anchor before unit projection."""
import argparse
import copy
from pathlib import Path
from . import feature_conditioned_trial as parent
from . import multiview_trial as trial
from .variance_trial import load_scale

RUN='20261101-phase1-Restore-PowerAnchor-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.power_anchor_trial'
ARM='MSAT_PowerAnchor'
TEMPLATE=parent.configuration(parent.ARM)


def configuration(arm):
    if arm!=ARM:raise ValueError('Only the fixed source amplitude-anchor candidate is allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,raw_power_anchor_weight=.1,
        raw_power_anchor_reference='same-record sourceclean complex RMS/detached',
        objective='IQNMSE+.05STFT+.1(F0outputcos+sourcevarMSE)+.1(latentcos+sourcevarMSE)+.1(rawRMS/cleanRMS-1)^2',
        matched_control='20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01/MSAT_FeatureCond',
        scientific_change='Only add .1 sourceclean raw-complex-RMS ratio anchor before unchanged unit-power output;all architecture/latent-output guides/order/optimization/budget matched')
    return c


def power_anchor_loss(aux,clean):
    raw=aux['raw_restored']
    raw_power=raw.square().sum(1).mean(1)
    reference_power=clean.detach().square().sum(1).mean(1)
    ratio=(raw_power/reference_power).sqrt();error=(ratio-1).square()
    return .1*error,dict(raw_complex_power=raw_power,raw_clean_RMS_ratio=ratio,raw_power_anchor_error=error)


def install(scale):
    parent.RUN=RUN;parent.MODULE=MODULE;parent.ARM=ARM
    parent.configuration=configuration;parent.OUTPUT_REGULARIZER=power_anchor_loss
    parent.install(scale)


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);a,_=p.parse_known_args()
    install(load_scale(a.root));trial.main()
