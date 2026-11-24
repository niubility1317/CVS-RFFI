"""Compare independently tracked LoRa shape supervision against the frozen matched fit."""
import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from .action_support_source import independent_score
from .antithetic_detail_source import evaluate, reserve, smoke, timing
from .carrier_coupled import packet_frame
from .carrier_coupled_source import calibrate
from .channel_blind_source import fit
from .complex_vector_source import publish
from .gated_slope_source import dispatch
from .relative_carrier import mix, training_objective
from .relative_carrier_source import TRAIN_ARMS, RUN as MATCHED_RUN
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-independent-shape-source-m3-r01'
NEW_ARM = 'own_shape'
CONTROL = TRAIN_ARMS['relative_coupled']
RECIPES = {CONTROL.name:CONTROL,NEW_ARM:replace(CONTROL,name=NEW_ARM)}
REUSED = {CONTROL.name:MATCHED_RUN}
CALCULATE = partial(training_objective,source_shape=True)
MODES = ('direct','reference','rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse','half','full')))
PAIRS = tuple((f'{NEW_ARM}_{stage}',f'{CONTROL.name}_{stage}') for stage in ('coarse','half','full'))
PAIRS += (('own_shape_half','rank_parent_half'),('relative_coupled_half','rank_parent_half'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest='action',required=True)
    pub = steps.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for action in ('dispatch','smoke','fit','timing','evaluate','score'):
        step=steps.add_parser(action)
        step.add_argument('runtime',type=Path)
        if action!='dispatch':
            choices=(NEW_ARM,) if action=='fit' else (*ROUTES,'reserve') if action=='evaluate' else ('source',)
            step.add_argument('row',choices=choices)
    options=vars(parser.parse_args())
    action=options.pop('action')
    if action=='publish':
        return publish(**options,run=RUN,runner=__spec__.name)
    rt=read(options['runtime'])
    if action=='dispatch':
        return dispatch(rt,fit_arms=(NEW_ARM,))
    if action=='score':
        return independent_score(rt,modes=MODES,comparisons=PAIRS)
    activate(rt)
    common = dict(recipes=RECIPES,reused=REUSED,reframe=packet_frame,mixer=mix)
    if action=='smoke':
        return smoke(rt,modes=MODES,split_risk=(False,False),calculations={NEW_ARM:CALCULATE},
                     calibrate_fn=calibrate,**common)
    if action=='fit':
        return fit(rt,recipe=RECIPES[NEW_ARM],calculate=CALCULATE,receiver_batches=True)
    if action=='timing':
        return timing(rt,**common)
    if options['row']=='reserve':
        return reserve(rt,modes=MODES,**common)
    return evaluate(rt,options['row'],modes=MODES,**common)


if __name__=='__main__':
    main()
