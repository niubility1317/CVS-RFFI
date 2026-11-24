"""Fit LoRa relative carrier controls with a frozen power-only comparator."""
import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from .action_support_source import independent_score
from .antithetic_detail_source import evaluate, reserve, smoke, timing
from .carrier_coupled import packet_frame
from .carrier_coupled_source import RECIPES as LEGACY, RUN as LEGACY_RUN, calibrate
from .channel_blind_source import FitRecipe, fit
from .complex_vector_source import publish
from .gated_slope_source import dispatch
from .relative_carrier import RelativeCarrierRecovery, mix, training_objective
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, WIDTHS


RUN = '20261124-phase1-lora-relative-carrier-source-m5-r01'
TRAIN_ARMS = {
    name: FitRecipe(name, STEPS, partial(RelativeCarrierRecovery,WIDTHS,coupled=coupled),
                    colored_observation,RelativeCarrierRecovery.forward,initialize=calibrate)
    for name,coupled in (('relative_independent',False),('relative_coupled',True))
}
RECIPES = {**TRAIN_ARMS, 'coupled_gain': LEGACY['coupled_gain']}
REUSED = {'coupled_gain': LEGACY_RUN}
MODES = ('direct','reference','rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse','half','full')))
PAIRS = tuple((f'relative_coupled_{stage}',f'relative_independent_{stage}') for stage in ('coarse','half','full'))
PAIRS += (('relative_coupled_half','coupled_gain_half'),('relative_coupled_half','rank_parent_half'),
          ('relative_independent_half','rank_parent_half'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    pub = commands.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for action in ('dispatch','smoke','fit','timing','evaluate','score'):
        step=commands.add_parser(action)
        step.add_argument('runtime',type=Path)
        if action!='dispatch':
            choices=TRAIN_ARMS if action=='fit' else (*ROUTES,'reserve') if action=='evaluate' else ('source',)
            step.add_argument('row',choices=choices)
    options=vars(parser.parse_args())
    action=options.pop('action')
    if action=='publish':
        return publish(**options,run=RUN,runner=__spec__.name)
    rt=read(options['runtime'])
    if action=='dispatch':
        return dispatch(rt,fit_arms=TRAIN_ARMS)
    if action=='score':
        return independent_score(rt,modes=MODES,comparisons=PAIRS)
    activate(rt)
    common = dict(recipes=RECIPES,reused=REUSED,reframe=packet_frame,mixer=mix)
    if action=='smoke':
        calculators=dict.fromkeys(TRAIN_ARMS,training_objective)
        return smoke(rt,modes=MODES,split_risk=(False,False,False),calculations=calculators,
                     calibrate_fn=calibrate,**common)
    if action=='fit':
        return fit(rt,recipe=TRAIN_ARMS[options['row']],calculate=training_objective,receiver_batches=True)
    if action=='timing':
        return timing(rt,**common)
    if options['row']=='reserve':
        return reserve(rt,modes=MODES,**common)
    return evaluate(rt,options['row'],modes=MODES,**common)


if __name__=='__main__':
    main()
