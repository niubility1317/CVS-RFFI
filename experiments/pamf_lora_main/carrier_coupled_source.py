"""Train and assess matched amplitude-coupled LoRa restoration."""
import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from .action_support_source import independent_score
from .antithetic_detail_source import evaluate, reserve, smoke, timing
from .carrier_coupled import CarrierCoupledRecovery, observed_frame, packet_frame, training_objective
from .channel_blind_source import FitRecipe, fit
from .complex_vector_source import publish
from .data import SOURCE
from .factorized_chirps_source import PACKETS_PER_RECEIVER
from .gated_slope_source import dispatch
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, WIDTHS


RUN = '20261124-phase1-lora-carrier-coupled-source-m4-r01'


@torch.no_grad()
def calibrate(model, tracker, source_u):
    balanced = torch.cat([rx[:PACKETS_PER_RECEIVER] for rx in source_u.chunk(len(SOURCE))])
    spectra = []
    for batch in balanced.split(20):
        measured, _, _, _ = observed_frame(tracker,batch)
        spectra.append(torch.fft.fft(measured,norm='ortho'))
    model.clean_detail.fit(torch.cat(spectra))


RECIPES = {
    name: FitRecipe(name, STEPS, partial(CarrierCoupledRecovery,WIDTHS,coupled=coupled),
                    colored_observation,CarrierCoupledRecovery.forward,initialize=calibrate)
    for name,coupled in (('independent_gain',False),('coupled_gain',True))
}
MODES = ('direct','reference','rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse','half','full')))
PAIRS = tuple((f'coupled_gain_{stage}',f'independent_gain_{stage}') for stage in ('coarse','half','full'))
PAIRS += (('coupled_gain_half','rank_parent_half'),('independent_gain_half','rank_parent_half'))


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
            choices=RECIPES if action=='fit' else (*ROUTES,'reserve') if action=='evaluate' else ('source',)
            step.add_argument('row',choices=choices)
    options=vars(parser.parse_args())
    action=options.pop('action')
    if action=='publish':
        return publish(**options,run=RUN,runner=__spec__.name)
    rt=read(options['runtime'])
    if action=='dispatch':
        return dispatch(rt,fit_arms=RECIPES)
    if action=='score':
        return independent_score(rt,modes=MODES,comparisons=PAIRS)
    activate(rt)
    if action=='smoke':
        calculators=dict.fromkeys(RECIPES,training_objective)
        return smoke(rt,recipes=RECIPES,modes=MODES,calculations=calculators,
                     reframe=packet_frame,calibrate_fn=calibrate)
    if action=='fit':
        return fit(rt,recipe=RECIPES[options['row']],calculate=training_objective,receiver_batches=True)
    if action=='timing':
        return timing(rt,recipes=RECIPES,reframe=packet_frame)
    if options['row']=='reserve':
        return reserve(rt,recipes=RECIPES,modes=MODES,reframe=packet_frame)
    return evaluate(rt,options['row'],recipes=RECIPES,modes=MODES,reframe=packet_frame)


if __name__=='__main__':
    main()
