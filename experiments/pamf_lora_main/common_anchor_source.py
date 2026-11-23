"""Fit same-budget LoRa observation solves with and without a fixed chirp mean."""

from functools import partial

from .channel_blind_source import FitRecipe
from .common_anchor import CommonAnchor, objective
from .joint_spectrum_source import evaluate, smoke
from .reference_cancel import generic_observation
from .spectral_block_source import main


RUN = '20261124-phase1-lora-common-anchor-source-m4-r01'
RECIPES = {name:FitRecipe(name,40_000,partial(CommonAnchor,anchor_common=name=='anchored'),
                          generic_observation,None,coarse_first=True,batch=24,fit_per_rx=4000,
                          validation_steps=(1800,6000,12000,20000,30000))
           for name in ('free','anchored')}
STAGES = ('coarse','full','half')
MODES = ('direct','reference',*(f'{arm}_{mode}' for arm in RECIPES for mode in STAGES),
         'anchored_matched','free_matched')
FIELDS = ('active','outer_common_power','estimated_effective_variance','tracker_coherence',
          'relative_action_magnitude','full_edit_rms_ratio','matched_edit_rms_ratio')
COMPARISONS = (('anchored_matched','free_matched'),('anchored_full','anchored_coarse'),
               *((f'anchored_{mode}',f'free_{mode}') for mode in STAGES),
               *((mode,base) for mode in MODES[2:] for base in ('reference','direct')))


if __name__=='__main__':
    main(run=RUN,runner=__spec__.name,recipes=RECIPES,modes=MODES,comparisons=COMPARISONS,
         calculate=objective,run_smoke=partial(smoke,switch='anchor_common'),
         run_evaluate=partial(evaluate,switch='anchor_common',fields=FIELDS))
