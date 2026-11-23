"""Matched-dose LoRa recovery with observation-derived repeat protection."""

from functools import partial

from .spectral_block_recovery import SpectralBlockRecovery, varied_observation
from .spectral_block_source import FitRecipe, main, restore


RUN = '20261123-phase1-lora-repeat-protection-source-m4-r01'
ARMS = ('scalar','protected')
MODES = ('direct','reference',*(f'{arm}_{mode}' for arm in ARMS for mode in ('coarse','full','half')))
RECIPES = {arm: FitRecipe(arm,1800,partial(SpectralBlockRecovery,masked=True,protection=protect),
                         varied_observation,restore)
           for arm, protect in zip(ARMS,(False,True),strict=True)}
COMPARISONS = (('protected_coarse','scalar_coarse'), ('protected_full','scalar_full'),
               ('protected_half','scalar_half'), ('protected_full','protected_coarse'),
               ('protected_half','direct'))


if __name__=='__main__':
    main(run=RUN,runner=__spec__.name,recipes=RECIPES,modes=MODES,comparisons=COMPARISONS)
