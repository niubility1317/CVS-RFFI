"""Run matched long-budget LoRa complex geometry and waveform controls."""

from functools import partial

from .complex_geometry import ComplexGeometry, objective
from .local_phase import generic_phase_observation
from .spectral_block_source import FitRecipe, main


RUN = '20261124-phase1-lora-complex-geometry-source-m4-r01'
ARMS = ('waveform','geometry')
READOUTS = ('coarse','full','half')
RECIPES = {arm: FitRecipe(arm,40_000,partial(ComplexGeometry,geometry=arm=='geometry'),
                          generic_phase_observation,None,batch=24,fit_per_rx=4000,
                          validation_steps=(1800,6000,12000,20000,30000)) for arm in ARMS}
MODES = ('direct','reference',*(f'{arm}_{kind}' for arm in ARMS for kind in READOUTS))
COMPARISONS = (*(('geometry_'+kind,'waveform_'+kind) for kind in READOUTS),
               ('geometry_full','geometry_coarse'),('geometry_half','direct'),
               *((name,'reference') for name in MODES[2:]))


if __name__=='__main__':
    main(run=RUN,runner=__spec__.name,recipes=RECIPES,modes=MODES,comparisons=COMPARISONS,calculate=objective)
