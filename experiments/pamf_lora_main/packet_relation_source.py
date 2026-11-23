"""Run matched LoRa waveform and cross-packet relation objectives."""

from functools import partial

from .local_phase import generic_phase_observation
from .packet_relation import PacketRelation, objective
from .spectral_block_source import FitRecipe, main, restore


RUN = '20261124-phase1-lora-packet-relation-source-m4-r01'
ARMS = ('waveform', 'relation')
READOUTS = ('coarse', 'full', 'half')
RECIPES = {
    arm: FitRecipe(arm, 40_000, partial(PacketRelation, preserve_relations=arm=='relation'),
                   generic_phase_observation, restore, coarse_first=True, batch=24, fit_per_rx=4000,
                   validation_steps=(1800, 6000, 12000, 20000, 30000))
    for arm in ARMS
}
MODES = ('direct', 'reference', *(f'{arm}_{kind}' for arm in ARMS for kind in READOUTS))
COMPARISONS = (*(('relation_' + kind, 'waveform_' + kind) for kind in READOUTS),
               ('relation_full', 'relation_coarse'), ('relation_half', 'direct'),
               *((name, 'reference') for name in MODES[2:]))


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective)
