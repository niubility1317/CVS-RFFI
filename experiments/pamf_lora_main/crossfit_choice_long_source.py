"""Train the matched LoRa direction arms with expanded U and continuous budgets."""

from dataclasses import replace

from .crossfit_choice import objective
from .crossfit_choice_source import COMPARISONS, MODES, RECIPES, evaluate, smoke
from .spectral_block_source import main


RUN = '20261124-phase1-lora-crossfit-choice-long-source-m4-r01'
LONG = {name: replace(recipe, steps=40_000, batch=24, fit_per_rx=4000,
                      validation_steps=(1800, 6000, 12000, 20000, 30000))
        for name, recipe in RECIPES.items()}


if __name__=='__main__':
    main(run=RUN, runner=__spec__.name, recipes=LONG, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=smoke, run_evaluate=evaluate)
