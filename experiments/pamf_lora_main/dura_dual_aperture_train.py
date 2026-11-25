"""Fit shared LoRa restoration weights on narrow and W24 observations in one loop."""

from .dura_apertures import RECIPE, ApertureTrackers, objective
from .dura_w24_train import main


RUN = '20261125-phase1-lora-dura-dual-aperture-s2026112991-r01'


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipe=RECIPE, calculate=objective, make_tracker=ApertureTrackers,
         description=__doc__)
