"""Score fixed differential timing with the registered WiSig Main split reader."""

from .wisig_differential_timing import ARMS, observations, readouts
from .wisig_smooth_preamble_source import main


RUN = '20261125-phase1-wisig-differential-timing-source-s2026092701-r01'
READOUTS = ('direct', *ARMS)
COMPARISONS = (('differential', 'smooth'), ('joint', 'smooth'), ('differential', 'direct'),
               ('differential', 'r69'), ('joint', 'r69'))


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, names=READOUTS, comparisons=COMPARISONS, analyze=observations,
         render=readouts, check='differential')
