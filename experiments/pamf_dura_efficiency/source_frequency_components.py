"""Separate strong-quartile phase and magnitude on the original source panel."""
from . import source_frequency_oracle as base

RUN='20261031-diagnostic-DURA-source-frequency-components-s2026092701-r01'
CASES=('phase_Q1','magnitude_Q1','phase_Q2','magnitude_Q2')


def configure():
    base.RUN=RUN
    base.CASES=CASES


if __name__=='__main__':
    configure()
    base.main()
