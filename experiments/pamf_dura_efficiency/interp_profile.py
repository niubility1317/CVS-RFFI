"""Zero-update matched source benchmark for fixed-grid decoder interpolation."""
from . import gpu_profile as base
from .linear2x import install_switch

RUN='20261030-diagnostic-DURA-interpolation-s2026092701-r01'
MODULE='experiments.pamf_dura_efficiency.interp_profile'

if __name__=='__main__':
    base.RUN=RUN
    base.MODULE=MODULE
    base.CASES=('native_linear','slice_linear2x')
    base.SWITCH_FACTORY=install_switch
    base.METRIC_MODE='individual'
    base.main()
