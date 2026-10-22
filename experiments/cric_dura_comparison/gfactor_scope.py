"""Lightweight, predeclared fairness scope; no planning/vendor imports."""
from .core_groups import SEEDS

GFACTOR='20261022-phase1-PAMF-PL-Gfactor-readouts-m51-r01'
MAIN='20261021-phase1-T1-DURA-comparison-m153-r01'
REPAIRED_CONTROLS='20261022-phase1-T1-DURA-F0-controls-binding-m6-r01'
PROTOCOL_IDS=('F0','F1','F2','chrono','ManyRx','label01','label05')+tuple(
    'm%d-%s'%(m,b) for m in (1,2,3,4,6) for b in ('natural','fixed'))


def main_source(pid,regime,seed):
    if pid=='F0' and regime!='native_cric':
        return dict(run_id=REPAIRED_CONTROLS,row_id='F0-'+regime+'-s'+str(seed))
    return dict(run_id=MAIN,row_id=pid+'-'+regime+'-s'+str(seed))
