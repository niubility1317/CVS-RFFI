"""Source replay and ground prediction for the fixed expanded-source model."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq

SOURCE='20261103-phase1-iterative-large-source-s2026092701-r01'


def main():
    names=('DynamicVolterra','SOURCE','SCHEMA','LOG_PREFIX','FIT_SIZE')
    old=tuple(getattr(base,k) for k in names)
    values=(IterativeTimeFreq,SOURCE,'phase1_iterative_large_v1','ITERATIVE_LARGE',32768)
    for k,v in zip(names,values):setattr(base,k,v)
    try:base.main()
    finally:
        for k,v in zip(names,old):setattr(base,k,v)


if __name__=='__main__':main()
