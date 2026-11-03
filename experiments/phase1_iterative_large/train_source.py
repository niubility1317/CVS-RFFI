"""Standalone iterative recovery trained on nested expanded source fit."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq
from experiments.phase1_volterra_aligned.objective import source_loss
from experiments.phase1_iterative_large.sampling import select_large_fit


def main():
    names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_loss','FIT_SIZE','select_fit')
    old=tuple(getattr(base,k) for k in names)
    values=(IterativeTimeFreq,'phase1_iterative_large_v1','ITERATIVE_LARGE',source_loss,32768,select_large_fit)
    for k,v in zip(names,values):setattr(base,k,v)
    try:base.main()
    finally:
        for k,v in zip(names,old):setattr(base,k,v)


if __name__=='__main__':main()
