"""Train the standalone iterative time/frequency restorer on paired source IQ."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq
from experiments.phase1_volterra_aligned.objective import source_loss


def main():
    old=base.DynamicVolterra,base.SCHEMA,base.LOG_PREFIX,base.source_loss
    base.DynamicVolterra,base.SCHEMA,base.LOG_PREFIX,base.source_loss=(
        IterativeTimeFreq,'phase1_iterative_timefreq_v1','ITERATIVE_TIMEFREQ',source_loss)
    try:base.main()
    finally:base.DynamicVolterra,base.SCHEMA,base.LOG_PREFIX,base.source_loss=old


if __name__=='__main__':main()
