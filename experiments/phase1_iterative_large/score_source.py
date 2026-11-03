"""Truth-last scoring for the fixed large-source model."""
from experiments.phase1_dynamic_volterra import score_source as base


def main():
    old=base.SCHEMA,base.FIT_SIZE
    base.SCHEMA,base.FIT_SIZE='phase1_iterative_large_v1',32768
    try:base.main()
    finally:base.SCHEMA,base.FIT_SIZE=old


if __name__=='__main__':main()
