"""Expose a fixed independent source-trained feature projector for Phase1."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_projector_aligned.train_source import make_model
SOURCE='20261103-phase1-projector-aligned-source-s2026092701-r01'


def main():
 old=base.DynamicVolterra,base.SOURCE,base.SCHEMA,base.LOG_PREFIX,base.USES_DURA
 base.DynamicVolterra,base.SOURCE,base.SCHEMA,base.LOG_PREFIX,base.USES_DURA=(
     make_model,SOURCE,'phase1_projector_aligned_v1','PROJECTOR_ALIGNED',False)
 try:base.main()
 finally:base.DynamicVolterra,base.SOURCE,base.SCHEMA,base.LOG_PREFIX,base.USES_DURA=old


if __name__=='__main__':main()
