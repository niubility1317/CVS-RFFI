"""Load fixed wider-range physical fusion for Phase1 inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_dura_physical_wide.model import WidePhysicalFusion as DynamicVolterra
from experiments.phase1_dura_physical_fusion.pipeline import load_components,predict_batch
SOURCE='20261103-phase1-dura-physical-wide-source-s2026092701-r01'

def main():
 names=('DynamicVolterra','SOURCE','SCHEMA','LOG_PREFIX','USES_DURA','load_components','predict_batch')
 old=tuple(getattr(base,n) for n in names)
 values=(DynamicVolterra,SOURCE,'phase1_dura_physical_wide_v1','DURA_PHYSICAL_WIDE',True,load_components,predict_batch)
 for name,value in zip(names,values):setattr(base,name,value)
 try:base.main()
 finally:
  for name,value in zip(names,old):setattr(base,name,value)

if __name__=='__main__':main()
