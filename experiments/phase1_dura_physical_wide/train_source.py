"""Train a wider-range physical fusion under the matched source protocol."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_physical_wide.model import WidePhysicalFusion
from experiments.phase1_dura_physical_fusion.train_source import fusion_training_objective
from experiments.phase1_dura_physical_fusion.pipeline import load_components,input_noise,predict_batch


def main():
 names=('DynamicVolterra','SCHEMA','LOG_PREFIX','training_source_objective','USES_DURA',
        'load_components','input_noise','predict_batch','extra_frozen_components')
 old=tuple(getattr(base,n) for n in names)
 new=(WidePhysicalFusion,'phase1_dura_physical_wide_v1','DURA_PHYSICAL_WIDE',fusion_training_objective,True,
      load_components,input_noise,predict_batch,lambda dura,physical:(dura,physical))
 for name,value in zip(names,new):setattr(base,name,value)
 try:base.main()
 finally:
  for name,value in zip(names,old):setattr(base,name,value)


if __name__=='__main__':main()
