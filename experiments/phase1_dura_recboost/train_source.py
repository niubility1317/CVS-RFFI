"""Train larger DURA residuals with source-only pseudo-class guidance."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_recboost.model import WideContextRecovery as DynamicVolterra
from experiments.phase1_dura_recboost.objective import recognition_source_objective
from experiments.phase1_dura_volterra.pipeline import load_components,input_noise,predict_batch

def main():
    names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA',
           'load_components','input_noise','predict_batch','extra_frozen_components')
    old=tuple(getattr(base,n) for n in names)
    values=(DynamicVolterra,'phase1_dura_recboost_v1','DURA_RECBOOST',recognition_source_objective,
            True,load_components,input_noise,predict_batch,lambda dura,eye:(dura,))
    for name,value in zip(names,values):setattr(base,name,value)
    try:base.main()
    finally:
        for name,value in zip(names,old):setattr(base,name,value)

if __name__=='__main__':main()
