"""Train DURA correction against hard paired source-channel views."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_context.model import ContextRecovery as DynamicVolterra
from experiments.phase1_dura_viewcvar.objective import viewcvar_source_objective,COUNTS
from experiments.phase1_dura_volterra.pipeline import load_components,input_noise,predict_batch

def main():
    COUNTS[:]=[0,0,0,0]
    names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','PAIRED_VIEWS','USES_DURA',
           'load_components','input_noise','predict_batch','extra_frozen_components')
    old=tuple(getattr(base,n) for n in names)
    values=(DynamicVolterra,'phase1_dura_viewcvar_v1','DURA_VIEWCVAR',viewcvar_source_objective,
            True,True,load_components,input_noise,predict_batch,lambda dura,eye:(dura,))
    for name,value in zip(names,values):setattr(base,name,value)
    try:base.main()
    finally:
        for name,value in zip(names,old):setattr(base,name,value)
        print('DURA_VIEWCVAR_COUNTS',*COUNTS,flush=True)

if __name__=='__main__':main()
