"""Source training of same-window frozen-reader feature feedback."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_feedback.model import FeedbackRecovery as DynamicVolterra
from experiments.phase1_dura_feedback.pipeline import load_components,input_noise,predict_batch
from experiments.phase1_volterra_aligned.objective import source_loss

def feedback_source_objective(reader,output,packed,clean,clean_features,groups):
    if packed.shape!=(len(output),3,256):
        raise ValueError('Same-record packed DURA and reader condition required')
    return source_loss(reader,output,clean,clean_features,groups)

def main():
    names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA',
           'load_components','input_noise','predict_batch','extra_frozen_components')
    old=tuple(getattr(base,name) for name in names)
    values=(DynamicVolterra,'phase1_dura_feedback_v1','DURA_FEEDBACK',feedback_source_objective,
            True,load_components,input_noise,predict_batch,lambda dura,condition_reader:(dura,condition_reader))
    for name,value in zip(names,values):setattr(base,name,value)
    try:base.main()
    finally:
        for name,value in zip(names,old):setattr(base,name,value)

if __name__=='__main__':main()
