"""Train wide DURA/physical fusion with an additional source-only local teacher."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_blendgate.model import align_to_anchor
from experiments.phase1_dura_physical_wide.model import WidePhysicalFusion
from experiments.phase1_dura_physical_fusion.train_source import fusion_training_objective
from experiments.phase1_dura_physical_fusion.pipeline import load_components, input_noise, predict_batch
from experiments.phase1_dura_physical_local_teacher.objective import local_teacher_loss


def combined_objective(reader, output, packed, clean, clean_features, groups, received, log_variance):
    protected, terms=fusion_training_objective(reader,output,packed,clean,clean_features,groups,received,log_variance)
    anchor=packed[:,:2]
    physical=align_to_anchor(packed[:,2:],anchor,preserve_edges=True)
    aligned_clean=align_to_anchor(clean,anchor,preserve_edges=True)
    local=local_teacher_loss(output,anchor,physical,aligned_clean,groups)
    return protected+.25*local,dict(terms,local_teacher=local)


def main():
    names=('DynamicVolterra','SCHEMA','LOG_PREFIX','training_source_objective','USES_DURA',
           'load_components','input_noise','predict_batch','extra_frozen_components')
    old=tuple(getattr(base,n) for n in names)
    new=(WidePhysicalFusion,'phase1_dura_physical_local_teacher_v1','DURA_PHYSICAL_LOCAL_TEACHER',
         combined_objective,True,load_components,input_noise,predict_batch,lambda dura,physical:(dura,physical))
    for name,value in zip(names,new):setattr(base,name,value)
    try:base.main()
    finally:
        for name,value in zip(names,old):setattr(base,name,value)


if __name__=='__main__':main()
