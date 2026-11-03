"""Source-only clean-reader-guided fusion of DURA and physical posterior."""
import torch
from torch.nn import functional as F
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_blendgate.model import align_to_anchor
from experiments.phase1_dura_physical_fusion.model import PhysicalFusion
from experiments.phase1_dura_physical_fusion.pipeline import load_components,input_noise,predict_batch
from experiments.phase1_dura_safe_memory.objective import protected_source_objective


def rescue_mask(clean_logits,anchor_logits,physical_logits):
 if clean_logits.ndim!=2 or clean_logits.shape[1]!=6 or anchor_logits.shape!=clean_logits.shape or physical_logits.shape!=clean_logits.shape:
  raise ValueError('Matching six-class frozen reader logits required')
 confidence,target=clean_logits.softmax(1).max(1)
 return (confidence>=.95)&(anchor_logits.argmax(1)!=target)&(physical_logits.argmax(1)==target),target


def fusion_training_objective(reader,output,packed,clean,clean_features,groups,received,log_variance):
 n=len(output)
 if packed.shape!=(n,4,256) or groups.shape!=(n,):raise ValueError('One same-record physical and DURA waveform required')
 anchor=packed[:,:2]
 protected,terms=protected_source_objective(reader,output,anchor,clean,clean_features,groups)
 with torch.no_grad():
  physical=align_to_anchor(packed[:,2:],anchor,preserve_edges=True)
  clean_logits=reader.classify_features(clean_features)
  anchor_logits=reader.classify_features(reader.features(anchor))
  physical_logits=reader.classify_features(reader.features(physical))
  mask,target=rescue_mask(clean_logits,anchor_logits,physical_logits)
  anchor_ce=F.cross_entropy(anchor_logits,target,reduction='none')
 output_logits=reader.classify_features(reader.features(output))
 gain=F.relu(.2+F.cross_entropy(output_logits,target,reduction='none')-anchor_ce)
 per_group=[gain[mask&(groups==g)].mean() for g in groups.unique() if (mask&(groups==g)).any()]
 rescue=torch.stack(per_group).mean() if per_group else output.sum()*0.
 if not torch.isfinite(rescue):raise FloatingPointError('Finite source-only rescue loss required')
 return protected+.3*rescue,dict(terms,physical_rescue=rescue,rescue_count=mask.sum())


def main():
 names=('DynamicVolterra','SCHEMA','LOG_PREFIX','training_source_objective','USES_DURA',
        'load_components','input_noise','predict_batch','extra_frozen_components')
 old=tuple(getattr(base,n) for n in names)
 new=(PhysicalFusion,'phase1_dura_physical_fusion_v1','DURA_PHYSICAL_FUSION',fusion_training_objective,True,
      load_components,input_noise,predict_batch,lambda dura,physical:(dura,physical))
 for name,value in zip(names,new):setattr(base,name,value)
 try:base.main()
 finally:
  for name,value in zip(names,old):setattr(base,name,value)


if __name__=='__main__':main()
