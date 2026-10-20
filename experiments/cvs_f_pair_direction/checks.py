"""Known geometric controls, common-offset invariance and source-role negatives."""
import copy
import numpy as np
from . import analysis as a

def smoke():
 mu=np.tile(np.eye(6)[:,None,:],(1,5,1));s=a.pair_stats(mu,1,3)
 assert abs(s['mean_direction_cosine']-1.)<1e-12 and abs(s['cross_to_self']-1.)<1e-12
 return mu

def run():
 mu=smoke();neg=0
 rng=np.random.default_rng(77);shifted=mu+rng.normal(size=(1,5,6));before=a.pair_stats(mu,1,3);after=a.pair_stats(shifted,1,3)
 assert abs(before['mean_direction_cosine']-after['mean_direction_cosine'])<1e-12 and abs(before['self_energy']-after['self_energy'])<1e-12
 orth=mu.copy()
 for r in range(5):orth[1,r]=.5*np.eye(6)[r];orth[3,r]=-.5*np.eye(6)[r]
 v=a.pair_stats(orth,1,3);assert abs(v['mean_direction_cosine'])<1e-12 and abs(v['cross_to_self'])<1e-12
 zero=mu.copy();zero[3]=zero[1];assert a.pair_stats(zero,1,3)['valid_RX_pairs']==0 and not a.analyze(zero,a.SEEDS[0])['prospective_direction_route_qualified']
 coherent=[a.analyze(mu,s) for s in a.SEEDS];incoherent=[a.analyze(orth,s) for s in a.SEEDS]
 assert not a.decision(coherent)['PASS'] and a.decision(incoherent)['PASS'];one=copy.deepcopy(incoherent);one[2]['prospective_direction_route_qualified']=False;assert not a.decision(one)['PASS']
 cases=[lambda:a.scope('U_s',a.SEEDS[0],a.CACHE_RUN),lambda:a.scope('target',a.SEEDS[0],a.CACHE_RUN),lambda:a.scope('V',1,a.CACHE_RUN),lambda:a.scope('V',a.SEEDS[0],'unknown'),lambda:a.decision(coherent[:1]),lambda:a.pair_stats(np.zeros((2,5,6)),0,1)]
 base=dict(row_id='BASE-s'+str(a.SEEDS[0]),model_seed=a.SEEDS[0],status='SOURCE_GEOMETRY_COMPLETE',source_role='V',count=27000,head_fit_updates=0,optimizer_updates=0,prototype_classifier_decisions=0,target_reads=0,truth_reads=0,U_TX_reads=0,model_state_unchanged=True,source_code_commit=a.REFERENCE_COMMIT,geometry=dict(class_names=list(a.CLASSES)))
 a.metadata(base,a.SEEDS[0])
 for key,value in [('source_role','U_s'),('target_reads',1),('U_TX_reads',1),('head_fit_updates',1),('optimizer_updates',1),('source_code_commit','wrong')]:
  bad=copy.deepcopy(base);bad[key]=value;cases.append(lambda b=bad:a.metadata(b,a.SEEDS[0]))
 for case in cases:
  try:case()
  except ValueError:neg+=1
  else:raise AssertionError('Source/provenance negative accepted')
 return dict(status='PASS',protocol_negative_cases=neg,known_same_directions_cosine1=True,known_orthogonal_directions_cosine0=True,RX_common_offset_cancels=True,degenerate_pair_not_promoted=True,all3_required=True,source_cache_only=True,new_model_forwards=0,checkpoint_loads=0,target_reads=0,head_fit_updates=0)
if __name__=='__main__':
 import json;print(json.dumps(run()))
