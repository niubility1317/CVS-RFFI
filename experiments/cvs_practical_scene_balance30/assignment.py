"""Exact64-position training scene allocation;only a private NumPy RNG."""
import numpy as np
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
NAMESPACE='scene_balance_assignment_v1'
def allocation(cumulative_step,namespace,train_seed,stable_seed,*,count=64):
 if type(cumulative_step) is not int or cumulative_step not in range(4441,6661) or count!=64 or namespace!=f'FcrossRX/train/e{(cumulative_step-1)//222+1}/step{cumulative_step}/L_S':raise ValueError('Only original fixedE21..30 As64 training namespace')
 extra=(cumulative_step-1)%3;counts=[21,21,21];counts[extra]+=1
 seed=int(stable_seed(train_seed,namespace,NAMESPACE));permutation=np.random.default_rng(seed).permutation(64)
 ordered=np.repeat(np.arange(3,dtype=np.int64),counts);indices=ordered[permutation]
 return dict(scene_indices=indices.tolist(),planned_counts=counts,extra_scene_index=extra,assignment_seed=seed,permutation=permutation.tolist(),assignment_permutations=1)
