import json,copy
import numpy as np
import torch
from . import design as d,measure as f

def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError,KeyError):return
    raise AssertionError('Illegal diagnostic accepted')

def main():
    torch.set_num_threads(2)
    fit=np.array([(y,rx,day) for y in range(6) for rx in (3,4,6,8) for day in (1,2,3) for _ in range(2)],dtype=np.int64)
    held=np.array([(y,1,day) for y in range(6) for day in (1,2,3) for _ in range(70)],dtype=np.int64)
    labels=np.r_[fit[:,0],held[:,0]];z=np.eye(160,dtype=np.float64)[labels]
    args=[fit[:,0],fit[:,1],fit[:,2],held[:,0],held[:,1],held[:,2]]
    r=f.relations(z,*args);g=r['groups'][0]
    assert g['same_pairs']==10080 and g['other_pairs']==50400 and g['triplets']==3528000 and g['strict_wins']==3528000 and g['ties']==0 and g['relation_rate']==1 and g['mean_distance_gap']==2
    assert all(x['anchors']==2 for x in r['groups'] if x['group']=='cell')
    same=f.relations(np.ones_like(z),*args)['groups'][0]
    assert same['strict_wins']==0 and same['ties']==3528000 and same['relation_rate']==.5 and same['mean_distance_gap']==0
    swapped=z.copy();swapped[:144]=np.eye(160)[(fit[:,0]+1)%6]
    reversed=f.relations(swapped,*args)['groups'][0]
    assert reversed['strict_wins']==0 and reversed['relation_rate']==.4
    perm=np.array([2,5,0,3,1,4]);other_args=[perm[args[0]],args[1],args[2],perm[args[3]],args[4],args[5]]
    permuted=f.relations(z[:,::-1].copy()*np.linspace(.5,2.,1404)[:,None],*other_args)['groups'][0]
    assert permuted==g
    negative=0
    for bad in (np.zeros_like(z),z[:,:159],z[:-1],z.astype(np.int64),z*float('nan')):
        reject(lambda:f.relations(bad,*args));negative+=1
    for i,value in ((0,args[0]+1),(1,np.ones(144,dtype=np.int64)),(5,np.zeros(1260,dtype=np.int64)),(3,args[3].astype(np.float64))):
        bad=list(args);bad[i]=value;reject(lambda:f.relations(z,*bad));negative+=1
    for key,value in (('target_access',True),('optimizer_steps',1),('head_fit_updates',1),('model_conditions',3)):
        bad=copy.deepcopy(d.request());bad[key]=value;reject(lambda:d.validate(bad));negative+=1
    from experiments.cvs_sigtime_rdrop_completion20 import model as api
    from .runtime import extract,snapshot,require_unchanged
    m=api.build(d.parent.config(d.parent.rows()[0]),'cpu').eval().requires_grad_(False)
    saved=snapshot(m);torch.manual_seed(2026101625);x=torch.randn(4,2,256)
    first=extract(m,x);w=api.classifier(m).weight;old=w.clone()
    with torch.no_grad():w.add_(torch.linspace(-.1,.1,w.numel()).reshape_as(w))
    second=extract(m,x)
    with torch.no_grad():w.copy_(old)
    assert first.shape==(4,160) and torch.equal(first,second)
    require_unchanged(m,saved)
    m.train();reject(lambda:extract(m,x));negative+=1
    print(json.dumps(dict(status='PASS',device='cpu',negative_cases=negative,ideal_fixture_triplets=3528000,
        exact_wins_ties_and_reverse_fixture=True,group_counts=True,label_and_coordinate_permutation_scale_invariance=True,
        actual_state_tensors=len(saved['state']),actual_joint160_independent_of_default_W=True,all_parameters_buffers_gradslots_counters_unchanged=True,
        synthetic_identity_forward_packets=8,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,head_fit_updates=0,target_access=False)))

if __name__=='__main__':main()
