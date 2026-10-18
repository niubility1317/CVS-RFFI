"""Private CPU waterfill over existing L128 positions;no Torch/model/channel imports."""
import hashlib,json,time
import numpy as np
SEED=20261016421
NAMESPACE='FcrossRX/AsAllocation/waterfill64/v1'
RX=(1,3,4,6,8)

def stable_seed(seed,*parts):
 raw=json.dumps([int(seed),*parts],ensure_ascii=False,separators=(',',':'))
 return int.from_bytes(hashlib.sha256(raw.encode('utf8')).digest()[:8],'little')

def waterfill(labels,rx,priorities,rng):
 y=np.asarray(labels);r=np.asarray(rx);p=np.asarray(priorities,dtype=np.float64)
 if y.shape!=(128,) or r.shape!=(128,) or y.dtype.kind not in 'iu' or r.dtype.kind not in 'iu' or np.any((y<0)|(y>=6)) or not np.isin(r,RX).all() or p.shape!=(128,) or not np.isfinite(p).all() or np.any((p<0)|(p>=1)) or not isinstance(rng,np.random.Generator):raise ValueError('Visible trueL128 TX/RX,private priorities and generator required')
 grouped={}
 for i,(yy,rr) in enumerate(zip(y,r)):grouped.setdefault((int(yy),int(rr)),[]).append(i)
 members=[sorted(ix,key=lambda i:(float(p[i]),i)) for ix in grouped.values()]
 # A group's order derives from its packet priorities,never the numeric TX/RX name.
 members.sort(key=lambda ix:(float(p[ix[0]]),ix[0]));quotas=[0]*len(members);chosen=[];rounds=[]
 while len(chosen)<64:
  active=[g for g,ix in enumerate(members) if quotas[g]<len(ix)]
  order=[int(v) for v in rng.permutation(np.asarray(active,dtype=np.int64))];used=[]
  for g in order:
   if len(chosen)==64:break
   chosen.append(members[g][quotas[g]]);quotas[g]+=1;used.append(g)
  rounds.append(dict(active_groups=active,visit_order=order,selected_groups=used))
 selected=sorted(chosen);mask=np.zeros(128,dtype=bool);mask[selected]=True
 cells=[]
 for g,ix in enumerate(members):
  cells.append(dict(canonical_index=g,TX=int(y[ix[0]]),RX=int(r[ix[0]]),available=len(ix),selected=quotas[g],input_L_positions=sorted(ix),priority_ordered_L_positions=ix,selected_L_positions=sorted(i for i in ix if mask[i])))
 return mask,dict(cells=cells,rounds=rounds,selected_L_positions=selected,ordinary_L_positions=np.flatnonzero(~mask).tolist(),active_cells=len(cells),quota_min=min(quotas),quota_max=max(quotas),priority_ties=int(128-len(np.unique(p))),tie_rule='originalposition onlyfor exactpriority ties')

def select(labels,rx,cumulative_epoch,cumulative_step):
 start=time.perf_counter()
 if type(cumulative_epoch)is not int or type(cumulative_step)is not int or cumulative_epoch not in range(21,31) or cumulative_step not in range(4441,6661) or cumulative_epoch!=(cumulative_step-1)//222+1:raise ValueError('Only fixed cumulativeE21..30/step4441..6660')
 seed=stable_seed(SEED,NAMESPACE,cumulative_epoch,cumulative_step);rng=np.random.default_rng(seed);priorities=rng.random(128)
 mask,record=waterfill(labels,rx,priorities,rng);old=(np.arange(128)+cumulative_step)%2!=0
 for cell in record['cells']:
  cell['old_selected_L_positions']=[i for i in cell['input_L_positions'] if old[i]];cell['old_selected']=len(cell['old_selected_L_positions'])
 record.update(cumulative_epoch=cumulative_epoch,cumulative_step=cumulative_step,augmentation_seed=SEED,allocation_namespace=NAMESPACE,allocation_seed=seed,priorities=priorities.tolist(),old_selected_L_positions=np.flatnonzero(old).tolist(),selected_packets=64,ordinary_packets=64,changed_positions=int(np.count_nonzero(mask!=old)),allocation_cpu_wall_seconds=time.perf_counter()-start,private_CPU_rng='numpy.default_rng(PCG64)',global_RNG_used=False,sampler_changed=False,physical_ID_uniqueness_required=False,labels_used='trueL TX only',RX_used='existing originalRX',confidence_or_quality_used=False,U_truth_used=False,extra_NN=0,new_losses=0,formal_test_changed=False)
 return mask,record
