"""Recount all actual L scene decisions independently;no IQ/checkpoint/NN/target."""
import gzip,hashlib,json,math,statistics,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[2]/'local_artifacts/F_physical_scene_cycle_E20_20261017';arms=('B00','C_IID','B01');names=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(q):return json.loads(q.read_text(encoding='utf8'))
def records(q):
 with gzip.open(q,'rt',encoding='utf8') as f:return [json.loads(s) for s in f if s.strip()]
def seed(*parts):return int.from_bytes(hashlib.sha256(json.dumps(list(parts),ensure_ascii=False,separators=(',',':')).encode('utf8')).digest()[:8],'little')
def recount(arm):
 started=time.perf_counter();src=p/'final'/(arm+'-RX1-s2026092701')/'source';audit=read(src/'scene_cycle_audit.json');trace=records(src/'scene_cycle_steps.jsonl.gz');seq=[v for v in records(src/'training_sequence.jsonl.gz') if v['role']=='L'];steps=records(src/'satmix_steps.jsonl.gz');mode={'B00':'none','C_IID':'iid','B01':'balanced'}[arm];assert len(seq)==len(steps)==4440
 counts={};visits=Counter();perms={};global_counts=Counter();rxday_scene=Counter();new_permutations=0
 assert audit['config']['mode']==mode and audit['config']['cycle_length']==6 and audit['config']['permutation_seed']==202610175001 and audit['teacher_training_calls']==audit['reader_training_calls']==0 and audit['all_legal_L_supervised'] and audit['source_eval_unchanged'] and not audit['target_access']
 assert len(trace)==(4440 if mode!='none' else 0)
 if mode!='none':
  for i,(s,t,details) in enumerate(zip(seq,trace,steps),1):
   epoch=(i-1)//222+1;namespace='train/epoch%d/step%d'%(epoch,i);assert t['namespace']==namespace and len(t['scene_indices'])==len(t['visits_before'])==len(s['ids'])==128 and details['step']==i
   step_counts=Counter()
   for k,j,prior in zip(s['ids'],t['scene_indices'],t['visits_before']):
    eq,rx,day,sig,base=map(int,k);pid=(rx,day,eq,sig,base);assert rx in (3,4,6,8) and day in (1,2,3) and eq==1;v=visits[pid];assert prior==v
    iid=int(np.random.default_rng(seed(20261016101,namespace,'scene',pid)).integers(6))
    if mode=='balanced':
     if v%6==0:perms[pid]=np.random.default_rng(seed(202610175001,'physical_scene_cycle',pid,v//6)).permutation(6).tolist();new_permutations+=1
     expected=perms[pid][v%6]
    else:expected=iid
    assert j==expected;counts.setdefault(pid,[0]*6)[j]+=1;visits[pid]+=1;global_counts[names[j]]+=1;step_counts[names[j]]+=1;rxday_scene[(rx,day,names[j])]+=1
   assert dict(step_counts)==details['extra_scene_counts']
 assert sum(visits.values())==(568320 if mode!='none' else 0) and len(visits)==(5040 if mode!='none' else 0)
 actual={tuple(v['pid']):v['scene_counts'] for v in audit['per_packet_counts']};assert actual==counts
 q=audit['summary'];assert q['mode']==mode and q['training_calls']==len(trace) and q['training_packets']==sum(visits.values()) and q['unique_physical_L']==len(visits) and q['permutation_cycles_constructed']==new_permutations
 spread=[max(v)-min(v) for v in counts.values()];assert q['max_minus_min']==max(spread,default=0)
 if mode=='balanced':assert max(spread)<=1
 coverage=read(src/'satellite_supervision_coverage.json');assert coverage['total']==sum(global_counts.values());agg=Counter();g=Counter()
 for c in coverage['cells']:agg[(c['RX'],c['day'],c['scene'])]+=c['count'];g[c['scene']]+=c['count']
 assert agg==rxday_scene and g==global_counts
 def distribution(z):return dict(min=min(z),median=statistics.median(z),mean=statistics.mean(z),max=max(z)) if z else None
 result=dict(status='VERIFIED',arm=arm,mode=mode,training_calls=len(trace),training_packets=sum(visits.values()),unique_physical_L=len(visits),all_actual_decisions_replayed=True,per_packet_final_counts_identical=True,global_scene_counts=dict(global_counts),permutation_cycles_constructed=new_permutations,per_packet_appearance_distribution=distribution(list(visits.values())),per_packet_max_minus_min_distribution=distribution(spread),missing_packet_scene_cells=sum(c==0 for v in counts.values() for c in v),all_TX_RX_day_state_coverage_runtime_report=coverage,independent_RX_day_scene_counts=[dict(RX=r,day=d,scene=s,count=n) for (r,d,s),n in sorted(rxday_scene.items())],runtime_generation_wall_seconds=q['training_generation_wall_seconds'],recount_wall_seconds=time.perf_counter()-started,IQ_reads=0,checkpoint_loads=0,NN=0,new_updates=0,target_access=False)
 (src/'schedule_recount.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({k:result[k] for k in ('status','arm','training_calls','training_packets','unique_physical_L','per_packet_max_minus_min_distribution','recount_wall_seconds')}))
if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in arms;recount(sys.argv[1])
