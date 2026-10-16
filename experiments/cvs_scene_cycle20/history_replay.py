"""Historical L metadata only;no IQ,NN,checkpoint or new simulation."""
import ast,gzip,hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];p=root/'local_artifacts/F_crossrx_satmix20_20261016';src=p/'final/B01-RX1-s2026092701/source';start=time.perf_counter()
if (p/'scene_coverage_replay_summary.json').exists() or (p/'scene_coverage_replay_packets_private.jsonl.gz').exists():raise FileExistsError('Historical analysis exists;reuse original,never overwrite')
module=ast.parse((root/'experiments/adv3b02_xuc/code/leo_practical/channel.py').read_text(encoding='utf8'));node=next(v for v in module.body if isinstance(v,ast.FunctionDef) and v.name=='stable_seed');env=dict(json=json,hashlib=hashlib);exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_stable_seed_only','exec'),env);stable_seed=env['stable_seed']
scenes=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def records(path):
 with gzip.open(path,'rt',encoding='utf8') as f:
  for line in f:yield json.loads(line)
L=[v['ids'] for v in records(src/'training_sequence.jsonl.gz') if v['role']=='L'];steps=list(records(src/'satmix_steps.jsonl.gz'));assert len(L)==len(steps)==4440
counts={};metadata={};global_counts=Counter()
for i,(ids,row) in enumerate(zip(L,steps),1):
 assert row['step']==i and row['epoch']==(i-1)//222+1 and row['As_packets']==len(ids)==128
 observed=Counter();namespace='train/epoch%d/step%d'%(row['epoch'],i)
 for values in ids:
  eq,rx,day,sig,base=map(int,values);assert rx in (3,4,6,8) and day in (1,2,3)
  pid=(rx,day,eq,sig,base);seed=stable_seed(20261016101,namespace,'scene',pid);selected=int(np.random.default_rng(seed).integers(6));counts.setdefault(pid,np.zeros(6,dtype=np.int64))[selected]+=1;observed[scenes[selected]]+=1
 assert observed==Counter(row['scene_counts']),(i,observed,row['scene_counts'])
 global_counts.update(observed)
assert len(counts)==5040 and sum(global_counts.values())==568320
stored=json.loads((src/'satellite_supervision_coverage.json').read_text(encoding='utf8'));expected=Counter()
for v in stored['cells']:expected[v['scene']]+=v['count']
assert expected==global_counts
A=np.array(list(counts.values()));N=A.sum(1);mn=A.min(1);diff=A.max(1)-mn;tv=np.abs(A/N[:,None]-1/6).sum(1)/2
quant=lambda x:dict(min=float(np.min(x)),q05=float(np.quantile(x,.05)),median=float(np.median(x)),q95=float(np.quantile(x,.95)),max=float(np.max(x)),mean=float(np.mean(x)))
by={}
for rx in (3,4,6,8):
 for day in (1,2,3):
  B=np.array([v for pid,v in counts.items() if pid[0]==rx and pid[1]==day]);assert len(B)==420;Q=B.sum(1)
  by[f'RX{rx}/day{day}']=dict(unique_L=len(B),observations=int(Q.sum()),occurrences=quant(Q),per_scene_totals={s:int(B[:,i].sum()) for i,s in enumerate(scenes)},per_packet_min_scene=quant(B.min(1)),per_packet_max_minus_min=quant(B.max(1)-B.min(1)),low_urban=quant(B[:,5]),mid_urban=quant(B[:,4]))
summary=dict(status='VERIFIED_HISTORICAL_SCENE_REPLAY',source_run='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01',row='B01-RX1-s2026092701',all4440_step_scene_counts_exact=True,coverage_aggregate_exact=True,unique_L=5040,L_observations=568320,all_packets_all6scenes_seen=bool((A>0).all()),missing_packet_scene_cells=int((A==0).sum()),global_scene_counts=dict(global_counts),per_packet_occurrences=quant(N),per_packet_min_scene=quant(mn),per_packet_max_minus_min=quant(diff),per_packet_scene_TV_from_uniform=quant(tv),per_scene_counts={s:quant(A[:,i]) for i,s in enumerate(scenes)},per_packet_imbalance_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(diff,return_counts=True))},RX_day=by,wall_seconds=time.perf_counter()-start,IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,channel_IQ_generated=0,U_truth_read=False,target_access=False,scope='Historical L metadata scene replay only;not recognition efficacy or source nuisance proof',counterfactual_property='A six-scene permutation per six appearances would bound each packet countmax-countmin<=1;not measured recognition benefit')
(p/'scene_coverage_replay_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with gzip.open(p/'scene_coverage_replay_packets_private.jsonl.gz','xt',encoding='utf8') as f:
 for pid,v in sorted(counts.items()):f.write(json.dumps(dict(pid=pid,scene_counts=v.tolist()))+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('RX_day','per_packet_imbalance_histogram','per_scene_counts')},ensure_ascii=False))
