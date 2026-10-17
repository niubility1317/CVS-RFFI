"""Replay visible-L metadata only; no IQ, weights, U truth, or target inputs."""
import argparse,collections,gzip,hashlib,json,math,random
from pathlib import Path
FIT=(1,3,4,6,8)
REFERENCE='20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02'

def repartition(packets,mode,seed):
 if len(packets)!=768 or mode not in ('ORIGINAL','RANDOM','TX_RX_BALANCED'):raise ValueError('Fixed6x128 visible-L pool required')
 if any(p['tx'] not in range(6) or p['rx'] not in FIT or p['day'] not in (1,2,3) for p in packets):raise ValueError('Original visible source metadata required')
 if mode=='ORIGINAL':return [packets[i:i+128] for i in range(0,768,128)]
 rng=random.Random(seed)
 if mode=='RANDOM':
  order=list(packets);rng.shuffle(order);return [order[i:i+128] for i in range(0,768,128)]
 cells=collections.defaultdict(list)
 for p in packets:cells[(p['tx'],p['rx'])].append(p)
 # Uniform floor in every output batch;remainders form a bipartite degree problem.
 batches=[[] for _ in range(6)];base=sum(len(v)//6 for v in cells.values());capacity=[128-base]*6;groups=list(cells.values());rng.shuffle(groups);groups.sort(key=lambda v:len(v)%6,reverse=True)
 for values in groups:
  q,r=divmod(len(values),6);offset=0
  for batch in batches:batch.extend(values[offset:offset+q]);offset+=q
  tie=list(range(6));rng.shuffle(tie);rank={v:k for k,v in enumerate(tie)};selected=sorted(range(6),key=lambda j:(-capacity[j],rank[j]))[:r]
  for j in selected:
   if capacity[j]<=0:raise AssertionError('Remainder assignment infeasible')
   batches[j].append(values[offset]);offset+=1;capacity[j]-=1
  assert offset==len(values)
 if any(capacity) or any(len(b)!=128 for b in batches):raise AssertionError('Fixed batch sizes lost')
 for b in batches:rng.shuffle(b)
 if collections.Counter(p['occurrence'] for b in batches for p in b)!=collections.Counter(p['occurrence'] for p in packets):raise AssertionError('Physical occurrence changed')
 return batches

def geometry(batch):
 tx=collections.Counter(p['tx'] for p in batch);rx=collections.Counter(p['rx'] for p in batch);cell=collections.Counter((p['tx'],p['rx']) for p in batch);positive=[tx[p['tx']]-cell[(p['tx'],p['rx'])] for p in batch]
 return dict(cell_squared_imbalance=sum((cell[(t,r)]-128/30)**2 for t in range(6) for r in FIT)/30,TX_squared_imbalance=sum((tx[t]-128/6)**2 for t in range(6))/6,RX_squared_imbalance=sum((rx[r]-128/5)**2 for r in FIT)/5,empty_TX_RX_cells=sum(cell[(t,r)]==0 for t in range(6) for r in FIT),valid_anchors=sum(v>0 for v in positive),positive_min=min(positive),positive_max=max(positive),positive_sum=sum(positive))

def checks():
 rng=random.Random(20261018120);state=random.getstate();cases=0
 for k in range(100):
  values=[dict(occurrence=i,tx=rng.randrange(6),rx=rng.choice(FIT),day=rng.choice((1,2,3))) for i in range(768)]
  b=repartition(values,'TX_RX_BALANCED',k);assert [len(z) for z in b]==[128]*6
  for t in range(6):
   for r in FIT:
    counts=[sum(p['tx']==t and p['rx']==r for p in z) for z in b];assert max(counts)-min(counts)<=1
  assert b==repartition(values,'TX_RX_BALANCED',k);cases+=1
 for pairs in [((0,1),),((0,1),(5,8)),tuple((t,r) for t in range(6) for r in FIT)]:
  values=[dict(occurrence=i,tx=pairs[i%len(pairs)][0],rx=pairs[i%len(pairs)][1],day=1) for i in range(768)];repartition(values,'TX_RX_BALANCED',9);cases+=1
 assert state==random.getstate()
 try:repartition(values[:-1],'TX_RX_BALANCED',9)
 except ValueError:pass
 else:raise AssertionError('Invalid size accepted')
 return dict(status='PASS',synthetic_pools=cases,exact_occurrences=True,every_cell_across_batches_difference_at_most1=True,global_RNG_unchanged=True,IQ=0,CP=0,NN=0,updates=0)

def run(project):
 project=Path(project);c=json.loads((project/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json').read_text());allowed=set(c['role_ids']['L_s'])
 if len(allowed)!=6300 or c['source_rxs']!=list(FIT) or c['source_days']!=[1,2,3] or c['num_classes']!=6:raise ValueError('Original full L6300 source handles required')
 sequence=project/'runs'/REFERENCE/'CONCAT-s2026092701/source/training_sequence.jsonl.gz';source=[];seen=set();batches=0;lookup={}
 n_per_cell=sum(len(v) for v in c['role_ids'].values())//90
 if n_per_cell!=1000:raise ValueError('Original untruncated1000 per source cell required')
 with gzip.open(sequence,'rt',encoding='utf8') as stream:
  for line in stream:
   row=json.loads(line)
   if row['role']!='L':continue # Never resolve hidden U TX.
   if len(row['ids'])!=128:raise ValueError('Original L128 required')
   for raw in row['ids']:
    if len(raw)!=5:raise ValueError('Original metadata tuple required')
    eq,rx,day,sig,base=map(int,raw);key=(eq,rx,day,sig,base)
    if key not in lookup:
     # Original source dataset index iterates TX,RX,day,eq,sig;visible L only.
     # Do not decode or join any U record. Confirm every component and opaque L ID.
     cell,canonical_sig=divmod(base,n_per_cell);tx_rx,di=divmod(cell,3);t,ri=divmod(tx_rx,5)
     if t not in range(6) or (FIT[ri],di+1,1,canonical_sig)!=(rx,day,eq,sig):raise ValueError('Original source index metadata differ')
     opaque=hashlib.sha256(f'{t}/{rx}/{day}/{eq}/{sig}'.encode()).hexdigest()[:24]
     if opaque not in allowed:raise ValueError('Resolved record not in visibleL6300')
     lookup[key]=(t,opaque)
    t,opaque=lookup[key];seen.add(opaque);source.append(dict(occurrence=len(source),tx=t,rx=rx,day=day,base_index=base,physical_id=opaque))
   batches+=1
 if batches!=4440 or len(source)!=568320 or seen!=allowed:raise ValueError('Complete4440 steps/all6300L exposure required')
 metrics={mode:[] for mode in ('ORIGINAL','RANDOM','TX_RX_BALANCED')};pool_ok=True
 for i in range(0,len(source),768):
  pool=source[i:i+768];original=collections.Counter(p['physical_id'] for p in pool)
  for mode in metrics:
   parts=repartition(pool,mode,20261018121+i//768);assert collections.Counter(p['physical_id'] for b in parts for p in b)==original
   metrics[mode].extend(geometry(b) for b in parts)
 result={}
 for mode,rows in metrics.items():
  result[mode]={k:dict(mean=sum(v[k] for v in rows)/len(rows),minimum=min(v[k] for v in rows),maximum=max(v[k] for v in rows)) for k in rows[0]}
 return dict(status='METADATA_REPLAY_COMPLETE',reference_run=REFERENCE,reference_row='CONCAT-s2026092701',source_RX=list(FIT),source_days=[1,2,3],source_RX_holdout=None,source_L_unique=len(seen),source_L_batches=batches,source_L_exposures=len(source),six_batch_pools=len(source)//768,all_physical_occurrences_preserved_within_each_pool=True,methods=result,model_or_augmentation_executed=False,new_Ag=False,Ag_efficacy=False,As_efficacy=False,recognition_gain_proved=False,IQ=0,CP=0,NN=0,updates=0,target_access=False,U_TX_resolved=False,limits='Only batch-composition geometry;cannot establish contrastive gradient variance or identity recognition gain. No candidate promotion or target access.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project');p.add_argument('--checks',action='store_true');a=p.parse_args();print(json.dumps(checks() if a.checks else run(a.project),allow_nan=False))
