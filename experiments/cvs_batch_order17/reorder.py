"""Preserve every L packet in six updates; balance TX/RX counts within this buffer."""
import math,random
from collections import Counter,defaultdict
BLOCK=6
BATCH=128
SEED=202610175101
RX=(3,4,6,8)

def valid(item):
 if not isinstance(item,tuple) or len(item)!=7 or any(type(v)is not int for v in item):raise ValueError('L tuple y,eq,RX,day,sig,base,TXindex required')
 y,eq,r,day,sig,base,tx=item
 if y not in range(6) or tx not in range(6) or eq!=1 or r not in RX or day not in (1,2,3) or min(sig,base)<0:raise ValueError('Exact visible fit L only')
 return item

def reorder(batches,block_index):
 if type(block_index)is not int or block_index<0 or len(batches)!=BLOCK or any(len(b)!=BATCH for b in batches):raise ValueError('One fixed six by128 L buffer required')
 pool=[valid(x) for b in batches for x in b];cells=defaultdict(list)
 for p in pool:cells[(p[0],p[2])].append(p)
 rng=random.Random(SEED+block_index);keys=sorted(cells);rng.shuffle(keys)
 quota={k:[len(cells[k])//BLOCK]*BLOCK for k in keys};loads=[0]*BLOCK
 for k in keys:
  candidates=list(range(BLOCK));rng.shuffle(candidates);candidates.sort(key=lambda b:loads[b])
  for b in candidates[:len(cells[k])%BLOCK]:quota[k][b]+=1;loads[b]+=1
 if len(set(loads))!=1:raise ValueError('Remainder transportation failed')
 rows=[[] for _ in range(BLOCK)]
 for k in keys:
  order=list(range(BLOCK));rng.shuffle(order);offset=0
  for b in order:
   n=quota[k][b];rows[b].extend(cells[k][offset:offset+n]);offset+=n
  assert offset==len(cells[k])
 for row in rows:rng.shuffle(row)
 if any(len(b)!=BATCH for b in rows) or Counter(pool)!=Counter(x for b in rows for x in b):raise ValueError('Physical L multiset changed')
 for k in keys:
  counts=[sum((p[0],p[2])==k for p in b) for b in rows]
  if max(counts)-min(counts)>1:raise ValueError('Within-buffer cell quota differs')
 return rows

def metrics(batch):
 for x in batch:valid(x)
 n=len(batch);yr=Counter((x[0],x[2]) for x in batch);ys=Counter(x[0] for x in batch);rs=Counter(x[2] for x in batch)
 positives=[ys[x[0]]-yr[(x[0],x[2])] for x in batch]
 valid_counts=[p for x,p in zip(batch,positives) if p and n-ys[x[0]]>0]
 mean=sum(positives)/n
 return dict(valid_anchors=len(valid_counts),total_anchors=n,positive_min=min(valid_counts,default=0),positive_max=max(valid_counts,default=0),positive_mean=mean,positive_variance=sum((v-mean)**2 for v in positives)/n,TX_count_range=max(ys.get(t,0) for t in range(6))-min(ys.get(t,0) for t in range(6)),RX_count_range=max(rs.get(r,0) for r in RX)-min(rs.get(r,0) for r in RX),missing_TX_RX_cells=sum(yr[(t,r)]==0 for t in range(6) for r in RX),missing_TX_RX_day_cells=sum(not any((x[0],x[2],x[3])==(t,r,d) for x in batch) for t in range(6) for r in RX for d in (1,2,3)),batch_TX_RX_mutual_information_nats=sum(v/n*math.log(v*n/(ys[t]*rs[r])) for (t,r),v in yr.items()))

def labeled_records(records):
 for obj in records:
  if obj['role']=='U':continue
  if obj['role']!='L':raise ValueError('Only original L/U stream roles')
  yield obj['ids']
