"""Transient visible-L donor bank; no model buffers or checkpoint state."""
from collections import Counter
import torch
from experiments.cvs_same_tx_iq16.iqmix import mix
KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
_BANK=None

class Bank:
 def __init__(self,x,labels,rx,day,keys,opaque_ids):
  if x.shape!=(5040,2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or len(keys)!=5040 or len(set(keys))!=5040 or len(opaque_ids)!=5040 or len(set(opaque_ids))!=5040 or any(len(v)!=5040 for v in (labels,rx,day)) or set(labels)!=set(range(6)) or set(rx)!={3,4,6,8} or set(day)!={1,2,3}:raise ValueError('Complete finite visible fitL5040 bank required')
  self.x=x.detach().cpu().clone();self.labels=list(labels);self.rx=list(rx);self.day=list(day);self.keys=list(keys);self.opaque_ids=list(opaque_ids);self.lookup={k:i for i,k in enumerate(keys)};cells={}
  for i,k in enumerate(zip(labels,rx,day)):cells.setdefault(k,[]).append(i)
  if len(cells)!=72 or any(len(v)!=70 for v in cells.values()):raise ValueError('Every original visibleL cell must contain70')
  for v in cells.values():v.sort(key=lambda i:opaque_ids[i])
  rank={i:j for v in cells.values() for j,i in enumerate(v)};self.same=[];self.cross=[]
  for i,(y,r,d) in enumerate(zip(labels,rx,day)):
   j=rank[i];self.same.append([cells[(y,r,d)][(j+k)%70] for k in (1,2,3)]);self.cross.append([cells[(y,q,d)][(j+1)%70] for q in sorted(set(rx)) if q!=r])
  self.visits=[0]*5040;self.calls=0;self.generated=0;self.gathered=0;self.coverage=Counter();self.anchor_checked=False
 def apply(self,raw,labels,meta,mode,step):
  if mode not in ('raw','sameRX','crossRX') or step!=self.calls+1 or raw.ndim!=3 or raw.shape[1:]!=(2,256) or raw.dtype!=torch.float32 or labels.shape!=(len(raw),) or bool(((labels<0)|(labels>5)).any()):raise ValueError('One ordered legal L IQmix call required')
  values={k:torch.as_tensor(meta[k]).detach().cpu().tolist() for k in KEYS}
  if any(len(v)!=len(raw) for v in values.values()):raise ValueError('Every anchor requires its original physical metadata')
  anchor=[];donor=[];slots=[];y=labels.detach().cpu().tolist()
  for j,key in enumerate(zip(*(values[k] for k in KEYS))):
   if key not in self.lookup:raise ValueError('Anchor outside visible trainL bank')
   i=self.lookup[key]
   if self.labels[i]!=y[j]:raise ValueError('Only visible sameTX donor pairing permitted')
   slot=self.visits[i]%3;self.visits[i]+=1;d=(self.same if mode=='sameRX' else self.cross)[i][slot]
   if d==i or self.labels[d]!=y[j] or self.day[d]!=self.day[i] or (mode=='sameRX' and self.rx[d]!=self.rx[i]) or (mode=='crossRX' and self.rx[d]==self.rx[i]):raise ValueError('Donor role/TX/day/RX differs')
   anchor.append(i);donor.append(d);slots.append(slot)
  if not self.anchor_checked:
   if not torch.equal(raw.detach().cpu(),self.x[anchor]):raise ValueError('Native raw anchors differ from actual bank IQ')
   self.anchor_checked=True
  if mode=='raw':view=raw;info={};actual_donors=[]
  else:
   view,info=mix(raw,self.x[donor].to(device=raw.device),.1);actual_donors=donor;self.generated+=len(raw);self.gathered+=len(raw)
   for i,k in zip(anchor,donor):self.coverage[(self.labels[i],self.rx[i],self.day[i],self.rx[k])]+=1
  self.calls+=1
  record=dict(anchor_indices=anchor,donor_indices=actual_donors,visit_slot=slots,bank_unique_L=5040,IQmix_packets=0 if mode=='raw' else len(raw),physical_label_rule='visible sameTX/day',variant_count_per_anchor=3)
  if info:
   for name in ('relative_IQ_change','relative_power_error'):
    v=torch.tensor(info[name],dtype=torch.float64);record[name]=dict(minimum=float(v.min()),mean=float(v.mean()),median=float(v.quantile(.5)),maximum=float(v.max()))
  return view,record

def install(bank):
 global _BANK
 if _BANK is not None:raise ValueError('One transient bank per worker')
 _BANK=bank

def current():
 if _BANK is None:raise ValueError('Native visible-L bank must exist before training hook')
 return _BANK
