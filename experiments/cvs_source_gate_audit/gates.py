"""L/V-only replica of existing confidence/physical-neighbor/strong gates."""
import math
from types import SimpleNamespace
import torch

FIELDS={'rx_i','day_i','eq_i','sig_i','base_index'}
def physical_metadata(data,role):
 if role not in ('L_s','V'):raise ValueError('Gate precision is restricted to labeled source roles')
 expected=6300 if role=='L_s' else 27000
 if len(data['ids'])!=expected or data['y'].shape!=(expected,) or data['y'].min()<0 or data['y'].max()>=6:raise ValueError('L/V role/label dimensions differ')
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 from cvsrffi.game_tracking.data import opaque_id
 # No U indices/IDs/labels are enumerated or decoded.
 ranges=range(70) if role=='L_s' else range(700,1000)
 first=dict(tx_i=int(data['y'][0]),rx_i=int(data['rx'][0]),day_i=int(data['day'][0]))
 eq_matches=[eq for eq in (0,1) if any(opaque_id(SimpleNamespace(**first,eq_i=eq,sig_i=s))==data['ids'][0] for s in ranges)]
 if len(eq_matches)!=1:raise ValueError('Native physical source equalization index is not identifiable from allowed L/V record')
 eq=eq_matches[0];rxs=[1,3,4,6,8];days=[1,2,3];lookup={}
 for tx in range(6):
  for ri,rx in enumerate(rxs):
   for di,day in enumerate(days):
    for sig in ranges:
     item=SimpleNamespace(tx_i=tx,rx_i=rx,day_i=day,eq_i=eq,sig_i=sig)
     lookup[opaque_id(item)]=dict(rx_i=rx,day_i=day,eq_i=eq,sig_i=sig,base_index=tx*15000+ri*3000+di*1000+sig)
 if set(data['ids'])!=set(lookup):raise ValueError('Existing L/V physical metadata mapping differs')
 # Source labels are used only to verify the approved cache, never sent to gates.
 meta={k:torch.tensor([lookup[v][k] for v in data['ids']],dtype=torch.long) for k in FIELDS}
 if not torch.equal(meta['rx_i'],data['rx']) or not torch.equal(meta['day_i'],data['day']):raise ValueError('Cache/physical source RX/day mismatch')
 return meta

def masks(native,teacher_logits,strong_logits,meta,domains,args):
 if set(meta)!=FIELDS or teacher_logits.shape!=strong_logits.shape or teacher_logits.ndim!=2 or teacher_logits.shape[1]!=6:raise ValueError('Label-blind gate inputs differ')
 if any(p.ndim!=1 or len(p)!=len(teacher_logits) for p in meta.values()) or not torch.isfinite(teacher_logits).all() or not torch.isfinite(strong_logits).all():raise ValueError('Invalid source gate tensors')
 conf,pseudo=teacher_logits.softmax(1).max(1)
 confidence=native._threshold_mask(conf,domains,args)
 extra=(None,meta)
 temporal=native._temporal_mask_tensor(pseudo,conf,extra,args,torch.device('cpu'))
 strong=strong_logits.argmax(1)==pseudo
 geometry=native._temporal_mask_tensor(torch.zeros_like(pseudo),torch.ones_like(conf),extra,args,torch.device('cpu'))
 return pseudo,dict(CONFIDENCE=confidence,CONFIDENCE_STRONG=confidence&strong,FULL_OLD=confidence&temporal&strong,GEOMETRIC_NEIGHBOR=geometry),dict(confidence=confidence,temporal=temporal,strong=strong,conf=conf)

def expected_neighbor_visibility(pool,cell_samples,window=2,batch=256):
 if pool<cell_samples or cell_samples<2 or window!=2 or batch!=256:raise ValueError('Unexpected source sampling geometry')
 remainder=pool%batch
 def probability(m,b):
  no=1.
  for j in range(m):no*=((pool-b-j)/(pool-1-j))
  return 1-no
 values=[]
 for i in range(cell_samples):
  neighbors=min(window,i)+min(window,cell_samples-1-i)
  values.append((pool-remainder)/pool*probability(neighbors,batch)+(remainder/pool*probability(neighbors,remainder) if remainder else 0))
 return dict(pool=pool,physical_cell_samples=cell_samples,window=window,batch=batch,expected_neighbor_visibility=sum(values)/len(values),scope='Analytic uniform random batching geometry only; no teacher predictions or U labels; not pseudo accuracy')
