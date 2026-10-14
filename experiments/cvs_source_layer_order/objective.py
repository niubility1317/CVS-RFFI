"""Fixed source-label relations, no class W, learned projector or RNG."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from . import design as d

def metadata(y,rx,day,ids,role='L_s'):
 if role!='L_s':raise ValueError('Only visible legal L_s is permitted')
 if any(v.ndim!=1 or v.dtype!=torch.long for v in (y,rx,day)) or len(y)!=len(rx) or len(y)!=len(day) or len(ids)!=len(y):raise ValueError('Source metadata shape/type differs')
 if not set(y.tolist())<=set(range(6)) or not set(rx.tolist())<=set(d.RX) or not set(day.tolist())<=set(d.DAYS):raise ValueError('Source metadata out of scope')
 if len(set(ids))!=len(ids):raise ValueError('Repeated physical packets are not eligible')

@contextmanager
def capture(model):
 core=model.id_backbone.encoder.core.id_backbone;maps={};handles=[]
 try:
  for name in ('t1','t2'):
   def hook(module,inputs,out,name=name):
    if name in maps or out.ndim!=3 or not torch.isfinite(out).all():raise ValueError('Actual layer map differs')
    maps[name]=out
   handles.append(getattr(core,name).register_forward_hook(hook))
  yield maps
  if tuple(maps)!=('t1','t2'):raise ValueError('Shallow/middle actual forward missing')
 finally:
  for h in handles:h.remove()

def views(maps,out):
 z={n:F.normalize(F.adaptive_avg_pool1d(maps[n],4).flatten(1),dim=1,eps=1e-4) for n in ('t1','t2')}
 z['joint160']=F.normalize(out['id_feat_joint'],dim=1,eps=1e-4)
 return z

def relation_indices(y,rx,day,ids,role='L_s'):
 metadata(y,rx,day,ids,role);groups=[];triples=[]
 for i in range(len(y)):
  for r in d.RX:
   if r==int(rx[i]):continue
   ref=torch.where((rx==r)&(day==day[i]))[0]
   if len(ref)!=6 or set(y[ref].tolist())!=set(range(6)):raise ValueError('One legal packet perTX/referenceRX/day required')
   p=ref[y[ref]==y[i]].item();neg=ref[y[ref]!=y[i]]
   groups.append((i,p,ref.tolist()));triples.extend((i,p,int(n)) for n in neg)
 return torch.tensor(triples,dtype=torch.long),groups

def losses(z,q,groups):
 if z.ndim!=2 or len(z)!=90 or not torch.isfinite(z).all():raise ValueError('Balanced90 feature view required')
 q=q.to(z.device);s=z@z.T;delta=s[q[:,0],q[:,1]]-s[q[:,0],q[:,2]];gap=d.MARGIN-delta
 order=F.relu(gap).mean();terms=[]
 for i,p,ref in groups:terms.append(torch.logsumexp(s[i,ref]/d.TEMPERATURE,0)-s[i,p]/d.TEMPERATURE)
 supcon=torch.stack(terms).mean()
 return order,supcon,dict(gap=gap.detach().cpu(),delta=delta.detach().cpu(),cosine=s.detach().cpu())
