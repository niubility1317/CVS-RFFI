"""Read-only source descriptor negative masks and full feature gradients."""
import torch
from torch.nn import functional as F
PREFIX='id_backbone.encoder.core.id_backbone.'
LEVELS=('time_down','t2.pool');RXS=(1,3,4,6,8);SEEDS=(2026092701,2026092702,2026092703)
REF_RUN='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01';REF_COMMIT='31467086c3ced30237f3ac86e91eae68645425d7'
RUN='20261020-diagnostic-FcrossRX-RXnegative-sourceL-m3-local-r01';PARTITION_SEED=2026102010;NEGATIVE_SEED=2026102011
OBJECTIVES=('CE_TX1','CE_TX3','CE_OTHER4','SC_ALL','SC_RANDOM_MATCHED','SC_SAME_RX')

def masks(y,rx,batch):
 if y.dtype!=torch.long or rx.dtype!=torch.long or y.ndim!=1 or rx.shape!=y.shape or not len(y) or not bool(((y>=0)&(y<6)).all()) or not all(int(r) in RXS for r in rx.detach().cpu().tolist()) or not isinstance(batch,int) or batch<1:raise ValueError('Visible source L TX/RX and fixed batch ordinal required')
 same=y[:,None]==y[None,:];same_rx=rx[:,None]==rx[None,:];pos=same&~same_rx;negative=~same;local=negative&same_rx;counts=local.sum(1).cpu();gen=torch.Generator(device='cpu').manual_seed(NEGATIVE_SEED+batch);scores=torch.rand((len(y),len(y)),generator=gen).masked_fill(~negative.detach().cpu(),float('inf'));order=scores.argsort(1);chosen=torch.arange(len(y))[None,:]<counts[:,None];random=torch.zeros_like(scores,dtype=torch.bool).scatter(1,order,chosen).to(y.device);valid=pos.any(1)&local.any(1)
 if not torch.equal(random.sum(1).cpu(),counts) or bool((random&~negative).any()):raise ValueError('Exact negative count matching failed')
 return pos,dict(ALL=negative,RANDOM_MATCHED=random,SAME_RX=local),valid

def losses(maps,y,rx,batch):
 if tuple(maps)!=LEVELS:raise ValueError('Both actual shallow and middle source maps required')
 pos,neg,valid=masks(y,rx,batch);n=pos.sum(1);values={key:[] for key in neg};stats=[]
 for level in LEVELS:
  h=maps[level]
  if h.ndim!=3 or len(h)!=len(y) or not bool(torch.isfinite(h).all()):raise ValueError('Finite actual current source maps required')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4);unit=F.normalize(z,dim=1,eps=1e-4);cos=unit@unit.T;logits=cos/.2
  for key,negative in neg.items():
   value=(torch.logsumexp(logits[valid].masked_fill(~(pos|negative)[valid],-torch.inf),1)-(logits[valid]*pos[valid]).sum(1)/n[valid]-n[valid].to(z.dtype).log()).mean() if bool(valid.any()) else z.sum()*0.;values[key].append(value)
  different=neg['ALL']&~(rx[:,None]==rx[None,:]);eligible=valid&different.any(1);per=[]
  for c in range(6):
   mask=(y==c)&eligible;count=int(mask.sum());count_all=int((y==c).sum());item=dict(TX=c,total=count_all,eligible=count)
   for key,mm in dict(same_RX_negative=neg['SAME_RX'],other_RX_negative=different,cross_RX_positive=pos).items():
    v=(cos*mm).sum(1)/mm.sum(1).clamp_min(1);item[key+'_sum']=float(v[mask].detach().sum())
   per.append(item)
  stats.append(dict(level=level,valid_anchors=int(valid.sum()),source_anchors=len(y),per_TX=per))
 result={key:torch.stack(v).mean() for key,v in values.items()}
 if not all(bool(torch.isfinite(v)) for v in result.values()):raise FloatingPointError('Nonfinite source-only contrastive statistic')
 return result,stats

def select(m):
 out=[(n,p) for n,p in m.named_parameters() if n.startswith('id_backbone.') and not n.startswith(PREFIX+'cls_head.')]
 if not out or any(not p.requires_grad for n,p in out) or not any(n.endswith('sinc.low_coordinate') for n,p in out):raise ValueError('Full actual deployed F feature parameters required')
 return out

def objectives(output,maps,y,rx,batch):
 if output['tx_logits'].shape!=(len(y),6) or output['id_feat_joint'].shape!=(len(y),160):raise ValueError('Original defaulthead and deployed joint160 required')
 ce=F.cross_entropy(output['tx_logits'],y,reduction='none',label_smoothing=.01);vv,info=losses(maps,y,rx,batch);values=[ce[y==1].sum()/1050,ce[y==3].sum()/1050,ce[(y!=1)&(y!=3)].sum()/4200]+[vv[key]*len(y)/6300 for key in ('ALL','RANDOM_MATCHED','SAME_RX')]
 return values,info

def vectors(values,params):
 out=[]
 for i,value in enumerate(values):
  grads=torch.autograd.grad(value,[p for n,p in params],retain_graph=i+1<len(values),allow_unused=True);parts=[]
  for (n,p),g in zip(params,grads):
   g=torch.zeros_like(p) if g is None else g
   if not bool(torch.isfinite(g).all()):raise FloatingPointError('Nonfinite source feature gradient')
   parts.append(g.detach().reshape(-1))
  out.append(torch.cat(parts))
 return torch.stack(out).double()

def summary(raw,params):
 out={}
 for metric,factor in (('CLIP_PHYSICAL',1/25000),('NORMALIZED_HZ',1000.)):
  scale=torch.cat([torch.full((p.numel(),),factor if n.endswith(('sinc.low_coordinate','sinc.band_coordinate')) else 1.,device=raw.device,dtype=raw.dtype) for n,p in params]);g=raw*scale;gram=g@g.T;norm=gram.diag().clamp_min(0).sqrt();den=norm[:,None]*norm[None,:];cos=gram/den.clamp_min(1e-300)
  out[metric]=dict(objectives=list(OBJECTIVES),norm=norm.cpu().tolist(),cosine=[[float(cos[i,j].clamp(-1,1)) if den[i,j]>0 else None for j in range(6)] for i in range(6)])
 return out

def decision(rows):
 checks=[]
 for row in rows:
  weak=[]
  for c in (1,3):
   items=[s['per_TX'][c] for s in row['descriptor_stats']];same=sum(x['same_RX_negative_sum'] for x in items);other=sum(x['other_RX_negative_sum'] for x in items);n=sum(x['eligible'] for x in items);total=sum(x['total'] for x in items);gap=(same-other)/n if n else None;fraction=n/total if total else 0.;weak.append(dict(TX=c,negative_cosine_RX_gap=gap,eligible_fraction=fraction,PASS=gap is not None and gap>=.02 and fraction>=.99))
  checks.append(dict(model_seed=row['model_seed'],weak=weak,PASS=all(x['PASS'] for x in weak)))
 return dict(status='SOURCE_NEGATIVE_STRUCTURE_DECISION',PASS=len(rows)==3 and all(r['PASS'] for r in checks),rows=checks,target_access=False,rule='Both source-defined weakTX, all3 existing BASE modelseeds: sameRX differentTX negatives have >=.02 higher mean unit-descriptor cosine than otherRX negatives across fixed shallow/middle, eligiblefraction>=.99. Gradient proxies fullyreported, not used to choose metric or tune. Descriptive motivation only for a separately preregistered matched training study, no accuracy or invariance guarantee.')
