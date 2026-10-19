"""Frozen full-feature gradients on source L; no model or optimizer update."""
import math
import torch
from torch.nn import functional as F
PREFIX='id_backbone.encoder.core.id_backbone.'
OBJECTIVES=tuple(['CE_TX'+str(c) for c in range(6)]+['BASE_SC','NEW_MEAN','NEW_TAIL','JOINT_SC_TX1','JOINT_SC_TX3'])
NAMES=OBJECTIVES+('CE_OTHER4','CE_WEAK2','CE_ALL6')
METRICS=('CLIP_PHYSICAL','NORMALIZED_HZ')
REF_RUN='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01'
REF_COMMIT='31467086c3ced30237f3ac86e91eae68645425d7'
def reference_scope(run,commit):
 if run!=REF_RUN or commit!=REF_COMMIT:raise ValueError('Exact original local BASE source/checkpoint implementation required')
def scope(role,y,rx):
 if role!='L_s' or y.dtype!=torch.long or rx.dtype!=torch.long or y.ndim!=1 or rx.shape!=y.shape or not len(y) or not bool(((y>=0)&(y<6)).all()) or not all(int(v) in (1,3,4,6,8) for v in rx.tolist()):raise ValueError('Known source L TX/RX only')
def unit_scale(name,metric):
 if metric not in METRICS:raise ValueError('Fixed physical coordinate metrics only')
 if name.endswith(('sinc.low_coordinate','sinc.band_coordinate')):return 1/25000 if metric=='CLIP_PHYSICAL' else 1000.
 return 1.
def derived(g):return torch.cat((g,g[[0,2,4,5]].mean(0,keepdim=True),g[[1,3]].mean(0,keepdim=True),g[:6].mean(0,keepdim=True)),0)
def gram_summary(gram,names=NAMES):
 g=gram.detach().cpu().double();norm=g.diag().clamp_min(0).sqrt();den=norm[:,None]*norm[None,:];cos=g/den.clamp_min(1e-300)
 if g.shape!=(len(names),len(names)) or not bool(torch.isfinite(g).all()):raise ValueError('Finite Gram matrix required')
 return dict(objectives=list(names),norms=norm.tolist(),cosine=[[float(cos[i,j].clamp(-1,1)) if den[i,j]>0 else None for j in range(len(names))] for i in range(len(names))])
def select(model):
 values=[(n,p) for n,p in model.named_parameters() if n.startswith('id_backbone.') and not n.startswith(PREFIX+'cls_head.')]
 if not values or any(not p.requires_grad for n,p in values) or not any(n.endswith('sinc.low_coordinate') for n,p in values):raise ValueError('Actual full deployed identity features with centered Sinc required')
 values.sort(key=lambda v:(block_name(v[0]),v[0]));return values
def joint_each(f,y,rx):
 z=F.normalize(f,dim=1,eps=1e-4);same=y[:,None]==y[None,:];positive=same&(rx[:,None]!=rx[None,:]);negative=~same;valid=positive.any(1)&negative.any(1);out=z.sum(1)*0
 if bool(valid.any()):
  logits=(z@z.T/.2)[valid];pos=positive[valid];n=pos.sum(1).to(z.dtype);v=torch.logsumexp(logits.masked_fill(~(positive|negative)[valid],-torch.inf),1)-(logits*pos).sum(1)/n-n.log();out=out.index_copy(0,torch.where(valid)[0],v)
 return out,valid
def objectives(output,maps,y,rx,base_penalty,tail):
 scope('L_s',y,rx);logits=output['tx_logits'];joint=output['id_feat_joint']
 if logits.shape!=(len(y),6) or joint.shape!=(len(y),160) or tuple(maps)!=('time_down','t2.pool'):raise ValueError('Actual default logits/deployed joint160/shallow/middle required')
 ce=F.cross_entropy(logits,y,reduction='none',label_smoothing=.01);values=[ce[y==c].sum()/1050 for c in range(6)]
 old,_=base_penalty(maps,y,rx,'CROSS_RX');values.append(old*len(y)/6300)
 risk_values={mode:[] for mode in ('MEAN_WEAK_RX','TAIL_WEAK_RX')};coverage=[]
 for level,h in maps.items():
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4);each,valid=tail.anchor_loss(z,y,rx)
  for mode in risk_values:
   loss,info=tail.risk(each,y,rx,valid,mode);risk_values[mode].append(loss)
   if mode=='MEAN_WEAK_RX':coverage.append(info['eligible'])
 for mode in risk_values:values.append(torch.stack(risk_values[mode]).mean()*len(y)/6300)
 each,valid=joint_each(joint,y,rx)
 for c in (1,3):values.append(each[(y==c)&valid].sum()/1050)
 if len(values)!=len(OBJECTIVES) or not all(bool(torch.isfinite(v)) for v in values):raise FloatingPointError('Finite declared source objectives required')
 return values,dict(source_packets=len(y),risk_levels_eligible=coverage,joint_valid_weak={str(c):int(((y==c)&valid).sum()) for c in (1,3)})
def vectors(model,values,parameters=None):
 parameters=select(model) if parameters is None else parameters;ps=[p for _,p in parameters];rows=[];unused=set()
 for i,value in enumerate(values):
  grads=torch.autograd.grad(value,ps,retain_graph=i+1<len(values),create_graph=False,allow_unused=True);parts=[]
  for (name,p),g in zip(parameters,grads):
   if g is None:unused.add(name);g=torch.zeros_like(p)
   if not bool(torch.isfinite(g).all()):raise FloatingPointError('Nonfinite feature VJP')
   parts.append(g.detach().reshape(-1))
  rows.append(torch.cat(parts))
 return torch.stack(rows).double(),sorted(unused)
def block_name(name):
 if name.startswith(PREFIX):return 'core/'+name[len(PREFIX):].split('.')[0]
 prefix='id_backbone.encoder.';short=name[len(prefix):] if name.startswith(prefix) else name
 if short.startswith('sequence_pool.views.'):return '.'.join(short.split('.')[:3])
 return '.'.join(short.split('.')[:2])
def blocks(parameters):
 groups=[];offset=0
 for name,p in parameters:
  key=block_name(name);end=offset+p.numel()
  if groups and groups[-1]['name']==key:groups[-1]['end']=end;groups[-1]['parameters'].append(name)
  else:groups.append(dict(name=key,start=offset,end=end,parameters=[name]))
  offset=end
 return groups
def grams(raw,parameters):
 g=derived(raw);groups=blocks(parameters);result={}
 scales={metric:torch.cat([torch.full((p.numel(),),unit_scale(n,metric),device=g.device,dtype=g.dtype) for n,p in parameters]) for metric in METRICS}
 for metric,scale in scales.items():
  v=g*scale;parts={block['name']:v[:,block['start']:block['end']]@v[:,block['start']:block['end']].T for block in groups};total=sum(parts.values());result[metric]=dict(total=total,blocks=parts)
 return result
