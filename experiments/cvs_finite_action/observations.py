"""All lawful L/U actual maps and finite differences; no U TX or target."""
import math,torch
from . import design as d
from .operators import apply,quality,rms
from experiments.cvs_crossrx_relation.core import capture
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
def load(c):
 from experiments.cvs_nocfo_foundation.source import load_source
 L,unused,V,contract=load_source(c['source_data_config']);del V,unused
 ec=d.e.config(next(r for r in d.e.rows() if r['arm']=='SSL' and r['model_seed']==c['model_seed']))
 U=torch.load(ec['unlabeled_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
 if len(U['ids'])!=56700 or set(L['ids'])&set(U['ids']):raise ValueError('Legal L/U role differs')
 L['role']='L_s';U['role']='U_s'
 return L,U,contract
@torch.no_grad()
def readout(model,x):
 with capture(model,d.LEVELS) as features:out=model.forward_identity_only(x)
 features['id_feat_joint']=out['id_feat_joint']
 if set(features)!=set(d.LEVELS) or any(v.requires_grad or not torch.isfinite(v).all() for v in features.values()):raise ValueError('Actual finite frozen features required')
 return {k:v.detach().cpu() for k,v in features.items()},out['tx_logits'].detach().cpu()
def extract(model,data,c,device,role,progress=None):
 if role not in ('L_s','U_s') or data.get('role')!=role:raise ValueError('Lawful source role required')
 if role=='U_s' and any(k in data for k in ('y','label','labels','tx','TX','truth')):raise ValueError('U TX is forbidden')
 n=len(data['ids']);batch=c['batch_size'];feature_chunks={k:[] for k in d.LEVELS};qs=[];forwards=occurrences=0;clean_correct=0
 for start in range(0,n,batch):
  x=data['x'][start:start+batch].to(device);h,logits=readout(model,x)
  for k in d.LEVELS:feature_chunks[k].append(h[k])
  qs.append(quality(x).cpu());forwards+=1;occurrences+=len(x)
  if role=='L_s':clean_correct+=int((logits.argmax(1)==data['y'][start:start+batch]).sum())
 h={k:torch.cat(v) for k,v in feature_chunks.items()};q=torch.cat(qs);targets={};metrics=[]
 for kind in d.KINDS:
  points=list(range(len(c['points'][kind]))) if role=='L_s' else c['U_points'][kind]
  means={k:[] for k in d.LEVELS};variances={k:[] for k in d.LEVELS}
  for p in points:
   a=c['points'][kind][p];repeats=c['noise_repeats'] if kind=='noise' else 1;arrays={k:[] for k in d.LEVELS};vs={k:[] for k in d.LEVELS};correct=count=0;rmsratios=[];relative_changes=[];quality_delta=[]
   for start in range(0,n,batch):
    x=data['x'][start:start+batch].to(device);deltas={k:[] for k in d.LEVELS}
    for repeat in range(repeats):
     salt=0 if role=='L_s' else 700000003;seed=c['model_seed']+salt+start*8191+p*104729+repeat*65537+list(d.KINDS).index(kind)*10000019
     y=apply(x,kind,a,seed);out,logits=readout(model,y);forwards+=1;occurrences+=len(y);count+=len(y)
     for k in d.LEVELS:deltas[k].append(out[k]-h[k][start:start+batch])
     if role=='L_s':correct+=int((logits.argmax(1)==data['y'][start:start+batch]).sum())
     rmsratios.extend((rms(y)/rms(x)).squeeze(1).cpu().tolist());relative_changes.extend(((y-x).square().sum((1,2))/x.square().sum((1,2)).clamp_min(1e-12)).cpu().tolist());quality_delta.append((quality(y)-quality(x)).cpu())
    for k in d.LEVELS:
     value=torch.stack(deltas[k],1);arrays[k].append(value.mean(1));vs[k].append(value.var(1,unbiased=False))
   for k in d.LEVELS:means[k].append(torch.cat(arrays[k]));variances[k].append(torch.cat(vs[k]))
   metric=dict(kind=kind,point=p,a=a,repeats=repeats,packet_occurrences=count,source_role=role,head_accuracy=correct/count if role=='L_s' else None,RMS_ratio_mean=sum(rmsratios)/len(rmsratios),RMS_ratio_min=min(rmsratios),RMS_ratio_max=max(rmsratios),relative_IQ_change_energy=sum(relative_changes)/len(relative_changes),quality_delta_mean=torch.cat(quality_delta).mean(0).tolist(),digital_controlled_operation=True,real_RX_counterfactual=False)
   metrics.append(metric)
   if progress:progress(metric)
  targets[kind]=dict(points=points,mean={k:torch.stack(v,1) for k,v in means.items()},variance={k:torch.stack(v,1) for k,v in variances.items()})
 return dict(role=role,ids=list(data['ids']),h=h,q=q,targets=targets,operation_metrics=metrics,counts=dict(model_forwards=forwards,packet_occurrences=occurrences),clean_head_accuracy=clean_correct/n if role=='L_s' else None,source_only=True,U_truth=False)
