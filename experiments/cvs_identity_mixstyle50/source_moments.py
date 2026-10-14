import json,time
from pathlib import Path
import torch
from . import design as d
from .predict import source_proof,own_model
from experiments.cvs_e20_budget_fishr.multilevel import weak_mixstyle_u
from experiments.cvs_equivariant_identity.precision import numerical_context

LAYERS=('t1','t2')
def descriptor(x):
 if x.ndim!=3 or not x.is_floating_point() or not torch.isfinite(x).all():raise ValueError('Invalid actual layer map')
 return torch.cat((x.mean(-1),torch.sqrt(x.var(-1,unbiased=False)+1e-6)),1)

def metadata(y,rx,day,ids):
 if y.dtype!=torch.long or rx.dtype!=torch.long or day.dtype!=torch.long or y.ndim!=1 or not len(y):raise ValueError('Invalid L source metadata')
 if any(len(v)!=len(y) for v in (rx,day,ids)) or not set(y.tolist())<=set(range(6)) or not set(rx.tolist())<={1,3,4,6,8} or not set(day.tolist())<={1,2,3} or len(set(ids))!=len(ids):raise ValueError('Invalid actual source identity/physical IDs')

def mix(x,y,rx,day,ids,seed):
 metadata(y,rx,day,ids);out,detail=weak_mixstyle_u(x,y,rx,day,ids,seed,p=.5,alpha=.1,eta=1.)
 for i,j in enumerate(detail['donors']):
  if i!=j and (y[i]!=y[j] or day[i]!=day[j] or rx[i]==rx[j] or ids[i]==ids[j]):raise ValueError('Unprotected style donor')
 return out,detail

def moments(values,L):
 total=((values-values.mean(0)).square().sum()).double()
 # Conditional source RX shifts are measured within the same TX/day.
 rx_ss=values.new_zeros(());tx_ss=values.new_zeros(())
 for tx in range(6):
  for day in (1,2,3):
   keep=(L['y']==tx)&(L['day']==day);center=values[keep].mean(0)
   for rx in (1,3,4,6,8):
    cell=keep&(L['rx']==rx);rx_ss+=int(cell.sum())*(values[cell].mean(0)-center).square().sum()
 for rx in (1,3,4,6,8):
  for day in (1,2,3):
   keep=(L['rx']==rx)&(L['day']==day);center=values[keep].mean(0)
   for tx in range(6):
    cell=keep&(L['y']==tx);tx_ss+=int(cell.sum())*(values[cell].mean(0)-center).square().sum()
 if total<=0 or not torch.isfinite(total):raise ValueError('Degenerate actual source moment descriptor')
 return dict(dimension=values.shape[1],source_packets=len(values),total_centered_SS=float(total),RX_within_TX_day_SS=float(rx_ss),TX_within_RX_day_SS=float(tx_ss),RX_within_TX_day_fraction=float(rx_ss/total),TX_within_RX_day_fraction=float(tx_ss/total),interpretation='Descriptive conditional mean/std variation; not RX shortcut, causal nuisance decomposition, feature rank or transfer proof')

@torch.no_grad()
def run(c):
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 d.validate(c);source_proof(c);torch.set_num_threads(2);start=time.time()
 L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V
 if len(L['ids'])!=6300:raise ValueError('Only complete approved L6300')
 model=own_model(c,'cpu').eval().requires_grad_(False);saved={k:v.clone() for k,v in model.state_dict().items()};b=model.id_backbone.encoder.core.id_backbone
 order=torch.randperm(6300,generator=torch.Generator().manual_seed(c['model_seed']+95001));packed={n:[] for n in LAYERS};labels=[];receivers=[];days=[];batches=[];clean_ok=styled_ok=changed=correct_to_wrong=wrong_to_correct=0
 with numerical_context(d.FULL_FP32_POLICY):
  for start_index in range(0,6300,128):
   ix=order[start_index:start_index+128];x=L['x'][ix];y=L['y'][ix];rx=L['rx'][ix];day=L['day'][ix];ids=[L['ids'][i] for i in ix.tolist()];metadata(y,rx,day,ids)
   captured={};hooks=[getattr(b,n).register_forward_hook(lambda m,i,o,n=n:captured.update({n:o.detach().clone()})) for n in LAYERS]
   try:clean=model.forward_identity_only(x)
   finally:
    for h in hooks:h.remove()
   if set(captured)!=set(LAYERS):raise ValueError('Actual source layers bypassed')
   for n in LAYERS:packed[n].append(descriptor(captured[n]))
   labels.append(y);receivers.append(rx);days.append(day);details={};hooks=[]
   def styled_hook(name):
    def hook(module,args,out):
     mixed,detail=mix(out,y,rx,day,ids,c['model_seed']+(start_index//128+1)*701+LAYERS.index(name)*1709);details[name]={k:v for k,v in detail.items() if k!='donors'};return mixed
    return hook
   try:
    for n in LAYERS:hooks.append(getattr(b,n).register_forward_hook(styled_hook(n)))
    styled=model.forward_identity_only(x)
   finally:
    for h in hooks:h.remove()
   cp=clean['tx_logits'].argmax(1);sp=styled['tx_logits'].argmax(1);cz=clean['id_feat_joint'];sz=styled['id_feat_joint']
   if cz.shape!=sz.shape or cz.shape[1]!=160 or not torch.isfinite(sz).all():raise ValueError('Actual joint160 differs')
   clean_ok+=int((cp==y).sum());styled_ok+=int((sp==y).sum());changed+=int((cp!=sp).sum());correct_to_wrong+=int(((cp==y)&(sp!=y)).sum());wrong_to_correct+=int(((cp!=y)&(sp==y)).sum())
   batches.append(dict(batch=start_index//128+1,count=len(ix),layers=details,source_joint160_L2_mean=float((sz-cz).norm(dim=1).mean()),source_joint160_cosine=float(torch.nn.functional.cosine_similarity(cz,sz,dim=1,eps=1e-4).mean())))
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Source observer changed actual state')
 grouping=dict(y=torch.cat(labels),rx=torch.cat(receivers),day=torch.cat(days))
 return dict(status='VERIFIED',run_id=d.RUN,row_id=c['row_id'],arm=c['arm'],model_seed=c['model_seed'],models=1,source_only=True,source_role='L_s',source_packets=6300,source_batches=50,packet_forwards=12600,source_count_checks=dict(clean_correct=clean_ok,styled_correct=styled_ok,prediction_changes=changed,correct_to_wrong=correct_to_wrong,wrong_to_correct=wrong_to_correct),moments={n:moments(torch.cat(packed[n]),grouping) for n in LAYERS},batches=batches,model_parameters_buffers_unchanged=True,accumulated_gradients=False,optimizer_steps=0,head_fit=False,U_IQ_read=False,U_truth_read=False,query_read=False,V_cache_materialized=True,V_forward=False,forced_feature_hook_in_eval=True,p=.5,alpha=.1,strength=1.,teacher_forward=False,seconds=time.time()-start,interpretation='Source descriptive moments and a fixed frozen feature intervention. No new training/target result; augmented source error does not prove training efficacy or disqualify a learnable augmentation automatically.')


def completed_source_rows():
 rows=d.rows()
 for row in rows:
  c=d.config(row);source_proof(c)
 return rows

def complete():
 out=d.BASE/'source_style_mechanism.json'
 if out.exists():raise FileExistsError('Existing immutable same-row source mechanism result; reuse it')
 rows=completed_source_rows();results=[]
 for row in rows:
  c=d.config(row)
  if row['arm']==d.BASELINE_ARM:
   # Reuse the already complete identical source observer; never rerun it.
   old=d.BASE.parent/'20261014-phase1-source-style-profile-manysig-s01-m1-r01'/'SOURCE_STYLE-s2026092701'
   resolved=d.read(old/'resolved_config.json');previous=d.read(old/'results.json')
   from experiments.cvs_neighbor_pair50.predict import source_proof as previous_proof
   if previous_proof(resolved['config']['source_config'])!=source_proof(c) or resolved['config']['shuffle_seed']!=c['model_seed']+95001 or previous['model_seed']!=c['model_seed'] or previous['status']!='VERIFIED' or previous['source_packets']!=6300 or previous['packet_forwards']!=12600 or not previous['model_parameters_buffers_unchanged'] or previous['head_fit'] or previous['query_read']:raise ValueError('Existing identical source profile not reusable')
   for batch in previous['batches']:
    for detail in batch['layers'].values():detail.pop('donors',None)
   previous.update(original_run_id=previous['run_id'],run_id=d.RUN,row_id=row['row_id'],arm=row['arm'],source_profile_reused=True,source_observer_execution_commit=resolved['execution_commit'],new_packet_forwards=0)
   results.append(previous)
  else:
   value=run(c);value.update(source_profile_reused=False,new_packet_forwards=value['packet_forwards']);results.append(value)
 baseline=next(v for v in results if v['arm']=='MATCHED_CONTROL');candidate=next(v for v in results if v['arm']==d.PRIMARY)
 comparisons={name:dict(RX_conditional_fraction_difference=candidate['moments'][name]['RX_within_TX_day_fraction']-baseline['moments'][name]['RX_within_TX_day_fraction'],TX_conditional_fraction_difference=candidate['moments'][name]['TX_within_RX_day_fraction']-baseline['moments'][name]['TX_within_RX_day_fraction']) for name in LAYERS}
 result=dict(status='VERIFIED',run_id=d.RUN,models=len(results),source_packets_per_model=6300,source_batches_per_model=50,packet_forwards=sum(v['new_packet_forwards'] for v in results),reused_profile_packet_forwards=sum(v['packet_forwards'] for v in results if v['source_profile_reused']),rows=results,comparisons_to_matched_control=comparisons,head_fit=False,head_updates=0,optimizer_steps=0,query_read=False,U_truth_read=False,U_IQ_read=False,V_cache_materialized=True,V_forward=False,scope='Post-training same-row source descriptive mechanism; all source RX were seen in backbone, no held-out RX or new TX claim; forced eval intervention is not training efficacy. Does not change any registered threshold, selection or candidate.')
 d.write(out,result);return result
if __name__=='__main__':print(json.dumps(complete()))
