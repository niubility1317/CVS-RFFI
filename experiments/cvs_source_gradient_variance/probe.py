"""No optimizer, auxiliary classifier, U truth, or target access."""
import json,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_feature_multiscale50.predict import source_proof,own_model
from experiments.cvs_equivariant_identity.precision import numerical_context

def objects(model):
 if any(m.training for m in model.modules()):raise ValueError('Read-only observer requires eval')
 head=model.id_backbone.encoder.core.id_backbone.cls_head
 if type(head).__name__!='ResidualPhysicalHead' or head.weight.shape!=(6,160) or head.scale!=30:raise ValueError('Default head differs')
 named=dict(model.named_parameters())
 if any(n not in named or '.cls_head.' in n for n in d.PARAMETERS):raise ValueError('Feature gradient object differs')
 params=[named[n] for n in d.PARAMETERS]
 if [p.numel() for p in params]!=[3456,6912]:raise ValueError('Actual t1/t2 pointwise shapes differ')
 model.requires_grad_(False)
 for p in params:p.requires_grad_(True)
 return params

def select(L,role='L_s'):
 if role!='L_s':raise ValueError('Only approved source L labels are permitted')
 if len(L['ids'])!=6300 or set(L['y'].tolist())!=set(range(6)) or set(L['rx'].tolist())!=set(d.RX) or set(L['day'].tolist())!=set(d.DAYS):raise ValueError('Approved source pool differs')
 order=torch.randperm(6300,generator=torch.Generator().manual_seed(d.SEED+97501));selected=[]
 for tx in range(6):
  for day in d.DAYS:
   for rx in d.RX:
    ix=order[(L['y'][order]==tx)&(L['day'][order]==day)&(L['rx'][order]==rx)]
    if len(ix)!=70:raise ValueError('Source physical cell differs')
    selected.extend(ix[:2].tolist())
 if len(selected)!=180 or len(set(selected))!=180 or len({L['ids'][i] for i in selected})!=180:raise ValueError('Source subset has repeated physical IDs')
 return torch.tensor(selected,dtype=torch.long)

def statistic(G,y,rx,day):
 if G.ndim!=2 or len(G)!=180 or not torch.isfinite(G).all():raise ValueError('Invalid actual gradient matrix')
 G=G.double();residual=G.clone()
 for tx in range(6):
  for dd in d.DAYS:
   keep=(y==tx)&(day==dd)
   if int(keep.sum())!=10:raise ValueError('TX/day balance differs')
   residual[keep]-=G[keep].mean(0)
 result={}
 for mode,values in [('raw',G),('TX_day_centered',residual)]:
  variances=[];records=[]
  for receiver in d.RX:
   keep=rx==receiver
   if int(keep.sum())!=36:raise ValueError('RX balance differs')
   variance=values[keep].var(0,unbiased=True);variances.append(variance)
   records.append(dict(receiver=receiver,count=36,mean_gradient_L2=float(values[keep].mean(0).norm()),variance_trace=float(variance.sum()),coordinate_variance_mean=float(variance.mean()),variance_vector_L2=float(variance.norm())))
  variances=torch.stack(variances);average=variances.mean(0);denominator=float(average.square().mean())
  result[mode]=dict(records=records,coordinate_variance_discrepancy=float((variances-average).square().mean()),normalized_coordinate_variance_discrepancy=float((variances-average).square().mean())/denominator if denominator>0 else None,gradient_elements=G.shape[1],variance_estimator='unbiased within each RX (n=36)',conditional_centering='TX/day mean across all5RX, removes shared TX/day mean only; not causal disentanglement' if mode!='raw' else None)
 return result

def smoke(model):
 params=objects(model);saved={k:v.clone() for k,v in model.state_dict().items()}
 for label in (0,1):
  out=model.forward_identity_only(torch.zeros(1,2,256));loss=F.cross_entropy(out['tx_logits'],torch.tensor([label]),label_smoothing=.01)
  gs=torch.autograd.grad(loss,params,allow_unused=False)
  if out['id_feat_joint'].shape!=(1,160) or not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gs):raise ValueError('No-query derivative smoke failed')
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Smoke mutated model')
 return dict(status='PASS',real_checkpoint_query_read=False,zero_IQ_packet_forwards=2,head_fit=False,optimizer_steps=0)

def measure(c,L):
 d.validate(c);ix=select(L);proof=source_proof(c['source_config']);model=own_model(c['source_config'],'cpu');params=objects(model);saved={k:v.clone() for k,v in model.state_dict().items()};start=time.time();gradients=[[],[]];losses=[];correct=0
 with numerical_context(d.FULL_FP32_POLICY):
  for i in ix.tolist():
   out=model.forward_identity_only(L['x'][i:i+1]);loss=F.cross_entropy(out['tx_logits'],L['y'][i:i+1],label_smoothing=.01);gs=torch.autograd.grad(loss,params,allow_unused=False)
   if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gs):raise ValueError('Nonfinite actual source derivative')
   for target,g in zip(gradients,gs):target.append(g.detach().cpu().flatten())
   losses.append(float(loss.detach()));correct+=int(out['tx_logits'].argmax(1)==L['y'][i])
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Source observer changed model or accumulated gradients')
 gradients=[torch.stack(g) for g in gradients];y,rx,day=(L[k][ix] for k in ('y','rx','day'))
 for tx in range(6):
  for dd in d.DAYS:
   for receiver in d.RX:
    if int(((y==tx)&(day==dd)&(rx==receiver)).sum())!=2:raise ValueError('Actual sample balance differs')
 result=dict(status='VERIFIED',run_id=d.RUN,row_id=c['row_id'],model_seed=d.SEED,source_only=True,source_role='L_s',source_packets=180,source_pool_packets=6300,packet_forwards=180,domain_count=5,domain_packets=36,parameter_names=list(d.PARAMETERS),statistics={n:statistic(g,y,rx,day) for n,g in zip(d.PARAMETERS,gradients)},source_subset_CE_mean=sum(losses)/180,source_subset_correct=correct,parameters_buffers_unchanged=True,accumulated_gradients=False,optimizer_steps=0,head_fit=False,head_updates=0,training_epochs=0,U_IQ_read=False,U_truth_read=False,query_read=False,V_cache_materialized=True,V_forward=False,checkpoint_budget=dict(epochs=50,updates=11100),checkpoint_path=str(proof/'final_ssdg.pth'),seconds=time.time()-start,scope='Fixed first model seed, 180 balanced source packets, actual t1/t2 feature-weight derivatives of default CE; no training gradients, model-seed uncertainty, Fishr efficacy, causal RX invariance or new-TX transfer claim')
 return result,dict(gradients=gradients,y=y,rx=rx,day=day,source_indices=ix,source_physical_ids=[L['ids'][i] for i in ix.tolist()],parameter_names=list(d.PARAMETERS),config=c,losses=losses)

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2)
 if d.BASE.exists():raise FileExistsError('Existing immutable source variance root; reconcile instead of rerun')
 d.BASE.mkdir(parents=True);c=d.config(d.rows()[0]);out=Path(c['output_root']);out.mkdir()
 try:
  source_proof(c['source_config']);L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V
  result,raw=measure(c,L);torch.save(raw,out/'gradients.pt');d.write(out/'results.json',result)
  # Recount the saved derivative artifact, without another IQ forward.
  saved=torch.load(out/'gradients.pt',map_location='cpu',weights_only=False)
  if saved['config']!=c or saved['parameter_names']!=list(d.PARAMETERS):raise ValueError('Saved derivative context differs')
  recounted={n:statistic(g,saved['y'],saved['rx'],saved['day']) for n,g in zip(d.PARAMETERS,saved['gradients'])}
  if recounted!=result['statistics']:raise ValueError('Saved gradient variance recount differs')
  import subprocess
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip() if (d.ROOT/'.git').exists() else (d.ROOT/'release_commit.txt').read_text(encoding='utf-8').strip()
  d.write(out/'resolved_config.json',dict(config=c,execution_commit=commit,CUDA_VISIBLE_DEVICES='',parameters_buffers_unchanged=True))
  done=dict(status='ANALYZED',models=1,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0,source_packets=180,source_gradient_recount='VERIFIED')
  d.write(out/'completion.json',done);d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True));raise
if __name__=='__main__':main()
