import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .objective import transform,metrics,cosine
from experiments.cvs_source_layer_order.probe import cohorts
from experiments.cvs_backbone_fishr50.objective import validate_layout
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_signature_stats50.predict import source_proof,own_model

def load_model(row):
 parent=d.parent_design(row);c=d.source_config(row);source_proof(c,design=parent)
 if row['model_seed']==d.SEEDS[0]:
  from experiments.cvs_receiver_dro50.model import build
 else:
  from experiments.cvs_receiver_dro_seeds.model import build
 return own_model(c,'cpu',design=parent,build_fn=build)

def prepare(model):
 model.train().requires_grad_(False)
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.id_backbone.modules()):raise ValueError('Source probe would change BatchNorm state')
 named=dict(model.named_parameters());pairs=[(n,p) for n,p in named.items() if n.startswith('id_backbone.') and n!=d.CLASS_W]
 if sum(p.numel() for n,p in pairs)!=d.FEATURE_COUNT or d.CLASS_W not in named or named[d.CLASS_W].numel()!=960:raise ValueError('Actual deployed feature/class boundary differs')
 if not any('.cls_head.base_norm.' in n for n,p in pairs) or not any('.cls_head.pa_norm.' in n for n,p in pairs) or not any('.cls_head.gain' in n for n,p in pairs):raise ValueError('Feature normalization/gain wrongly removed')
 for n,p in pairs:p.requires_grad_(True)
 head=named[d.CLASS_W];head.requires_grad_(True)
 return pairs,head

def derivative(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
 return [torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1) for p,g in zip(params,gs)]

def measure_batch(model,x,y,rx,seed,random_seed):
 pairs,head=prepare(model);params=[p for n,p in pairs];sizes=[p.numel() for p in params]
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);strong=native_modules()._strong_augment(x,.015);out=model.forward_identity_only(strong)
  ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);all_ce=derivative(ce,params+[head]);g=torch.cat(all_ce[:-1]).cpu()
  losses=[F.cross_entropy(out['tx_logits'][rx==r],y[rx==r],label_smoothing=.01) for r in d.RX]
  per_rx=torch.stack([torch.cat(derivative(loss,params)).cpu() for loss in losses]);v=transform(per_rx,g,sizes,random_seed=random_seed)
 return dict(logits=out['tx_logits'].detach().cpu(),features=out['id_feat_joint'].detach().cpu(),loss=float(ce.detach()),receiver_losses=[float(loss.detach()) for loss in losses],parameters=[dict(name=n,shape=list(p.shape),elements=p.numel()) for n,p in pairs],per_rx=per_rx,ce=g,head_ce=all_ce[-1].cpu(),**v)

def summarize(raw):
 original=torch.stack([x['ce'].double() for x in raw]).mean(0);common=torch.stack([x['masked'].double() for x in raw]).mean(0);random=torch.stack([x['random'].double() for x in raw]).mean(0);groups=torch.stack([x['per_rx'].double() for x in raw]).mean(0)
 mc=[cosine(common,g) for g in groups];rc=[cosine(random,g) for g in groups]
 global_stats=dict(original_norm=float(original.norm()),masked_norm=float(common.norm()),random_norm=float(random.norm()),retained_gradient_norm_ratio=float(common.norm()/original.norm()) if float(original.norm()) else 0.,masked_CE_cosine=cosine(common,original),random_CE_cosine=cosine(random,original),masked_RX_cosines=mc,random_RX_cosines=rc,masked_min_RX_cosine=min(mc) if all(x is not None for x in mc) else None,random_min_RX_cosine=min(rc) if all(x is not None for x in rc) else None)
 batch_stats=[];body=[]
 for x in raw:batch_stats.append(dict(cohort=x['cohort'],source_packets=180,source_loss=x['loss'],receiver_losses=x['receiver_losses'],**metrics(x['per_rx'],x['ce'],x)))
 offset=0;body_offsets={}
 for p in raw[0]['parameters']:
  if p['name'] in d.BODY:body_offsets[p['name']]=(offset,offset+p['elements'])
  offset+=p['elements']
 for name in d.BODY:
  lo,hi=body_offsets[name];ce=torch.stack([x['ce'][lo:hi].double() for x in raw]).mean(0);mm=torch.stack([x['masked'][lo:hi].double() for x in raw]).mean(0)
  body.append(dict(parameter=name,elements=hi-lo,masked_nonzero_cohorts=sum(float(x['masked'][lo:hi].double().norm())>1e-12 for x in raw),masked_norm=float(mm.norm()),CE_cosine=cosine(ce,mm),changed_direction_cohorts=sum(not torch.equal(x['masked'][lo:hi],x['ce'][lo:hi]) for x in raw)))
 return dict(global_feature_gradient=global_stats,cohorts=batch_stats,body=body,feature_parameters=d.FEATURE_COUNT,class_W_masked=False,proposed_class_W_gradient='Original CE gradient, excluded from mask; no update performed')

def qualification(s):
 g=s['global_feature_gradient'];q=d.QUALIFICATION
 ok=(len(s['cohorts'])==5 and g['masked_min_RX_cosine'] is not None and g['random_min_RX_cosine'] is not None and g['masked_min_RX_cosine']>=q['minimum_global_masked_CE_cosine_each_RX'] and g['masked_min_RX_cosine']>g['random_min_RX_cosine'] and g['retained_gradient_norm_ratio']>=q['global_retained_gradient_norm_ratio_min'] and all(q['eligible_keep_fraction_each_cohort_min']<=b['eligible_keep_fraction']<=q['eligible_keep_fraction_each_cohort_max'] for b in s['cohorts']) and all(b['masked_nonzero_cohorts']==5 for b in s['body']))
 return dict(status='PASS' if ok else 'FAIL',fixed_rules=q,source_only=True,target_metrics_used=False,no_automatic_training_release=True)

def recount(root,row):
 p=root/row['row_id'];report=d.read(p/'analysis.json');d.validate(report['config'])
 if report['config']!=d.config(row):raise ValueError('Source attribution differs')
 raw=torch.load(p/'actual_source_gradients.pt',map_location='cpu',weights_only=False)
 if len(raw)!=5 or [x['cohort'] for x in raw]!=list(range(5)):raise ValueError('Source cohorts incomplete')
 ids=[];partition=raw[0]['parameters']
 if sum(p['elements'] for p in partition)!=d.FEATURE_COUNT or len({p['name'] for p in partition})!=len(partition) or any(not p['name'].startswith('id_backbone.') or p['name']==d.CLASS_W for p in partition):raise ValueError('Saved feature partition invalid')
 for x in raw:
  validate_layout(x['y'],x['rx'],x['day'],x['source_physical_ids'],'L_s');ids.extend(x['source_physical_ids'])
  if x['parameters']!=partition or x['per_rx'].shape!=(5,d.FEATURE_COUNT) or x['ce'].shape!=(d.FEATURE_COUNT,) or x['head_ce'].shape!=(960,) or not torch.isfinite(x['head_ce']).all():raise ValueError('Full feature/defaultW derivatives incomplete')
  if not torch.allclose(x['ce'],x['per_rx'].mean(0),atol=1e-6,rtol=1e-4):raise ValueError('Balanced physical cohort mean CE gradient differs')
  value=transform(x['per_rx'],x['ce'],[p['elements'] for p in partition],random_seed=report['config']['random_mask_rng_seed']+x['cohort'])
  if any(not torch.equal(x[k],v) for k,v in value.items()):raise ValueError('Saved mask or gradient transform differs')
  offset=0
  for param in partition:
   n=param['elements']
   if int(x['mask'][offset:offset+n].sum())!=int(x['random_mask'][offset:offset+n].sum()):raise ValueError('Random sparsity mismatch')
   offset+=n
  ce=F.cross_entropy(x['logits'],x['y'],label_smoothing=.01);rr=[float(F.cross_entropy(x['logits'][x['rx']==r],x['y'][x['rx']==r],label_smoothing=.01)) for r in d.RX]
  if abs(float(ce)-x['loss'])>1e-6 or any(abs(a-b)>1e-6 for a,b in zip(rr,x['receiver_losses'])) or x['features'].shape!=(180,160) or not torch.isfinite(x['features']).all():raise ValueError('Saved CE or deployed features invalid')
 s=summarize(raw)
 if len(ids)!=900 or len(set(ids))!=900 or s!=report['summary'] or qualification(s)!=report['qualification']:raise ValueError('Source summary/qualification or coverage differs')
 return dict(status='VERIFIED',source_packets=900,model_forward_packets=900,gradient_calls=30,selected_layer_records=20,saved_full_feature_or_W_vectors=45,feature_parameters=d.FEATURE_COUNT,default_W_excluded_only=True,new_forwards=0,new_gradients=0,optimizer_steps=0,head_fit_updates=0,query_read=False)

def measure(row,L):
 c=d.config(row);d.validate(c);p=Path(c['output_root']);p.mkdir();started=time.time();ixs=cohorts(L);indices=[torch.cat((ixs[2*j],ixs[2*j+1])) for j in range(5)];raw=[]
 with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
  model=load_model(row);before={k:v.clone() for k,v in model.state_dict().items()}
  d.write(p/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_roles='EXACT_MATCH',source_only=True,query_read=False,head_fit_updates=0))
  for j,ix in enumerate(indices):
   y,rx,day=(L[k][ix] for k in ('y','rx','day'));ids=[L['ids'][i] for i in ix.tolist()];validate_layout(y,rx,day,ids,'L_s')
   raw.append(dict(cohort=j,y=y.cpu(),rx=rx.cpu(),day=day.cpu(),source_physical_ids=ids,**measure_batch(model,L['x'][ix],y,rx,c['strong_rng_seed']+j,c['random_mask_rng_seed']+j)))
   print(json.dumps(dict(row_id=row['row_id'],source_cohorts=j+1,source_packets=(j+1)*180,model_forward_packets=(j+1)*180)),flush=True)
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Read-only probe changed model state or accumulated .grad')
 s=summarize(raw);report=dict(status='ANALYZED',config=c,summary=s,qualification=qualification(s),source_packets=900,model_forward_packets=900,model_parameters_buffers_unchanged=True,head_fit_updates=0,optimizer_steps=0,U_IQ_read=False,U_truth_read=False,V_cache_materialized_then_discarded=True,V_forward=False,query_read=False,seconds=time.time()-started,limitation=c['limitation'])
 torch.save(raw,p/'actual_source_gradients.pt');d.write(p/'analysis.json',report);return report

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2);spec=d.expand_registration(d.read(d.ROOT/'registration/experiment.json'))
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered full source plan differs')
 if d.BASE.exists():raise FileExistsError('Immutable run exists; inspect instead of resubmitting')
 d.BASE.mkdir(parents=True)
 try:
  L,U,V,contract=load_source(nd.config(nd.rows()[0]))
  if U is not None:raise ValueError('U tensors materialized')
  del V
  reports=[];ids0=None
  for row in d.rows():
   if reports and reports[0]['qualification']['status']!='PASS':break
   report=measure(row,L);reports.append(report)
   subprocess.run([sys.executable,'-u','-m','experiments.cvs_source_rx_consensus.audit','--recount',str(d.BASE),'--row',row['row_id']],check=True)
   raw=torch.load(d.BASE/row['row_id']/'actual_source_gradients.pt',map_location='cpu',weights_only=False);ids=[i for x in raw for i in x['source_physical_ids']]
   if ids0 is not None and ids!=ids0:raise ValueError('Different ordered source cohorts across models')
   ids0=ids
   d.write(d.BASE/row['row_id']/'completion.json',dict(status='ANALYZED',source_packets=900,forward_packets=900,optimizer_steps=0,head_updates=0))
  q=dict(status='PASS' if len(reports)==3 and all(x['qualification']['status']=='PASS' for x in reports) else 'FAIL',fixed_rules=d.QUALIFICATION,source_only=True,target_metrics_used=False,no_automatic_training_release=True)
  skipped=[r['row_id'] for r in d.rows()[len(reports):]];result=dict(status='ANALYZED',run_id=d.RUN,models=len(reports),model_seeds=[x['config']['model_seed'] for x in reports],rows=reports,qualification=q,conditionally_skipped_rows=skipped,source_model_observations=900*len(reports),unique_source_physical_ids=900,model_forward_packets=900*len(reports),same_ordered_source_cohorts=True,optimizer_steps=0,head_fit_updates=0,U_IQ_read=False,U_truth_read=False,V_forward=False,query_read=False,goal84_tested=False)
  d.write(d.BASE/'analysis.json',result);d.write(d.BASE/'source_qualification.json',q)
  done=dict(status='ANALYZED',models=len(reports),source_packets=900*len(reports),forward_packets=900*len(reports),gradient_batches=5*len(reports),layer_records=20*len(reports),saved_vectors=45*len(reports),source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0,conditionally_skipped_rows=skipped)
  d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as exc:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recount',type=Path);p.add_argument('--row');a=p.parse_args()
 if a.recount:
  row=next((r for r in d.rows() if r['row_id']==a.row),None)
  if row is None:raise ValueError('Exact registered row required')
  torch.set_num_threads(2);value=recount(a.recount,row);d.write(a.recount/row['row_id']/'independent_artifact_recount.json',value);print(json.dumps(value))
 elif a.row:p.error('--row requires --recount')
 else:main()
