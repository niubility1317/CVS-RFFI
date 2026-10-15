import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .objective import symmetric_kl
from experiments.cvs_source_soft_relation.audit import prepare
from experiments.cvs_source_self_challenge.audit import source_model
from experiments.cvs_source_layer_order.probe import cohorts,derivative
from experiments.cvs_source_self_challenge.objective import gradient_stats
from experiments.cvs_signature_stats50.predict import source_proof
from experiments.cvs_backbone_fishr50.objective import validate_layout
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_equivariant_identity.precision import numerical_context

def measure_batch(model,x,y,seed):
 params=prepare(model)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed)
  strong=native_modules()._strong_augment(x,.015)
  a=model.forward_identity_only(strong);b=model.forward_identity_only(strong)
  ce=.5*(F.cross_entropy(a['tx_logits'],y,label_smoothing=.01)+F.cross_entropy(b['tx_logits'],y,label_smoothing=.01))
  kl=symmetric_kl(a['tx_logits'],b['tx_logits'])
  vectors={name:derivative(loss,params) for name,loss in (('PAIRED_CE',ce),('SYMMETRIC_KL',kl))}
  model.eval()
  with torch.no_grad():reference=model.forward_identity_only(strong)
  return dict(logits1=a['tx_logits'].detach().cpu(),logits2=b['tx_logits'].detach().cpu(),eval_logits=reference['tx_logits'].cpu(),
   features1=a['id_feat_joint'].detach().cpu(),features2=b['id_feat_joint'].detach().cpu(),eval_features=reference['id_feat_joint'].cpu(),
   losses=dict(PAIRED_CE=float(ce.detach()),SYMMETRIC_KL=float(kl.detach())),
   vectors={k:[g.cpu() for g in v] for k,v in vectors.items()})

def summarize(raw):
 layers=[];rows=[]
 for k,name in enumerate(d.PARAMETERS):
  ce=torch.stack([r['vectors']['PAIRED_CE'][k].double() for r in raw]).mean(0)
  aux=torch.stack([r['vectors']['SYMMETRIC_KL'][k].double() for r in raw]).mean(0)
  layers.append(dict(parameter=name,cohorts=len(raw),**gradient_stats(ce,aux),
   KL_nonzero_cohorts=sum(float(r['vectors']['SYMMETRIC_KL'][k].norm())>1e-12 for r in raw)))
 for r in raw:
  a,b,e=r['logits1'],r['logits2'],r['eval_logits'];y=r['y'];z1,z2=r['features1'],r['features2']
  rows.append(dict(cohort=r['cohort'],packets=len(y),losses=r['losses'],stochastic1_correct=int((a.argmax(1)==y).sum()),
   stochastic2_correct=int((b.argmax(1)==y).sum()),eval_correct=int((e.argmax(1)==y).sum()),
   stochastic_disagreements=int((a.argmax(1)!=b.argmax(1)).sum()),
   joint160_pair_cosine_mean=float(F.cosine_similarity(z1.double(),z2.double(),dim=1).mean()),
   joint160_pair_L2_mean=float((z1.double()-z2.double()).norm(dim=1).mean())))
 return dict(layers=layers,cohorts=rows)

def qualification(summary):
 body=summary['layers'][:3]
 passed=len(summary['cohorts'])==5 and all(r['losses']['SYMMETRIC_KL']>0 for r in summary['cohorts']) and all(r['KL_nonzero_cohorts']==5 and r['cosine'] is not None and r['cosine']>=0 for r in body)
 return dict(status='PASS' if passed else 'FAIL',fixed_rules=d.QUALIFICATION,source_only=True,
  no_automatic_training_release=True,target_metrics_used=False)

def recount(root):
 p=root/d.rows()[0]['row_id'];report=d.read(p/'analysis.json');d.validate(report['config'])
 raw=torch.load(p/'actual_source_gradients.pt',map_location='cpu',weights_only=False)
 if len(raw)!=5 or [r['cohort'] for r in raw]!=list(range(5)):raise ValueError('Fixed source cohorts missing')
 ids=[]
 for r in raw:
  validate_layout(r['y'],r['rx'],r['day'],r['source_physical_ids'],'L_s');ids.extend(r['source_physical_ids'])
  ce=.5*(F.cross_entropy(r['logits1'],r['y'],label_smoothing=.01)+F.cross_entropy(r['logits2'],r['y'],label_smoothing=.01))
  kl=symmetric_kl(r['logits1'],r['logits2'])
  if abs(float(ce)-r['losses']['PAIRED_CE'])>1e-6 or abs(float(kl)-r['losses']['SYMMETRIC_KL'])>1e-6:raise ValueError('Saved same-input CE/KL recount differs')
  for name in ('PAIRED_CE','SYMMETRIC_KL'):
   if [g.numel() for g in r['vectors'][name]]!=[3456,6912,576,960] or any(not torch.isfinite(g).all() for g in r['vectors'][name]):raise ValueError('Actual parameter vector shape/finite values differ')
  if any(r[k].shape!=(180,160) or not torch.isfinite(r[k]).all() for k in ('features1','features2','eval_features')):raise ValueError('Actual joint160 witness missing')
 summary=summarize(raw)
 if len(ids)!=900 or len(set(ids))!=900 or summary!=report['summary'] or qualification(summary)!=report['qualification']:raise ValueError('Saved source coverage/summary/qualification differs')
 return dict(status='VERIFIED',source_packets=900,model_forward_packets=2700,gradient_batches=5,parameter_layer_records=20,saved_vectors=40,
  new_forwards=0,new_gradient_calls=0,head_fit_updates=0,optimizer_steps=0,U_IQ_read=False,U_truth_read=False,query_read=False)

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2);c=d.config(d.rows()[0]);d.validate(c)
 spec=d.expand_registration(d.read(d.ROOT/'registration/experiment.json'))
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[c]:raise ValueError('Full source registration differs')
 if d.BASE.exists():raise FileExistsError('Immutable output exists; inspect original run')
 d.BASE.mkdir(parents=True);p=Path(c['output_root']);p.mkdir();started=time.time()
 try:
  source_proof(c['source_config'],design=d.parent)
  L,U,V,contract=load_source(nd.config(nd.rows()[0]))
  if U is not None:raise ValueError('U tensor materialized')
  del V
  old=cohorts(L);indices=[torch.cat((old[2*j],old[2*j+1])) for j in range(5)]
  with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
   model=source_model();before={k:v.clone() for k,v in model.state_dict().items()};raw=[]
   d.write(p/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_roles='EXACT_MATCH',
    checkpoint_parent=d.parent.RUN,source_only=True,query_read=False,head_fit_updates=0))
   for j,ix in enumerate(indices):
    y,rx,day=(L[k][ix] for k in ('y','rx','day'));ids=[L['ids'][i] for i in ix.tolist()];validate_layout(y,rx,day,ids,'L_s')
    result=measure_batch(model,L['x'][ix],y,c['strong_rng_seed']+j)
    raw.append(dict(cohort=j,y=y.cpu(),rx=rx.cpu(),day=day.cpu(),source_physical_ids=ids,**result))
    print(json.dumps(dict(source_cohorts=j+1,source_packets=(j+1)*180,model_forward_packets=(j+1)*540)),flush=True)
   if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(v.grad is not None for v in model.parameters()):raise ValueError('Source diagnostic mutated weights/buffers or accumulated gradients')
  summary=summarize(raw);report=dict(status='ANALYZED',config=c,summary=summary,qualification=qualification(summary),source_packets=900,
   model_forward_packets=2700,model_parameters_buffers_unchanged=True,head_fit_updates=0,optimizer_steps=0,U_IQ_read=False,U_truth_read=False,
   V_cache_materialized_then_discarded=True,V_forward=False,V_gradient=False,query_read=False,seconds=time.time()-started,limitation=c['limitation'])
  torch.save(raw,p/'actual_source_gradients.pt');d.write(p/'analysis.json',report)
  subprocess.run([sys.executable,'-u','-m','experiments.cvs_source_rdrop.audit','--recount',str(d.BASE)],check=True)
  d.write(d.BASE/'source_qualification.json',report['qualification']);d.write(d.BASE/'analysis.json',report)
  done=dict(status='ANALYZED',models=1,source_packets=900,forward_packets=2700,gradient_batches=5,layer_records=20,saved_vectors=40,
   source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)
  d.write(p/'completion.json',done);d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as exc:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--recount',type=Path);args=parser.parse_args()
 if args.recount:
  torch.set_num_threads(2);value=recount(args.recount);d.write(args.recount/'independent_artifact_recount.json',value);print(json.dumps(value))
 else:main()
