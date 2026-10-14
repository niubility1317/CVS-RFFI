import argparse,copy,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .objective import relation,MODES
from experiments.cvs_source_self_challenge.audit import source_model
from experiments.cvs_source_layer_order.probe import cohorts,derivative
from experiments.cvs_source_self_challenge.objective import gradient_stats
from experiments.cvs_signature_stats50.predict import source_proof
from experiments.cvs_backbone_fishr50.objective import validate_layout
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_equivariant_identity.precision import numerical_context

def prepare(student):
 student.train().requires_grad_(False)
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.id_backbone.modules()):raise ValueError('Diagnostic would mutate BatchNorm')
 named=dict(student.named_parameters());params=[named[k] for k in d.PARAMETERS]
 for p in params:p.requires_grad_(True)
 return params

def measure_batch(student,teacher,x,y,rx,seed):
 params=prepare(student);teacher.eval().requires_grad_(False)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed)
  with torch.no_grad():weak=teacher.forward_identity_only(x)
  strong=student.forward_identity_only(native_modules()._strong_augment(x,.015))
  prob=weak['tx_logits'].softmax(1).detach()
  ce=F.cross_entropy(strong['tx_logits'],y,label_smoothing=.01)
  vectors={'CE':derivative(ce,params)};losses={'CE':float(ce.detach())};graphs={}
  for mode in MODES:
   loss,graph=relation(strong['id_feat_joint'],prob,rx,mode)
   if torch.autograd.grad(loss,params[-1],allow_unused=True,retain_graph=True)[0] is not None:raise ValueError('Graph objective depends on existing classW')
   losses[mode]=float(loss.detach());vectors[mode]=derivative(.16*loss,params)
   graphs[mode]={k:v.cpu() for k,v in graph.items()}
  if not all(float(g.norm())>0 for g in vectors['CE']):raise ValueError('Matched identity CE actual gradients missing')
  return dict(student_features=strong['id_feat_joint'].detach().cpu(),teacher_prob=prob.cpu(),strong_logits=strong['tx_logits'].detach().cpu(),losses=losses,vectors={k:[g.cpu() for g in v] for k,v in vectors.items()},graphs=graphs,auxiliary_W_gradient_none=True)

def summarize(raw):
 layers=[]
 for k,name in enumerate(d.PARAMETERS):
  ce=torch.stack([r['vectors']['CE'][k].double() for r in raw]).mean(0)
  entry=dict(parameter=name,cohorts=len(raw))
  for mode in MODES:
   gg=torch.stack([r['vectors'][mode][k].double() for r in raw]).mean(0)
   entry[mode]=dict(**gradient_stats(ce,gg),nonzero_cohorts=sum(float(r['vectors'][mode][k].norm())>1e-12 for r in raw))
  layers.append(entry)
 rows=[]
 for r in raw:
  y=r['y'];same=y[:,None]==y[None,:];row=dict(cohort=r['cohort'],packets=len(y),teacher_correct=int((r['teacher_prob'].argmax(1)==y).sum()),strong_default_correct=int((r['strong_logits'].argmax(1)==y).sum()),losses=r['losses'],graphs={})
  for mode in MODES:
   graph=r['graphs'][mode];valid=graph['valid'];row['graphs'][mode]=dict(allowed_edges=int(graph['allowed'].sum()),valid_rows=int(valid.sum()),true_identity_mass=float((graph['target']*same).sum(1)[valid].double().mean()),wrong_identity_mass=float((graph['target']*~same).sum(1)[valid].double().mean()))
  rows.append(row)
 return dict(layers=layers,cohorts=rows)

def qualification(summary):
 mass=sum(r['graphs']['XR_GRAPH']['true_identity_mass'] for r in summary['cohorts'])/5
 body=summary['layers'][:3]
 passed=len(summary['cohorts'])==5 and mass>=.90 and all(r['XR_GRAPH']['nonzero_cohorts']==5 and r['XR_GRAPH']['cosine'] is not None and r['XR_GRAPH']['cosine']>=0 for r in body)
 return dict(status='PASS' if passed else 'FAIL',fixed_rules=d.QUALIFICATION,source_only=True,XR_mean_true_identity_mass=mass,no_automatic_training_release=True,target_metrics_used=False)

def recount(root):
 p=root/d.rows()[0]['row_id'];report=d.read(p/'analysis.json');d.validate(report['config'])
 raw=torch.load(p/'actual_source_gradients.pt',map_location='cpu',weights_only=False)
 if len(raw)!=5 or [r['cohort'] for r in raw]!=list(range(5)):raise ValueError('Fixed source cohorts missing')
 ids=[]
 for r in raw:
  validate_layout(r['y'],r['rx'],r['day'],r['source_physical_ids'],'L_s');ids.extend(r['source_physical_ids'])
  z=r['student_features'].requires_grad_(True)
  if abs(float(F.cross_entropy(r['strong_logits'],r['y'],label_smoothing=.01))-r['losses']['CE'])>1e-6:raise ValueError('Matched CE recount differs')
  for mode in MODES:
   loss,graph=relation(z,r['teacher_prob'],r['rx'],mode)
   if abs(float(loss.detach())-r['losses'][mode])>1e-6 or any(not torch.equal(v,r['graphs'][mode][k]) for k,v in graph.items()):raise ValueError('Saved semantic graph/loss recount differs')
   expected=32220 if mode=='ALL_GRAPH' else 25920
   if int(graph['allowed'].sum())!=expected or not bool(graph['valid'].all()):raise ValueError('Fixed balanced source edges missing')
  for key in ('CE',*MODES):
   if [g.numel() for g in r['vectors'][key]]!=[3456,6912,576,960] or any(not torch.isfinite(g).all() for g in r['vectors'][key]):raise ValueError('Actual parameter vector shape/finite check differs')
  if not r['auxiliary_W_gradient_none'] or any(r['vectors'][mode][-1].count_nonzero() for mode in MODES):raise ValueError('Auxiliary W dependency')
 summary=summarize(raw)
 if len(ids)!=900 or len(set(ids))!=900 or summary!=report['summary'] or qualification(summary)!=report['qualification']:raise ValueError('Saved coverage/summary/qualification differs')
 return dict(status='VERIFIED',source_packets=900,model_forward_packets=1800,gradient_batches=5,parameter_layer_records=20,saved_vectors=60,new_forwards=0,new_gradient_calls=0,head_fit_updates=0,optimizer_steps=0,U_IQ_read=False,U_truth_read=False,query_read=False)

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2);c=d.config(d.rows()[0]);d.validate(c)
 spec=d.expand_registration(d.read(d.ROOT/'registration/experiment.json'))
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[c]:raise ValueError('Full source registration differs')
 if d.BASE.exists():raise FileExistsError('Immutable output exists; reconcile before retry')
 d.BASE.mkdir(parents=True);p=Path(c['output_root']);p.mkdir();started=time.time()
 try:
  source=source_proof(c['source_config'],design=d.parent)
  L,U,V,contract=load_source(nd.config(nd.rows()[0]))
  if U is not None:raise ValueError('U tensor materialized')
  del V
  old=cohorts(L);indices=[torch.cat((old[2*j],old[2*j+1])) for j in range(5)]
  with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
   student=source_model();teacher=copy.deepcopy(student)
   checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
   if checkpoint['epoch']!=50 or checkpoint['ema_model'] is None or checkpoint['a1_ema_successful_updates']!=11100:raise ValueError('Fixed ownEMA source unverified')
   teacher.load_state_dict(checkpoint['ema_model'],strict=True);teacher.eval().requires_grad_(False)
   before={k:v.clone() for k,v in student.state_dict().items()};before_t={k:v.clone() for k,v in teacher.state_dict().items()};raw=[]
   d.write(p/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_roles='EXACT_MATCH',checkpoint_parent=d.parent.RUN,ownEMA_same_checkpoint=True,source_only=True,query_read=False,head_fit_updates=0))
   for j,ix in enumerate(indices):
    y,rx,day=(L[k][ix] for k in ('y','rx','day'));ids=[L['ids'][i] for i in ix.tolist()];validate_layout(y,rx,day,ids,'L_s')
    result=measure_batch(student,teacher,L['x'][ix],y,rx,c['strong_rng_seed']+j)
    raw.append(dict(cohort=j,y=y.cpu(),rx=rx.cpu(),day=day.cpu(),source_physical_ids=ids,**result))
    print(json.dumps(dict(source_cohorts=j+1,source_packets=(j+1)*180,model_forward_packets=(j+1)*360)),flush=True)
   if any(not torch.equal(v,student.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in before_t.items()) or any(v.grad is not None for v in student.parameters()):raise ValueError('Source diagnostic mutated weights/buffers or accumulated gradients')
  summary=summarize(raw);report=dict(status='ANALYZED',config=c,summary=summary,qualification=qualification(summary),source_packets=900,model_forward_packets=1800,model_parameters_buffers_unchanged=True,teacher_ownEMA_same_checkpoint=True,head_fit_updates=0,optimizer_steps=0,U_IQ_read=False,U_truth_read=False,V_forward=False,V_gradient=False,query_read=False,seconds=time.time()-started,limitation=c['limitation'])
  torch.save(raw,p/'actual_source_gradients.pt');d.write(p/'analysis.json',report)
  subprocess.run([sys.executable,'-u','-m','experiments.cvs_source_soft_relation.audit','--recount',str(d.BASE)],check=True)
  d.write(d.BASE/'source_qualification.json',report['qualification']);d.write(d.BASE/'analysis.json',report)
  done=dict(status='ANALYZED',models=1,source_packets=900,forward_packets=1800,gradient_batches=5,layer_records=20,saved_vectors=60,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)
  d.write(p/'completion.json',done);d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as exc:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--recount',type=Path);args=parser.parse_args()
 if args.recount:
  torch.set_num_threads(2);value=recount(args.recount);d.write(args.recount/'independent_artifact_recount.json',value);print(json.dumps(value))
 else:main()
