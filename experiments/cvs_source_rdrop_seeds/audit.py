import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_source_rdrop import audit as shared
from experiments.cvs_receiver_dro_seeds import predict as model_api
from experiments.cvs_source_layer_order.probe import cohorts
from experiments.cvs_backbone_fishr50.objective import validate_layout
from experiments.cvs_equivariant_identity.precision import numerical_context

def measure(row,L):
 c=d.config(row);d.validate(c);p=Path(c['output_root']);p.mkdir();started=time.time()
 old=cohorts(L);indices=[torch.cat((old[2*j],old[2*j+1])) for j in range(5)]
 with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
  model=model_api.own_model(c['source_config'],'cpu');before={k:v.clone() for k,v in model.state_dict().items()};raw=[]
  d.write(p/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),
   source_roles='EXACT_MATCH',checkpoint_parent=d.parent.RUN,source_only=True,query_read=False,head_fit_updates=0))
  for j,ix in enumerate(indices):
   y,rx,day=(L[k][ix] for k in ('y','rx','day'));ids=[L['ids'][i] for i in ix.tolist()];validate_layout(y,rx,day,ids,'L_s')
   result=shared.measure_batch(model,L['x'][ix],y,c['strong_rng_seed']+j)
   raw.append(dict(cohort=j,y=y.cpu(),rx=rx.cpu(),day=day.cpu(),source_physical_ids=ids,**result))
   print(json.dumps(dict(row_id=row['row_id'],cohorts=j+1,source_packets=(j+1)*180,forward_packets=(j+1)*540)),flush=True)
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(v.grad is not None for v in model.parameters()):raise ValueError('Source diagnostic changed model state or accumulated gradients')
 summary=shared.summarize(raw)
 report=dict(status='ANALYZED',config=c,summary=summary,qualification=shared.qualification(summary),source_packets=900,
  model_forward_packets=2700,model_parameters_buffers_unchanged=True,head_fit_updates=0,optimizer_steps=0,
  U_IQ_read=False,U_truth_read=False,V_cache_materialized_then_discarded=True,V_forward=False,V_gradient=False,query_read=False,
  seconds=time.time()-started,limitation=c['limitation'])
 torch.save(raw,p/'actual_source_gradients.pt');d.write(p/'analysis.json',report);return report

def source_ids(root,row):
 raw=torch.load(root/row['row_id']/'actual_source_gradients.pt',map_location='cpu',weights_only=False)
 ids=[i for r in raw for i in r['source_physical_ids']]
 if len(ids)!=900 or len(set(ids))!=900:raise ValueError('Fixed900 source physical IDs required')
 return ids

def combined(base):
 first=d.first.read(d.first.BASE/'analysis.json');d.first.validate(first['config']);d.validate_qualification(first['qualification'])
 reports=[first];first_ids=source_ids(d.first.BASE,d.first.rows()[0])
 for row in d.rows():
  report=d.read(base/row['row_id']/'analysis.json');d.validate(report['config'])
  if report['config']!=d.config(row) or source_ids(base,row)!=first_ids:raise ValueError('Source model attribution or identical ordered physical cohorts differ')
  reports.append(report)
 return dict(status='ANALYZED',source_only=True,model_seeds=[2026092701,*d.SEEDS],models=3,
  model_source_observations=2700,unique_source_physical_ids=900,identical_ordered_cohorts_across_models=True,
  first_seed_remeasured=False,total_source_forward_packets=8100,new_source_forward_packets=5400,
  total_gradient_batches=15,total_parameter_layer_records=60,total_saved_vectors=120,
  rows=[dict(model_seed=r['config']['model_seed'],summary=r['summary'],qualification=r['qualification'],
   model_parameters_buffers_unchanged=r['model_parameters_buffers_unchanged'],seconds=r['seconds']) for r in reports],
  qualification=dict(status='PASS' if all(r['qualification']['status']=='PASS' for r in reports) else 'FAIL',
   fixed_per_model_rules=d.QUALIFICATION,no_automatic_training_release=True,target_metrics_used=False),
  training_epochs=0,head_updates=0,optimizer_steps=0,U_IQ_read=False,U_truth_read=False,query_read=False,
  limitation='Three frozen final50 source models, same900L samples; not training, U correctness, DG or84 evidence')

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2);d.require_source();spec=d.expand_registration(d.read(d.ROOT/'registration/experiment.json'))
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Remaining source registration differs')
 if d.BASE.exists():raise FileExistsError('Immutable output exists; inspect instead of retrying')
 d.BASE.mkdir(parents=True)
 try:
  L,U,V,contract=load_source(nd.config(nd.rows()[0]))
  if U is not None:raise ValueError('Unlabelled tensor materialized')
  del V
  reports=[];recounts=[]
  for row in d.rows():
   reports.append(measure(row,L))
   subprocess.run([sys.executable,'-u','-m','experiments.cvs_source_rdrop_seeds.audit','--recount',str(d.BASE),'--row',row['row_id']],check=True)
   p=d.BASE/row['row_id'];recounts.append(d.read(p/'independent_artifact_recount.json'))
   d.write(p/'completion.json',dict(status='ANALYZED',source_packets=900,forward_packets=2700,source_only=True,optimizer_steps=0,head_updates=0))
  total=combined(d.BASE)
  d.write(d.BASE/'independent_artifact_recount.json',dict(status='VERIFIED',models=2,source_packets=1800,
   model_forward_packets=5400,parameter_layer_records=40,saved_vectors=80,new_forwards=0,new_gradient_calls=0,
   optimizer_steps=0,head_fit_updates=0,U_IQ_read=False,U_truth_read=False,query_read=False,rows=recounts))
  d.write(d.BASE/'combined_three_seed_source_analysis.json',total);d.write(d.BASE/'source_qualification.json',total['qualification'])
  d.write(d.BASE/'analysis.json',dict(status='ANALYZED',rows=reports,combined=total));d.write(d.BASE/'completion.json',d.COMPLETION)
  print(json.dumps(d.COMPLETION),flush=True)
 except Exception as exc:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recount',type=Path);p.add_argument('--row');a=p.parse_args()
 if a.recount:
  row=next((r for r in d.rows() if r['row_id']==a.row),None)
  if row is None:raise ValueError('Exact registered source row required')
  torch.set_num_threads(2);v=shared.recount(a.recount,design=d,row=row);d.write(a.recount/row['row_id']/'independent_artifact_recount.json',v);print(json.dumps(v))
 elif a.row:p.error('--row requires --recount')
 else:main()
