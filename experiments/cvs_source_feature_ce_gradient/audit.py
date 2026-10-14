import copy,json,time
from pathlib import Path
import torch
from . import design as d
from .metrics import observe,summarize_vectors
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation import design as nd

def recount(path):
 saved=torch.load(path/'L_source_gradient_vectors.pt',map_location='cpu',weights_only=False);report=d.read(path/'analysis.json');counts={key:0 for key in d.GROUPS};sums=None;packets=0
 if len(saved)!=25 or len(report['gradient_batches'])!=25:raise ValueError('Full L gradient batches incomplete')
 for raw,record in zip(saved,report['gradient_batches']):
  n=raw['count'];packets+=n
  if raw['begin']!=record['begin'] or n!=record['count'] or raw['groups']!=record['groups'] or summarize_vectors(raw['vectors'],raw['groups'],d.PARAMETERS)!=record['layers']:raise ValueError('Saved same-graph gradient summary differs')
  if len(raw['vectors'])!=11 or any(len(v)!=3 for v in raw['vectors'].values()):raise ValueError('Declared gradient objects incomplete')
  if sums is None:sums={key:[torch.zeros_like(x,dtype=torch.float64) for x in vv] for key,vv in raw['vectors'].items()}
  for key,vv in raw['vectors'].items():
   for k,x in enumerate(vv):sums[key][k]+=n*x.double()
  for key,value in raw['groups'].items():counts[key]+=value
 if packets!=6300 or sum(counts.values())!=6300:raise ValueError('Source packet accounting differs')
 for key,vv in sums.items():sums[key]=[x/packets for x in vv]
 if summarize_vectors(sums,counts,d.PARAMETERS)!=report['global_gradients']:raise ValueError('Complete source gradient average differs')
 return dict(status='VERIFIED',source_packets=packets,gradient_batches=25,layer_records=75,saved_vectors=825,groups=counts)

def run(c,L,contract):
 d.validate(c);out=Path(c['output_root'])
 if out.exists():raise FileExistsError('Existing immutable diagnostic output')
 if len(L['y'])!=6300:raise ValueError('Exact labeled-source role required')
 row=next(r for r in d.rows() if r['row_id']==c['row_id']);api=d.model_api(row);prior=c['source_config'];source=api.source_proof(prior);args=d.prior(row).make_args(prior,'cpu')
 if args.strong_noise_std!=.015 or args.epochs!=50 or args.label_epochs!=33 or args.pseudo_epochs!=17:raise ValueError('Actual source training recipe differs')
 started=time.time();torch.set_num_threads(2)
 with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
  student=api.own_model(prior,'cpu');teacher=copy.deepcopy(student);ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if ck['epoch']!=50 or ck['ema_model'] is None or ck['a1_ema_successful_updates']!=11100:raise ValueError('Actual ownEMA lineage differs')
  teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False);student.train().requires_grad_(False)
  if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.modules()):raise ValueError('Diagnostic cannot update BatchNorm state')
  params=[dict(student.named_parameters())[name] for name in d.PARAMETERS]
  for p in params:p.requires_grad_(True)
  before={k:v.clone() for k,v in student.state_dict().items()};before_t={k:v.clone() for k,v in teacher.state_dict().items()}
  out.mkdir(parents=True,exist_ok=False);d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),checkpoint_source=str(source/'final_ssdg.pth'),source_roles='EXACT_MATCH',default_head_only=True,new_head_fit=False,source_only=True,query_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False))
  order=torch.randperm(6300,generator=torch.Generator().manual_seed(c['shuffle_seed']));torch.manual_seed(c['strong_dropout_seed']);native=native_modules();raw=[];records=[];values={k:[] for k in ('feature','CE','weak_right','strong_right','rx','day')};sums=None;counts={key:0 for key in d.GROUPS};total_losses={}
  for begin in range(0,6300,c['batch']):
   ix=order[begin:begin+c['batch']];x=L['x'][ix]
   with torch.no_grad():weak=teacher.forward_identity_only(x)
   strong=student.forward_identity_only(native._strong_augment(x,args.strong_noise_std));r=observe(strong['tx_logits'],weak['tx_logits'],strong['id_feat_joint'],weak['id_feat_joint'],L['y'][ix],params,'L_s');n=r['count']
   raw.append(dict(begin=begin,count=n,groups=r['groups'],vectors=r['vectors']))
   records.append(dict(begin=begin,count=n,weak_right=r['weak_right'],strong_right=r['strong_right'],groups=r['groups'],losses=r['losses'],decomposition_max_abs_error=r['decomposition_max_abs_error'],layers=summarize_vectors(r['vectors'],r['groups'],d.PARAMETERS)))
   if sums is None:sums={key:[torch.zeros_like(v,dtype=torch.float64) for v in vv] for key,vv in r['vectors'].items()}
   for key,vv in r['vectors'].items():
    for k,v in enumerate(vv):sums[key][k]+=n*v.double()
   for key,v in r['groups'].items():counts[key]+=v
   for key,v in r['losses'].items():total_losses[key]=total_losses.get(key,0.)+n*v
   a=torch.nn.functional.normalize(strong['id_feat_joint'].detach().float(),dim=1,eps=1e-8);b=torch.nn.functional.normalize(weak['id_feat_joint'].float(),dim=1,eps=1e-8)
   for key,v in dict(feature=.5*(a-b).square().sum(1),CE=torch.nn.functional.cross_entropy(strong['tx_logits'].detach(),L['y'][ix],reduction='none',label_smoothing=.01),weak_right=weak['tx_logits'].argmax(1)==L['y'][ix],strong_right=strong['tx_logits'].detach().argmax(1)==L['y'][ix],rx=L['rx'][ix],day=L['day'][ix]).items():values[key].append(v)
  for key,vv in sums.items():sums[key]=[v/6300 for v in vv]
  all_values={key:torch.cat(vv) for key,vv in values.items()};strata=[]
  for name in ('rx','day'):
   for value in sorted(all_values[name].unique().tolist()):
    mask=all_values[name]==value;strata.append(dict(dimension=name,stratum=int(value),count=int(mask.sum()),feature_loss=float(all_values['feature'][mask].double().mean()),true_CE=float(all_values['CE'][mask].double().mean()),weak_accuracy=float(all_values['weak_right'][mask].double().mean()),strong_accuracy=float(all_values['strong_right'][mask].double().mean())))
  if any(not torch.equal(v,student.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in before_t.items()) or any(p.grad is not None for p in student.parameters()):raise ValueError('Read-only diagnostic mutated parameters/buffers or accumulated gradients')
 torch.save(raw,out/'L_source_gradient_vectors.pt')
 result=dict(status='ANALYZED',config=c,source_packets=6300,forward_packets=12600,forward_batches=50,gradient_batches=records,group_counts=counts,mean_losses={key:v/6300 for key,v in total_losses.items()},global_gradients=summarize_vectors(sums,counts,d.PARAMETERS),source_strata=strata,seconds=time.time()-started,model_parameters_buffers_unchanged=True,head_updates=0,optimizer_steps=0,model_state_updates=0,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,limitation='Fixed final50 seen-L proxy using same strong graph, student train without BN updates/teacher eval. No actual U correctness, optimizer trajectory, target DG, head fitting, source threshold or coefficient selection.')
 d.write(out/'analysis.json',result);audit=recount(out);d.write(out/'source_gradient_recount.json',audit);d.write(out/'completion.json',dict(status='ANALYZED',row_id=c['row_id'],source_packets=6300,source_only=True,query_read=False,head_updates=0,optimizer_steps=0));return result

def dispatch():
 d.BASE.mkdir(parents=True,exist_ok=False);spec=d.read(d.ROOT/'registration/experiment.json')
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Frozen six-model source matrix differs')
 torch.set_num_threads(2);L,U,V,contract=load_source(nd.config(nd.rows()[0]))
 if U is not None or len(L['y'])!=6300 or len(V['y'])!=27000:raise ValueError('Existing source role handles differ')
 del V;results=[]
 for row in d.rows():
  try:result=run(d.config(row),L,contract);results.append(result)
  except Exception as exc:d.write(d.BASE/'failure.json',dict(status='FAILED',row_id=row['row_id'],error=str(exc),no_automatic_retry=True));raise
  print(json.dumps(dict(row_id=row['row_id'],status=result['status'],seconds=result['seconds'])),flush=True)
 d.write(d.BASE/'analysis.json',dict(status='ANALYZED',run_id=d.RUN,models=6,rows=results));d.write(d.BASE/'completion.json',dict(status='ANALYZED',models=6,source_packets=37800,forward_packets=75600,gradient_batches=150,layer_records=450,saved_vectors=4950,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0))
if __name__=='__main__':dispatch()
