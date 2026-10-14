import copy,json,time
from pathlib import Path
import torch
from . import design as d
from .metrics import policy,policy_summary,observe,summarize_vectors
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation import design as nd

def qualification(policies):
 rows=[]
 for p in policies:
  helpful=p['groups']['WEAK_RIGHT_STRONG_WRONG'];harmful=p['groups']['WEAK_WRONG_STRONG_RIGHT'];identifiable=helpful['count']>0 and harmful['count']>0
  rows.append(dict(row_id=p['row_id'],helpful=helpful,harmful=harmful,identifiable=identifiable,passed=identifiable and helpful['retention']>harmful['retention']))
 return dict(status='PASS' if len(rows)==6 and all(r['passed'] for r in rows) else 'FAIL',source_only=True,role='V',rows=rows,fixed_rules=d.QUALIFICATION,no_automatic_training_release=True,threshold_scan=False,query_read=False,U_truth_read=False)

def recount(path):
 report=d.read(path/'analysis.json');observations=torch.load(path/'source_policy_observations.pt',map_location='cpu',weights_only=False);roles={r['role']:r for r in report['roles']}
 if set(observations)!={'L_s','V'} or set(roles)!={'L_s','V'}:raise ValueError('Exact source roles required')
 for role,n in [('L_s',6300),('V',27000)]:
  if len(observations[role]['truth'])!=n or policy_summary(observations[role])!=roles[role]['policy']:raise ValueError('Independent source policy/counts differ')
 if roles['V']['gradient_batches'] or roles['V']['global_gradients'] is not None:raise ValueError('V gradients forbidden')
 raw=torch.load(path/'L_source_gradient_vectors.pt',map_location='cpu',weights_only=False);records=roles['L_s']['gradient_batches'];sums=None;counts={k:0 for k in d.GROUPS};retained={k:0 for k in d.GROUPS};packets=0
 if len(raw)!=25 or len(records)!=25:raise ValueError('Full L gradient stream required')
 for a,b in zip(raw,records):
  n=a['count'];begin=a['begin'];packets+=n;part=policy_summary({k:v[begin:begin+n] for k,v in observations['L_s'].items()})
  if a['begin']!=b['begin'] or n!=b['count'] or a['groups']!={k:v['count'] for k,v in part['groups'].items()} or a['retained_groups']!={k:v['retained'] for k,v in part['groups'].items()}:raise ValueError('Batch role/policy observations differ')
  if len(a['vectors'])!=16 or any(len(v)!=3 or any(x.shape!=(size,) for x,size in zip(v,(3456,6912,576))) for v in a['vectors'].values()):raise ValueError('Actual parameter derivative objects differ')
  if summarize_vectors(a['vectors'],a['groups'],a['retained_groups'],d.PARAMETERS)!=b['layers']:raise ValueError('Saved confidence-gated gradients differ')
  if sums is None:sums={key:[torch.zeros_like(v,dtype=torch.float64) for v in vv] for key,vv in a['vectors'].items()}
  for key,vv in a['vectors'].items():
   for k,v in enumerate(vv):sums[key][k]+=n*v.double()
  for key in d.GROUPS:counts[key]+=a['groups'][key];retained[key]+=a['retained_groups'][key]
 if packets!=6300 or sum(counts.values())!=6300:raise ValueError('Source gradient accounting differs')
 for key,vv in sums.items():sums[key]=[v/6300 for v in vv]
 if summarize_vectors(sums,counts,retained,d.PARAMETERS)!=roles['L_s']['global_gradients']:raise ValueError('Weighted complete-L gradient differs')
 return dict(status='VERIFIED',source_packets=33300,forward_packets=66600,gradient_batches=25,layer_records=75,saved_vectors=1200,V_gradients=0,policy_recount='VERIFIED')

def role(native,student,teacher,data,c,args,name):
 n=c['counts'][name]
 if name not in ('L_s','V') or len(data['y'])!=n:raise ValueError('Unregistered source role')
 order=torch.randperm(n,generator=torch.Generator().manual_seed(c['shuffle_seeds'][name]));torch.manual_seed(c['strong_dropout_seeds'][name]);params=[dict(student.named_parameters())[p] for p in d.PARAMETERS];raw=[];records=[];obs={k:[] for k in ('keep','teacher_confidence','student_confidence','weak_pred','strong_pred','truth')};sums=None;counts={k:0 for k in d.GROUPS};retained={k:0 for k in d.GROUPS}
 for begin in range(0,n,c['batch']):
  ix=order[begin:begin+c['batch']];x=data['x'][ix]
  with torch.no_grad():weak=teacher.forward_identity_only(x)
  with torch.set_grad_enabled(name=='L_s'):strong=student.forward_identity_only(native._strong_augment(x,args.strong_noise_std))
  rule=policy(weak['tx_logits'],strong['tx_logits']) # Fixed before labels.
  observation=dict(**rule,weak_pred=weak['tx_logits'].argmax(1),strong_pred=strong['tx_logits'].detach().argmax(1),truth=data['y'][ix])
  for key,v in observation.items():obs[key].append(v.detach().cpu())
  if name=='L_s':
   r=observe(strong['tx_logits'],weak['tx_logits'],strong['id_feat_joint'],weak['id_feat_joint'],data['y'][ix],params,'L_s');m=r['count'];raw.append(dict(begin=begin,count=m,groups=r['groups'],retained_groups=r['retained_groups'],vectors=r['vectors']))
   records.append(dict(begin=begin,count=m,policy=r['policy'],losses=r['losses'],layers=summarize_vectors(r['vectors'],r['groups'],r['retained_groups'],d.PARAMETERS)))
   if sums is None:sums={key:[torch.zeros_like(v,dtype=torch.float64) for v in vv] for key,vv in r['vectors'].items()}
   for key,vv in r['vectors'].items():
    for k,v in enumerate(vv):sums[key][k]+=m*v.double()
   for key in d.GROUPS:counts[key]+=r['groups'][key];retained[key]+=r['retained_groups'][key]
  elif strong['tx_logits'].requires_grad or strong['id_feat_joint'].requires_grad:raise ValueError('V graph created')
 all_obs={key:torch.cat(vv) for key,vv in obs.items()};global_gradients=None
 if sums:
  for key,vv in sums.items():sums[key]=[v/n for v in vv]
  global_gradients=summarize_vectors(sums,counts,retained,d.PARAMETERS)
 return dict(role=name,count=n,forward_packets=2*n,forward_batches=2*((n+255)//256),policy=policy_summary(all_obs),gradient_batches=records,global_gradients=global_gradients,gradient_role='L_s only' if name=='L_s' else 'NONE'),raw,all_obs

def run(c,L,V):
 d.validate(c);out=Path(c['output_root'])
 if out.exists():raise FileExistsError('Immutable source output exists')
 row=next(r for r in d.rows() if r['row_id']==c['row_id']);api=d.model_api(row);prior=c['source_config'];source=api.source_proof(prior);args=d.prior(row).make_args(prior,'cpu');started=time.time()
 if (args.epochs,args.label_epochs,args.pseudo_epochs,args.strong_noise_std)!=(50,33,17,.015):raise ValueError('Actual recipe differs')
 with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
  student=api.own_model(prior,'cpu');teacher=copy.deepcopy(student);ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if ck['epoch']!=50 or ck['ema_model'] is None or ck['a1_ema_successful_updates']!=11100:raise ValueError('Actual ownEMA lineage differs')
  teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False);student.train().requires_grad_(False)
  if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.modules()):raise ValueError('BN updates forbidden')
  for name,p in student.named_parameters():p.requires_grad_(name in d.PARAMETERS)
  before={k:v.clone() for k,v in student.state_dict().items()};before_t={k:v.clone() for k,v in teacher.state_dict().items()};out.mkdir(parents=True,exist_ok=False)
  d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),checkpoint_source=str(source/'final_ssdg.pth'),source_roles='EXACT_MATCH',default_head_only=True,new_head_fit=False,query_read=False,U_truth_read=False))
  results=[];observations={}
  for name,data in [('L_s',L),('V',V)]:
   result,raw,observations[name]=role(native_modules(),student,teacher,data,c,args,name);results.append(result)
   if name=='L_s':torch.save(raw,out/'L_source_gradient_vectors.pt')
  if any(not torch.equal(v,student.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in before_t.items()) or any(p.grad is not None for p in student.parameters()):raise ValueError('Frozen diagnostic mutated state/grad')
 torch.save(observations,out/'source_policy_observations.pt');result=dict(status='ANALYZED',config=c,roles=results,seconds=time.time()-started,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,V_gradient=False,head_updates=0,optimizer_steps=0,model_parameters_buffers_unchanged=True,limitation='Fixed seen source checkpoint and one view seed; confidence rule is a hypothesis, not U correctness or DG proof; no fitted head/threshold scan/new training')
 d.write(out/'analysis.json',result);d.write(out/'independent_artifact_recount.json',recount(out));d.write(out/'completion.json',dict(status='ANALYZED',row_id=c['row_id'],source_packets=33300,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0));return result

def dispatch():
 d.BASE.mkdir(parents=True,exist_ok=False);spec=d.read(d.ROOT/'registration/experiment.json')
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Fixed confidence source matrix differs')
 torch.set_num_threads(2);L,U,V,contract=load_source(nd.config(nd.rows()[0]))
 if U is not None or len(L['y'])!=6300 or len(V['y'])!=27000:raise ValueError('Source roles differ')
 results=[]
 for row in d.rows():
  try:r=run(d.config(row),L,V);results.append(r)
  except Exception as exc:d.write(d.BASE/'failure.json',dict(status='FAILED',row_id=row['row_id'],error=str(exc),no_automatic_retry=True));raise
  print(json.dumps(dict(row_id=row['row_id'],status=r['status'],seconds=r['seconds'])),flush=True)
 policies=[dict(row_id=r['config']['row_id'],**next(role['policy'] for role in r['roles'] if role['role']=='V')) for r in results]
 d.write(d.BASE/'source_qualification.json',qualification(policies));d.write(d.BASE/'analysis.json',dict(status='ANALYZED',run_id=d.RUN,models=6,rows=results));d.write(d.BASE/'completion.json',dict(status='ANALYZED',models=6,source_packets=199800,forward_packets=399600,gradient_batches=150,layer_records=450,saved_vectors=7200,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0))
if __name__=='__main__':dispatch()
