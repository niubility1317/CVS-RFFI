"""User-authorized clean-only evaluation of the four already frozen F models."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from statistics import mean,stdev
import numpy as np
from . import design as d
RUN='20261019-eval-FcrossRX-TXgeometry-clean-local-m4-r01';VIEWS=('clean',);COUNT=168000

def evaluation(c,source_commit):return dict(run_id=RUN,training_run=d.RUN,row_id=c['row_id'],source_commit=source_commit,training_config=c,views=list(VIEWS),count=COUNT,user_override='clean only; practical cancelled',head_fit_updates=0)

def source_proof(c,source_base):
 from .predict import source_proof as original
 done=original(c,source_base)
 if done['commit']!='4679f66babb40c17b96285c7393225787ed13c08':raise ValueError('Original frozen source commit differs')
 if d.read(Path(source_base)/'source_expansion.json')['status']!='SOURCE_EXPANSION_FIXED':raise ValueError('Source-only decision must already be fixed')
 return done

def predict(c,profile,source_base,base):
 import torch
 from . import model as api
 from .predict import observation_ids
 from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY
 d.validate(c);done=source_proof(c,source_base);source=Path(source_base)/c['row_id']/'source';out=Path(base)/c['row_id']/'prediction';out.mkdir(parents=True,exist_ok=False)
 started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(c['model_seed']);ev=evaluation(c,done['commit'])
 with numerical_context(FULL_FP32_POLICY):
  cp=torch.load(source/'final.pt',weights_only=False,map_location='cpu')
  expected=dict(config=c,commit=done['commit'],epoch=20,optimizer_steps=4440,checkpoint_selection='fixed_final20',source_roles='EXACT_MATCH',scratch_only=True,ancestors=[],target_access=False,default_head_joint_training=True,head_fit_updates=0)
  if any(cp.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
  model=api.build(c,'cuda:0');model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False);versions=[p._version for p in model.parameters()];buffers={n:b.clone() for n,b in model.named_buffers()}
  with torch.no_grad():z=model.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
  if z['tx_logits'].shape!=(2,6) or not torch.isfinite(z['tx_logits']).all():raise ValueError('Actual final checkpoint no-query smoke failed')
  d.write(out/'smoke.json',dict(status='PASS',checkpoint_row=c['row_id'],checkpoint_epoch=20,query_reads=0,truth_reads=0,default_head=[6,160]))
  ids=observation_ids(profile);array=np.load(Path(profile['dataset']['target_capsule'])/'clean.npy',mmap_mode='r',allow_pickle=False)
  if array.shape!=(COUNT,2,256) or array.dtype!=np.float32:raise ValueError('Existing validated clean observations differ')
  pred=np.empty(COUNT,dtype=np.int64);torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():
   for begin in range(0,COUNT,256):
    x=torch.from_numpy(np.array(array[begin:begin+256],copy=True)).to('cuda:0');result=model.forward_identity_only(x);logits=result['tx_logits']
    if logits.shape!=(len(x),6) or not torch.isfinite(logits).all() or not torch.isfinite(result['id_feat_joint']).all():raise FloatingPointError('Invalid fixed default-head prediction')
    pred[begin:begin+len(x)]=logits.argmax(1).cpu().numpy()
  if versions!=[p._version for p in model.parameters()] or any(not torch.equal(b,dict(model.named_buffers())[n]) for n,b in buffers.items()):raise ValueError('Prediction changed frozen model state')
  np.savez_compressed(out/'predictions.npz',ids=ids,clean=pred);torch.cuda.synchronize()
  d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',evaluation=ev,views=['clean'],count=COUNT,truth_read=False,query_fit=False,head_fit_updates=0,seconds=time.perf_counter()-started,peak_cuda_MiB=torch.cuda.max_memory_allocated()/2**20))
 print('CLEAN_PREDICTED '+c['row_id'],flush=True)

def score_preflight(c,source_base,base):
 source=source_proof(c,source_base);out=Path(base)/c['row_id']/'prediction';done=d.read(out/'complete.json');ev=evaluation(c,source['commit'])
 if any(done.get(k)!=v for k,v in dict(status='PREDICTIONS_COMPLETE',evaluation=ev,views=['clean'],count=COUNT,truth_read=False,query_fit=False,head_fit_updates=0).items()):raise ValueError('Keep truth closed: incomplete clean prediction')
 with np.load(out/'predictions.npz',allow_pickle=False) as values:
  if set(values.files)!={'ids','clean'}:raise ValueError('Keep truth closed: exact clean scope required')
  ids=values['ids'].copy();pred=values['clean'].copy()
 if ids.shape!=(COUNT,) or len(set(ids.tolist()))!=COUNT or pred.shape!=(COUNT,) or pred.dtype.kind not in 'iu' or pred.min()<0 or pred.max()>=6:raise ValueError('Keep truth closed: prediction coverage invalid')
 return ids,pred,done

def score(c,profile,source_base,base):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU-only truth-side scorer required')
 ids,pred,done=score_preflight(c,source_base,base);out=Path(base)/c['row_id']/'scores'
 if out.exists():raise FileExistsError('Existing clean scores preserved')
 from .score import metrics
 from experiments.cvs_report_ssl.evaluate import truth_arrays
 truth=d.read(profile['dataset']['scoring_truth']);y,rx,days=truth_arrays(truth,ids);metric=metrics(pred,y,rx,days);out.mkdir()
 d.write(out/'metrics.json',dict(status='MODEL_PREDICTED_SCORED',evaluation=done['evaluation'],views=['clean'],count=COUNT,records=dict(clean=metric),truth_last=True,independent_CPU_score=True,head_fit_updates=0))
 print('CLEAN_SCORED '+c['row_id'],flush=True)

def aggregate(source_base,base):
 records=[];cost=[]
 for row in d.rows(d.SEEDS[:1]):
  c=d.config(row);source=source_proof(c,source_base);sc=d.read(base/c['row_id']/'scores/metrics.json');pr=d.read(base/c['row_id']/'prediction/complete.json')
  if sc['evaluation']!=evaluation(c,source['commit']) or sc['truth_last'] is not True or sc['views']!=['clean']:raise ValueError('Clean score matrix incomplete')
  records.append(dict(**row,metrics=sc['records']));cost.append(dict(row_id=row['row_id'],epochs=20,optimizer_steps=4440,source_seconds=source['seconds'],prediction_seconds=pr['seconds'],peak_cuda_MiB=source['peak_cuda_MiB'],L_exposure=20*222*128,U_exposure=0,satellite_examples=source['satellite_examples'],satellite_applied_examples=source['satellite_applied_examples']))
 summary={arm:dict(clean=dict(accuracy=dict(n=1,mean=next(r['metrics']['clean']['accuracy'] for r in records if r['arm']==arm),sample_SD=None))) for arm in d.ARMS}
 result=dict(status='ARTIFACTS_COMPLETE',run_id=RUN,training_run_id=d.RUN,views=['clean'],records=records,summary=summary,cost=cost,source_expansion=d.read(source_base/'source_expansion.json'),head_fit_updates=0,target_feedback=False,truth_last=True,scope_change='User explicitly cancels practical tests; source training preserved; all source arms failed original expansion; initial4 target comparisons remain single-seed')
 d.write(base/'result.json',result)

def driver(a):
 profile=d.read(a.profile);base=Path(a.base).resolve();source=Path(a.source_base).resolve()
 if base.name!=RUN or source.name!=d.RUN or d.ROOT.resolve()!=Path.cwd().resolve() or (d.ROOT/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Wrong clean-only immutable release/run')
 rows=d.rows(d.SEEDS[:1])
 if d.read(source/'frozen_matrix.json')['rows']!=rows:raise ValueError('Original frozen comparison must be exactly initial4')
 for row in rows:source_proof(d.config(row),source)
 lock=Path(a.profile).parent/'gpu0.worker.lock'
 with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),run_id=RUN,cwd=str(d.ROOT),max_GPU_workers=3),f)
 base.mkdir(parents=True,exist_ok=False);d.write(base/'execution.json',dict(run_id=RUN,commit=a.commit,source_commit='4679f66babb40c17b96285c7393225787ed13c08',source_run=d.RUN,max_GPU_workers=3,views=['clean'],pid=os.getpid(),cwd=str(d.ROOT)))
 env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(d.ROOT),str(d.ROOT/'code'))),TORCHINDUCTOR_CACHE_DIR=str(base/'torch_cache'));active=[];queue=list(rows);scored=set()
 def launch(role,row):
  argv=[sys.executable,'-X','utf8','-u','-m','experiments.cvs_f_txgeometry20.clean_eval','--role',role,'--row',row['row_id'],'--profile',a.profile,'--source-base',str(source),'--base',str(base),'--commit',a.commit]
  log=base/(row['row_id']+'-'+role+'.log')
  with log.open('x',encoding='utf-8') as f:p=subprocess.Popen(argv,cwd=d.ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='' if role=='score' else '0'),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
  evidence=dict(pid=p.pid,role=role,row_id=row['row_id'],argv=argv,cwd=str(d.ROOT));d.write(base/(row['row_id']+'-'+role+'-start.json'),evidence);active.append(dict(process=p,**evidence));print(json.dumps(dict(event='STARTED',**evidence)),flush=True)
 try:
  while queue or active or len(scored)<len(rows):
   for job in list(active):
    code=job['process'].poll()
    if code is None:continue
    active.remove(job);d.write(base/(job['row_id']+'-'+job['role']+'-exit.json'),dict(pid=job['pid'],returncode=code))
    if code:raise RuntimeError('Preserved clean '+job['role']+' failure '+job['row_id'])
    if job['role']=='score':scored.add(job['row_id'])
   while queue and sum(j['role']=='predict' for j in active)<3:launch('predict',queue.pop(0))
   if not any(j['role']=='score' for j in active):
    for row in rows:
     rid=row['row_id']
     if (base/rid/'prediction/complete.json').exists() and rid not in scored and not any(j['row_id']==rid for j in active):launch('score',row);break
   snapshot=dict(status='RUNNING',controller_pid=os.getpid(),max_GPU_workers=3,views=['clean'],active=[{k:v for k,v in j.items() if k!='process'} for j in active],scored=sorted(scored));tmp=base/'progress.tmp';tmp.write_text(json.dumps(snapshot,indent=2),encoding='utf-8');tmp.replace(base/'progress.json');time.sleep(1)
  aggregate(source,base);tmp=base/'progress.tmp';tmp.write_text(json.dumps(dict(status='ARTIFACTS_COMPLETE',rows=4,views=['clean'])),encoding='utf-8');tmp.replace(base/'progress.json');print('CLEAN_FACTORIAL_COMPLETE',flush=True)
 finally:
  if lock.exists() and d.read(lock).get('pid')==os.getpid():lock.unlink()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--role',choices=['driver','predict','score'],required=True);p.add_argument('--row');p.add_argument('--profile',required=True);p.add_argument('--source-base',required=True);p.add_argument('--base',required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.role=='driver':driver(a)
 else:
  c=d.config(next(r for r in d.rows(d.SEEDS[:1]) if r['row_id']==a.row));(predict if a.role=='predict' else score)(c,d.read(a.profile),a.source_base,a.base)
