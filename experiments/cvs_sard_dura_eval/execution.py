import argparse,os,subprocess,sys,time,traceback,statistics
from pathlib import Path
from . import plan as p
from .runtime import read,write
ROOT=Path(__file__).resolve().parents[2]

def configure():
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.set_float32_matmul_precision('highest')

def predict(r):
 import numpy as np,torch
 from .runtime import observations,load_reader,smoke,existing_predictions
 from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
 out=p.BASE/r['row_id'];out.mkdir();pred=out/'prediction';pred.mkdir()
 reader,native=load_reader(r,'cuda:0')
 write(out/'smoke.json',smoke(r,reader,native,'cuda:0'));del native
 ids,raw,restored=observations();before={k:v.clone() for k,v in reader.state_dict().items()}
 decisions={};references={};times={};new_cells=0
 with torch.inference_mode():
  for mode,arrays in [('raw',raw),('dura',restored)]:
   for view in p.VIEWS:
    key=mode+'__'+view;tic=time.perf_counter()
    if p.reused(r,mode,view):
     values,ref=existing_predictions(r,mode,view,ids);references[key]=ref
    else:
     a=np.load(arrays[view],mmap_mode='r',allow_pickle=False)
     logits=np.empty((p.COUNT,6),np.float32)
     for begin in range(0,p.COUNT,128):
      x=array_tensor(a[begin:begin+128],'cuda:0');z=reader(x)
      if z.shape!=(len(x),6) or not torch.isfinite(z).all():raise ValueError('Invalid frozen default-head logits')
      logits[begin:begin+len(x)]=z.cpu().numpy()
     np.save(pred/(key+'.npy'),logits,allow_pickle=False);values=logits.argmax(1).astype(np.int64);new_cells+=1
    if values.shape!=(p.COUNT,) or values.dtype.kind not in 'iu' or values.min()<0 or values.max()>5:raise ValueError('Invalid fixed decisions')
    decisions[key]=values;times[key]=time.perf_counter()-tic
    print({'event':'VIEW_COMPLETE','row':r['row_id'],'readout':mode,'view':view,'reused':p.reused(r,mode,view),'seconds':times[key]},flush=True)
 if any(not torch.equal(v,reader.state_dict()[k]) for k,v in before.items()):raise ValueError('Inference changed original reader state')
 np.savez_compressed(pred/'predictions.npz',ids=ids,**decisions)
 write(pred/'complete.json',dict(status='PREDICTIONS_COMPLETE',row=r,policy=p.policy(),count=p.COUNT,views=list(p.VIEWS),readouts=['raw','dura'],
  source_checkpoint=str(p.source(r)/'final_ssdg.pth'),references=references,new_prediction_cells=new_cells,reused_prediction_cells=14-new_cells,
  G_new_predictions=0,DURA_original_IQ_reused=True,truth_read=False,state_updates=0,head_fit_updates=0,seconds=times,commit=(ROOT/'release_commit.txt').read_text().strip()))

def preflight(r):
 import numpy as np
 from .runtime import observations,existing_predictions
 out=p.BASE/r['row_id']/'prediction';done=read(out/'complete.json');p.validate(done['policy']);ids,_,_=observations()
 if done['status']!='PREDICTIONS_COMPLETE' or done['row']!=r or done['views']!=list(p.VIEWS) or done['count']!=p.COUNT or done['truth_read'] or done['state_updates'] or done['head_fit_updates']:
  raise ValueError('Truth closed: incomplete fixed readout matrix')
 with np.load(out/'predictions.npz',allow_pickle=False) as f:
  keys={m+'__'+v for m in ('raw','dura') for v in p.VIEWS}
  if set(f.files)!={'ids',*keys} or not np.array_equal(f['ids'],ids):raise ValueError('Truth closed: IDs/readout set differ')
  values={k:f[k].copy() for k in keys}
 for mode in ('raw','dura'):
  for view in p.VIEWS:
   key=mode+'__'+view;z=values[key]
   if z.shape!=(p.COUNT,) or z.dtype.kind not in 'iu' or z.min()<0 or z.max()>5:raise ValueError('Truth closed: invalid class decisions')
   if p.reused(r,mode,view):expected,_=existing_predictions(r,mode,view,ids)
   else:
    a=np.load(out/(key+'.npy'),allow_pickle=False)
    if a.shape!=(p.COUNT,6) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Truth closed: invalid new logits')
    expected=a.argmax(1)
   if not np.array_equal(z,expected):raise ValueError('Truth closed: decisions/logits mismatch')
 return ids,values

def score(r):
 import numpy as np
 from experiments.cvs_t1_e50_confirm import design as d
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 from comparison_suite.score import metrics
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
 ids,values=preflight(r)
 # No truth handle is opened until all fourteen cells of this row are fixed.
 y,rx,days=truth_arrays(read(d.TRUTH),ids);records=[];paired=[]
 for view in p.VIEWS:
  baseline=values['raw__'+view];restored=values['dura__'+view]
  for dim,groups in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(days))),('transmitter',range(6))]:
   for group in groups:
    mask=np.ones(p.COUNT,bool) if dim=='overall' else rx==group if dim=='receiver' else days==group if dim=='day' else y==group
    a=baseline[mask]==y[mask];b=restored[mask]==y[mask]
    paired.append(dict(view=view,dimension=dim,stratum=str(group),count=int(mask.sum()),repaired=int((b&~a).sum()),damaged=int((a&~b).sum()),net=int(b.sum()-a.sum()),accuracy_delta_pp=100*float(b.mean()-a.mean())))
    for mode in ('raw','dura'):
     predicted=values[mode+'__'+view];result=metrics(y[mask],predicted[mask],6)
     cm=np.bincount(y[mask]*6+predicted[mask],minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
     f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()
     if abs(result['accuracy']-np.trace(cm)/cm.sum())>1e-12 or abs(result['macro_f1']-f1)>1e-12:raise ValueError('Independent metric recount mismatch')
     records.append(dict(row=r,readout=mode,view=view,dimension=dim,stratum=str(group),**result))
 out=p.BASE/r['row_id']/'scoring';out.mkdir()
 write(out/'scores.json',dict(results=records,paired=paired,truth_last=True,target_feedback=False))
 write(out/'complete.json',dict(status='SCORED_COMPLETE',row=r,views=list(p.VIEWS),metric_records=len(records),paired_records=len(paired),truth_last=True))
 write(p.BASE/r['row_id']/'complete.json',dict(status='ANALYZED',row=r,truth_last=True,all14cells_before_truth=True,training_updates=0))

def collect():
 cells={};rows=[]
 for r in p.rows():
  root=p.BASE/r['row_id'];done=read(root/'complete.json')
  if done['status']!='ANALYZED':raise ValueError('Six complete rows required')
  scores=read(root/'scoring/scores.json');rows.append(dict(row=r,scores=scores))
  for v in scores['results']:
   key=(r['arm'],v['readout'],v['view'],v['dimension'],v['stratum']);cells.setdefault(key,[]).append(dict(seed=r['seed'],accuracy=v['accuracy'],macro_f1=v['macro_f1'],count=v['query_count']))
 summary=[]
 for key,values in cells.items():
  if len(values)!=3 or len({x['count'] for x in values})!=1:raise ValueError('Three comparable seeds required')
  summary.append(dict(arm=key[0],readout=key[1],view=key[2],dimension=key[3],stratum=key[4],model_seed_count=3,count_per_seed=values[0]['count'],values=values,
   accuracy_mean=statistics.mean(x['accuracy'] for x in values),accuracy_std=statistics.stdev(x['accuracy'] for x in values),
   macro_f1_mean=statistics.mean(x['macro_f1'] for x in values),macro_f1_std=statistics.stdev(x['macro_f1'] for x in values)))
 write(p.BASE/'results.json',dict(rows=rows,summary=summary,std_ddof=1,shared_DURA_model_seed=2026092701,recognizer_model_seeds=list(p.SEEDS),
  restorer_seed_variability_tested=False,training_updates=0,target_feedback=False,benchmark_exposed=True,cached_DURA_output_reused=True))
 write(p.BASE/'complete.json',dict(status='ANALYZED',rows=[r['row_id'] for r in p.rows()],views=list(p.VIEWS),model_seed_count_per_arm=3,new_prediction_cells=47,reused_prediction_cells=37,truth_last=True,training_updates=0))

def dispatch():
 from experiments.cvs_receiver_residual_capacity4.control import lock
 from experiments.cvs_multi_state_action.dispatch import capacity
 if p.BASE.exists():raise FileExistsError('Reconcile existing immutable run;never relaunch blindly')
 p.BASE.mkdir();(p.BASE/'logs').mkdir();write(p.BASE/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,owner='codex/root/SARD-DURA',policy=p.policy()))
 pending=p.rows();active={};launched=[]
 try:
  while pending or active:
   for rid,info in list(active.items()):
    code=info['process'].poll()
    if code is None:continue
    if code or not (p.BASE/rid/'complete.json').is_file():raise RuntimeError('Own technical failure;preserve other healthy rows,stop pending;no automatic retry: '+rid)
    del active[rid]
   if pending and len(active)<2:
    with lock(p.PROJECT/'runs/receiver_residual_capacity4.lock'):
     state=capacity();used={v['gpu'] for v in active.values()}
     slots=sorted((len(v['pids']),-v['free_mb'],g) for g,v in state.items() if g in range(8) and g not in used and len(v['pids'])<3 and v['free_mb']>=12000)
     while slots and pending and len(active)<2:
      gpu=slots.pop(0)[2];r=pending.pop(0);rid=r['row_id'];cmd=[sys.executable,'-u',str(ROOT/'experiments/cvs_sard_dura_eval/runner.py'),'--mode','worker','--row',rid]
      log=p.BASE/'logs'/(rid+'.log')
      with log.open('x') as f:child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      active[rid]=dict(process=child,gpu=gpu);launched.append(dict(row=r,pid=child.pid,gpu=gpu,cwd=str(ROOT),argv=cmd,log=str(log)));write(p.BASE/'workers.json',dict(workers=launched))
   write(p.BASE/'queue.json',dict(active=list(active),pending=pending,updated_at=time.time()))
   if active or pending:time.sleep(3)
  collect()
 except BaseException as error:
  write(p.BASE/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),preserve_outputs=True,no_automatic_retry=True));raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['worker','score']);parser.add_argument('--row');args=parser.parse_args();configure()
 if not args.mode:return dispatch()
 r=p.row(args.row)
 if args.mode=='worker':
  from experiments.cvs_receiver_residual_capacity4.control import lock
  from experiments.cvs_multi_state_action.dispatch import capacity
  with lock(p.PROJECT/'runs/receiver_residual_capacity4.lock'):
   gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
   if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise RuntimeError('All-owner capacity exceeds3')
  predict(r)
  os.execve(sys.executable,[sys.executable,'-u',str(ROOT/'experiments/cvs_sard_dura_eval/runner.py'),'--mode','score','--row',r['row_id']],dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 else:
  from experiments.cvs_receiver_residual_capacity4.control import lock
  with lock(p.BASE/'CPU_score.lock'):score(r)
