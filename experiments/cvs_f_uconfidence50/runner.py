"""Frozen source-U confidence analysis. No training, head fitting or target reads."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REF_RUN='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01'
REF_COMMIT='31467086c3ced30237f3ac86e91eae68645425d7'
RUN='20261019-diagnostic-FcrossRX-Uconfidence-source-m6-local-r01'
SEEDS=(2026092701,2026092702,2026092703)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def source_scope(role):
 if role!='U_s':raise ValueError('Only hidden-label source U_s; no V/L/target analysis role')
def reference_scope(source_run,source_release):
 if Path(source_run).name!=REF_RUN or (Path(source_release)/'release_commit.txt').read_text(encoding='utf-8').strip()!=REF_COMMIT:raise ValueError('Fixed local source baseline and its original code only')
def row_scope(row):
 if row not in tuple(a+'-s'+str(s) for s in SEEDS for a in ('BASE','BOTH')):raise ValueError('All6 fixed BASE/BOTH models only; no target-based row selection')

def run(a):
 import numpy as np
 import torch
 from torch.utils.data import DataLoader
 from confidence import describe
 out=Path(a.output).resolve();source_run=Path(a.source_run).resolve();origin=Path(a.source_release).resolve();profile_path=Path(a.profile).resolve()
 if os.name!='nt' or Path.cwd().resolve()!=ROOT or out.name!=RUN or (ROOT/'release_commit.txt').read_text(encoding='utf-8').strip()!=a.commit:raise ValueError('Fixed immutable local diagnostic release/output required')
 reference_scope(source_run,origin);source_scope('U_s');profile=read(profile_path)
 if profile['physical_gpu']!=0 or profile['source_data_status']!='READY':raise ValueError('Existing local profile required')
 sys.path[1:1]=[str(origin),str(origin/'code')]
 from experiments.cvs_f_crossrx50_local import design as d,model as api
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 from experiments.cvs_f_crossrx50_local.driver import resources
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from tools.local_phase1_data import SourceRoleDataset
 if d.ROOT.resolve()!=origin or d.RUN!=REF_RUN:raise ValueError('Wrong frozen source implementation')
 configs=[]
 for seed in SEEDS:
  for arm in ('BASE','BOTH'):
   row=arm+'-s'+str(seed);row_scope(row);c=read(source_run/row/'source/local_source_complete.json')['config']
   if Path(c['runtime_base']).resolve()!=source_run or Path(c['runtime_profile']).resolve()!=profile_path or c['model_seed']!=seed or c['treatment']!=arm:raise ValueError('Source/profile reference mismatch')
   source_proof(c);configs.append(c)
 out.mkdir(parents=True,exist_ok=False);write(out/'execution.json',dict(run_id=RUN,commit=a.commit,reference_code_commit=REF_COMMIT,pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,data_role='U_s',U_truth_read=False,target_reads=0,head_fit_updates=0,optimizer_updates=0,EMA_updates=0))
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 while True:
  try:
   gpu,ram=resources()
   if len(gpu-{os.getpid()})<3 and ram>=4.5:break
   print('WAIT_RESOURCE',flush=True)
  except Exception as e:print('WAIT_RESOURCE_UNKNOWN '+repr(e),flush=True)
  time.sleep(10)
 results=[];dataset=None;started=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY):
  for c in configs:
   cp=torch.load(Path(c['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(cp,c);folder=out/c['row_id'];folder.mkdir();predictions=[]
   for role,key in (('student','model'),('ownEMA','ema_model')):
    tick=time.perf_counter();torch.manual_seed(c['model_seed']);m=api.build(c,'cuda:0');m.load_state_dict(cp[key],strict=True);m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()];buffers={n:v.clone() for n,v in m.named_buffers()}
    with torch.no_grad():smoke=m.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
    if smoke['tx_logits'].shape!=(2,6) or not torch.isfinite(smoke['tx_logits']).all() or not torch.isfinite(smoke['id_feat_joint']).all():raise ValueError('Real frozen checkpoint no-query smoke failed')
    write(folder/(role+'-smoke.json'),dict(status='PASS',snapshot_role=role,target_reads=0,U_truth_read=False,head_fit_updates=0,optimizer_updates=0,EMA_updates=0))
    if dataset is None:
     dataset=SourceRoleDataset(profile['dataset']['source_cache'],'U_s')
     if dataset.role!='U_s' or len(dataset)!=56700 or dataset.y is not None:raise ValueError('Existing hidden-label sourceU only')
    loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0,drop_last=False);probabilities=np.empty((56700,6),dtype=np.float32);rxs=np.empty(56700,dtype=np.int64);days=np.empty_like(rxs);offset=0;torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
     for x,y,domain,meta in loader:
      if set(meta['split_source'])!={'U_s'} or meta['tx_label_visible'].any() or not (y==-1).all():raise ValueError('SourceU must expose no TX truth')
      logits=m.forward_identity_only(x.to('cuda:0'))['tx_logits'];n=len(x)
      if logits.shape!=(n,6) or not torch.isfinite(logits).all():raise FloatingPointError('Invalid frozen default-head output')
      probabilities[offset:offset+n]=logits.softmax(1).cpu().numpy();rxs[offset:offset+n]=meta['rx_i'].numpy();days[offset:offset+n]=meta['day_i'].numpy();offset+=n
    if offset!=56700 or set(rxs)!={1,3,4,6,8} or set(days)!={1,2,3}:raise ValueError('Full existing U group coverage differs')
    if versions!=[p._version for p in m.parameters()] or any(not torch.equal(v,dict(m.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Frozen diagnostic updated model state')
    summary=describe(probabilities,rxs,days,d.CLASSES);predictions.append(probabilities.argmax(1));result=dict(row_id=c['row_id'],model_seed=c['model_seed'],treatment=c['treatment'],snapshot_role=role,status='SOURCE_U_CONFIDENCE_COMPLETE',summary=summary,count=56700,head_fit_updates=0,optimizer_updates=0,EMA_updates=0,U_truth_read=False,target_reads=0,truth_reads=0,model_state_unchanged=True,seconds=time.perf_counter()-tick,peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2)
    write(folder/(role+'-sourceU.json'),result);results.append(result);print('SOURCE_U_COMPLETE '+c['row_id']+' '+role,flush=True);del m,smoke,buffers,probabilities;torch.cuda.empty_cache()
   write(folder/'student_EMA_agreement.json',dict(count=56700,argmax_agreement=float(np.mean(predictions[0]==predictions[1])),U_truth_read=False));del cp,predictions
 result=dict(status='ARTIFACTS_COMPLETE',run_id=RUN,commit=a.commit,reference_run=REF_RUN,reference_code_commit=REF_COMMIT,records=results,unique_sourceU_samples=56700,snapshot_forward_examples=680400,head_fit_updates=0,optimizer_updates=0,EMA_updates=0,U_truth_read=False,target_reads=0,truth_reads=0,seconds=time.perf_counter()-started,limitation='Exploratory frozen sourceU confidence analysis after benchmark exposure. Not true class coverage, PL precision, historical acceptance, target performance or Phase2 claim; ordinary training augmentation/temporal/strong-agreement gates are not replayed. All6 prior-fixed BASE/BOTH final50 models included, no score-based selection. Current margin training unchanged.')
 write(out/'result.json',result);write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE',run_id=RUN,snapshots=12,U_truth_read=False,target_reads=0,head_fit_updates=0,optimizer_updates=0,EMA_updates=0));print('SOURCE_U_ALL_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--source-run',required=True);p.add_argument('--source-release',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True);run(p.parse_args())
