"""Frozen source-V feature analysis. No training, head fitting or target reads."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REF_RUN='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01'
REF_COMMIT='31467086c3ced30237f3ac86e91eae68645425d7'
RUN='20261019-diagnostic-FcrossRX-feature-geometry-sourceV-m3-local-r01'
SEEDS=(2026092701,2026092702,2026092703)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def source_scope(role):
 if role!='V':raise ValueError('Only known-label source V; no L/U/target analysis role')
def reference_scope(source_run,source_release):
 if Path(source_run).name!=REF_RUN or (Path(source_release)/'release_commit.txt').read_text(encoding='utf-8').strip()!=REF_COMMIT:raise ValueError('Fixed local source baseline and its original code only')
def row_scope(row):
 if row not in tuple('BASE-s'+str(s) for s in SEEDS):raise ValueError('All3 fixed BASE models only; no target-based row selection')

def run(a):
 import numpy as np
 import torch
 from torch.utils.data import DataLoader
 from geometry import describe,aggregate
 out=Path(a.output).resolve();source_run=Path(a.source_run).resolve();origin=Path(a.source_release).resolve();profile_path=Path(a.profile).resolve()
 if os.name!='nt' or Path.cwd().resolve()!=ROOT or out.name!=RUN or (ROOT/'release_commit.txt').read_text(encoding='utf-8').strip()!=a.commit:raise ValueError('Use fixed immutable local diagnostic release/output')
 reference_scope(source_run,origin);source_scope('V');profile=read(profile_path)
 if profile['physical_gpu']!=0 or profile['source_data_status']!='READY':raise ValueError('Existing local source/GPU0 profile required')
 # Model code and all checkpoint guards come from the exact training release.
 # This standalone analysis module is separate and never patches that release.
 sys.path[1:1]=[str(origin),str(origin/'code')]
 from experiments.cvs_f_crossrx50_local import design as d,model as api
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 from experiments.cvs_f_crossrx50_local.driver import resources
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from tools.local_phase1_data import SourceRoleDataset
 if d.ROOT.resolve()!=origin or d.RUN!=REF_RUN:raise ValueError('Wrong reference implementation')
 configs=[]
 for seed in SEEDS:
  row='BASE-s'+str(seed);row_scope(row);c=read(source_run/row/'source/local_source_complete.json')['config']
  if Path(c['runtime_base']).resolve()!=source_run or Path(c['runtime_profile']).resolve()!=profile_path or c['model_seed']!=seed or c['treatment']!='BASE':raise ValueError('Source/profile/checkpoint reference mismatch')
  source_proof(c);configs.append(c)
 out.mkdir(parents=True,exist_ok=False);write(out/'execution.json',dict(run_id=RUN,analysis_commit=a.commit,reference_code_commit=REF_COMMIT,pid=os.getpid(),cwd=str(Path.cwd().resolve()),argv=sys.argv,source_only=True,data_role='V',target_reads=0,U_TX_reads=0,head_fit_updates=0,optimizer_updates=0))
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 while True:
  try:
   gpu,ram=resources();others=gpu-{os.getpid()}
   if len(others)<3 and ram>=4.5:break
   print('WAIT_RESOURCE '+json.dumps(dict(other_GPU_workers=len(others),free_GiB=ram)),flush=True)
  except Exception as e:print('WAIT_RESOURCE_UNKNOWN '+repr(e),flush=True)
  time.sleep(10)
 results=[];dataset=None;started=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY):
  for c in configs:
   torch.manual_seed(c['model_seed']);tick=time.perf_counter();row=c['row_id'];folder=out/row;folder.mkdir();cp=torch.load(Path(c['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(cp,c)
   m=api.build(c,'cuda:0');m.load_state_dict(cp['model'],strict=True);m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()];buffers={n:v.clone() for n,v in m.named_buffers()}
   with torch.no_grad():smoke=m.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
   if smoke['tx_logits'].shape!=(2,6) or smoke['id_feat_joint'].shape!=(2,160) or not torch.isfinite(smoke['tx_logits']).all() or not torch.isfinite(smoke['id_feat_joint']).all():raise ValueError('Actual final checkpoint no-query smoke failed')
   write(folder/'smoke.json',dict(status='PASS',checkpoint='same-row fixed final50 student/defaulthead',source_role='V',target_reads=0,optimizer_updates=0,head_fit_updates=0))
   if dataset is None:
    dataset=SourceRoleDataset(profile['dataset']['source_cache'],'V')
    if dataset.role!='V' or len(dataset)!=27000 or dataset.y is None:raise ValueError('Existing full sourceV only')
   loader=DataLoader(dataset,batch_size=256,shuffle=False,drop_last=False,num_workers=0);features=np.empty((27000,160),dtype=np.float32);ys=np.empty(27000,dtype=np.int64);rxs=np.empty_like(ys);days=np.empty_like(ys);offset=0;torch.cuda.reset_peak_memory_stats()
   with torch.no_grad():
    for x,y,domain,meta in loader:
     if set(meta['split_source'])!={'V'} or not bool(meta['tx_label_visible'].all()) or (y<0).any() or (y>=6).any():raise ValueError('Known sourceV labels only; never hidden U or target labels')
     z=m.forward_identity_only(x.to('cuda:0'))['id_feat_joint'];n=len(x)
     if z.shape!=(n,160) or not torch.isfinite(z).all():raise FloatingPointError('Nonfinite/deployed-feature shape mismatch')
     features[offset:offset+n]=z.cpu().numpy();ys[offset:offset+n]=y.numpy();rxs[offset:offset+n]=meta['rx_i'].numpy();days[offset:offset+n]=meta['day_i'].numpy();offset+=n
   if offset!=27000 or set(rxs)!={1,3,4,6,8} or set(days)!={1,2,3}:raise ValueError('Declared full sourceV groups differ')
   if versions!=[p._version for p in m.parameters()] or any(not torch.equal(v,dict(m.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Frozen source diagnostic changed model state')
   geometry=describe(features,ys,rxs,days,d.CLASSES);torch.cuda.synchronize();result=dict(row_id=row,model_seed=c['model_seed'],status='SOURCE_GEOMETRY_COMPLETE',geometry=geometry,source_role='V',count=27000,head_fit_updates=0,optimizer_updates=0,prototype_classifier_decisions=0,target_reads=0,truth_reads=0,U_TX_reads=0,model_state_unchanged=True,seconds=time.perf_counter()-tick,peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2,source_code_commit=REF_COMMIT)
   np.savez_compressed(folder/'source_features.npz',features=features,y=ys,rx=rxs,day=days);write(folder/'source_geometry.json',result);results.append(result);print('SOURCE_GEOMETRY_COMPLETE '+row,flush=True);del m,cp,features,smoke,buffers;torch.cuda.empty_cache()
 result=dict(status='ARTIFACTS_COMPLETE',run_id=RUN,analysis_commit=a.commit,reference_run=REF_RUN,reference_code_commit=REF_COMMIT,records=results,summary=aggregate(results),unique_sourceV_samples=27000,model_forward_examples=81000,source_role='V',target_reads=0,truth_reads=0,U_TX_reads=0,head_fit_updates=0,optimizer_updates=0,prototype_classifier_decisions=0,seconds=time.perf_counter()-started,limitation='Exploratory source-only feature geometry after some frozen target results were exposed; all3 BASE references fixed by prior preregistration; no target-dependent class ranking, no fitted head, no new target performance or Phase2 claim')
 write(out/'result.json',result);write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE',run_id=RUN,rows=3,source_only=True,head_fit_updates=0,optimizer_updates=0,target_reads=0));print('SOURCE_GEOMETRY_ALL_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--source-run',required=True);p.add_argument('--source-release',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True);run(p.parse_args())
