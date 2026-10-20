"""Full sourceL only: original3BASE geometry and no-update feature gradients."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from core import *
from loaders import imports,read,ROOT
from checks import run as math_checks

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

class CountingRole(torch.utils.data.Dataset):
 def __init__(self,wrapped):self.wrapped=wrapped;self.samples_read=0;self.role=wrapped.role;self.y=wrapped.y
 def __len__(self):return len(self.wrapped)
 def __getitem__(self,i):self.samples_read+=1;return self.wrapped[i]

def collect(m,dataset,seed,folder):
 start=time.perf_counter();before_reads=dataset.samples_read;params=select(m);before={n:v.detach().clone() for n,v in m.state_dict().items()};modes=[v.training for v in m.modules()];rng=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all();hookcount=sum(len(v._forward_hooks) for v in m.modules());order=np.random.default_rng(PARTITION_SEED).permutation(6300).tolist();loader=DataLoader(Subset(dataset,order),batch_size=128,shuffle=False,num_workers=0,drop_last=False,generator=torch.Generator().manual_seed(PARTITION_SEED));counts=np.zeros(6,dtype=np.int64);acc=None;batch_records=[];stats={level:[dict(TX=c,total=0,eligible=0,same_RX_negative_sum=0.,other_RX_negative_sum=0.,cross_RX_positive_sum=0.) for c in range(6)] for level in LEVELS};loss_sum=np.zeros(6)
 for batch,(x,y,domain,meta) in enumerate(loader,1):
  if set(meta['split_source'])!={'L_s'} or not bool(meta['tx_label_visible'].all()):raise ValueError('Visible source L only; no U/V/target')
  counts+=np.bincount(y.numpy(),minlength=6);maps={};handles=[]
  for level in LEVELS:handles.append(m.get_submodule(PREFIX+level).register_forward_hook(lambda module,args,out,level=level:maps.__setitem__(level,out)))
  try:
   output=m.forward_identity_only(x.to('cuda:0'));values,detail=objectives(output,maps,y.to('cuda:0'),meta['rx_i'].to('cuda:0'),batch);loss_sum+=np.array([float(v.detach()) for v in values]);raw=vectors(values,params);acc=raw.clone() if acc is None else acc+raw;batch_records.append(dict(batch=batch,count=len(y),metrics=summary(raw,params)))
   for ss in detail:
    for c,item in enumerate(ss['per_TX']):
     for key in ('total','eligible','same_RX_negative_sum','other_RX_negative_sum','cross_RX_positive_sum'):stats[ss['level']][c][key]+=item[key]
  finally:
   for h in handles:h.remove()
  del output,values,raw,maps
  if batch%10==0:print('SOURCE_RX_NEGATIVE_PROGRESS '+json.dumps(dict(seed=seed,batch=batch,packets=int(counts.sum()))),flush=True)
 if len(batch_records)!=50 or counts.tolist()!=[1050]*6 or dataset.samples_read-before_reads!=6300:raise ValueError('Full existing sourceL6300 required')
 if any(not torch.equal(v,m.state_dict()[n]) for n,v in before.items()) or any(p.grad is not None for p in m.parameters()) or modes!=[v.training for v in m.modules()] or not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all())) or hookcount!=sum(len(v._forward_hooks) for v in m.modules()):raise ValueError('Read-only diagnostic altered model/RNG/grad/modes/hooks')
 row=dict(model_seed=seed,source_role='L_s',count=6300,class_counts=counts.tolist(),batches=50,objective_names=list(OBJECTIVES),objective_values=loss_sum.tolist(),feature_parameter_count=sum(p.numel() for n,p in params),feature_parameter_names=[n for n,p in params],metrics=summary(acc,params),descriptor_stats=[dict(level=level,per_TX=stats[level]) for level in LEVELS],batch_records=batch_records,seconds=time.perf_counter()-start,peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2,model_state_RNG_grad_modes_hooks_unchanged=True,head_fit_updates=0,optimizer_updates=0,target_reads=0,U_reads=0,V_reads=0)
 write(folder/'summary.json',row);return row

def run(a):
 if os.name!='nt' or ROOT.resolve()!=Path.cwd().resolve() or Path(a.output).name!=RUN or (ROOT/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Fixed native immutable source diagnostic release/output required')
 torch.set_num_threads(2);torch.set_num_interop_threads(1);checks=math_checks();print('MATH_CHECKS_PASS',flush=True);d,api,validate,precision,configs=imports(a);profile=read(a.profile)
 if profile['physical_gpu']!=0 or profile['source_data_status']!='READY':raise ValueError('Verified localGPU0 existing data required')
 from experiments.cvs_f_crossrx50_local.driver import resources
 from tools.local_phase1_data import SourceRoleDataset
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write(out/'execution.json',dict(run_id=RUN,commit=a.commit,pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,source_role='L_s',CPU_threads=2,GPU_workers=1,target_reads=0));write(out/'checks.json',checks)
 while True:
  try:
   live,free=resources()
   if len(live-{os.getpid()})<3 and free>=4.5:break
   print('WAIT_RESOURCE '+json.dumps(dict(other_workers=len(live-{os.getpid()}),free_GiB=free)),flush=True)
  except Exception as error:print('WAIT_RESOURCE_UNKNOWN '+repr(error),flush=True)
  time.sleep(10)
 dataset=CountingRole(SourceRoleDataset(profile['dataset']['source_cache'],'L_s'))
 if dataset.role!='L_s' or dataset.y is None or len(dataset)!=6300:raise ValueError('Existing visible L6300 only')
 rows=[]
 with precision(d.FULL_FP32_POLICY):
  for c in configs:
   folder=out/c['row_id'];folder.mkdir();cp=torch.load(Path(c['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False);validate(cp,c);m=api.build(c,'cuda:0');m.load_state_dict(cp['model'],strict=True);m.eval();torch.cuda.reset_peak_memory_stats();rows.append(collect(m,dataset,c['model_seed'],folder));print('SOURCE_RX_NEGATIVE_COMPLETE '+str(c['model_seed']),flush=True);del m,cp;torch.cuda.empty_cache()
 if dataset.samples_read!=18900:raise ValueError('Exact source-only occurrence budget differs')
 result=dict(status='ARTIFACTS_COMPLETE',run_id=RUN,analysis_code_commit=a.commit,reference_run=REF_RUN,reference_commit=REF_COMMIT,rows=rows,decision=decision(rows),unique_sourceL=6300,model_forward_examples=18900,source_IQ_read_occurrences=18900,feature_VJP_calls=900,target_reads=0,U_reads=0,V_reads=0,truth_reads=0,optimizer_updates=0,head_fit_updates=0,limitation='Frozen clean eval-mode seen-sourceL descriptors/feature gradients, not stochastic actual RDrop or AdamW updates. Same source packets across3models; no classifier fitting, unknownRX/target accuracy or DCoDR invariance guarantee. Random negatives countmatched peranchor by private generator; allmodes share validanchors. Both physicalclip and normalizedHz gradient metrics reported, not tuning feedback.')
 write(out/'result.json',result);write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE',models=3,source_only=True,model_forward_examples=18900,target_reads=0,optimizer_updates=0,head_fit_updates=0));print('SOURCE_RX_NEGATIVE_ALL_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--source-run',required=True);p.add_argument('--source-release',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True);run(p.parse_args())
