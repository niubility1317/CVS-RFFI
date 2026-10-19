"""Three fixed BASE checkpoints; full source L gradients; no training or target access."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from core import *
from loaders import imports,read,ROOT,SEEDS
RUN='20261020-diagnostic-FcrossRX-weakTX-gradient-sourceL-m3-local-r01'
DIAG_SEED=2026102007
class CountingRole(torch.utils.data.Dataset):
 def __init__(self,wrapped):self.wrapped=wrapped;self.samples_read=0;self.role=wrapped.role;self.y=wrapped.y
 def __len__(self):return len(self.wrapped)
 def __getitem__(self,index):
  value=self.wrapped[index];self.samples_read+=1;return value
def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def collect(m,dataset,base_penalty,tail,folder):
 start_reads=dataset.samples_read;params=select(m);before={n:v.detach().clone() for n,v in m.state_dict().items()};modes=[v.training for v in m.modules()];rng=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all();hooks=sum(len(v._forward_hooks) for v in m.modules());gen=torch.Generator().manual_seed(DIAG_SEED);order=np.random.default_rng(DIAG_SEED).permutation(6300).tolist();loader=DataLoader(Subset(dataset,order),batch_size=128,shuffle=False,num_workers=0,drop_last=False,generator=gen);acc=None;batch_records=[];counts=np.zeros(6,dtype=np.int64);unused=set();seen=0;scalar_sum=np.zeros(len(OBJECTIVES));tick=time.perf_counter()
 for index,(x,y,domain,meta) in enumerate(loader,1):
  if set(meta['split_source'])!={'L_s'} or not bool(meta['tx_label_visible'].all()):raise ValueError('Only visible source L batches permitted')
  rx=meta['rx_i'];scope('L_s',y,rx);counts+=np.bincount(y.numpy(),minlength=6);seen+=len(y);maps={};handles=[]
  for level in ('time_down','t2.pool'):handles.append(m.get_submodule(PREFIX+level).register_forward_hook(lambda module,args,out,level=level:maps.__setitem__(level,out)))
  try:
   output=m.forward_identity_only(x.to('cuda:0'));values,info=objectives(output,maps,y.to('cuda:0'),rx.to('cuda:0'),base_penalty,tail);scalar_sum+=np.array([float(v.detach()) for v in values]);raw,missing=vectors(m,values,params);unused.update(missing);acc=raw.clone() if acc is None else acc+raw;gg=grams(raw,params);batch_records.append(dict(batch=index,count=len(y),details=info,metrics={key:gram_summary(v['total']) for key,v in gg.items()}))
  finally:
   for handle in handles:handle.remove()
  del output,values,raw,maps,gg
  if index%10==0:print(json.dumps(dict(event='SOURCE_GRADIENT_PROGRESS',batch=index,packets=seen,total=6300)),flush=True)
 if seen!=6300 or dataset.samples_read-start_reads!=6300 or len(batch_records)!=50 or counts.tolist()!=[1050]*6:raise ValueError('Full existing visible L pool required')
 if any(not torch.equal(v,m.state_dict()[n]) for n,v in before.items()) or any(p.grad is not None for p in m.parameters()) or modes!=[v.training for v in m.modules()] or not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())) or hooks!=sum(len(v._forward_hooks) for v in m.modules()):raise ValueError('Frozen diagnostic changed model/RNG/grad/mode/hooks')
 overall=grams(acc,params);result=dict(source_role='L_s',count=6300,class_counts=counts.tolist(),batches=50,default_head_fit_updates=0,optimizer_updates=0,model_state_RNG_grads_modes_hooks_unchanged=True,feature_parameter_count=sum(p.numel() for n,p in params),feature_parameter_names=[n for n,p in params],unused_feature_parameter_names=sorted(unused),objectives=list(OBJECTIVES),objective_values=scalar_sum.tolist(),metrics={key:dict(total=gram_summary(v['total']),blocks={name:gram_summary(g) for name,g in v['blocks'].items()}) for key,v in overall.items()},batch_records=batch_records,target_reads=0,U_reads=0,V_reads=0,IQ_read_occurrences=6300,seconds=time.perf_counter()-tick)
 write(folder/'gradient.json',result);return result
def run(a):
 out=Path(a.output).resolve();profile=read(a.profile)
 if os.name!='nt' or ROOT.resolve()!=Path.cwd().resolve() or out.name!=RUN or (ROOT/'release_commit.txt').read_text(encoding='utf-8').strip()!=a.commit:raise ValueError('Fixed immutable local diagnostic release/output required')
 if profile['physical_gpu']!=0 or profile['source_data_status']!='READY':raise ValueError('Verified local GPU0 source handle required')
 d,api,validate,base_penalty,tail,precision,configs=imports(a)
 from experiments.cvs_f_crossrx50_local.driver import resources
 from tools.local_phase1_data import SourceRoleDataset
 out.mkdir(parents=True,exist_ok=False);write(out/'execution.json',dict(run_id=RUN,commit=a.commit,pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,source_role='L_s',optimizer_updates=0,target_reads=0))
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 while True:
  try:
   gpu,ram=resources()
   if len(gpu-{os.getpid()})<3 and ram>=4.5:break
   print('WAIT_RESOURCE '+json.dumps(dict(other_workers=len(gpu-{os.getpid()}),free_GiB=ram)),flush=True)
  except Exception as e:print('WAIT_RESOURCE_UNKNOWN '+repr(e),flush=True)
  time.sleep(10)
 records=[];dataset=CountingRole(SourceRoleDataset(profile['dataset']['source_cache'],'L_s'))
 if dataset.role!='L_s' or len(dataset)!=6300 or dataset.y is None:raise ValueError('Existing visible L6300 only')
 with precision(d.FULL_FP32_POLICY):
  for c in configs:
   torch.manual_seed(c['model_seed']);folder=out/c['row_id'];folder.mkdir();cp=torch.load(Path(c['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False);validate(cp,c);m=api.build(c,'cuda:0');m.load_state_dict(cp['model'],strict=True);m.eval();torch.cuda.reset_peak_memory_stats();v=collect(m,dataset,base_penalty,tail,folder);v.update(row_id=c['row_id'],model_seed=c['model_seed'],peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2);records.append(v);print('SOURCE_GRADIENT_COMPLETE '+c['row_id'],flush=True);del m,cp;torch.cuda.empty_cache()
 result=dict(status='ARTIFACTS_COMPLETE',run_id=RUN,analysis_commit=a.commit,reference_run=REF_RUN,reference_code_commit=REF_COMMIT,records=records,source_role='L_s',unique_source_samples=6300,model_forward_examples=18900,source_IQ_read_occurrences=dataset.samples_read,source_V_reads=0,U_reads=0,target_reads=0,truth_reads=0,head_fit_updates=0,optimizer_updates=0,diagnostic_seed=DIAG_SEED,limitation='Clean eval-mode source-L gradient geometry, fixed shuffled finite minibatches, not actual stochastic RDrop/AdamW updates, unseenRX validation, target performance, causal proof or a fitted classifier. Gradients never become saved model updates. Default W read only; no separate head. CLIP_PHYSICAL uses gq/25000; NORMALIZED_HZ uses gq*25000000/25000, with all other coordinates unchanged. Coordinate metrics differ; report both.')
 if dataset.samples_read!=18900:raise ValueError('Source diagnostic occurrence budget differs')
 write(out/'result.json',result);write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE',models=3,source_only=True,target_reads=0,head_fit_updates=0,optimizer_updates=0));print('SOURCE_WEAK_GRADIENT_ALL_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--source-run',required=True);p.add_argument('--source-release',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True);run(p.parse_args())
