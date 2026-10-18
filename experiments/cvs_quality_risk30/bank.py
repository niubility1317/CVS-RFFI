"""One immutable complete-practical sourceL proposal bank;private risk sampling only."""
from collections import Counter
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from . import design as d
FIELDS=('eq_i','rx_i','day_i','sig_i','base_index')
NAMESPACE='FcrossRX/qualityRisk/bank_sampling/v1'
def stable_seed(seed,*parts):return int.from_bytes(hashlib.sha256(json.dumps([int(seed),*parts],sort_keys=True,separators=(',',':')).encode()).digest()[:8],'little')
def bins(values):
 a=np.asarray(values,dtype=np.float64)
 if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Real finite signal_to_added_noise_db required for every proposal')
 return np.searchsorted(np.asarray(d.BIN_EDGES),a,side='right').astype(np.int8)
def select_indices(groups,mode,epoch,step):
 if mode not in ('IID','STRATIFIED') or epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1:raise ValueError('Fixed private tail sampling context required')
 a=np.asarray(groups)
 if a.ndim!=1 or len(a)==0 or not np.isin(a,np.arange(4)).all():raise ValueError('Four-bin finite bank required')
 count=np.bincount(a.astype(np.int64),minlength=4)
 if (count<=0).any():raise ValueError('Fixed four-bin bank contains an empty bin;no rescue/rejection')
 start=time.perf_counter();rng=np.random.default_rng(stable_seed(d.BANK_SEED,NAMESPACE,mode,epoch,step));mass=count/count.sum()
 if mode=='IID':indices=rng.integers(0,len(a),size=64);w=np.ones(64,dtype=np.float64)
 else:
  indices=np.concatenate([rng.choice(np.flatnonzero(a==g),16,replace=True) for g in range(4)]);rng.shuffle(indices);w=4*mass[a[indices]]
 return indices.astype(np.int64),w,dict(mode=mode,cumulative_epoch=int(epoch),cumulative_step=int(step),sampling_seed=int(d.BANK_SEED),sampling_namespace=NAMESPACE,private_seed=stable_seed(d.BANK_SEED,NAMESPACE,mode,epoch,step),bank_bin_counts=count.tolist(),nominal_bin_mass=mass.tolist(),bank_indices=indices.tolist(),bin=a[indices].astype(int).tolist(),bin_counts=np.bincount(a[indices].astype(int),minlength=4).tolist(),weights=w.tolist(),bank_weight_sum=float(w.sum()),sampling_cpu_wall_seconds=time.perf_counter()-start,with_replacement=True,global_RNG_used=False,extra_NN=0)

class Bank:
 def __init__(self,path):
  self.path=Path(path);self.complete=d.read(self.path/'complete.json')
  if self.complete['status']!='SOURCE_L_BANK_COMPLETE' or self.complete['run_id']!=d.RUN or self.complete['count']!=d.BANK_SIZE or self.complete['role']!='L' or self.complete['target_access'] or self.complete['U_truth_used'] or self.complete['NN'] or self.complete['each_L_views']!=40 or self.complete['bin_edges']!=list(d.BIN_EDGES):raise ValueError('Completed own shared sourceL-only bank required')
  self.iq=np.load(self.path/'iq.npy',mmap_mode='r',allow_pickle=False)
  with np.load(self.path/'metadata.npz',allow_pickle=False) as f:self.meta={k:f[k].copy() for k in f.files}
  if self.iq.shape!=(d.BANK_SIZE,2,256) or self.iq.dtype!=np.float32 or self.meta['TX'].shape!=(d.BANK_SIZE,) or self.meta['sample_ids'].shape!=(d.BANK_SIZE,5) or np.bincount(self.meta['bin'],minlength=4).tolist()!=self.complete['bin_counts']:raise ValueError('Bank physical/index shape differs')
 def draw(self,mode,epoch,step):
  indices,w,record=select_indices(self.meta['bin'],mode,epoch,step);tic=time.perf_counter();iq=np.array(self.iq[indices],dtype=np.float32,copy=True,order='C');m={k:v[indices].copy() for k,v in self.meta.items()};record['cache_gather_cpu_wall_seconds']=time.perf_counter()-tic
  return iq,m,record

class Sampler:
 def __init__(self,mode,path):self.mode=mode;self.path=Path(path);self.bank=None;self.calls=0;self.packets=0;self.coverage=Counter()
 def __deepcopy__(self,memo):
  obj=type(self)(self.mode,self.path);memo[id(self)]=obj;return obj
 def draw(self,epoch,step):
  if self.bank is None:self.bank=Bank(self.path)
  iq,meta,record=self.bank.draw(self.mode,epoch+20,step+4440);self.calls+=1;self.packets+=64
  for r,day,s,t,l in zip(meta['RX'],meta['day'],meta['scene'],meta['state'],meta['lock']):self.coverage[(int(r),int(day),str(s),str(t),str(l))]+=1
  return iq,meta,record
 def audit(self):return dict(mode=self.mode,bank=str(self.path),complete_practical_generated_once=True,training_simulations=0,read_only_bank=True,formal_test_changed=False)
 def summary(self):return dict(mode=self.mode,calls=self.calls,packets=self.packets,bank_cache_reads=self.packets,training_propagation_simulations=0,bank=str(self.path),strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(self.coverage.items())],dropped=0,confidence_filter=False,ordinary_before_satellite=False,formal_test_changed=False)

def _prepare_bank():
 """Dispatcher-only CPU generation after synthetic smoke,before spawning either row."""
 import gc,torch
 from .channel import tensor_iq
 from experiments.cvs_phase1_stack.runtime import installed
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 from experiments.cvs_clock_factorial30.channel import Practical
 if d.BANK.exists():raise FileExistsError('Immutable shared bank already exists;reconcile,no automatic rebuild')
 start=time.perf_counter();d.BANK.mkdir();c=d.config(d.rows()[0]);a=d.make_args(c,'cpu');torch.set_num_threads(2)
 cache_started=time.perf_counter()
 with installed(c) as native:
  ctx=native._build_ssdg_wisig_data(a,torch.device('cpu'))
  if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=d.read(d.SOURCE)['role_ids']:raise ValueError('Bank only exact existing source role handles')
  ds=ctx['train_loader'].dataset
  if len(ds)!=6300 or ctx['num_domains']!=15:raise ValueError('Complete all5RX L6300 required')
  raw=np.empty((6300,2,256),np.float32);tx=np.empty(6300,np.int64);dom=np.empty(6300,np.int64);ids=np.empty((6300,5),np.int64)
  for i in range(6300):
   x,y,domain,m=ds[i];raw[i]=x.tolist();tx[i]=int(y);dom[i]=int(ctx['domain_label_map'][int(domain)]);ids[i]=[int(m[k]) for k in FIELDS]
  if set(tx.tolist())!=set(range(6)) or set(ids[:,1].tolist())!=set(d.RX) or set(ids[:,2].tolist())!={1,2,3} or set(ids[:,0].tolist())!={1} or not np.isfinite(raw).all() or len(set(map(tuple,ids.tolist())))!=6300:raise ValueError('Exact visible L raw/physical contract differs')
  del ds,ctx
 gc.collect();raw_cache_seconds=time.perf_counter()-cache_started
 n=d.BANK_SIZE;iq=np.lib.format.open_memmap(d.BANK/'iq.npy',mode='w+',dtype=np.float32,shape=(n,2,256));channel=Practical('train')
 arrays={k:np.empty(n,dtype=dtype) for k,dtype in [('TX',np.int64),('RX',np.int64),('day',np.int64),('domain',np.int64),('bin',np.int8),('scene','U32'),('state','U16'),('lock','U16'),('signal_to_added_noise_db',np.float64),('quality_snr_db',np.float64),('cumulative_epoch',np.int64),('replica',np.int64),('context_step',np.int64),('source_L_index',np.int64)]};arrays['sample_ids']=np.empty((n,5),np.int64)
 offset=0;simulation_seconds=copy_seconds=metadata_seconds=0.
 with gzip.open(d.BANK/'physics.jsonl.gz','xt',encoding='utf8',compresslevel=3) as stream:
  for context in d.bank_contexts():
   epoch=context['cumulative_epoch'];step=context['cumulative_step'];rep=context['replica']
   for begin in range(0,6300,64):
    end=min(6300,begin+64);count=end-begin;metadata={k:torch.tensor(ids[begin:end,j].tolist(),dtype=torch.long) for j,k in enumerate(FIELDS)};tic=time.perf_counter();out,observed=channel.apply(tensor_iq(raw[begin:end],'cpu'),metadata,epoch=epoch,step=step);simulation_seconds+=time.perf_counter()-tic
    ratios=np.asarray([o['signal_to_added_noise_db'] for o in observed],np.float64);quality=np.asarray([o['quality_snr_db'] for o in observed],np.float64);bb=bins(ratios);sl=slice(offset,offset+count);tic=time.perf_counter();iq[sl]=out.tolist();copy_seconds+=time.perf_counter()-tic
    tic=time.perf_counter();arrays['TX'][sl]=tx[begin:end];arrays['RX'][sl]=ids[begin:end,1];arrays['day'][sl]=ids[begin:end,2];arrays['domain'][sl]=dom[begin:end];arrays['sample_ids'][sl]=ids[begin:end];arrays['bin'][sl]=bb;arrays['scene'][sl]=[o['scene'] for o in observed];arrays['state'][sl]=[o['state_end'] for o in observed];arrays['lock'][sl]=[o['receiver_lock_status'] for o in observed];arrays['signal_to_added_noise_db'][sl]=ratios;arrays['quality_snr_db'][sl]=quality;arrays['cumulative_epoch'][sl]=epoch;arrays['replica'][sl]=rep;arrays['context_step'][sl]=step;arrays['source_L_index'][sl]=np.arange(begin,end)
    for j,o in enumerate(observed):stream.write(json.dumps(dict(bank_index=offset+j,TX=int(tx[begin+j]),domain=int(dom[begin+j]),source_L_index=begin+j,bin=int(bb[j]),replica=rep,cumulative_epoch=epoch,context_step=step,physics=o),allow_nan=False)+'\n')
    metadata_seconds+=time.perf_counter()-tic;offset+=count
   print(json.dumps(dict(event='BANK_CONTEXT_COMPLETE',context=context,generated=offset,NN=0,U_truth_used=False,target_access=False)),flush=True)
 if offset!=n:raise ValueError('Shared bank exact size differs')
 iq.flush();del iq;counts=np.bincount(arrays['bin'],minlength=4)
 if (counts<=0).any():raise ValueError('Fixed proposal bank has empty quality bin;preserve,no rescue')
 tic=time.perf_counter();np.savez(d.BANK/'metadata.npz',**arrays);metadata_seconds+=time.perf_counter()-tic
 strata=Counter(zip(arrays['TX'].tolist(),arrays['RX'].tolist(),arrays['day'].tolist(),arrays['bin'].tolist(),arrays['scene'].tolist(),arrays['state'].tolist(),arrays['lock'].tolist()))
 value=dict(status='SOURCE_L_BANK_COMPLETE',run_id=d.RUN,role='L',count=n,source_L=6300,each_L_views=40,replicas_per_epoch=4,contexts=d.bank_contexts(),shape=[n,2,256],dtype='float32',IQ_bytes=n*2*256*4,bin_field='signal_to_added_noise_db',bin_edges=list(d.BIN_EDGES),bin_counts=counts.tolist(),nominal_bin_mass=(counts/n).tolist(),sample_id_fields=list(FIELDS),physics_file='physics.jsonl.gz',index_file='metadata.npz',IQ_file='iq.npy',source_contract=str(d.SOURCE),role_ids=d.read(d.SOURCE)['role_ids']['L_s'],source_role_key='L_s',channel=channel.audit(),channel_generation=channel.summary(),source_L_index_counts=np.bincount(arrays['source_L_index'],minlength=6300).tolist(),strata=[dict(TX=k[0],RX=k[1],day=k[2],bin=k[3],scene=k[4],state=k[5],lock=k[6],count=v) for k,v in sorted(strata.items())],raw_dataset_cache_cpu_wall_seconds=raw_cache_seconds,simulation_adapter_cpu_wall_seconds=simulation_seconds,IQ_cache_write_cpu_wall_seconds=copy_seconds,metadata_write_cpu_wall_seconds=metadata_seconds,wall_seconds=time.perf_counter()-start,timing='CPU perf_counter;simulation_adapter includes original normalization/bridge/process;disjoint IQ/metadata cache writes;not isolated residual kernel benchmarking',propagation_simulations=n,NN=0,checkpoint_reads=0,U_truth_used=False,U_iterations=0,target_access=False,quality_rejection=False,retries=0,bank_online_risk_equivalence_claim=False)
 d.write(d.BANK/'complete.json',value);return value


def prepare_bank():
 import random,torch
 before_np=np.random.get_state();before_py=random.getstate()
 try:
  with torch.random.fork_rng(devices=[]):return _prepare_bank()
 finally:np.random.set_state(before_np);random.setstate(before_py)
