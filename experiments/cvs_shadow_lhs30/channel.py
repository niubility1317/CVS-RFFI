"""Full original practical;only independent shadow initialvalue sampling is organized."""
from collections import Counter
from dataclasses import asdict
import gzip,json,math,time
import numpy as np
from . import design as d
from .sampling import apply_shadow
def serial(v):
 if isinstance(v,np.generic):return serial(v.item())
 if isinstance(v,np.ndarray):return serial(v.tolist())
 if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [serial(x) for x in v]
 if isinstance(v,complex):return [v.real,v.imag]
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
class Practical:
 def __init__(self,mode='NATIVE'):
  if mode not in ('NATIVE','IID','LHS'):raise ValueError('Only fixed shadow mode')
  import leo_practical.channel as ch
  import leo_practical.residual as residual
  self.mode=mode;self.ch=ch;self.residual=residual;self.calls=0;self.packets=0;self.coverage=Counter();self.sessions={};self.log=None;self.sampling_seconds=0.;self.sampling_groups=0;self.shadow_changed=0;self.strata_failures=0
  self.configs={s:ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.SCENES}
 def __deepcopy__(self,memo):
  v=type(self)(self.mode);memo[id(self)]=v;return v
 def attach_logger(self,out):
  if self.log is not None:raise ValueError('One owned shadow logger only')
  self.log=gzip.open(out/'shadow_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3)
 def flush(self):
  if self.log is not None:self.log.flush()
 def close(self):
  if self.log is not None:self.log.close();self.log=None
 def audit(self):return dict(channel_module_file=self.ch.__file__,residual_module_file=self.residual.__file__,configurations={s:asdict(c) for s,c in self.configs.items()},all_original_RX=list(d.FIT),shadow_sampling=dict(mode=self.mode,variable='latent.z only',conditional_groups='scene,state',seed=d.SAMPLING_SEED),ordinary_before_satellite=False,formal_test_changed=False)
 def process_numpy(self,raw,meta,epoch,step):
  raw=np.asarray(raw,dtype=np.float64);keys=('eq_i','rx_i','day_i','sig_i','base_index')
  if epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1 or raw.ndim!=3 or raw.shape[1:]!=(2,256) or not np.isfinite(raw).all():raise ValueError('Finite L256/cumulativeE21..30 only')
  if any(k not in meta or len(meta[k])!=len(raw) for k in keys):raise ValueError('Physical source metadata required')
  v={k:list(meta[k]) for k in keys}
  if set(map(int,v['rx_i']))-set(d.FIT) or set(map(int,v['day_i']))-{1,2,3} or set(map(int,v['eq_i']))!={1}:raise ValueError('Original5RX/day/EQ1 required')
  namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S';snapshots=[];info=[];groups=[]
  for i in range(len(raw)):
   pid=tuple(int(v[k][i]) for k in keys);rx=int(v['rx_i'][i]);day=int(v['day_i'][i]);session=f'rx{rx}:day{day}:eq{int(v["eq_i"][i])}';rng=np.random.default_rng(self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,'scene',pid));scene=str(rng.choice(d.TRAIN_SCENES));cfg=self.configs[scene];seed=self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,scene,pid);hw=self.ch.receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],session);hardware=asdict(hw)
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('Original session hardware changed')
   snapshot=self.residual.ResidualChannel(cfg,seed,hw);snapshots.append(snapshot);groups.append(3*d.TRAIN_SCENES.index(scene)+int(snapshot.latent.state));info.append((pid,rx,day,session,scene))
  t=time.perf_counter();seed=self.ch.stable_seed(d.SAMPLING_SEED,'LooShadow/conditional/scene-state',epoch,step);sampling=apply_shadow(self.mode,[s.latent for s in snapshots],np.asarray(groups,dtype=np.int64),seed);seconds=time.perf_counter()-t;self.sampling_seconds+=seconds;group_stats=[]
  for key in sorted(set(groups)):
   ix=[i for i,g in enumerate(groups) if g==key];n=len(ix);self.sampling_groups+=1
   if self.mode=='LHS' and sorted(sampling[i]['stratum'] for i in ix)!=list(range(n)):self.strata_failures+=1;raise ValueError('Conditional LHS probability strata incomplete')
   group_stats.append(dict(group=key,count=n,full_probability_strata=self.mode=='LHS'))
  out=[];records=[];sampler_rows=[]
  for i,(snapshot,record) in enumerate(zip(snapshots,sampling)):
   pid,rx,day,session,scene=info[i];normalized,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);signal,m=snapshot.process(normalized)
   if not np.isfinite(signal).all() or m['channel_equalization_applied'] or m['quality_source']!='white_input_expected_power_proxy':raise ValueError('Original full residual/quality/noEQ/finite output required;no rejection')
   self.shadow_changed+=int(record['shadow_used']!=record['shadow_original']);self.coverage[(rx,day,scene,m['state_end'],m['receiver_lock_status'])]+=1;out.append(np.stack((signal.real,signal.imag)));records.append(serial(dict(m,scene=scene,sample_id=pid,session_id=session,namespace=namespace,shadow_sampling=record)));sampler_rows.append(dict(record,sample_id=pid))
  self.calls+=1;self.packets+=len(raw)
  if self.log is not None:self.log.write(json.dumps(dict(step=step,epoch=epoch,mode=self.mode,packets=len(raw),groups=group_stats,sampling_seconds=seconds,records=sampler_rows),allow_nan=False)+'\n')
  return np.ascontiguousarray(out,dtype=np.float32),records
 def apply(self,x,meta,epoch,step):
  import torch
  raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);v={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')};a,records=self.process_numpy(raw,v,epoch,step)
  return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device),records
 def summary(self):return dict(calls=self.calls,packets=self.packets,session_count=len(self.sessions),sessions=self.sessions,strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(self.coverage.items())],all_original_source_RX=list(d.FIT),dropped=0,confidence_filter=False,ordinary_before_satellite=False,formal_test_changed=False,shadow_sampling=dict(mode=self.mode,groups=self.sampling_groups,changed=self.shadow_changed,strata_failures=self.strata_failures,sampling_seconds=self.sampling_seconds,original_singlepacket_prior=True,new_Ag=False))
