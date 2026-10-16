"""Frozen current F+CROSS_RX weights; fit-L gradients only, never updates."""
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import gzip,json,time
import numpy as np
import torch
from . import design as d
from .core import measure,decision
from experiments.cvs_f_crossrx_practical20 import model as api
from experiments.cvs_f_crossrx_practical20.supcon import LEVELS,PREFIX,penalty
from experiments.cvs_ordinary_fold20.streams import stream
from experiments.cvs_urban_sigtime.channel import imports,numpy_iq,tensor_iq
from experiments.cvs_equivariant_identity.precision import numerical_context

GROUPS=('default_head','SIG_TIME','Sinc_Hz','time_path','remaining')
def group(name):
 if '.cls_head.' in name:return 'default_head'
 if '.sequence_pool.' in name:return 'SIG_TIME'
 if '.sinc.' in name:return 'Sinc_Hz'
 if any('.'+s+'.' in name for s in ('time_stem','time_down','t1','t2')):return 'time_path'
 return 'remaining'
def gradients(value,named,retain=False):
 g=torch.autograd.grad(value,[p for n,p in named],allow_unused=True,retain_graph=retain)
 result={k:[] for k in GROUPS}
 for (n,p),v in zip(named,g):
  v=torch.zeros_like(p) if v is None else v.detach()
  if n.endswith(('.sinc.low_coordinate','.sinc.band_coordinate')):v=v/25000.
  result[group(n)].append(v.reshape(-1).double())
 return {k:torch.cat(v) if v else torch.empty(0,device=named[0][1].device,dtype=torch.float64) for k,v in result.items()}
def primary(model,x,y,domains,rx,named,call):
 maps=[{},{}];phase=[0]
 with ExitStack() as hooks:
  for level in LEVELS:
   def capture(mod,args,out,level=level):maps[phase[0]][level]=out
   handle=model.get_submodule(PREFIX+level).register_forward_hook(capture);hooks.callback(handle.remove)
  with stream(d.SEEDS['dropout_primary'],call,x.device):
   a=model.forward_identity_only(x,domain_labels=domains)['tx_logits'];phase[0]=1
   b=model.forward_identity_only(x,domain_labels=domains)['tx_logits']
  rd,ce,kl,_,_=api.raw.base.legacy.terms(a,b,y,1.,.01)
  aux=torch.stack([penalty(v,y,rx,'CROSS_RX')[0] for v in maps]).mean()
  gr=gradients(rd,named,retain=True);ga=gradients(aux,named)
 return {k:.5*gr[k]+.1*ga[k] for k in GROUPS},dict(RD=float(rd.detach()),CE=float(ce.detach()),KL=float(kl.detach()),SupCon=float(aux.detach())),((a+b)*.5).detach()
def secondary(model,x,y,domains,named,call):
 with stream(d.SEEDS['dropout_secondary'],call,x.device):
  a=model.forward_identity_only(x,domain_labels=domains)['tx_logits'];b=model.forward_identity_only(x,domain_labels=domains)['tx_logits']
 rd,ce,kl,_,_=api.raw.base.legacy.terms(a,b,y,1.,.01)
 g=gradients(.5*rd,named)
 return g,dict(RD=float(rd.detach()),CE=float(ce.detach()),KL=float(kl.detach())),((a+b)*.5).detach()
def geometry(p,s):
 result={}
 for k in GROUPS:result[k]=measure(float(p[k].square().sum()),float(s[k].square().sum()),float((p[k]*s[k]).sum()))
 result['whole']=measure(sum(r['primary_squared'] for r in result.values()),sum(r['secondary_squared'] for r in result.values()),sum(r['dot'] for r in result.values()))
 return result

def references():
 contracts={}
 for arm,path in d.REFERENCES.items():contracts[arm]=d.verify_reference(arm,d.read(path/'completion.json'),d.read(path/'initialization.json'),d.read(path/'source_contract.json'),d.read(path/'source_frozen.json'))
 if contracts['C0']['role_ids']!=contracts['C1']['role_ids']:raise ValueError('Current C0/C1 physical roles differ')
 return contracts

def load_L(contracts):
 expected=d.read(d.SOURCE);prepared=d.read(d.PREPARATION/'preparation_complete.json')
 if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']:raise ValueError('Legal existing source preparation required')
 packed=torch.load(d.PREPARATION/'source_cache.pt',map_location='cpu',weights_only=False)
 for actual in (packed['contract'],prepared['source_contract']):
  if any(actual.get(k)!=v for k,v in expected.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing carrier differs')
 L=packed['labeled'];ids=L['ids']
 if L['role']!='L_s' or len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(expected['role_ids']['L_s']) or set(ids)&(set(expected['role_ids']['U_s'])|set(expected['role_ids']['V'])):raise ValueError('Original legal L6300 required')
 keep=[i for i,r in enumerate(L['rx'].tolist()) if int(r) in d.FIT]
 if len(keep)!=5040 or {ids[i] for i in keep}!=set(contracts['C0']['role_ids']['L_s']):raise ValueError('Exact fit L5040 only')
 out={k:L[k][keep].clone() for k in ('x','y','rx','day','domain')};out['ids']=[ids[i] for i in keep]
 if out['x'].shape!=(5040,2,256) or not bool(torch.isfinite(out['x']).all()) or out['y'].dtype!=torch.long:raise ValueError('Finite native normalized IQ256 and true L labels required')
 if any(r not in d.FIT for r in out['rx'].tolist()) or any(v not in (1,2,3) for v in out['day'].tolist()):raise ValueError('Only fit source metadata allowed')
 return out

class Channel:
 def __init__(self):
  self.ch,self.residual=imports();self.sessions={};self.packets=0
  self.configs={s:self.ch.Config(fs_hz=25_000_000.,scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.prior.TRAIN_SCENES}
 def apply(self,x,ids,rx,day,rep):
  raw=numpy_iq(x);out=[];physics=[]
  for i,pid in enumerate(ids):
   namespace=f'FcrossRX/gradient_diagnostic/rep{rep}/fitL';rng=np.random.default_rng(self.ch.stable_seed(d.SEEDS['channel'],namespace,'scene',pid));scene=str(rng.choice(d.prior.TRAIN_SCENES));cfg=self.configs[scene]
   session=f'rx{int(rx[i])}:day{int(day[i])}:eq1';hw=self.ch.receiver_for_session(cfg,d.SEEDS['receiver'],session);hardware=asdict(hw)
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('RX hardware varied within physical session')
   z,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);value,m=self.residual.ResidualChannel(cfg,self.ch.stable_seed(d.SEEDS['channel'],namespace,scene,pid),hw).process(z)
   if m['channel_equalization_applied']:raise ValueError('Full residual must retain EQoff G1')
   m.update(sample_id=pid,session_id=session,namespace=namespace,scene=scene,ordinary_before_satellite=False);physics.append(m);out.append(np.stack((value.real,value.imag)))
  self.packets+=len(ids);return tensor_iq(np.stack(out),x.device),physics
 def audit(self):return dict(channel_module_file=self.ch.__file__,residual_module_file=self.residual.__file__,configurations={k:asdict(v) for k,v in self.configs.items()},EQ_off_G_equals_one=True,session_count=len(self.sessions),packets=self.packets,formal_test_changed=False)

def prepare(L,device,out):
 from SSDG import train_ssdg as native
 from experiments.cvs_f_crossrx_practical20.channel import serial
 args=d.prior.make_args(d.prior.config(d.prior.rows()[0]),str(device));cfg=native.build_aug_base_cfg(args)
 if cfg!=d.read(d.REFERENCES['C0']/'ordinary_base_config.json'):raise ValueError('Actual ordinary base config differs from current C0')
 with stream(d.SEEDS['ordinary'],0,device):augmentor=native.make_augmentor(cfg)
 augstate=native.configure_augmentor_for_epoch(augmentor,cfg,13,args)
 channel=Channel();baseline_configs=d.read(d.REFERENCES['C1']/'practical_channel_audit.json')['configurations']
 if any(json.loads(json.dumps(asdict(cfg)))!=baseline_configs[s] for s,cfg in channel.configs.items()):raise ValueError('Full original training channel config differs')
 order=np.random.default_rng(d.SEEDS['order']).permutation(5040).tolist();batches=[]
 with gzip.open(out/'observations.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as trace:
  for rep in d.REPS:
   for batch,start in enumerate(range(0,5040,d.BATCH)):
    indices=order[start:start+d.BATCH];call=rep*40+batch+1;x=L['x'][indices].to(device);y=L['y'][indices].to(device);ids=[L['ids'][i] for i in indices];rx=L['rx'][indices];day=L['day'][indices]
    with stream(d.SEEDS['ordinary'],call,device):ordinary=augmentor(x,labels=y,no_pa=(not bool(args.aug_enable_pa_normal)))
    sat,physics=channel.apply(x,ids,rx,day,rep)
    if not bool(torch.isfinite(ordinary).all() and torch.isfinite(sat).all()):raise ValueError('Nonfinite actual view')
    batches.append(dict(rep=rep,batch=batch,call=call,indices=indices,ordinary=ordinary.detach().cpu(),As=sat.detach().cpu(),physics=serial(physics)))
    trace.write(json.dumps(dict(rep=rep,batch=batch,ids=ids,physics=serial(physics)),allow_nan=False)+'\n')
 d.write(out/'observation_audit.json',dict(ordinary_base=cfg,ordinary_epoch_state=augstate,native_no_pa=not bool(args.aug_enable_pa_normal),channel=channel.audit(),source_L_unique=5040,ordinary_calls=80,ordinary_packets=10080,full_As_packets=10080,sourceV_used=False,U_used=False,target_access=False))
 return batches

def record_predictions(file,model,rep,batch,kind,logits,L,indices,physics=None):
 y=L['y'][indices].to(logits.device);margin=(logits.gather(1,y[:,None]).flatten()-logits.masked_fill(torch.nn.functional.one_hot(y,6).bool(),-torch.inf).max(1).values)/30
 for j,i in enumerate(indices):
  row=dict(checkpoint=model,rep=rep,batch=batch,view=kind,physical_id=L['ids'][i],TX=int(L['y'][i]),RX=int(L['rx'][i]),day=int(L['day'][i]),prediction=int(logits[j].argmax()),margin_over30=float(margin[j]))
  if physics is not None:row['physics']=physics[j]
  file.write(json.dumps(row,allow_nan=False)+'\n')

def run(device):
 c=d.config(d.rows()[0]);d.validate(c);out=Path(c['output_root']);out.mkdir(exist_ok=False);started=time.perf_counter();torch.set_num_threads(2)
 contracts=references();d.write(out/'resolved_config.json',c);rows=[];forwards=0
 with numerical_context(d.FULL_FP32_POLICY):
  models={};before={}
  for arm,path in d.REFERENCES.items():
   model=api.build(d.prior.config(d.prior.rows()[('C0','C1').index(arm)]),device);payload=torch.load(path/'final_ssdg.pth',map_location='cpu',weights_only=False)
   if payload['epoch']!=20 or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or payload['checkpoint_selection']!='final_only':raise ValueError('Frozen current final20 checkpoint payload differs')
   model.load_state_dict(payload['model'],strict=True);del payload
   if any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in model.modules()):raise ValueError('This diagnostic requires no BatchNorm state')
   model.eval()
   with torch.no_grad():smoke=model.forward_identity_only(torch.zeros(2,2,256,device=device))['tx_logits']
   if smoke.shape!=(2,6) or not bool(torch.isfinite(smoke).all()):raise ValueError('Real checkpoint no-query smoke failed')
   forwards+=2;before[arm]={k:v.detach().clone() for k,v in model.state_dict().items()};models[arm]=model
  d.write(out/'checkpoint_noquery_smoke.json',dict(status='PASS',real_checkpoints=2,synthetic_packets=4,IQ_reads_before_smoke=0,query_reads=0,optimizer_updates=0))
  L=load_L(contracts);batches=prepare(L,torch.device(device),out)
  with gzip.open(out/'gradients.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as gout,gzip.open(out/'predictions.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as pout:
   for arm,model in models.items():
    model.train();named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    for obs in batches:
     ix=obs['indices'];y=L['y'][ix].to(device);dom=L['domain'][ix].to(device);rx=L['rx'][ix].to(device);ordinary=obs['ordinary'].to(device)
     p,terms,logits=primary(model,ordinary,y,dom,rx,named,obs['call']);forwards+=2*len(ix)
     record_predictions(pout,arm,obs['rep'],obs['batch'],'primary_ordinary',logits,L,ix)
     for kind in ('same_ordinary','As'):
      x=ordinary if kind=='same_ordinary' else obs['As'].to(device);s,sterms,slogits=secondary(model,x,y,dom,named,obs['call']);forwards+=2*len(ix)
      for k,stats in geometry(p,s).items():
       row=dict(checkpoint=arm,rep=obs['rep'],batch=obs['batch'],count=len(ix),secondary=kind,group=k,primary_terms=terms,secondary_terms=sterms,**stats);rows.append(row);gout.write(json.dumps(row,allow_nan=False)+'\n')
      record_predictions(pout,arm,obs['rep'],obs['batch'],kind,slogits,L,ix,obs['physics'] if kind=='As' else None)
     print(json.dumps(dict(checkpoint=arm,rep=obs['rep'],batch=obs['batch'],completed_batches=len(rows)//12,NN_packets=forwards)),flush=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before[arm].items()) or any(p.grad is not None for n,p in named):raise ValueError('Frozen parameter/buffer/gradient state changed')
  if forwards!=c['expected_model_packets'] or len(rows)!=1920:raise ValueError('Complete fixed gradient budget required')
 summary=dict(status='COMPLETE_SOURCE_DIAGNOSTIC',decision=decision(rows),NN_packets=forwards,gradient_rows=len(rows),predictions=c['expected_decisions'],source_L_unique=5040,model_seed=d.SEED,checkpoint_loads=2,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,parameters_and_buffers_unchanged=True,source_only=True,target_access=False,V_used=False,U_used=False,wall_seconds=time.perf_counter()-started,goal_completed=False)
 d.write(out/'summary.json',summary);d.write(d.BASE/'completion.json',{**summary,'status':'ARTIFACTS_COMPLETE'});return summary
