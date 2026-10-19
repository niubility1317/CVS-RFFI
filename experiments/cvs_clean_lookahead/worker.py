import os,time,traceback,gc,random
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .data import observations
from .virtual import discardable_step
from experiments.cvs_F_nontrue20 import design as base,model as api
from experiments.cvs_rdrop_sinc_units import optimizer as units
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def gradients(loss,named):return dict(zip(named,torch.autograd.grad(loss,tuple(named.values()),allow_unused=True)))
def combine(a,b):return {k:None if a[k] is None and b[k] is None else .5*((0. if a[k] is None else a[k])+(0. if b[k] is None else b[k])) for k in a}
def geometry(a,b):
 dot=aa=bb=0.
 for k in a:
  x=a[k];y=b[k];scale=25000. if k in units.QNAMES else 1.
  if x is not None:aa+=float((x.double()/scale).square().sum())
  if y is not None:bb+=float((y.double()/scale).square().sum())
  if x is not None and y is not None:dot+=float(((x.double()/scale)*(y.double()/scale)).sum())
 return dict(cosine=None if aa==0 or bb==0 else dot/(aa*bb)**.5,norm_a=aa**.5,norm_b=bb**.5,physical_Sinc_metric=True)

def batch_response(m,train,response,y,ry,forward,step,counts):
 ce=lambda z,y:torch.nn.functional.cross_entropy(z,y,label_smoothing=.01)
 def evaluate(model):
  result={}
  with torch.no_grad():
   for view,x in response.items():
    z=forward(model,x);result[view]=dict(CE=float(ce(z,ry)),correct=int((z.argmax(1)==ry).sum()),n=len(ry),logits=z.tolist())
  return result
 named=dict(m.named_parameters());zr=forward(m,train['raw']);graw=gradients(ce(zr,y),named)
 zs=[forward(m,train[v]) for v in d.SCENES];gas=gradients(sum(ce(z,y) for z in zs)/2,named);counts['network_backward_packets']+=3*len(y)
 measured=dict(BASE=evaluate(m));raw_step,wr=step(m,graw)
 shifted=[forward(raw_step,train[v]) for v in d.SCENES];shiftg=gradients(sum(ce(z,y) for z in shifted)/2,dict(raw_step.named_parameters()));counts['network_backward_packets']+=2*len(y)
 measured['RAW_STEP']=evaluate(raw_step);del raw_step
 joint,wj=step(m,combine(graw,gas));measured['JOINT_STEP']=evaluate(joint);del joint
 look,wl=step(m,combine(graw,shiftg));measured['FIRST_ORDER_LOOKAHEAD']=evaluate(look);del look
 return measured,dict(As_before_after_raw_step=geometry(gas,shiftg),raw_As_at_original=geometry(graw,gas)),dict(RAW_STEP=wr,JOINT_STEP=wj,FIRST_ORDER_LOOKAHEAD=wl)
def qualify(rows):
 by={}
 for arm in ('B0','B1'):
  rs=[r for r in rows if r['arm']==arm]
  urban=[sum(r['response']['FIRST_ORDER_LOOKAHEAD'][v]['CE']-r['response']['JOINT_STEP'][v]['CE'] for v in d.SCENES)/2 for r in rs]
  clean=[r['response']['FIRST_ORDER_LOOKAHEAD']['raw']['CE']-r['response']['JOINT_STEP']['raw']['CE'] for r in rs]
  by[arm]=dict(urban_CE_delta=float(np.mean(urban)),raw_CE_delta=float(np.mean(clean)),urban_better_batches=sum(v < -1e-6 for v in urban),batches=len(rs))
 passed=all(v['batches']==6 and v['urban_CE_delta'] < -1e-6 and v['raw_CE_delta']<=1e-6 and v['urban_better_batches']>=4 for v in by.values())
 return dict(rationale_pass=passed,rows=by,automatic_training=False,E50_qualified=False,recognition_benefit_proved=False,decision='MECHANISTIC_SUPPORT_ONLY; freeze a separate E20 recipe before any training' if passed else 'CLOSE_FIXED_FRESH_ADAM_RAW_LOOKAHEAD_RATIONALE; no step-size/inner-state/seed rescue',target_access=False)

def run():
 with lock(d.BASE/'owner.lock',nonblocking=True):return _run()

def _run():
 d.validate(d.config());out=d.BASE;start=time.perf_counter();torch.set_num_threads(2);counts=dict(NN=0,network_backward_packets=0,virtual_updates=0);rows=[]
 def forward(m,x):counts['NN']+=len(x);return m.forward_identity_only(x)['tx_logits']
 def ce(z,y):return torch.nn.functional.cross_entropy(z,y,label_smoothing=.01)
 def step(m,g):
  result,witness=discardable_step(m,g);counts['virtual_updates']+=1;return result,witness
 try:
  if os.geteuid()==0:raise ValueError('Ordinary account required')
  allocation=d.read(out/'allocation.json')
  if allocation['pid']!=os.getpid() or allocation['owner']!=d.OWNER:raise ValueError('Single allocated launch owner required')
  gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
  with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
   if len(capacity()[gpu]['pids']|{os.getpid()})>2:raise ValueError('Global GPU cap2 includes other owners')
   torch.cuda.init()
  with numerical_context(base.FULL_FP32_POLICY):
   models={};snap={}
   # First: one real-checkpoint no-query smoke per actual fixed source model.
   for arm in ('B0','B1'):
    c=base.config(next(r for r in base.rows() if r['arm']==arm));m=api.own_model(c,'cuda:0').requires_grad_(True);models[arm]=m;snap[arm]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    with torch.random.fork_rng(devices=[0]):
     torch.manual_seed(20261019626);train={v:torch.randn(2,2,256,device='cuda:0') for v in d.VIEWS};response={v:torch.randn(2,2,256,device='cuda:0') for v in d.VIEWS};y=torch.tensor([0,4],device='cuda:0')
     measured,geo,witness=batch_response(m,train,response,y,y,forward,step,counts)
     if any(not np.isfinite(v['CE']) or np.asarray(v['logits']).shape!=(2,6) for result in measured.values() for v in result.values()):raise ValueError('Actual complete discarded response smoke differs')
    if any(p.grad is not None for p in m.parameters()) or any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in snap[arm].items()):raise ValueError('Real checkpoint smoke changed original state')
   d.write(out/'checkpoint_smoke.json',dict(status='PASS',synthetic_NN=counts['NN'],real_CP=2,virtual_updates=counts['virtual_updates'],original_updates=0,source_IQ_opened=False,target_access=False));print('SMOKE PASS two actual pureF checkpoints and discarded defaulthead/Sinc update; no query',flush=True)
   data=observations(out);rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone();pyrng=random.getstate();nrng=np.random.get_state()
   d.write(out/'resolved_config.json',dict(config=d.config(),commit=(d.ROOT/'release_commit.txt').read_text().strip(),checkpoint_provenance='Existing strict own_model/source contract/fixed FINAL20 reused',parameters={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},FP32_TF32_off=True,original_optimizer_not_loaded=True,U_truth_read=False,V_used=False,target_access=False))
   for arm,m in models.items():
    for batch in range(6):
     lo=batch*30;hi=lo+30;y=data['y'][lo:hi].to('cuda:0');ry=data['y'][lo+180:hi+180].to('cuda:0');train={v:x[lo:hi].to('cuda:0') for v,x in data['views'].items()};response={v:x[lo+180:hi+180].to('cuda:0') for v,x in data['views'].items()}
     measured,geo,witness=batch_response(m,train,response,y,ry,forward,step,counts)
     rec=dict(arm=arm,batch=batch,step_ids=data['ids'][lo:hi],response_ids=data['ids'][lo+180:hi+180],TX=ry.tolist(),RX=data['rx'][lo+180:hi+180].tolist(),day=data['day'][lo+180:hi+180].tolist(),response=measured,geometry=geo,virtual_witnesses=witness)
     rows.append(rec);d.write(out/'progress.json',dict(arm=arm,completed_batches=len(rows),expected_batches=12,**counts,original_updates=0,target_access=False));print(f'{arm} batch {batch+1}/6 NN={counts["NN"]} discardedSteps={counts["virtual_updates"]}',flush=True)
     gc.collect()
   if counts!=dict(NN=6188,network_backward_packets=1820,virtual_updates=42):raise ValueError('Full fixed source probe computation budget differs: '+str(counts))
   if not torch.equal(rng,torch.get_rng_state()) or not torch.equal(crng,torch.cuda.get_rng_state()) or pyrng!=random.getstate():raise ValueError('Frozen source probe changed global RNG')
   now=np.random.get_state()
   if nrng[0]!=now[0] or not np.array_equal(nrng[1],now[1]) or nrng[2:]!=now[2:]:raise ValueError('Frozen source probe changed NumPy RNG')
   for arm,m in models.items():
    if m.training or any(p.grad is not None for p in m.parameters()) or any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in snap[arm].items()):raise ValueError('Original checkpoint weights/buffers/grads changed')
    d.write(out/arm/'responses.json',dict(records=[r for r in rows if r['arm']==arm],source_only=True,target_access=False))
    d.write(out/arm/'complete.json',dict(status='SOURCE_DIAGNOSED',arm=arm,NN=3094,network_backward_packets=910,virtual_updates=21,original_model_updates=0,head_only_fit_updates=0,target_access=False,original_state_and_grads_unchanged=True))
   decision=qualify(rows);d.write(out/'responses.json',dict(records=rows,source_only=True,target_access=False));d.write(out/'source_decision.json',decision)
   d.write(out/'complete.json',dict(status='SOURCE_DIAGNOSED',config=d.config(),**counts,original_model_updates=0,new_trained_models=0,head_only_fit_updates=0,original_states_and_grads_unchanged=True,global_RNG_unchanged=True,source_L_unique=360,V_used=False,U_truth_read=False,target_access=False,wall_seconds=time.perf_counter()-start,decision=decision,goal_complete=False));print(str(decision),flush=True)
 except BaseException as error:
  d.write(out/'failure.json',dict(status='TECHNICAL_FAILED',error=str(error),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
