"""Source L input gradients, then fixed source V readouts;zero parameter updates."""
import os,sys,time
from collections import Counter
import numpy as np
import torch
from . import design as d,dsp

def select(packed,ready,contract):
 if ready.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or ready.get('target_access') is not False:raise ValueError('Matching existing source preparation required')
 for c in (packed['contract'],ready['source_contract']):
  if c.get('role_ids')!=contract['role_ids'] or c.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing source role handles differ')
 chosen={}
 for key,handle,n,per,take in [('labeled','L_s',6300,70,2),('validation','V',27000,300,20)]:
  role={k:packed[key][k] for k in ('x','y','rx','day','ids')};ids=role['ids']
  if tuple(role['x'].shape)!=(n,2,256) or len(ids)!=n or len(set(ids))!=n or set(ids)!=set(contract['role_ids'][handle]):raise ValueError('Exact complete original source role required')
  if any(set(ids)&set(contract['role_ids'][other]) for other in ('L_s','U_s','V') if other!=handle):raise ValueError('Source roles overlap')
  y,rx,day=(role[k].tolist() for k in ('y','rx','day'))
  if Counter(zip(y,rx,day))!=Counter({(t,r,z):per for t in range(6) for r in d.FIT for z in (1,2,3)}):raise ValueError('All90 original source cells required')
  indices=[]
  for t in range(6):
   for r in d.FIT:
    for z in (1,2,3):indices+=sorted((i for i in range(n) if (y[i],rx[i],day[i])==(t,r,z)),key=lambda i:ids[i])[:take]
  chosen[handle]={k:role[k][indices] for k in ('x','y','rx','day')};chosen[handle]['ids']=[ids[i] for i in indices]
 return chosen

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES') in ('',None):raise ValueError('Reserved one GPU required')
 d.validate(d.config());out=d.resolve(d.BASE);started=time.perf_counter();torch.set_num_threads(2)
 from experiments.cvs_fullsource_concat20 import design as ref,proof,model as api
 from experiments.cvs_equivariant_identity.precision import numerical_context
 contract=d.read(d.resolve(d.SOURCE));models={};states={};NN=0;gradient_calls=0;gradient_packets=0;gradient_unavailable=[]
 with numerical_context(ref.FULL_FP32_POLICY):
  # First model action:real-checkpoint no-query smoke,with no source IQ loaded.
  for arm in d.MODELS:
   c=ref.config(next(r for r in ref.rows() if r['row_id']==arm+'-s2026092701'));p=proof.source(c);sc=d.read(p/'source_contract.json')
   if sc['role_ids']!=contract['role_ids'] or sc.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
   cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
   if (cp['epoch'],cp['run_id'],cp['candidate_id'],cp['a1_ema_successful_updates'],cp['checkpoint_selection'])!=(20,d.REFERENCE_RUN,arm+'-s2026092701',4440,'final_only') or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt'] or cp['args'].get('source_fold_held_rx') is not None or cp['args']['freeze_backbone']:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
   m=api.build(c,'cuda:0');m.load_state_dict(cp['model'],strict=True);m.eval().requires_grad_(False);del cp
   head=m.id_backbone.encoder.core.id_backbone.cls_head
   if head.weight.shape!=(6,160) or head.scale!=30. or m.id_backbone.encoder.sequence_pool.modes!=('SIG_TIME',)*3 or m.id_backbone.encoder.response.contract()['explicit_CFO_identity_input'] is not False:raise ValueError('F/defaulthead/NoExplicitCFO required')
   t=torch.arange(256,device='cuda:0',dtype=torch.float32);smoke=torch.stack((torch.cos(t*.31),torch.sin(t*.31))).repeat(2,1,1)
   if arm=='REPEAT':
    smoke.requires_grad_(True);s=m.forward_identity_only(smoke)
    if s['tx_logits'].requires_grad:
     grad=torch.autograd.grad(s['tx_logits'][:,0].sum(),smoke,allow_unused=True)[0];gradient_calls+=1;gradient_packets+=2
     if grad is not None and not bool(torch.isfinite(grad).all()):raise ValueError('Nonfinite synthetic input gradient')
    else:gradient_unavailable.append('smoke')
   else:
    with torch.no_grad():s=m.forward_identity_only(smoke)
   NN+=2
   if s['tx_logits'].shape!=(2,6) or s['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(s['tx_logits']).all()):raise ValueError('Real-checkpoint smoke failed')
   models[arm]=m;states[arm]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};del s,smoke
  d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',CP=2,NN=4,real_IQ=0,query_read=False,updates=0));print('SMOKE PASS 2CP/defaulthead/noquery;continue sourceLtoV',flush=True)
  prep=d.resolve(d.PREPARATION);ready=d.read(prep/'preparation_complete.json');packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False);roles=select(packed,ready,contract);del packed
  L=roles['L_s'];V=roles['V'];derivatives=[]
  w=torch.tensor(dsp.windows().tolist(),device='cuda:0',dtype=torch.float32)
  for start in range(0,180,128):
   x=L['x'][start:start+128].float().to('cuda:0').detach().requires_grad_(True);y=L['y'][start:start+128].long().to('cuda:0');m=models['REPEAT'];s=m.forward_identity_only(x);NN+=len(x);logits=s['tx_logits']
   mask=torch.nn.functional.one_hot(y,6).bool();margin=(logits.gather(1,y[:,None]).squeeze(1)-logits.masked_fill(mask,float('-inf')).max(1).values)/30.
   if margin.requires_grad:
    g=torch.autograd.grad(margin.sum(),x,allow_unused=True)[0];gradient_calls+=1;gradient_packets+=len(x)
   else:g=None;gradient_unavailable.append('fit/'+str(start))
   if g is None:g=torch.zeros_like(x)
   if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite source input gradient')
   energy=torch.sqrt(torch.einsum('nct,bt->nb',x.detach().square(),w.square()));value=torch.einsum('nct,bt->nb',x.detach()*g,w)/energy.clamp_min(1e-24)
   derivatives+=value.detach().cpu().tolist();del s,x,g,value,logits,margin
  ranking=dsp.rank(derivatives,*[L[k].tolist() for k in ('y','rx','day')]);d.write(out/'source_L_ranking.json',dict(ids=L['ids'],y=L['y'].tolist(),rx=L['rx'].tolist(),day=L['day'].tolist(),derivatives=derivatives,ranking=ranking,unavailable=gradient_unavailable));print('SOURCE_L_MAP_FROZEN '+str(ranking['low_blocks']),flush=True)
  d.write(out/'resolved_config.json',dict(config=d.config(),source_role_counts={k:len(v) for k,v in contract['role_ids'].items()},L_selected_ids=L['ids'],V_selected_ids=V['ids'],classifier='original6x160cosine/scale30',feature='F/SIG_TIME_X3/CENTER_25K',numerical_policy=ref.FULL_FP32_POLICY,model_module=api.__file__,map_fitted_from='L180 REPEAT only,before V predictions',all_source_RX_used=True,source_RX_holdout=None,target_access=False,updates=0))
  # Value bridge avoids the pinned NumPy/Torch ABI;one generated IQ view shared by both heads.
  raw=np.asarray(V['x'].float().tolist(),dtype=np.float32);old=raw.copy();pred={a:{} for a in d.MODELS};numerics=[]
  def predict(key,array):
   nonlocal NN
   z=torch.frombuffer(bytearray(array.tobytes(order='C')),dtype=torch.float32).reshape(array.shape).clone().to('cuda:0')
   with torch.no_grad():
    for arm,m in models.items():
     logits=[]
     for start in range(0,len(z),128):
      s=m.forward_identity_only(z[start:start+128]);NN+=len(s['tx_logits'])
      if s['tx_logits'].shape!=(len(z[start:start+128]),6) or s['id_feat_joint'].shape!=(len(z[start:start+128]),160) or not bool(torch.isfinite(s['tx_logits']).all()):raise ValueError('Finite original default head required')
      logits+=s['tx_logits'].cpu().tolist()
     pred[arm][key]=logits
   print('VIEW_COMPLETE '+key,flush=True)
  predict('raw',raw)
  for block in range(16):
   for sign in (-1,1):
    z,stats=dsp.apply(raw,block,sign);numerics.append(stats);predict(str(block)+'/'+str(sign),z)
  if not np.array_equal(raw,old):raise ValueError('Raw source IQ mutated')
  for arm,m in models.items():
   if any(p.grad is not None or p.requires_grad for p in m.parameters()) or any(not torch.equal(v.detach().cpu(),states[arm][k]) for k,v in m.state_dict().items()):raise ValueError('Frozen model/parameter state changed')
  if NN!=d.config()['planned_NN'] or gradient_calls>3 or gradient_packets>182:raise ValueError('Fixed source budget differs')
  d.write(out/'predictions.json',dict(ids=V['ids'],y=V['y'].tolist(),rx=V['rx'].tolist(),day=V['day'].tolist(),logits=pred))
  d.write(out/'prediction_complete.json',dict(status='SOURCE_PREDICTIONS_COMPLETE',config=d.config(),NN=NN,CP=2,source_cache_loads=1,source_L_selected=180,source_V_selected=1800,input_gradient_packets=gradient_packets,input_backward_tensor_calls=gradient_calls,gradient_unavailable=gradient_unavailable,model_state_unchanged=True,numerics=numerics,ordinary_generations=0,operator_applications=57600,teacher_NN=0,updates=0,head_fit_updates=0,target_access=False,wall_seconds=time.perf_counter()-started))
 os.environ['CUDA_VISIBLE_DEVICES']='';os.execv(sys.executable,[sys.executable,'-u',str(d.ROOT/'experiments/cvs_identity_sensitivity18/runner.py'),'--analyze'])
