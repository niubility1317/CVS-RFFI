"""Frozen own-default-head source diagnosis;zero optimizer/head/EMA updates."""
from collections import Counter
from dataclasses import asdict
import gzip,json,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .channel import Channel
from .observer import run as observe,add_clean_cosines
from experiments.cvs_equivariant_identity.precision import numerical_context

def source_L(contract,carrier):
 prep=d.read(d.PREPARATION/'preparation_complete.json')
 if prep['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prep['target_access'] or prep['identity_updates']:raise ValueError('Existing legal source preparation required')
 packed=torch.load(d.PREPARATION/'source_cache.pt',map_location='cpu',weights_only=False)
 for actual in (packed['contract'],prep['source_contract']):
  if any(actual.get(k)!=v for k,v in carrier.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Unchanged VALIDATED_ONCE source carrier differs')
 L=packed['labeled'];ids=L['ids']
 if L['role']!='L_s' or len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(carrier['role_ids']['L_s']):raise ValueError('Original sourceL6300 carrier required')
 ix=[i for i,r in enumerate(L['rx'].tolist()) if int(r) in d.FIT]
 if len(ix)!=5040 or {ids[i] for i in ix}!=set(contract['role_ids']['L_s']):raise ValueError('Only exact matchingfitL5040 consumed')
 result={k:L[k][ix].clone() for k in ('x','y','rx','day','domain')};result['ids']=[ids[i] for i in ix]
 if result['x'].shape!=(5040,2,256) or not bool(torch.isfinite(result['x']).all()) or result['y'].dtype!=torch.long or set(result['y'].tolist())!=set(range(6)):raise ValueError('Finite originalfitL6TX IQ256 required')
 counts=Counter(zip(result['y'].tolist(),result['rx'].tolist(),result['day'].tolist()))
 if len(counts)!=72 or set(counts.values())!={70}:raise ValueError('Exact balanced source TXxfitRXxday required')
 return result

def run(row,device):
 c=d.validate(d.config(row));out=Path(c['output_root']);out.mkdir(exist_ok=False);started=time.perf_counter();torch.set_num_threads(2);d.write(out/'resolved_config.json',c)
 from experiments.cvs_f_noise_pair20 import model as api,design as parent
 p=d.reference(row['arm']);carrier=d.read(d.SOURCE);done=d.read(p/'completion.json');expected=parent.config(next(v for v in parent.rows() if v['arm']==row['arm']));result=d.read(p/'satmix_result.json');contract=d.verify_reference(row['arm'],done,d.read(p/'initialization.json'),d.read(p/'source_contract.json'),d.read(p/'source_frozen.json'),carrier,result,expected)
 args=d.read(p/'actual_native_args.json')
 if args['seed']!=d.SEED or args['epochs']!=20 or args['label_epochs']!=13 or args['pseudo_epochs']!=7 or args['baseline_ckpt'] or args['teacher_ckpt'] or args['source_fold_held_rx']!=1:raise ValueError('Actual matching source args differ')
 with numerical_context(d.FULL_FP32_POLICY):
  model=api.build(expected,device).eval();payload=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if payload['epoch']!=20 or payload['args']['seed']!=d.SEED or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or payload['checkpoint_selection']!='final_only':raise ValueError('Real source checkpoint provenance differs')
  model.load_state_dict(payload['model'],strict=True);del payload
  with torch.no_grad():zero=model.forward_identity_only(torch.zeros(2,2,256,device=device))['tx_logits']
  if zero.shape!=(2,6) or not bool(torch.isfinite(zero).all()) or model._noise_calls or model._noise_runtime is not None:raise ValueError('Real no-query default-head smoke failed')
  d.write(out/'checkpoint_noquery_smoke.json',dict(status='PASS',strict_load=True,actual_model_seed=d.SEED,NN=2,source_IQ_before_smoke=0,target_access=False,updates=0))
  before={k:v.detach().clone() for k,v in model.state_dict().items()};head=model.id_backbone.encoder.core.id_backbone.cls_head
  if head.weight.shape!=(6,160) or any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in model.modules()):raise ValueError('Original noBN joint160 defaulthead required')
  d.write(out/'model_audit.json',dict(parameters=sum(p.numel() for p in model.parameters()),type=type(model).__module__+'.'+type(model).__name__,default_head=type(head).__module__+'.'+type(head).__name__,scale=head.scale,NoExplicitCFO=True,learned_fixed_gains={n:dict(mean=float(p.detach().double().tanh().mean()),min=float(p.detach().double().tanh().min()),max=float(p.detach().double().tanh().max())) for n,p in model.named_parameters() if n.endswith('.gain') or n.endswith('.response_gain')},training_source=p.as_posix(),parent_model_epochs=20,parent_model_updates=4440))
  L=source_L(contract,carrier);d.write(out/'input_contract.json',contract);d.write(out/'source_labels.json',[dict(physical_id=pid,TX=int(L['y'][i]),RX=int(L['rx'][i]),day=int(L['day'][i])) for i,pid in enumerate(L['ids'])]);channel=Channel()
  from experiments.cvs_crossrx_satmix20.channel import PairedChannel
  original=json.loads(json.dumps(PairedChannel().audit()['actual_configs']['practical_low_urban']))
  if json.loads(json.dumps(asdict(channel.cfg)))!=original:raise ValueError('Actual original lowurban Config differs')
  order=np.random.default_rng(d.STREAMS['order']).permutation(5040).tolist();NN=2;decisions=0;autograd_calls=0;labels=[];by_view={v:[] for v in d.VIEWS};all_physics=[]
  with gzip.open(out/'predictions.jsonl.gz','xt',encoding='utf8',compresslevel=3) as pred,gzip.open(out/'physics.jsonl.gz','xt',encoding='utf8',compresslevel=3) as phys:
   for batch,start in enumerate(range(0,5040,d.BATCH)):
    ix=order[start:start+d.BATCH];ids=[L['ids'][i] for i in ix];raw=L['x'][ix];y=L['y'][ix].to(device);domains=L['domain'][ix].to(device);full,control,physics=channel.batch(raw,ids,L['rx'][ix].tolist(),L['day'][ix].tolist());clean=None;labels+=y.cpu().tolist();all_physics+=physics
    for v in physics:phys.write(json.dumps(v,allow_nan=False)+'\n')
    for view,x in zip(d.VIEWS,(raw,full,control)):
     rec,features=observe(model,x.to(device),y,domains);NN+=len(ix);decisions+=len(ix);autograd_calls+=1
     if clean is None:clean=features
     add_clean_cosines(rec,features,clean)
     for j,r in enumerate(rec):
      r.update(batch=batch,view=view,physical_id=ids[j],RX=int(L['rx'][ix[j]]),day=int(L['day'][ix[j]]));pred.write(json.dumps(r,allow_nan=False)+'\n');by_view[view].append(r['prediction'])
    pred.flush();phys.flush();d.write(out/'progress.json',dict(completed_batches=batch+1,total_batches=40,NN=NN,decisions=decisions,autograd_calls=autograd_calls,updates=0,source_only=True,target_access=False));print(json.dumps(dict(batch=batch+1,arm=row['arm'],NN=NN)),flush=True)
  if NN!=c['expected_NN'] or decisions!=c['expected_decisions'] or autograd_calls!=c['expected_autograd_calls'] or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in model.parameters()) or model._noise_calls or model._supcon_handles or model._supcon_pending is not None:raise ValueError('Frozen-state/defaulthead or full-budget invariant failed')
  d.write(out/'observation_audit.json',channel.audit());diag=d.recoverability(labels,by_view,all_physics)
 summary=dict(status='ARTIFACTS_COMPLETE',row_id=row['row_id'],arm=row['arm'],model_seed=d.SEED,source_only=True,target_access=False,NN=NN,decisions=decisions,source_L_unique=5040,source_cells=72,autograd_calls=autograd_calls,autograd_packet_equivalents=decisions,parameters_and_buffers_unchanged=True,optimizer_updates=0,head_fit_updates=0,EMA_updates=0,model_epochs=0,recoverability=diag,parent_epochs=20,parent_updates=4440,parent_NN=sum(result[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+result['held_forward_packets'],wall_seconds=time.perf_counter()-started,performance_improvement_proved=False)
 d.write(out/'summary.json',summary);return summary
