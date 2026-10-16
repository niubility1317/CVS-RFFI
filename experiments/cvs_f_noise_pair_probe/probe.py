"""Read-only fitL CE derivatives. No optimizer,backward(),teacher or new head."""
from collections import Counter
from dataclasses import asdict
import gzip,json,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .core import GROUPS,group,statistics,aggregate
from .noise import Channel
from experiments.cvs_equivariant_identity.precision import numerical_context

def provenance(seed):
 p=d.reference(seed);carrier=d.read(d.SOURCE)
 done=d.read(p/'completion.json');contract=d.verify_reference(seed,done,d.read(p/'initialization.json'),d.read(p/'source_contract.json'),d.read(p/'source_frozen.json'),carrier)
 args=d.read(p/'actual_native_args.json')
 if args['seed']!=seed or args['epochs']!=20 or args['label_epochs']!=13 or args['pseudo_epochs']!=7 or args['baseline_ckpt'] or args['teacher_ckpt'] or args['source_fold_held_rx']!=1:raise ValueError('Actual source baseline args differ')
 return done,contract,carrier

def source_L(contract,carrier):
 prep=d.read(d.PREPARATION/'preparation_complete.json')
 if prep['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prep['target_access'] or prep['identity_updates']:raise ValueError('Legal prior source preparation required')
 packed=torch.load(d.PREPARATION/'source_cache.pt',map_location='cpu',weights_only=False)
 for actual in (packed['contract'],prep['source_contract']):
  if any(actual.get(k)!=v for k,v in carrier.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing VALIDATED_ONCE source carrier differs')
 L=packed['labeled'];ids=L['ids']
 if L['role']!='L_s' or len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(carrier['role_ids']['L_s']):raise ValueError('Original L6300 carrier required')
 ix=[i for i,r in enumerate(L['rx'].tolist()) if int(r) in d.FIT]
 if len(ix)!=5040 or {ids[i] for i in ix}!=set(contract['role_ids']['L_s']):raise ValueError('Only exact source fitL5040 consumed')
 result={k:L[k][ix].clone() for k in ('x','y','rx','day','domain')};result['ids']=[ids[i] for i in ix]
 if result['x'].shape!=(5040,2,256) or not bool(torch.isfinite(result['x']).all()) or result['y'].dtype!=torch.long or set(result['y'].tolist())!=set(range(6)):raise ValueError('Finite native fitL IQ256 and6TX required')
 counts=Counter(zip(result['y'].tolist(),result['rx'].tolist(),result['day'].tolist()))
 if len(counts)!=72 or set(counts.values())!={70}:raise ValueError('Full TXxfitRXxday balanced source coverage required')
 return result

def gradient(model,x,y,domains,named):
 logits=model.forward_identity_only(x,domain_labels=domains)['tx_logits']
 loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.01)
 g=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
 buckets={k:[] for k in GROUPS}
 for (name,param),v in zip(named,g):
  v=torch.zeros_like(param) if v is None else v.detach()
  if name.endswith(('.sinc.low_coordinate','.sinc.band_coordinate')):v=v/25000.
  buckets[group(name)].append(v.flatten())
 result={key:torch.cat(value) for key,value in buckets.items()}
 if any(not bool(torch.isfinite(v).all()) for v in result.values()):raise FloatingPointError('Nonfinite source CE derivative')
 return result,logits.detach(),float(loss.detach())

def run(row,device):
 c=d.validate(d.config(row));out=Path(c['output_root']);out.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2)
 d.write(out/'resolved_config.json',c);seed=row['model_seed'];done,contract,carrier=provenance(seed)
 from experiments.cvs_additive_as20 import model as api,design as native_design
 expected=native_design.config(next(r for r in native_design.rows() if r['arm']=='D_raw'))
 if done['config']!=expected:raise ValueError('Actual immutable source backend config differs from reference')
 with numerical_context(d.FULL_FP32_POLICY):
  model=api.build(expected,device).eval();payload=torch.load(d.reference(seed)/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if payload['epoch']!=20 or payload['args']['seed']!=seed or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or payload['checkpoint_selection']!='final_only':raise ValueError('Real checkpoint seed/provenance differs')
  model.load_state_dict(payload['model'],strict=True);del payload
  with torch.no_grad():zero=model.forward_identity_only(torch.zeros(2,2,256,device=device))['tx_logits']
  if zero.shape!=(2,6) or not bool(torch.isfinite(zero).all()) or model._supcon_handles or model._supcon_pending is not None:raise ValueError('Real default-head checkpoint no-query smoke failed')
  d.write(out/'checkpoint_noquery_smoke.json',dict(status='PASS',strict_load=True,actual_model_seed=seed,synthetic_NN_packets=2,IQ_before_smoke=0,target_access=False,updates=0))
  named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];before={k:v.detach().clone() for k,v in model.state_dict().items()}
  if any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in model.modules()) or len(named)<1:raise ValueError('Fixed eval observer with no BatchNorm required')
  L=source_L(contract,carrier);d.write(out/'input_contract.json',contract);d.write(out/'source_labels.json',[dict(physical_id=pid,TX=int(L['y'][i]),RX=int(L['rx'][i]),day=int(L['day'][i])) for i,pid in enumerate(L['ids'])]);order=np.random.default_rng(d.STREAMS['order']).permutation(5040).tolist();channel=Channel()
  from experiments.cvs_crossrx_satmix20.channel import PairedChannel
  actual_original=json.loads(json.dumps(PairedChannel().audit()['actual_configs']))
  if any(json.loads(json.dumps(asdict(v)))!=actual_original[k] for k,v in channel.configs.items()):raise ValueError('Original channel Config differs')
  stats=[];NN=2;decisions=0
  with gzip.open(out/'noise_observations.jsonl.gz','xt',encoding='utf8',compresslevel=3) as obs,gzip.open(out/'gradient_statistics.jsonl.gz','xt',encoding='utf8',compresslevel=3) as records,gzip.open(out/'predictions.jsonl.gz','xt',encoding='utf8',compresslevel=3) as pred:
   for batch,start_index in enumerate(range(0,5040,d.BATCH)):
    ix=order[start_index:start_index+d.BATCH];ids=[L['ids'][i] for i in ix];x=L['x'][ix];y=L['y'][ix].to(device);domains=L['domain'][ix].to(device)
    values,physics=channel.batch(x,ids,L['rx'][ix].tolist(),L['day'][ix].tolist());obs.write(json.dumps(dict(batch=batch,ids=ids,records=physics),allow_nan=False)+'\n')
    grads={key:[[],[]] for key in GROUPS}
    for k in range(d.DRAWS):
     for j,sign in enumerate((1,-1)):
      array=np.ascontiguousarray(values[k,j]);view=torch.frombuffer(bytearray(array.tobytes()),dtype=torch.float32).reshape(array.shape).to(device)
      g,logits,loss=gradient(model,view,y,domains,named);NN+=len(ix);decisions+=len(ix)
      for key,value in g.items():grads[key][j].append(value)
      prediction=logits.argmax(1).cpu().tolist();margins=(logits.gather(1,y[:,None]).flatten()-logits.masked_fill(torch.nn.functional.one_hot(y,6).bool(),-torch.inf).max(1).values)/30
      for n,i in enumerate(ix):pred.write(json.dumps(dict(batch=batch,draw=k,sign=sign,physical_id=ids[n],TX=int(L['y'][i]),RX=int(L['rx'][i]),day=int(L['day'][i]),prediction=prediction[n],margin_over30=float(margins[n]),CE_batch=loss),allow_nan=False)+'\n')
    by_group={}
    for key,(plus,minus) in grads.items():
     r=dict(batch=batch,count=len(ix),group=key,**statistics(torch.stack(plus),torch.stack(minus)));by_group[key]=r;stats.append(r);records.write(json.dumps(r,allow_nan=False)+'\n')
    sums={k:sum(v[k] for v in by_group.values()) for k in ('Vplus','Vminus','noise_covariance','independent_pair_trace','antithetic_pair_trace')};whole=dict(batch=batch,count=len(ix),draws=d.DRAWS,group='whole',raw_derivative_gram=np.asarray([v['raw_derivative_gram'] for v in by_group.values()]).sum(0).tolist(),**sums,trace_ratio=sums['antithetic_pair_trace']/sums['independent_pair_trace'] if sums['independent_pair_trace']>0 else None);stats.append(whole);records.write(json.dumps(whole,allow_nan=False)+'\n');records.flush();pred.flush();obs.flush()
    print(json.dumps(dict(batch=batch+1,total_batches=40,model_seed=seed,NN=NN,trace_ratio=whole['trace_ratio'])),flush=True)
    d.write(out/'progress.json',dict(completed_batches=batch+1,total_batches=40,NN=NN,source_only=True,target_access=False))
    del grads,values,g
  if NN!=c['expected_NN'] or decisions!=c['expected_decisions'] or len(stats)!=240 or any(not torch.equal(v,model.state_dict()[key]) for key,v in before.items()) or any(p.grad is not None for _,p in named) or model._supcon_handles or model._supcon_pending is not None:raise ValueError('Frozen-state or complete-budget invariant failed')
  d.write(out/'observation_audit.json',channel.audit())
 summary=dict(status='ARTIFACTS_COMPLETE',row_id=row['row_id'],model_seed=seed,source_only=True,target_access=False,NN=NN,decisions=decisions,source_L_unique=5040,source_cells=72,gradient_records=len(stats),parameters_and_buffers_unchanged=True,optimizer_updates=0,head_fit_updates=0,EMA_updates=0,model_epochs=0,diagnostic=aggregate([v for v in stats if v['group']=='whole']),parent_epochs=20,parent_updates=4440,parent_NN=sum(d.read(d.reference(seed)/'satmix_result.json')[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+d.read(d.reference(seed)/'satmix_result.json')['held_forward_packets'],wall_seconds=time.perf_counter()-start,actual_model_class=type(model).__module__+'.'+type(model).__name__,performance_improvement_proved=False)
 d.write(out/'summary.json',summary);return summary
