"""Frozen As model,all fitL,exact matched H/noise and original-RX replay."""
from collections import Counter
from dataclasses import asdict
import gzip,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .physics import measure

def score(y,p):
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 f=[]
 for k in range(6):
  den=sum(cm[k])+sum(r[k] for r in cm);f.append(2*cm[k][k]/den if den else 0.)
 return dict(accuracy=sum(cm[k][k] for k in range(6))/len(y),macro_F1=sum(f)/6,confusion=cm)

def tensor(a):
 a=np.ascontiguousarray(a,dtype=np.float32)
 return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape)

def run():
 c=d.validate(d.config(d.rows()[0]))
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU2-only source frozen probe')
 if d.BASE.exists():raise FileExistsError('Reconcile immutable own output root,no relaunch')
 d.BASE.mkdir();out=Path(c['output_root']);out.mkdir();start=time.perf_counter();torch.set_num_threads(2)
 d.write(d.BASE/'execution.json',dict(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,CUDA_VISIBLE_DEVICES='',release_commit=(d.ROOT/'release_commit.txt').read_text().strip(),read_at=time.time()));d.write(out/'resolved_config.json',c)
 try:
  from experiments.cvs_as_timing20 import design as parent,model as api
  from experiments.cvs_equivariant_identity.precision import numerical_context
  p=Path(d.REFERENCE);carrier=d.read(d.SOURCE);result=d.read(p/'satmix_result.json');expected=parent.config(next(r for r in parent.rows() if r['arm']=='ALL'));contract=d.verify_reference(d.read(p/'completion.json'),d.read(p/'initialization.json'),d.read(p/'source_frozen.json'),d.read(p/'source_contract.json'),carrier,result,expected)
  with numerical_context(d.POLICY):
   torch.manual_seed(d.MODEL_SEED);model=api.build(expected,'cpu').eval();payload=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
   if payload['epoch']!=20 or payload['args']['seed']!=d.MODEL_SEED or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or payload['checkpoint_selection']!='final_only':raise ValueError('Actual checkpoint payload provenance differs')
   model.load_state_dict(payload['model'],strict=True);del payload
   with torch.no_grad():z=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or model._timing_calls or model._joint_calls:raise ValueError('Real-checkpoint noquery smoke failed')
   d.write(d.BASE/'checkpoint_noquery_smoke.json',dict(status='PASS',strict_load=True,NN=2,source_IQ_before_smoke=0,target_access=False,updates=0,checkpoint=str(p/'final_ssdg.pth')));print('REALCP_SMOKE PASS 2syntheticNN,0sourceIQ',flush=True)
   before={k:v.detach().clone() for k,v in model.state_dict().items()};head=model.id_backbone.encoder.core.id_backbone.cls_head
   if head.weight.shape!=(6,160) or float(head.scale)!=30. or len(before)!=206 or any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in model.modules()):raise ValueError('Original F/noBN/defaultcos30 joint160 required')
   from .source import load
   L=load(contract,carrier);d.write(out/'source_contract.json',contract);d.write(out/'source_labels.json',[dict(physical_id=pid,TX=int(L['y'][i]),RX=int(L['rx'][i]),day=int(L['day'][i]),eq=1) for i,pid in enumerate(L['ids'])]);d.write(out/'data_roles.json',dict(L=5040,U_IQ_or_truth=0,V_values=0,held_RX1_values=0,target=0,carrier_L6300_and_unusedV_oldfeatures_deserialized=True,only_fitL5040_consumed=True,old_cached_features_consumed=False))
   from experiments.cvs_crossrx_satmix20.channel import PairedChannel
   ch=model._joint_channel.ch;residual=model._joint_channel.residual;full_audit=PairedChannel().audit();configs={s:model._joint_channel.configs[s] for s in d.SCENES}
   d.write(out/'actual_modules.json',dict(channel=ch.__file__,residual=residual.__file__,model=api.__file__,original_channel=full_audit,parameters=sum(v.numel() for v in model.parameters()),state_tensors=len(before),NoExplicitCFO=True,default_head=str(type(head)),head_scale=float(head.scale)))
   if json.loads(json.dumps(full_audit['actual_configs']))!=d.read(p/'actual_channel.json')['actual_configs']:raise ValueError('Actual complete original six-practical configuration differs')
   decisions=0;calls=0;records=0;preds={('clean','raw'):[]};truth=[];receivers={};group_y={s:[] for s in d.SCENES}
   for s in d.SCENES:
    for b in d.BANKS:preds[s,b]=[]
   def infer(x,y,scene,bank,ids,rx,day,domains,stream):
    nonlocal decisions
    with torch.no_grad():zz=model.forward_identity_only(x,domain_labels=domains)['tx_logits'].double()
    if zz.shape!=(len(y),6) or not bool(torch.isfinite(zz).all()):raise ValueError('Complete finite original defaulthead logits required')
    logits=zz.tolist();ys=y.tolist();ps=zz.argmax(1).tolist();preds[scene,bank]+=ps;decisions+=len(ys)
    for j,(v,t,pr) in enumerate(zip(logits,ys,ps)):
     stream.write(json.dumps(dict(scene=scene,bank=bank,physical_id=ids[j],TX=int(t),RX=int(rx[j]),day=int(day[j]),logits=v,prediction=pr,margin_over30=(v[t]-max(z for k,z in enumerate(v) if k!=t))/30),allow_nan=False)+'\n')
   with gzip.open(out/'predictions.jsonl.gz','xt',encoding='utf8',compresslevel=3) as pf,gzip.open(out/'physics.jsonl.gz','xt',encoding='utf8',compresslevel=3) as mf:
    for batch,startix in enumerate(range(0,5040,d.BATCH)):
     stop=min(startix+d.BATCH,5040);raw=L['x'][startix:stop];y=L['y'][startix:stop];rx=L['rx'][startix:stop];day=L['day'][startix:stop];domains=L['domain'][startix:stop];ids=L['ids'][startix:stop];truth+=y.tolist()
     infer(raw,y,'clean','raw',ids,rx,day,domains,pf)
     source=[ch.normalize_input(z[0]+1j*z[1])[0] for z in np.asarray(raw.double().tolist(),dtype=np.float64)]
     for scene,cfg in configs.items():
      arrays={b:[] for b in d.BANKS}
      for j,x in enumerate(source):
       pid=str(ids[j]);eq=1;rr=int(rx[j]);dd=int(day[j])
       if pid!=ids[j] or rr not in d.FIT or dd not in d.DAYS:raise ValueError('Actual opaque physicalID/session metadata required')
       seed=ch.stable_seed(d.CHANNEL_SEED,'fit_rxsession/singleH',scene,pid);views,m=measure(ch,residual,cfg,x,seed,rr,dd,eq)
       for b,v in views.items():
        arrays[b].append(np.stack([v.real,v.imag]));rmeta=m['banks'][b]['receiver'];sid=rmeta['session_id']
        if receivers.setdefault(sid,rmeta)!=rmeta:raise ValueError('Injected RX session not stable acrossTX/packet/scene')
       m.update(scene=scene,physical_id=ids[j],RX=rr,day=dd,seed=int(seed));mf.write(json.dumps(m,allow_nan=False)+'\n');calls+=m['native_process_calls'];records+=1
      for bank,a in arrays.items():infer(tensor(np.stack(a)),y,scene,bank,ids,rx,day,domains,pf)
      group_y[scene]+=y.tolist()
     pf.flush();mf.flush();d.write(d.BASE/'progress.json',dict(completed_batches=batch+1,expected_batches=40,decisions=decisions,NN=decisions+2,native_process_calls=calls,physics_records=records,receiver_sessions=len(receivers),source_only=True,target_access=False,updates=0));print('BATCH '+str(batch+1)+'/40 NN='+str(decisions+2),flush=True)
   if decisions!=45360 or calls!=50400 or records!=10080 or len(receivers)!=48 or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(v.grad is not None for v in model.parameters()) or model._timing_calls or model._joint_calls or model._supcon_handles or model._supcon_pending is not None:raise ValueError('Frozen model/data budget/session invariants differ')
   groups={}
   for s in d.SCENES:
    yy=group_y[s];q={}
    for b in d.BANKS:
     pp=preds[s,b];q[b]=dict(**score(yy,pp),disagreement_vs_original=sum(a!=z for a,z in zip(pp,preds[s,'original']))/5040)
    groups[s]=dict(count=5040,banks=q)
   checks=d.criteria(groups);summary=dict(status='PASS_FIXED_SESSION_SENSITIVITY_RATIONALE_ONLY' if all(checks.values()) else 'FAIL_FIXED_SESSION_COVERAGE_RATIONALE_CLOSED',groups=groups,checks=checks,clean=score(truth,preds['clean','raw']),NN=decisions+2,decisions=decisions,physics_records=records,native_process_calls=calls,receiver_sessions=len(receivers),checkpoint_loads=1,source_L_unique=5040,source_cells=72,source_only=True,target_access=False,model_epochs=0,optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,EMA_updates=0,parameters_and_buffers_unchanged=True,gradients_unset=True,default_head_joint_training_history=True,parent_epochs=20,parent_updates=4440,parent_NN=sum(result[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+result['held_forward_packets'],wall_seconds=time.perf_counter()-start,automatic_training=False,performance_improvement_proved=False,goal_complete=False)
   d.write(out/'receivers.json',receivers);d.write(out/'summary.json',summary);d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',scientific_status=summary['status'],source_only=True,target_access=False,NN=decisions+2,native_process_calls=calls,model_epochs=0,optimizer_updates=0,wall_seconds=summary['wall_seconds']));print('COMPLETE '+summary['status'],flush=True)
 except BaseException as e:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),preserve_outputs=True,no_automatic_retry=True));raise
