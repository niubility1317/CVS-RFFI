"""Frozen source-V comparison only. Never opens U IQ, query IQ or target truth."""
import os,math,copy,random
from pathlib import Path
from collections import Counter
from . import design as d
from .provenance import source_proof
def readout(c):
 import numpy as np
 out=source_proof(c);v=d.read(out/'held_RX1_V.json')
 if v.get('status')!='SOURCE_TEACHER_COMPLETE' or v.get('config')!=c or v.get('epoch')!=20 or v.get('optimizer_steps')!=4440 or v.get('count')!=5400 or v.get('U_truth_read') is not False or v.get('target_access') is not False or v.get('head_fit_updates')!=0:raise ValueError('Complete legal final default-head source readout required')
 ids=v['physical_ids'];y=np.asarray(v['labels']);p=np.asarray(v['predictions']);lp=np.asarray(v['log_probabilities'],dtype=np.float64)
 if len(ids)!=5400 or len({tuple(x) for x in ids})!=5400 or y.shape!=(5400,) or p.shape!=(5400,) or lp.shape!=(5400,6) or y.dtype.kind not in 'iu' or set(y)!=set(range(6)) or not np.isfinite(lp).all() or not np.array_equal(p,lp.argmax(1)) or any(len(x)!=5 or x[1]!=1 for x in ids):raise ValueError('Source physical coverage/default-head probability invalid')
 cells=Counter((int(label),identity[2]) for label,identity in zip(y,ids))
 if set(cells)!={(t,day) for t in range(6) for day in (1,2,3)} or set(cells.values())!={300} or not np.allclose(np.exp(lp).sum(1),1,atol=1e-10,rtol=0):raise ValueError('Six TX and three source days required')
 error=int((p!=y).sum());ce=float(-lp[np.arange(5400),y].mean())
 done=d.read(out/'completion.json')
 if v['errors']!=error or abs(v['accuracy']-(1-error/5400))>1e-12 or abs(v['mean_CE']-ce)>1e-10 or v['source_health']!=(done['final_training_L_CE']<=.5):raise ValueError('Source readout recount differs')
 v['mean_true_probability']=float(np.exp(lp[np.arange(5400),y]).mean());v['mean_max_probability_on_errors']=float(np.exp(lp.max(1))[p!=y].mean()) if error else None
 return v
def original_A0_proxy(ids,labels):
 import torch
 from experiments.cvs_evidence_responsibility_train20 import design as a0,model as api
 from experiments.cvs_evidence_responsibility_train20.provenance import source_proof as proof,validate_checkpoint
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_evidence_compensation20.original_gate import gate
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=a0.config(next(r for r in a0.rows() if r['arm']=='BASE_SSL' and r['model_seed']==d.SEED));source=proof(c)
 ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c)
 with numerical_context(d.FULL_FP32_POLICY):
  student=api.build(c,'cpu').requires_grad_(False);student.load_state_dict(ck['model'],strict=True)
  teacher=api.build(c,'cpu').requires_grad_(False);teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval();student.train()
  if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.modules()):raise ValueError('Source replay may not update BatchNorm')
  states=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (student,teacher)]
  args=a0.make_args(c,'cpu');args.output_dir=str(d.BASE/'A0_proxy');args.num_workers=0
  if not args.a1_source_screen_only:raise ValueError('Source replay must use native source-screen-only construction')
  with installed(c) as native:
   ctx=native._build_ssdg_wisig_data(args,torch.device('cpu'))
   from cvsrffi.xuc_fusion.native import role_ids_from_native
   if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=d.read(d.SOURCE)['role_ids']:raise ValueError('A0 source replay carrier differs')
   ds=ctx['val_loader'].dataset;held=[i for i,v in enumerate(ds.index) if int(v.rx_i)==1]
   if len(held)!=5400:raise ValueError('Only original RX1 source V permitted')
   order=list(range(5400));random.Random(2026101502).shuffle(order)
   original_ids=[None]*5400;truth=[None]*5400;pseudo=[None]*5400;accepted=[None]*5400;confidence=[None]*5400;structural=[None]*5400
   with torch.random.fork_rng(devices=[]),torch.no_grad():
    torch.manual_seed(2026101503)
    for start in range(0,5400,256):
     indices=order[start:start+256];batch=[ds[held[i]] for i in indices];x=torch.stack([v[0] for v in batch])
     meta={key:torch.tensor([int(v[3][key]) for v in batch],dtype=torch.long) for key in ('eq_i','rx_i','day_i','sig_i','base_index')}
     domains=meta['day_i']-1
     with fresh_parameter_caches(teacher):weak=teacher.forward_identity_only(x,domain_labels=domains)['tx_logits']
     strong_x=native._strong_augment(x,float(args.strong_noise_std))
     with fresh_parameter_caches(student):strong=student.forward_identity_only(strong_x,domain_labels=domains)['tx_logits']
     if weak.shape!=(len(indices),6) or strong.shape!=weak.shape or not torch.isfinite(weak).all() or not torch.isfinite(strong).all():raise ValueError('Finite source replay predictions required')
     prediction,masks=gate(native,args,weak,strong,(None,meta),domains)
     for pos,i in enumerate(indices):
      original_ids[i]=[int(meta[k][pos]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')];truth[i]=int(batch[pos][1]);pseudo[i]=int(prediction[pos]);accepted[i]=bool(masks['accepted'][pos]);structural[i]=bool(masks['structural'][pos]);confidence[i]=float(weak[pos].softmax(0).max())
   if original_ids!=ids or truth!=labels:raise ValueError('A0 and teachers do not share exact source V physical IDs/labels')
   for m,state in zip((student,teacher),states):
    if any(not torch.equal(value,m.state_dict()[key]) for key,value in state.items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Source proxy changed model state or gradients')
 return dict(status='SOURCE_A0_GATE_PROXY_COMPLETE',A0_run_id=c['run_id'],A0_row_id=c['row_id'],A0_epoch=20,count=5400,physical_ids=ids,labels=labels,original_EMA_pseudo=pseudo,accepted=accepted,structural=structural,confidence=confidence,
  order_seed=2026101502,view_seed=2026101503,batch_size=256,teacher_weak='original clean U input',student_strong_std=float(args.strong_noise_std),student_dropout_active=True,metadata_TX_read_by_gate=False,parameters_buffers_gradients_unchanged=True,optimizer_steps=0,EMA_updates=0,head_fit_updates=0,U_IQ_reads=0,U_truth_reads=0,target_access=False,limitation='Frozen A0 final20 source V proxy; not actual U precision or early-stage teacher behavior')
def qualify(a,b,rejected):
 import numpy as np
 base=np.asarray(a['predictions'])==np.asarray(a['labels']);candidate=np.asarray(b['predictions'])==np.asarray(b['labels']);mask=np.asarray(rejected,dtype=bool)
 if mask.shape!=(5400,):raise ValueError('Fixed full source rejection region required')
 repair=int((~base&candidate).sum());broken=int((base&~candidate).sum());r=int((~base&candidate&mask).sum());n=int((base&~candidate&mask).sum())
 passed=a['source_health'] and b['source_health'] and b['errors']<a['errors'] and b['mean_CE']<=a['mean_CE'] and bool(mask.any()) and r>n
 return dict(qualified=bool(passed),source_healthy=bool(a['source_health'] and b['source_health']),overall_corrected=repair,overall_introduced=broken,rejected_count=int(mask.sum()),rejected_corrected=r,rejected_introduced=n,primary='EXCLUDE_RX1',control='SAMPLE_80',fixed_no_scan=True,teacher_bank_or_student_authorized=False)
def finish():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU source comparison required')
 import torch,numpy as np
 torch.set_num_threads(2);a,b=(readout(d.config(r)) for r in d.rows())
 if a['physical_ids']!=b['physical_ids'] or a['labels']!=b['labels']:raise ValueError('Matched source physical IDs/labels required')
 if (d.BASE/'completion.json').exists():raise FileExistsError('Preserve completed source comparison')
 proxy=original_A0_proxy(a['physical_ids'],a['labels']);d.write(d.BASE/'A0_gate_proxy.json',proxy)
 mask=~np.asarray(proxy['accepted'],dtype=bool);qualification=qualify(a,b,mask)
 y=np.asarray(a['labels']);pa=np.asarray(a['predictions']);pb=np.asarray(b['predictions']);base=pa==y;candidate=pb==y;days=np.asarray([x[2] for x in a['physical_ids']]);records=[]
 tp_a=np.exp(np.asarray(a['log_probabilities'])[np.arange(5400),y]);tp_b=np.exp(np.asarray(b['log_probabilities'])[np.arange(5400),y]);a0_ok=np.asarray(proxy['original_EMA_pseudo'])==y
 for dim,strata in (('all',['ALL']),('gate',['accepted','rejected']),('day',[1,2,3]),('TX',list(range(6)))):
  for value in strata:
   selected=np.ones(5400,bool) if dim=='all' else (~mask if value=='accepted' else mask) if dim=='gate' else days==value if dim=='day' else y==value
   n=int(selected.sum());repair=int((~base&candidate&selected).sum());broken=int((base&~candidate&selected).sum())
   records.append(dict(dimension=dim,stratum=value,count=n,SAMPLE_80_accuracy=float(base[selected].mean()) if n else None,EXCLUDE_RX1_accuracy=float(candidate[selected].mean()) if n else None,SAMPLE_80_mean_true_probability=float(tp_a[selected].mean()) if n else None,EXCLUDE_RX1_mean_true_probability=float(tp_b[selected].mean()) if n else None,A0_EMA_accuracy=float(a0_ok[selected].mean()) if n else None,corrected=repair,introduced=broken,accuracy_difference_pp=(repair-broken)*100/n if n else None))
 class_pairs=[dict(true_TX=t,wrong_TX=w,SAMPLE_80_errors=int(((y==t)&(pa==w)).sum()),EXCLUDE_RX1_errors=int(((y==t)&(pb==w)).sum()),corrected=int(((y==t)&(pa==w)&candidate).sum()),introduced=int(((y==t)&base&(pb==w)).sum())) for t in range(6) for w in range(6) if t!=w]
 public=lambda v:{k:value for k,value in v.items() if k not in ('physical_ids','labels','predictions','log_probabilities')}
 result=dict(status='SOURCE_MATRIX_COMPLETE',run_id=d.RUN,results={arm:public(v) for arm,v in zip(d.ARMS,(a,b))},qualification=qualification,records=records,class_pair_records=class_pairs,
  default_head_only=True,head_fit_updates=0,actual_epochs=40,actual_optimizer_steps=8880,teacher_train_L_packets=1136640,teacher_held_V_forward_packets=10800,A0_source_V_forward_packets=10800,A0_scope='Frozen complete original BASE_SSL final20 ownEMA/strong-student replay only; no teacher training inheritance',source_only=True,U_truth_read=False,target_access=False,goal_completed=False)
 d.write(d.BASE/'matrix_result.json',result);d.write(d.BASE/'completion.json',dict(status='SOURCE_MATRIX_COMPLETE',rows=2,qualified=qualification['qualified'],actual_optimizer_steps=8880,target_access=False));print('SOURCE_TEACHER_COMPARISON_COMPLETE',flush=True)
