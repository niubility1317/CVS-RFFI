"""Actual original heads/gates, labels used only by downstream source metrics."""
from copy import deepcopy
import argparse,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from . import design as d

def gate(n,args,teacher_logits,student_logits,extra,domain):
 meta=n._meta_from_extra(extra)
 if any(k in meta for k in d.FORBIDDEN_TX):raise ValueError('TX metadata passed to source proxy gate')
 if args.pseudo_domain_gate or not args.pseudo_temporal_gate or args.pseudo_temporal_mode!='batch_neighbor' or not args.pseudo_strong_agreement:raise ValueError('Expected actual historical gate')
 conf,pseudo=teacher_logits.softmax(1).max(1)
 confidence=n._threshold_mask(conf,domain,args);temporal=n._temporal_mask_tensor(pseudo,conf,extra,args,conf.device)
 agreement=student_logits.argmax(1)==pseudo
 structural=n._temporal_mask_tensor(torch.zeros_like(pseudo),torch.ones_like(conf),extra,args,conf.device)
 accepted=confidence&temporal&agreement
 if (accepted&~structural).any():raise ValueError('Accepted mask exceeds metadata structural bound')
 return pseudo,dict(confidence=confidence,temporal=temporal,agreement=agreement,base=confidence&temporal,accepted=accepted,structural=structural,all=torch.ones_like(accepted))

def update(records,pseudo,masks,truth,meta):
 # Truth is used after predictions/masks; it never enters the model or gate.
 truth=truth.cpu();pseudo=pseudo.cpu();rx=meta['rx_i'].cpu();day=meta['day_i'].cpu()
 selectors={'overall':torch.ones_like(truth,dtype=torch.bool)}
 for r in rx.unique().tolist():selectors[f'RX/{r}']=rx==r
 for z in day.unique().tolist():selectors[f'day/{z}']=day==z
 for y in range(6):selectors[f'TX/{y}']=truth==y
 for r,z in sorted(set(zip(rx.tolist(),day.tolist()))):selectors[f'RXday/{r}/{z}']=(rx==r)&(day==z)
 for name,ix in selectors.items():
  rec=records.setdefault(name,dict(count=0,stages={}))
  rec['count']+=int(ix.sum())
  for stage,mask in masks.items():
   selected=ix&mask.cpu();g=rec['stages'].setdefault(stage,dict(selected=0,correct=0,predicted_class=[0]*6,true_class=[0]*6))
   g['selected']+=int(selected.sum());g['correct']+=int((selected&(pseudo==truth)).sum())
   g['predicted_class']=[a+b for a,b in zip(g['predicted_class'],torch.bincount(pseudo[selected],minlength=6).tolist())]
   g['true_class']=[a+b for a,b in zip(g['true_class'],torch.bincount(truth[selected],minlength=6).tolist())]

def finish(records):
 for rec in records.values():
  for g in rec['stages'].values():g.update(coverage=g['selected']/rec['count'] if rec['count'] else None,precision=g['correct']/g['selected'] if g['selected'] else None)
 return records

def unchanged(model,before):
 if any(not torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items()):raise ValueError('Frozen source proxy mutated model parameters/buffers')

def native_role_reader():
 # A worker starts fresh; preflight imports cannot bootstrap its native path.
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 return role_ids_from_native

def run(row_id):
 c=d.config(next(r for r in d.rows() if r['row_id']==row_id));d.validate(c);out=Path(c['output_root'])
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(2);started=time.perf_counter()
 from experiments.cvs_feature_multiscale50.predict import source_proof,own_model
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from experiments.cvs_receiver_residual_capacity4.control import lock
 from experiments.cvs_multi_state_action.dispatch import capacity
 role_ids_from_native=native_role_reader()
 source=source_proof(c['source_config']);payload=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if payload['ema_model'] is None or payload['epoch']!=50 or payload['a1_ema_successful_updates']!=11100:raise ValueError('Actual own final EMA not available')
 student=own_model(c['source_config'],'cpu');teacher=deepcopy(student);teacher.load_state_dict(payload['ema_model'],strict=True);teacher.eval().requires_grad_(False);student.requires_grad_(False)
 before_student={k:v.detach().cpu().clone() for k,v in student.state_dict().items()};before_teacher={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
 with numerical_context(d.FULL_FP32_POLICY),installed(c['source_config']) as n:
  # Reserve and materialize one GPU context while holding the existing global lock.
  while True:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    states=capacity();ready=[(len(v['pids']),-v['free_mb'],g) for g,v in states.items() if len(v['pids'])<3 and v['free_mb']>=12000]
    if ready:
     gpu=min(ready)[2];device=torch.device(f'cuda:{gpu}');student.to(device);teacher.to(device);torch.cuda.synchronize(device);break
   time.sleep(3)
  args=configure(d.old.make_args(c['source_config'],str(device)));ctx=n._build_ssdg_wisig_data(args,device)
  if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=d.read(d.SOURCE)['role_ids']:raise ValueError('Exact legal source roles required')
  ds=ctx['train_loader'].dataset
  if len(ds)!=6300:raise ValueError('L_s physical count differs')
  loader=DataLoader(ds,batch_size=256,shuffle=bool(args.u_unlabeled_shuffle),drop_last=False,num_workers=0,collate_fn=ctx['train_loader'].collate_fn,generator=torch.Generator().manual_seed(c['sampling_seed']))
  torch.manual_seed(c['sampling_seed']);records={mode:{} for mode in c['student_modes']};batches=0;seen=0
  with torch.no_grad():
   for x,truth,domain,raw in loader:
    blind={k:v for k,v in raw.items() if k not in d.FORBIDDEN_TX};blind['tx_label_visible']=torch.zeros(len(truth),dtype=torch.bool)
    x,sentinels,extra=n.move_batch((x,torch.full_like(truth,-1),domain,blind),device)
    if (sentinels!=-1).any() or any(k in n._meta_from_extra(extra) for k in d.FORBIDDEN_TX):raise ValueError('Proxy gate received TX truth')
    du=n.domain_from_extra(extra,ctx['domain_label_map'],device);weak=teacher(x,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=du)['tx_logits']
    strong_x=n._strong_augment(x,float(args.strong_noise_std))
    for mode in c['student_modes']:
     student.train(mode=='train_dropout_no_grad');strong=student(strong_x,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=du)['tx_logits'];pseudo,masks=gate(n,args,weak,strong,extra,du);update(records[mode],pseudo,masks,truth,blind)
    batches+=1;seen+=len(truth)
    print(json.dumps(dict(row_id=row_id,source_L_seen=seen,batches=batches)),flush=True)
  student.eval();unchanged(student,before_student);unchanged(teacher,before_teacher)
  if seen!=6300 or batches!=25 or any(p.grad is not None for m in (student,teacher) for p in m.parameters()):raise ValueError('Source proxy count/gradient differs')
  result=dict(status='SOURCE_GATE_DIAGNOSTIC_COMPLETE',config=c,code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_checkpoint=str(source/'final_ssdg.pth'),source_provenance='EXACT_SOURCE_ROLES_FIXED_FINAL50_NO_TARGET_SELECTION',GPU=gpu,records={mode:finish(v) for mode,v in records.items()},student_teacher_parameters_buffers_unchanged=True,labels_enter_gate=False,target_IQ_or_truth_read=False,U_TX_truth_read=False,head_fit_updates=0,backbone_updates=0,EMA_updates=0,forward_packets=dict(teacher_weak=6300,student_strong_train=6300,student_strong_eval=6300),wall_seconds=time.perf_counter()-started)
  out.mkdir(parents=True,exist_ok=False);d.write(out/'summary.json',result)
 return result

def check():
 from experiments.cvs_phase1_stack.runtime import installed
 assert native_role_reader().__module__=='cvsrffi.xuc_fusion.native'
 c=d.baseline_config(d.SEEDS[0]);args=d.old.make_args(c,'cpu');meta={k:torch.tensor(v) for k,v in dict(rx_i=[1,1,1,1],day_i=[0,0,0,0],eq_i=[0,0,0,0],sig_i=[0,1,7,8],base_index=[0,1,7,8]).items()};extra=(torch.zeros(4,dtype=torch.long),meta);logits=torch.tensor([[9.,0.,0.,0.,0.,0.]]*4);domain=torch.zeros(4,dtype=torch.long)
 with installed(c) as n:
  pseudo,masks=gate(n,args,logits,logits,extra,domain);assert masks['structural'].all() and masks['accepted'].all();records={};update(records,pseudo,masks,torch.tensor([0,1,0,1]),meta);assert finish(records)['overall']['stages']['accepted']['precision']==.5
  contaminated=(extra[0],dict(meta,tx_i=torch.tensor([0]*4)))
  try:gate(n,args,logits,logits,contaminated,domain)
  except ValueError:pass
  else:raise AssertionError('TX gate metadata accepted')
  changed=torch.tensor([[0.,9.,0.,0.,0.,0.]]*4);_,different=gate(n,args,logits,changed,extra,domain);assert not different['accepted'].any() and different['structural'].all()
  missing=(extra[0],dict(meta,sig_i=torch.tensor([0,10,20,30]),base_index=torch.tensor([0,10,20,30])));_,separated=gate(n,args,logits,logits,missing,domain);assert not separated['structural'].any() and not separated['accepted'].any();empty={};update(empty,pseudo,separated,torch.tensor([0,1,0,1]),meta);assert finish(empty)['overall']['stages']['accepted']['precision'] is None
 bad=deepcopy(d.config(d.rows()[0]));bad['head_fit_updates']=1
 try:d.validate(bad)
 except ValueError:pass
 else:raise AssertionError('Head fit accepted')
 return dict(status='PASS',native_gate_executed=True,fresh_worker_native_import=True,TX_metadata_rejected=True,structural_upper_bound_and_strong_rejection=True,labels_only_downstream=True,empty_precision_null=True,head_fit_config_rejected=True,target_read=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--row');a=p.parse_args();v=check() if a.check else run(a.row);print(json.dumps(v if a.check else dict(status=v['status'],row_id=a.row,source_count=6300,head_fit_updates=0)),flush=True)
