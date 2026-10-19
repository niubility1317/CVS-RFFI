"""Frozen default-head sourceV perTX/RX/day audit; no parameter fitting."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .risk import calibrate
from experiments.cvs_f_supcon50 import design as original
from experiments.cvs_f_supcon50.provenance import source_proof,own_model
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_equivariant_identity.precision import numerical_context

def write(path,v):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def run():
 if os.geteuid()==0 or os.environ.get('CUDA_VISIBLE_DEVICES') not in d.GPUS:raise ValueError('Ordinary account and only physical GPU2/3 permitted')
 if d.BASE.exists():raise FileExistsError('Immutable source audit root exists; reconcile instead of repeat')
 d.BASE.mkdir();torch.set_num_threads(2);device=torch.device('cuda:0');started=time.time();rows=[];expected=original.read(original.SOURCE)
 configs=[original.config(next(r for r in original.rows() if r['arm']=='CROSS_RX' and r['model_seed']==seed)) for seed in d.SEEDS]
 # Existing role handles and already-validated provenance are reused, not rehashed.
 for c in configs:source_proof(c)
 with numerical_context(original.FULL_FP32_POLICY),installed(configs[0]) as native:
  args=original.make_args(configs[0],str(device));ctx=native._build_ssdg_wisig_data(args,device)
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Only exact original source data roles permitted')
  pooled=np.zeros((6,6),dtype=np.int64);rx_pooled={r:np.zeros((6,6),dtype=np.int64) for r in (1,3,4,6,8)}
  for c in configs:
   model=own_model(c,device);count=0;cm=np.zeros((6,6),dtype=np.int64);rx_cm={r:np.zeros((6,6),dtype=np.int64) for r in rx_pooled};day_cm={r:np.zeros((6,6),dtype=np.int64) for r in (1,2,3)};ce=[[] for _ in range(6)];margins=[[] for _ in range(6)]
   versions=[p._version for p in model.parameters()];buffers={n:b.detach().clone() for n,b in model.named_buffers()}
   with torch.no_grad():
    z=model(torch.zeros(2,2,256,device=device))
    if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real frozen checkpoint default-head no-query smoke failed')
    write(d.BASE/(c['row_id']+'-smoke.json'),dict(status='PASS',real_checkpoint=True,target_IQ_reads=0,head_fit_updates=0,optimizer_updates=0))
    for x,y,domain,meta in ctx['val_loader']:
     if len(y)!=len(x) or y.min()<0 or y.max()>=6:raise ValueError('Visible source V labels only')
     logits=model(x.to(device));pred=logits.argmax(1).cpu().numpy();truth=y.numpy();loss=F.cross_entropy(logits,y.to(device),reduction='none').cpu().numpy();other=logits.clone();other.scatter_(1,y.to(device)[:,None],-torch.inf);margin=((logits.gather(1,y.to(device)[:,None]).squeeze(1)-other.max(1).values)/30.).cpu().numpy()
     np.add.at(cm,(truth,pred),1);count+=len(y)
     for r,mat in rx_cm.items():
      mask=meta['rx_i'].numpy()==r;np.add.at(mat,(truth[mask],pred[mask]),1)
     for day,mat in day_cm.items():
      mask=meta['day_i'].numpy()==day;np.add.at(mat,(truth[mask],pred[mask]),1)
     for k in range(6):ce[k].extend(loss[truth==k].tolist());margins[k].extend(margin[truth==k].tolist())
   if count!=27000 or not np.all(cm.sum(1)==4500) or any(not np.all(v.sum(1)==900) for v in rx_cm.values()) or any(not np.all(v.sum(1)==1500) for v in day_cm.values()):raise ValueError('Complete original sourceV class/RX/day counts differ')
   if versions!=[p._version for p in model.parameters()] or any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen source V evaluation modified model state')
   value=dict(row_id=c['row_id'],model_seed=c['model_seed'],count=count,class_names=list(original.CLASSES),confusion=cm.tolist(),receiver_confusion={str(r):m.tolist() for r,m in rx_cm.items()},day_confusion={str(r):m.tolist() for r,m in day_cm.items()},per_TX=[dict(class_index=k,class_name=original.CLASSES[k],n=4500,recall=float(cm[k,k]/4500),worst_source_RX_recall=float(min(m[k,k]/900 for m in rx_cm.values())),CE=float(np.mean(ce[k])),margin_mean=float(np.mean(margins[k])),margin_q10=float(np.quantile(margins[k],.1))) for k in range(6)],checkpoint_role='fixed original final50; source analysis only; no new-model initialization',source_contract='EXACT_MATCH',target_IQ_reads=0,head_fit_updates=0,optimizer_updates=0,model_state_unchanged=True)
   write(d.BASE/(c['row_id']+'-sourceV.json'),value);rows.append(value);pooled+=cm
   for r in rx_cm:rx_pooled[r]+=rx_cm[r]
   write(d.BASE/'progress.json',dict(status='RUNNING',completed_models=len(rows),sourceV_predictions=count*len(rows),target_IQ_reads=0));del model
   torch.cuda.empty_cache()
  class_counts=[(int(pooled[k,k]),int(pooled[k].sum())) for k in range(6)];receiver_counts=[[(int(m[k,k]),int(m[k].sum())) for k in range(6)] for m in rx_pooled.values()];calibration=calibrate(class_counts,receiver_counts,d.BETA);calibration.update(class_names=list(original.CLASSES),class_counts=class_counts,receiver_counts=receiver_counts,source_rows=[r['row_id'] for r in rows],no_specific_TX_rule=True)
  write(d.BASE/'source_calibration.json',calibration)
  write(d.BASE/'result.json',dict(status='SOURCE_AUDIT_COMPLETE',run_id=d.RUN,rows=rows,calibration=calibration,sourceV_predictions=81000,synthetic_smoke_predictions=6,target_IQ_reads=0,head_fit_updates=0,optimizer_updates=0,seconds=time.time()-started))
  write(d.BASE/'completion.json',dict(status='SOURCE_AUDIT_COMPLETE',rows=3,sourceV_predictions=81000,target_IQ_reads=0,head_fit_updates=0,optimizer_updates=0))
 print('SOURCE_TX_AUDIT_COMPLETE',flush=True)
