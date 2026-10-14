"""Complete V diagnostics at the actual current teacher stage, without fitting."""
from types import SimpleNamespace
import torch
from . import design as d
from experiments.cvs_source_label_gate.audit import gate,update,finish
from experiments.cvs_trusted_correction.order import order

def snapshot(student,teacher,context,epoch,c,native,out):
 if teacher is None or teacher is student:raise ValueError('Current ownEMA required')
 dataset=context['val_loader'].dataset
 if len(dataset)!=27000 or getattr(dataset,'hide',False) or 'v_select' not in dataset.split_source:raise ValueError('Only existing complete source V proxy')
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.id_backbone.modules()):raise ValueError('V proxy may not update identity normalization')
 device=next(student.parameters()).device;devs=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
 saved=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (student,teacher)];saved_grads=[[None if p.grad is None else p.grad.detach().clone() for p in m.parameters()] for m in (student,teacher)];flags=[[(m,m.training) for m in obj.modules()] for obj in (student,teacher)]
 meta={k:torch.tensor([int(getattr(i,k)) for i in dataset.index],dtype=torch.long) for k in ('rx_i','day_i','eq_i','sig_i')};meta['base_index']=torch.tensor(dataset.selected.tolist(),dtype=torch.long)
 indices=order(meta,c['model_seed']+99431,'PAIRS');args=d.make_args(c,str(device));records={};pred=[];truth=[];masks={};confidences=[];ece_count=torch.zeros(15,dtype=torch.long);ece_conf=torch.zeros(15,dtype=torch.double);ece_correct=ece_conf.clone()
 # Native gate functions were captured before read-only U telemetry wrappers,
 # so this V diagnostic cannot create a pending U forward observation.
 n=SimpleNamespace(**native)
 with torch.random.fork_rng(devices=devs),torch.no_grad():
  torch.manual_seed(c['model_seed']+99437+epoch);student.train();teacher.eval()
  from experiments.cvs_phase1_overlay.model import native_modules
  strong=native_modules()._strong_augment
  for begin in range(0,27000,256):
   ix=indices[begin:begin+256];batch=[dataset[int(i)] for i in ix];x=torch.stack([v[0] for v in batch]).to(device);y=torch.tensor([int(v[1]) for v in batch]);metadata={k:v[ix].to(device) for k,v in meta.items()};domain=torch.searchsorted(torch.tensor([1,3,4,6,8],device=device),metadata['rx_i'])*3+metadata['day_i']-1
   weak=teacher.forward_identity_only(x)['tx_logits'];s=student.forward_identity_only(strong(x,.015))['tx_logits'];p,ms=gate(n,args,weak,s,(None,metadata),domain)
   update(records,p,ms,y,metadata);confidence=weak.softmax(1).amax(1).cpu();pc=p.cpu();bin_index=(confidence*15).long().clamp_max(14)
   for k in range(15):
    keep=bin_index==k;ece_count[k]+=int(keep.sum());ece_conf[k]+=confidence[keep].double().sum();ece_correct[k]+=(pc[keep]==y[keep]).double().sum()
   pred.append(pc);truth.append(y);confidences.append(confidence)
   for k,v in ms.items():masks.setdefault(k,[]).append(v.cpu())
 for group in flags:
  for module,flag in group:module.training=flag
 for obj,state,grads in zip((student,teacher),saved,saved_grads):
  if any(not torch.equal(v,obj.state_dict()[k]) for k,v in state.items()) or any((old is None)!=(p.grad is None) or (old is not None and not torch.equal(old,p.grad)) for old,p in zip(grads,obj.parameters())):raise ValueError('V proxy changed model/buffers/gradients')
 result=finish(records);pseudo=torch.cat(pred);y=torch.cat(truth);masked={k:torch.cat(z) for k,z in masks.items()};ece=float(sum(abs(ece_correct[k]-ece_conf[k]) for k in range(15))/27000)
 artifact=dict(source_indices=indices,pseudo=pseudo,truth=y,confidence=torch.cat(confidences),masks=masked,rx=meta['rx_i'][indices],day=meta['day_i'][indices]);path=out/f'current_teacher_V_epoch{epoch}.pt';torch.save(artifact,path);r=torch.load(path,map_location='cpu',weights_only=False)
 if len(r['source_indices'].unique())!=27000 or not torch.equal(r['truth'],y) or not torch.equal(r['pseudo'],pseudo) or any(not torch.equal(v,r['masks'][k]) for k,v in masked.items()):raise ValueError('Source teacher proxy tensor recount differs')
 summary=dict(status='SOURCE_CURRENT_TEACHER_PROXY_COMPLETE',epoch=epoch,teacher_successful_updates=(epoch-1)*222,source_role='V',source_count=27000,source_forward_packets=54000,teacher_accuracy=float((r['pseudo']==r['truth']).double().mean()),teacher_ECE15=ece,records=result,ownEMA_current_stage=True,final_teacher_injected=False,threshold_fit=False,model_buffers_parameters_gradients_unchanged=True,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,U_IQ_read=False,U_truth_read=False,target_access=False,limitation='Complete source V proxy at current training stage. PAIR batches/views are fixed source proxies, not hiddenU correctness; calibration is measured, never fitted.')
 d.write(out/f'current_teacher_V_epoch{epoch}.json',summary);return summary
