"""Reuse original data/optimizer/EMA scope and count the frozen class-teacher cost."""
import gzip,json
from pathlib import Path
import torch
from . import design as d
from .source import parent_payload,segment_train
from experiments.cvs_signal_bridge30.observer import Observer as Common

class Observer(Common):
 def __init__(self,c):self.c=c;self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c)
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222;e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  if (m._full_channel.calls!=n or len(m._distill_records)!=n or m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*512 or m._shape_NN!=n*256 or m._full_channel.packets!=n*64 or m._distill_NN!=n*64 or self.optimizer._sinc_unit_steps!=n or self.optimizer._sinc_physical_clip_calls!=n or m._supcon_calls!=n or m._supcon_applied!=n or m._supcon_handles or m._supcon_pending is not None or e._full_train_packets!=n*768+ep*56700 or t._full_eval_packets-self.teacher_offset['eval']!=ep*56700 or t._full_train_packets-self.teacher_offset['train']!=0):raise ValueError('Original student/U/EMA and class-teacher budgets differ')
  teacher=m.__dict__['_class_teacher']
  if teacher.training or any(v.requires_grad or v.grad is not None for v in teacher.parameters()) or len(m.state_dict())!=206:raise ValueError('Frozen class teacher entered student state/optimizer')
  d.write(out/'reference_progress.json',dict(epoch=ep,cumulative_epoch=ep+20,steps=n,cumulative_steps=n+4440,L_NN=n*768,L_As=n*64,frozen_class_teacher_NN=n*64,U_student_NN=ep*56700,U_EMA_NN=ep*56700,U_truth_read=False,target_access=False))
 def finish(self):
  m=self.model;done=d.read(self.out/'completion.json');cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);steps=self.c['updates']
  if cp['a1_ema_successful_updates']!=steps or done['optimizer_steps']!=steps or len(m._distill_records)!=steps or torch.equal(self.initial_head,m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach()):raise ValueError('Fixed joint default-head budget not completed')
  for name,records in [('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records),('shape_steps',m._shape_records),('distill_steps',m._distill_records)]:
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(dict(r,cumulative_step=4440+r['step']),allow_nan=False)+'\n')
  init=d.read(self.out/'initialization.json');init['training_class_teacher_sources']=[str(d.TEACHER/'last.pt')] if self.c['distill_mode']=='CSIL' else [];init['class_teacher_shadow_source']=str(d.TEACHER/'last.pt');init['class_teacher_source_contract_exact']=True;init['class_teacher_fixed_last_epoch']=200;d.write(self.out/'initialization.json',init)
  teacher=m.__dict__['_class_teacher']
  for k,v in m.__dict__['_class_teacher_initial'].items():
   if not torch.equal(v,teacher.state_dict()[k].detach().cpu()):raise ValueError('Frozen class-teacher weights/buffers changed')
  result=dict(status='SEGMENT_TRAINED',config=self.c,segment_epochs=10,optimizer_steps=steps,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,frozen_class_teacher_NN=m._distill_NN,teacher_historical_updates=9800,teacher_CE_gradient_updates=0,student_train_NN=m.id_backbone.encoder._full_train_packets,U_student_NN=567000,U_EMA_NN=567000,all_As_CE=True,As_KL_removed=True,distill_mode=self.c['distill_mode'],teacher_correct=sum(r['teacher_correct'] for r in m._distill_records),head_fit_updates=0,teacher_state_unchanged=True,new_student_NN=0,U_truth_read=False,target_access=False,wall_seconds=done['elapsed_seconds'])
  d.write(self.out/'result.json',result);return result

def train(c):
 from . import model
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
