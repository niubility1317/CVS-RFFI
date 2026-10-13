"""Read-only counts from the existing legacy U gate; never reads U truth."""
from contextlib import contextmanager
import inspect
from pathlib import Path
import torch

class Monitor:
 def __init__(self,c,write):self.c=c;self.write=write;self.epochs={}
 def observe(self,values):
  if not values.get('legacy_unlabeled_active') or values.get('phase')!='pseudo':return
  required=('epoch','mask','pseudo','receiver_u','day_u','out_s','y_u')
  if any(k not in values for k in required):raise ValueError('Native hidden-U diagnostic boundary differs')
  if bool((values['y_u']>=0).any()):raise ValueError('U labels exposed')
  mask=values['mask'].detach().cpu().bool();pred=values['pseudo'].detach().cpu().long()
  rx=values['receiver_u'].detach().cpu().long();day=values['day_u'].detach().cpu().long()
  agreement=(values['out_s']['tx_logits'].detach().argmax(1).cpu()==pred)
  epoch=int(values['epoch']);bucket=self.epochs.setdefault(epoch,{})
  for r in rx.unique().tolist():
   for day_value in day[rx==r].unique().tolist():
    take=(rx==r)&(day==day_value);key=f'{r}/{day_value}'
    record=bucket.setdefault(key,dict(count=0,accepted=0,student_strong_teacher_weak_agreement=0,predicted_class_counts=[0]*6,accepted_predicted_class_counts=[0]*6))
    record['count']+=int(take.sum());record['accepted']+=int((take&mask).sum());record['student_strong_teacher_weak_agreement']+=int((take&agreement).sum())
    for k,v in enumerate(torch.bincount(pred[take],minlength=6).tolist()):record['predicted_class_counts'][k]+=v
    for k,v in enumerate(torch.bincount(pred[take&mask],minlength=6).tolist()):record['accepted_predicted_class_counts'][k]+=v
 def close(self):
  self.write(Path(self.c['output_root'])/'source_U_epoch_diagnostic.json',dict(status='SOURCE_DIAGNOSTIC',epochs=self.epochs,U_truth_read=False,new_forward_calls=0,updates=0,label_stage_epochs=self.c['label_epochs'],student_teacher_agreement_scope='existing strong student vs weak EMA prediction; not same-view accuracy',physical_receiver_day_counts=True))

@contextmanager
def installed_monitor(native,c,write):
 previous=native._pseudo_gate_pass_rates;monitor=Monitor(c,write)
 def observe(*args,**kwargs):
  result=previous(*args,**kwargs)
  monitor.observe(inspect.currentframe().f_back.f_locals)
  return result
 native._pseudo_gate_pass_rates=observe
 try:yield monitor
 finally:
  native._pseudo_gate_pass_rates=previous
  if monitor.epochs:monitor.close()
