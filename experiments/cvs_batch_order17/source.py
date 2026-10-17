"""Matched private L loading, source-only fixed final20 and inherited seven views."""
from contextlib import contextmanager
import json
from . import design as d,model as api
from .pairing import Order,private_L_loader
from experiments.cvs_scene_cycle20.source import Observer as ParentObserver
from experiments.cvs_feature_sequence50.source import train as shared_train

class Observer(ParentObserver):
 def data(self,native,ctx,out):
  super().data(native,ctx,out)
  self.private_L=private_L_loader(ctx['train_loader'],self.c['model_seed']);self.order=Order(self.c['arm'],out)
  d.write(out/'L_batch_order_config.json',dict(config=self.c['batch_order'],actual_L_generators=self.private_L,original_U_modified=False,target_access=False))
 @contextmanager
 def native_scope(self,native):
  # Data are built inside native.train after this context starts. Resolve Order
  # only when the original native pair function is actually called.
  original=native._muse_epoch_pairs
  def delayed(*args,**kwargs):
   if not hasattr(self,'order'):raise ValueError('L source data must exist before epoch pairs')
   return self.order.wrap(original)(*args,**kwargs)
  native._muse_epoch_pairs=delayed
  try:
   with super().native_scope(native):yield
  finally:
   native._muse_epoch_pairs=original
   if hasattr(self,'order'):self.order.close()
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);epoch=len(rows)
  if self.order.epochs!=epoch or self.order.steps!=epoch*222 or self.order.blocks!=epoch*37:raise ValueError('Actual L buffer/update budget differs')
  if self.c['arm']=='O0' and self.order.moved:raise ValueError('Matched control order changed')
  d.write(out/'L_batch_order_progress.json',dict(epoch=epoch,steps=self.order.steps,blocks=self.order.blocks,moved_positions=self.order.moved,buffer_CPU_seconds=self.order.seconds,U_buffered=False,IQ_changed=False,target_access=False))
 def _final_record(self,done,payload,views):
  r=super()._final_record(done,payload,views);audit=self.order.summary()
  if (audit['epochs'],audit['steps'],audit['blocks'])!=(20,4440,740) or audit['U_buffered'] or audit['new_IQ_transform']:raise ValueError('Original L-only scope/budget differs')
  d.write(self.out/'L_batch_order_audit.json',dict(config=self.c['batch_order'],actual_L_generators=self.private_L,**audit,all_block_multisets_preserved=True,target_access=False))
  r.update(status='SOURCE_L_BATCH_ORDER_E20_COMPLETE',order_epochs=audit['epochs'],order_steps=audit['steps'],order_blocks=audit['blocks'],moved_L_positions=audit['moved_positions'],all_block_multisets_preserved=True,buffer_CPU_seconds=audit['buffer_CPU_seconds'],private_L_generators=self.private_L,Ag=False,As=False)
  d.write(self.out/'satmix_result.json',r);return r

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
