import json,torch
from . import design as d
from .objective import correction
from experiments.cvs_receiver_dro50.observer import Observer as Base
from experiments.cvs_local_order50.observer import derivative
from experiments.cvs_source_layer_order.probe import gradient_stats
class Observer(Base):
 def __init__(self,*,design=None):super().__init__(design=design or d)
 def labeled(self,*,model,args,epoch,batch_idx,logits,labels,receivers,clean_count,original_loss,role):
  dd=self.design;step=(epoch-1)*222+batch_idx
  if model is not self.model or step!=self.steps+1 or clean_count!=len(labels) or len(logits)!=clean_count or args.label_smoothing!=.01:raise ValueError('Native labelledgradient context differs')
  linear,info,witness=correction(model,logits,labels,receivers,self.c['receiver_gradient']['mode'],role=role);selected=original_loss+linear
  if float(linear.detach())!=0. or float(selected.detach())!=float(original_loss.detach()):raise ValueError('Gradient surrogate changed native CE scalar')
  self.steps=step
  if epoch in dd.DIAGNOSTIC_EPOCHS and batch_idx%16==0:
   objects=self.params+[self.head];vectors={key:derivative(loss,objects) for key,loss in [('original',original_loss),('selected',selected),('delta',linear)]};vectors['target']=[witness[n]['target'] for n in dd.PARAMETERS]+[vectors['original'][-1].clone()]
   for actual,target in zip(vectors['selected'],vectors['target']):
    if not torch.allclose(actual,target,atol=1e-6,rtol=1e-5):raise ValueError('Actual selected derivative differs from fullfeature correction')
   if torch.count_nonzero(vectors['delta'][-1]) or not torch.allclose(vectors['selected'][-1],vectors['original'][-1],atol=1e-6,rtol=1e-5):raise ValueError('DefaultW gradient changed by correction')
   self.grad.append(dict(step=step,epoch=epoch,vectors=vectors));self.records.append(dict(step=step,epoch=epoch,layers=[dict(parameter=n,selected=gradient_stats(vectors['original'][k],vectors['selected'][k]),delta=gradient_stats(vectors['original'][k],vectors['delta'][k])) for k,n in enumerate([*dd.PARAMETERS,'existing_default_W'])]))
  self.stream.write(json.dumps(dict(step=step,epoch=epoch,batch=batch_idx,mode=self.c['receiver_gradient']['mode'],source_packets=clean_count,original_CE=float(original_loss.detach()),selected_CE=float(selected.detach()),**info),allow_nan=False)+chr(10));return selected
 def epoch(self,model,ctx,rows,out,device):
  dd=self.design;epoch=int(rows[-1]['epoch']);self.stream.flush()
  if self.steps!=epoch*222:raise ValueError('Receivergradient budget differs')
  if epoch in dd.DIAGNOSTIC_EPOCHS:
   if len(self.grad)!=13:raise ValueError('Actualgradient witnesses incomplete')
   p=out/f'receiver_risk_gradients_e{epoch}.pt';torch.save(self.grad,p);raw=torch.load(p,map_location='cpu',weights_only=False);ref=[r for r in self.records if r['epoch']==epoch]
   for a,b in zip(raw,ref):
    for k,row in enumerate(b['layers']):
     for key in ('selected','delta'):
      if gradient_stats(a['vectors']['original'][k],a['vectors'][key][k])!=row[key]:raise ValueError('Savedgradient recount differs')
   self.saved.append(epoch);self.grad=[];dd.write(out/'receiver_risk_gradient_records.json',dict(status='VERIFIED',epochs=self.saved,batch_records=self.records,saved_gradient_recount='VERIFIED',head_fit_updates=0,default_W_normal_joint_CE=True,U_truth_read=False))
 def finish(self):
  dd=self.design
  if self.steps!=11100 or self.U_steps!=3774 or self.U_packets!=963900 or self.saved!=list(dd.DIAGNOSTIC_EPOCHS) or len(self.records)!=52:raise ValueError('Receivergradient nativebudget incomplete')
  value=dict(status='VERIFIED',mode=self.c['receiver_gradient']['mode'],steps=self.steps,U_steps=self.U_steps,U_packets=self.U_packets,U_accepted=self.U_accepted,gradient_batches=52,layer_records=208,saved_vectors=832,gradient_epochs=self.saved,extra_model_forwards=0,new_parameters=0,new_heads=0,head_fit_updates=0,default_head_jointly_trained=True,U_loss_returned_unchanged=True,U_truth_read=False,saved_gradient_recount='VERIFIED');dd.write(self.out/'receiver_risk_training_observation.json',value);return value
