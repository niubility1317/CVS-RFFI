import json,torch
from . import design as d
from .objective import objective
from experiments.cvs_receiver_dro50.observer import Observer as Base
from experiments.cvs_local_order50.observer import derivative
from experiments.cvs_source_layer_order.probe import gradient_stats
class Observer(Base):
 def __init__(self):super().__init__(design=d)
 def initialize(self,model,c,out):
  super().initialize(model,c,out);del self.logq
 def labeled(self,*,model,args,epoch,batch_idx,logits,labels,receivers,clean_count,original_loss,role):
  step=(epoch-1)*222+batch_idx
  if model is not self.model or step!=self.steps+1 or role!='L_s' or clean_count!=len(labels) or len(logits)!=clean_count or args.label_smoothing!=.01:raise ValueError('Original labelledcontext differs')
  alpha=self.c['receiver_risk']['coefficient'];selected,balanced,var,counts,means=objective(logits,labels,receivers,alpha,role=role)
  raw=(means*counts.to(means)).sum()/counts.sum()
  if abs(float(raw.detach())-float(original_loss.detach()))>2e-6:raise ValueError('OriginallabelledCE differs')
  self.steps=step
  if epoch in d.DIAGNOSTIC_EPOCHS and batch_idx%16==0:
   objects=self.params+[self.head];vectors={k:derivative(v,objects) for k,v in [('balanced',balanced),('variance',var),('selected',selected)]}
   self.grad.append(dict(step=step,epoch=epoch,vectors=vectors));self.records.append(dict(step=step,epoch=epoch,layers=[dict(parameter=n,selected=gradient_stats(vectors['balanced'][k],vectors['selected'][k]),variance=gradient_stats(vectors['balanced'][k],vectors['variance'][k])) for k,n in enumerate([*d.PARAMETERS,'existing_default_W'])]))
  self.stream.write(json.dumps(dict(step=step,epoch=epoch,batch=batch_idx,mode=self.c['receiver_risk']['mode'],source_packets=clean_count,group_counts=counts.detach().cpu().tolist(),group_mean_CE=means.detach().cpu().tolist(),original_CE=float(original_loss.detach()),balanced_CE=float(balanced.detach()),risk_variance=float(var.detach()),coefficient=alpha,selected_CE=float(selected.detach()),source_only=True),allow_nan=False)+chr(10));return selected
 def epoch(self,model,ctx,rows,out,device):
  epoch=int(rows[-1]['epoch']);self.stream.flush()
  if self.steps!=epoch*222:raise ValueError('VREXsourcebudgetincomplete')
  if epoch in d.DIAGNOSTIC_EPOCHS:
   if len(self.grad)!=13:raise ValueError('ActualVREXgradientsincomplete')
   path=out/f'receiver_risk_gradients_e{epoch}.pt';torch.save(self.grad,path);actual=torch.load(path,map_location='cpu',weights_only=False);expected=[r for r in self.records if r['epoch']==epoch]
   for a,b in zip(actual,expected):
    for k,row in enumerate(b['layers']):
     for name in ('selected','variance'):
      if gradient_stats(a['vectors']['balanced'][k],a['vectors'][name][k])!=row[name]:raise ValueError('SavedVREXgradientrecountdiffers')
   self.saved.append(epoch);self.grad=[];d.write(out/'receiver_risk_gradient_records.json',dict(status='VERIFIED',epochs=self.saved,batch_records=self.records,saved_gradient_recount='VERIFIED',head_fit_updates=0,default_W_normal_joint_objective=True,U_truth_read=False))
 def finish(self):
  if self.steps!=11100 or self.U_steps!=3774 or self.U_packets!=963900 or self.saved!=list(d.DIAGNOSTIC_EPOCHS) or len(self.records)!=52:raise ValueError('VREX/nativebudgetincomplete')
  value=dict(status='VERIFIED',mode=self.c['receiver_risk']['mode'],coefficient=self.c['receiver_risk']['coefficient'],steps=self.steps,U_steps=self.U_steps,U_packets=self.U_packets,U_accepted=self.U_accepted,gradient_batches=52,layer_records=208,saved_vectors=624,gradient_epochs=self.saved,extra_model_forwards=0,new_parameters=0,new_heads=0,head_fit_updates=0,default_head_jointly_trained=True,U_loss_returned_unchanged=True,U_truth_read=False,saved_gradient_recount='VERIFIED')
  d.write(self.out/'receiver_risk_training_observation.json',value);return value
