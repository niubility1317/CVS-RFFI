"""One frozen default-head source guard, with complete predictions and no fitting."""
import os,sys,time
import torch
from . import design as d
from .checkpoint import model,fit_data
from .iqmix import mix,donor_indices,qualify

def score(y,p):
 if not y or len(y)!=len(p) or any(v not in range(6) for v in y+p):raise ValueError('Complete six-class source predictions required')
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 f1=[]
 for i in range(6):
  den=sum(cm[i])+sum(row[i] for row in cm);f1.append(2*cm[i][i]/den if den else 0.)
 return dict(count=len(y),accuracy=sum(cm[i][i] for i in range(6))/len(y),macro_F1=sum(f1)/6,confusion=cm)

def run(seed):
 if seed not in d.SEEDS or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU two-model source guard only')
 torch.set_num_threads(2);d.validate(d.config());out=d.BASE/('s'+str(seed));out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cost=0
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,model_seed=seed))
 try:
  m,trained,provenance=model(seed);before={k:v.detach().clone() for k,v in m.state_dict().items()}
  if seed==d.SEEDS[0]:
   with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(20261016171)))['tx_logits']
   if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Real-checkpoint no-query smoke failed')
   cost+=2;d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',synthetic_NN_packets=2,source_IQ_reads=0,optimizer_updates=0,target_access=False));print('SMOKE PASS real matching default head;2 synthetic IQ;continue',flush=True)
  L=fit_data(trained);labels,rx,day=(L[k].tolist() for k in ('y','rx','day'));same,cross=donor_indices(labels,rx,day,L['ids']);d.write(out/'donor_indices.json',dict(same=same,cross=cross,physical_ids=L['ids'],labels=labels,RX=rx,day=day))
  d.write(out/'resolved_config.json',dict(config=d.config(),executed_commit=(d.ROOT/'release_commit.txt').read_text().strip(),model_seed=seed,checkpoint_provenance=provenance,fit_L_unique=5040,original_carrier_other_roles_features_deserialized_but_unused=True,held_RX1_used=False,python=sys.executable,torch_version=torch.__version__,source_only=True,target_access=False))
  (out/'predictions').mkdir();scores={};all_changes=[];all_errors=[];rawpred=None
  for view in ('raw','sameRX_sameTX','crossRX_sameTX'):
   anchors=list(range(5040)) if view!='crossRX_sameTX' else [i for i in range(5040) for _ in range(3)]
   donors=anchors if view=='raw' else same if view=='sameRX_sameTX' else [j for row in cross for j in row]
   logits=[];pred=[];observations=[];truth=[labels[i] for i in anchors]
   for start_index in range(0,len(anchors),128):
    ai=anchors[start_index:start_index+128];di=donors[start_index:start_index+128];x=L['x'][ai].clone()
    if view=='raw':mixed=x;info={}
    else:mixed,info=mix(x,L['x'][di],d.LAMBDA)
    with torch.no_grad():z=m.forward_identity_only(mixed)['tx_logits'].double()
    if z.shape!=(len(ai),6) or not bool(torch.isfinite(z).all()):raise ValueError('Complete finite existing-head predictions required')
    cost+=len(ai);logits+=z.tolist();pred+=z.argmax(1).tolist();observations.append(dict(start=start_index,count=len(ai),**info))
    if view=='crossRX_sameTX':all_changes+=info['relative_IQ_change'];all_errors+=info['relative_power_error']
    d.write(out/'progress.json',dict(model_seed=seed,view=view,view_completed=len(pred),view_total=len(anchors),NN_packets=cost,optimizer_updates=0,target_access=False))
   if view=='raw':rawpred=pred[:]
   value=score(truth,pred);value['mean_CE']=float(torch.nn.functional.cross_entropy(torch.tensor(logits,dtype=torch.float64),torch.tensor(truth)))
   for key,values,allowed in [('RX',rx,d.FIT),('TX',labels,range(6)),('day',day,(1,2,3))]:
    value['by_'+key]={str(v):score([truth[j] for j,i in enumerate(anchors) if values[i]==v],[pred[j] for j,i in enumerate(anchors) if values[i]==v]) for v in allowed}
   cells=[]
   for rr in d.FIT:
    for dd in (1,2,3):
     for yy in range(6):
      ix=[j for j,i in enumerate(anchors) if (rx[i],day[i],labels[i])==(rr,dd,yy)];cells.append(dict(RX=rr,day=dd,TX=yy,**score([truth[j] for j in ix],[pred[j] for j in ix])))
   value['cells']=cells;original_correct=[j for j,i in enumerate(anchors) if rawpred[i]==labels[i]];value['raw_correct_count']=len(original_correct);value['raw_correct_retention']=sum(pred[j]==truth[j] for j in original_correct)/len(original_correct) if original_correct else 0.;scores[view]=value
   d.write(out/'predictions'/(view+'.json'),dict(view=view,anchor_indices=anchors,donor_indices=donors,predictions=pred,logits=logits,labels=truth,observations=observations,source_only=True,target_access=False))
   print('VIEW '+view+' count='+str(len(pred))+' accuracy='+str(value['accuracy'])+' retention='+str(value['raw_correct_retention']),flush=True)
  expected=25200+(2 if seed==d.SEEDS[0] else 0)
  if cost!=expected or m.id_backbone.encoder._view_eval_packets!=cost or m.id_backbone.encoder._view_train_packets or m.training or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m._joint_calls or m._supcon_calls or m._joint_As_packets:raise ValueError('Frozen model/state or actual NN cost differs')
  decision=qualify(scores['raw'],scores['sameRX_sameTX'],scores['crossRX_sameTX'],all_changes,all_errors)
  result=dict(status='ANALYZED_SOURCE_IDENTITY_GUARD_ONLY',model_seed=seed,scores=scores,qualification=decision,actual_NN_packets=cost,source_decisions=25200,unique_source_L=5040,IQmix_generations=20160,checkpoint_loads=1,optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,model_state_unchanged=True,default_head_only=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-start)
  d.write(out/'result.json',result);d.write(out/'completion.json',dict(status=result['status'],decision=decision['decision'],model_seed=seed,actual_NN_packets=cost,source_decisions=25200,optimizer_updates=0,target_access=False,automatic_training=False));print('COMPLETE '+str(decision),flush=True)
 except Exception as error:d.write(out/'failure.json',dict(status='FAILED',error=repr(error),no_automatic_retry=True));raise
