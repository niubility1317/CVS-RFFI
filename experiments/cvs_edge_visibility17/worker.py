"""Two edge observations through the frozen original head;no fitting or target."""
import os,sys,time
import torch
from . import design as d
from .checkpoint import model,fit_data
from .window import apply,qualify

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
 torch.set_num_threads(2);d.validate(d.config());out=d.resolve(d.BASE)/('s'+str(seed));out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cost=0
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,model_seed=seed))
 try:
  m,trained,provenance=model(seed);before={k:v.detach().clone() for k,v in m.state_dict().items()}
  if seed==d.SEEDS[0]:
   with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(2026101717)))['tx_logits']
   if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Real-checkpoint no-query smoke failed')
   cost+=2;d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',synthetic_NN_packets=2,source_IQ_reads=0,optimizer_updates=0,target_access=False));print('SMOKE PASS real matching default head;2 synthetic IQ;continue',flush=True)
  L=fit_data(trained);labels,rx,day=(L[k].tolist() for k in ('y','rx','day'));d.write(out/'source_metadata.json',dict(physical_ids=L['ids'],labels=labels,RX=rx,day=day))
  for rr in d.FIT:
   for dd in (1,2,3):
    for yy in range(6):
     if sum((a,b,c)==(rr,dd,yy) for a,b,c in zip(rx,day,labels))!=70:raise ValueError('Exact72cells/70packets required')
  d.write(out/'resolved_config.json',dict(config=d.config(),executed_commit=(d.ROOT/'release_commit.txt').read_text().strip(),model_seed=seed,checkpoint_provenance=provenance,fit_L_unique=5040,source_cache_L_deserialized=6300,unused_source_V_features_deserialized=True,U_role_ID_metadata_only=True,U_IQ_or_labels_in_cache=False,held_RX1_used=False,python=sys.executable,torch_version=torch.__version__,source_only=True,target_access=False))
  (out/'predictions').mkdir();scores={};decisions={};rawpred=None
  for view in ('raw','left','right'):
   logits=[];pred=[];observations=[];allstats={k:[] for k in ('relative_IQ_change','power_increase','reference_unchanged','outside_support_unchanged')}
   for start in range(0,5040,128):
    x=L['x'][start:start+128].clone();saved=x.clone()
    if view=='raw':aug=x;info={}
    else:aug,info=apply(x,view)
    if not torch.equal(x,saved):raise ValueError('Original input changed in place')
    with torch.no_grad():z=m.forward_identity_only(aug)['tx_logits'].double()
    if z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()):raise ValueError('Complete finite default-head predictions required')
    cost+=len(x);logits+=z.tolist();pred+=z.argmax(1).tolist();observations.append(dict(start=start,count=len(x),**info))
    if view!='raw':
     for k in allstats:allstats[k]+=info[k]
    d.write(out/'progress.json',dict(model_seed=seed,view=view,view_completed=len(pred),view_total=5040,NN_packets=cost,optimizer_updates=0,target_access=False))
   if view=='raw':rawpred=pred[:]
   value=score(labels,pred);value['mean_CE']=float(torch.nn.functional.cross_entropy(torch.tensor(logits,dtype=torch.float64),torch.tensor(labels)))
   for key,values,allowed in [('RX',rx,d.FIT),('TX',labels,range(6)),('day',day,(1,2,3))]:value['by_'+key]={str(v):score([labels[i] for i in range(5040) if values[i]==v],[pred[i] for i in range(5040) if values[i]==v]) for v in allowed}
   cells=[]
   for rr in d.FIT:
    for dd in (1,2,3):
     for yy in range(6):
      ix=[i for i in range(5040) if (rx[i],day[i],labels[i])==(rr,dd,yy)];cells.append(dict(RX=rr,day=dd,TX=yy,**score([labels[i] for i in ix],[pred[i] for i in ix])))
   value['cells']=cells;correct=[i for i in range(5040) if rawpred[i]==labels[i]];value['raw_correct_count']=len(correct);value['raw_correct_retention']=sum(pred[i]==labels[i] for i in correct)/len(correct) if correct else 0.;scores[view]=value
   if view!='raw':decisions[view]=qualify(scores['raw'],value,allstats)
   d.write(out/'predictions'/(view+'.json'),dict(view=view,anchor_indices=list(range(5040)),predictions=pred,logits=logits,labels=labels,observations=observations,source_only=True,target_access=False));print('VIEW '+view+' count='+str(len(pred))+' accuracy='+str(value['accuracy'])+' retention='+str(value['raw_correct_retention']),flush=True)
  expected=15120+(2 if seed==d.SEEDS[0] else 0)
  if cost!=expected or m.id_backbone.encoder._view_eval_packets!=cost or m.id_backbone.encoder._view_train_packets or m.training or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m._joint_calls or m._supcon_calls or m._joint_As_packets:raise ValueError('Frozen model/state or actual NN cost differs')
  decision='PASS' if all(v['decision']=='PASS' for v in decisions.values()) else 'FAIL';result=dict(status='ANALYZED_SOURCE_IDENTITY_GUARD_ONLY',decision=decision,model_seed=seed,scores=scores,qualification=decisions,actual_NN_packets=cost,source_decisions=15120,unique_source_L=5040,edge_window_applications=10080,checkpoint_loads=1,optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,model_state_unchanged=True,default_head_only=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-started)
  d.write(out/'result.json',result);d.write(out/'completion.json',{k:result[k] for k in ('status','decision','model_seed','actual_NN_packets','source_decisions','optimizer_updates','target_access','automatic_training')});print('COMPLETE '+decision,flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),no_automatic_retry=True));raise
