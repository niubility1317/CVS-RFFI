"""Two existing defaultheads;allfitL/4draws/no learned diagnostic or updates."""
import gzip,json,os,sys,time,statistics
import torch
from . import guard_design as d
from .checkpoint import model,fit_data
from .iq import apply,choices

def score(y,p):
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 f=[]
 for i in range(6):
  den=sum(cm[i])+sum(z[i] for z in cm);f.append(2*cm[i][i]/den if den else 0.)
 return dict(count=len(y),accuracy=sum(cm[i][i] for i in range(6))/len(y),macro_F1=sum(f)/6)

def run(seed):
 if seed not in d.SEEDS or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU2 guard')
 torch.set_num_threads(2);out=d.resolve(d.BASE)/('s'+str(seed));out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,model_seed=seed,owner=d.OWNER))
 try:
  m,contract,provenance,args=model(seed);before={k:v.clone() for k,v in m.state_dict().items()};L=fit_data(contract);y=L['y'].tolist();rx=L['rx'].tolist();day=L['day'].tolist();d.write(out/'source_metadata.json',dict(physical_ids=L['ids'],labels=y,RX=rx,day=day));d.write(out/'resolved_config.json',dict(config=d.config(),provenance=provenance,model_seed=seed,python=sys.executable,actual_model_file=provenance['model_api'],code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),CP_used_as_train_init=False,source_only=True,target_access=False))
  nn=0;rawpred=[];rawlog=[]
  def infer(x):
   nonlocal nn
   with torch.no_grad():z=m.forward_identity_only(x)['tx_logits'].double()
   if z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite defaulthead six-class logits')
   nn+=len(x);return z.tolist()
  def metrics(log,ai):
   pred=[max(range(6),key=v.__getitem__) for v in log];truth=[y[i] for i in ai];q=score(truth,pred)
   for name,values,allowed in [('RX',rx,d.FIT),('TX',y,range(6)),('day',day,(1,2,3))]:q['by_'+name]={str(v):score([truth[j] for j,i in enumerate(ai) if values[i]==v],[pred[j] for j,i in enumerate(ai) if values[i]==v]) for v in allowed}
   if rawpred:
    correct=[j for j,i in enumerate(ai) if rawpred[i]==y[i]];q['raw_correct_retention']=sum(pred[j]==truth[j] for j in correct)/len(correct)
   return pred,q
  with gzip.open(out/'predictions.jsonl.gz','xt',encoding='utf8',compresslevel=3) as pf,gzip.open(out/'observations.jsonl.gz','xt',encoding='utf8',compresslevel=3) as of:
   def save(view,draw,ai,log):
    for i,z in zip(ai,log):pf.write(json.dumps(dict(view=view,draw=draw,anchor=i,logits=z),allow_nan=False)+'\n')
   for starti in range(0,5040,128):
    ai=list(range(starti,min(starti+128,5040)));log=infer(L['x'][ai].clone());rawlog+=log;save('raw',0,ai,log)
   rawpred,rawscore=metrics(rawlog,list(range(5040)));allm=[];allg=[];anchors=[];draws=[];iq=[];power=[];err=change=0.
   for draw,seedview in enumerate(d.DRAW_SEEDS,1):
    order=torch.randperm(5040,generator=torch.Generator().manual_seed(seedview+100)).tolist();ml=[];gl=[]
    for starti in range(0,5040,128):
     ai=order[starti:starti+128];x=L['x'][ai].clone();saved=x.clone();state=torch.get_rng_state();ks=choices(len(ai),seedview+starti);masked,gain,info=apply(x,ks,True);lz=infer(masked);gz=infer(gain)
     if not torch.equal(saved,x) or not torch.equal(state,torch.get_rng_state()):raise ValueError('Guard mutatedIQ or modelconsumed randomstream')
     ml+=lz;gl+=gz;save('MASK',draw,ai,lz);save('GAIN',draw,ai,gz);iq+=info['relative_IQ_RMS'];power+=info['power_match_relative_error'];err+=sum(info['boundary_error_energy']);change+=sum(info['change_energy']);of.write(json.dumps(dict(draw=draw,anchors=ai,**info),allow_nan=False)+'\n');d.write(out/'progress.json',dict(draw=draw,draw_processed=starti+len(ai),NN=nn,source_only=True,target_access=False,updates=0));pf.flush();of.flush()
    _,ms=metrics(ml,order);_,gs=metrics(gl,order);draws.append(dict(draw=draw,MASK=ms,GAIN=gs));allm+=ml;allg+=gl;anchors+=order;print('GUARD seed='+str(seed)+' draw='+str(draw)+' sourceNN='+str(nn),flush=True)
  _,ms=metrics(allm,anchors);_,gs=metrics(allg,anchors)
  def identity(q):
   checks=dict(accuracy=100*(q['accuracy']-rawscore['accuracy'])>=-.5-1e-12,F1=100*(q['macro_F1']-rawscore['macro_F1'])>=-.5-1e-12,retention=q['raw_correct_retention']>=.99)
   for name in ('RX','TX','day'):checks[name]=all(100*(v['accuracy']-rawscore['by_'+name][k]['accuracy'])>=-1.-1e-12 for k,v in q['by_'+name].items())
   return all(checks.values())
  checks=dict(pooled_identity=identity(ms),each_draw_identity=all(identity(v['MASK']) for v in draws),nontrivial=statistics.median(iq)>=.01,bounded=max(iq)<=.100001,power_match=max(power)<=1e-6,boundary=err/max(change,1e-30)<=.01)
  if nn!=45360 or m.id_backbone.encoder._view_eval_packets!=nn or m.id_backbone.encoder._view_train_packets or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m._joint_calls or m._supcon_calls:raise ValueError('Frozen defaulthead/state/NN budget differs')
  result=dict(status='SOURCE_GUARD_COMPLETE',model_seed=seed,qualified=all(checks.values()),checks=checks,raw=rawscore,MASK=ms,GAIN=gs,draws=draws,IQ_change_median=statistics.median(iq),IQ_change_max=max(iq),boundary_error_to_change=err/change,power_match_error_max=max(power),NN=nn,checkpoint_loads=1,source_L_unique=5040,mask_views=20160,gain_views=20160,DSP_long_references=20160,source_only=True,target_access=False,updates=0,head_fit_updates=0,model_state_unchanged=True,Ag_efficacy_proved=False,wall_seconds=time.perf_counter()-start);d.write(out/'result.json',result);d.write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE',qualified=result['qualified'],NN=nn,source_only=True,target_access=False));print('GUARD COMPLETE '+str(result['qualified']),flush=True)
 except BaseException as e:d.write(out/'failure.json',dict(error=repr(e),preserve_outputs=True,no_automatic_retry=True));raise
