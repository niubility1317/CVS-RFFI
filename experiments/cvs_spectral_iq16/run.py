"""Matching real-CP smoke first,fit-only statistics,then all25 source views."""
import os,sys,time
from pathlib import Path
import torch
from . import design as d
from .spectral import observe,basis,grid,audit
from .dsp import bank,apply
from .decision import aggregate,qualify
from experiments.cvs_commonphase_source.run import checkpoint_model
from experiments.cvs_commonphase_source.core import select_fit,score
def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Registered CPU-only probe')
 torch.set_num_threads(2);c=d.config();d.validate(c);out=d.BASE;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  m,trained=checkpoint_model();before={k:v.detach().clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(20261016302)))['tx_logits']
  if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or m._supcon_pending is not None or m._supcon_handles:raise ValueError('Matching realCP smoke failed')
  d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',strict_load=True,checkpoint_role='Own matching frozen F+CROSS_RX B00,not training initialization',synthetic_NN_packets=2,source_IQ_reads=0,target_access=False,optimizer_updates=0));print('SMOKE PASS matching real checkpoint;2synthetic;no query;continue',flush=True)
  carrier=d.read(d.SOURCE);prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
  if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']:raise ValueError('Validated source preparation differs')
  packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
  for actual in (packed['contract'],prepared['source_contract']):
   if any(actual.get(k)!=v for k,v in carrier.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Reused source handles differ')
  L=select_fit({k:packed['labeled'][k] for k in ('x','y','rx','day','ids')},carrier,trained);del packed;d.write(out/'fit_L_ids.json',L['ids'])
  d.write(out/'resolved_config.json',dict(config=c,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_commit=d.BACKEND_COMMIT,python=sys.executable,torch_version=torch.__version__,source_only=True,target_access=False,other_source_cache_roles_deserialized_but_never_used=True,old_fullL6300_spectral_coefficients_used=False,held_RX1_used=False,U_used=False,V_used=False,model_state_updates=0))
  labels=L['y'].tolist();rx=L['rx'].tolist();days=L['day'].tolist();raw_obs=observe(L['x']);raw_dct=raw_obs@basis();stat=audit(grid(raw_dct,L['y'],L['rx'],L['day']));d.write(out/'source_statistics.json',stat);d.write(out/'raw_DCT8.json',raw_dct.tolist());kernels=bank(stat['global_fit_effects'])
  d.write(out/'kernel_bank.json',{k:dict(**v['metadata'],h_real=v['h'].real.tolist(),h_imag=v['h'].imag.tolist(),desired_response_real=v['desired_response'].real.tolist(),desired_response_imag=v['desired_response'].imag.tolist()) for k,v in kernels.items()})
  print('INPUT VERIFIED fitL5040 only;18fold relativeMSE='+str(stat['relative_MSE'])+';continue all25 fixed views regardless scientific result',flush=True)
  raw_fft=torch.fft.fft(torch.complex(L['x'][:,0].double(),L['x'][:,1].double()));scores={};diagnostics={};batches={};(out/'predictions').mkdir()
  with torch.no_grad():
   for view in d.views():
    name=view['view'];pred=[];logits=[];ce=0.;records=[]
    for begin in range(0,5040,128):
     raw=L['x'][begin:begin+128];y=L['y'][begin:begin+128]
     if name=='identity':x=raw
     else:
      x,metrics=apply(raw,kernels[name],raw_obs[begin:begin+128],raw_fft[begin:begin+128]);records.append(dict(start=begin,**metrics))
     z=m.forward_identity_only(x.clone())['tx_logits'].double()
     if z.shape!=(len(y),6) or not bool(torch.isfinite(z).all()):raise ValueError('Complete finite default-head predictions required')
     pred+=z.argmax(1).tolist();logits+=z.tolist();ce+=float(torch.nn.functional.cross_entropy(z,y,reduction='sum'));d.write(out/'progress.json',dict(view=name,source_decisions=len(scores)*5040+len(pred),total_source_decisions=126000,optimizer_updates=0,target_access=False))
    s=score(labels,pred);s['mean_CE']=ce/5040
    for group,values,allowed in (('RX',rx,d.FIT),('TX',labels,range(6)),('day',days,(1,2,3))):s['by_'+group]={str(v):score([y for y,key in zip(labels,values) if key==v],[p for p,key in zip(pred,values) if key==v]) for v in allowed}
    scores[name]=s
    if records:batches[name]=records;diagnostics[name]=aggregate(records)
    d.write(out/'predictions'/(name+'.json'),dict(view=name,labels=labels,predictions=pred,logits=logits,RX=rx,day=days,physical_ids=L['ids'],head_fit_updates=0,target_access=False));d.write(out/'view_diagnostics.json',batches)
    print('VIEW '+name+' count5040 accuracy='+str(s['accuracy']),flush=True)
  if m.training or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m.id_backbone.encoder._view_eval_packets!=126002 or m.id_backbone.encoder._view_train_packets or m._joint_calls or m._joint_As_packets or m._supcon_calls:raise ValueError('Frozen model or actual NN cost differs')
  kmeta={k:v['metadata'] for k,v in kernels.items()};q=qualify(stat,kmeta,diagnostics,scores);result=dict(status='SOURCE_FILTER_COMPLETE',scores=scores,source_statistics=stat,kernels=kmeta,diagnostics=diagnostics,qualification=q,actual_NN_packets=126002,finite_filter_packets=120960,circular_oracle_packets=120960,actual_optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,As_generations=0,NoExplicitCFO=True,default_head_only=True,model_state_unchanged=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,elapsed_seconds=time.perf_counter()-start)
  d.write(out/'result.json',result);d.write(out/'completion.json',dict(status='SOURCE_FILTER_COMPLETE',decision=q['decision'],source_decisions=126000,views=25,actual_NN_packets=126002,optimizer_updates=0,target_access=False,automatic_training=False));print('COMPLETE '+str(q['conditions'])+' decision='+q['decision'],flush=True)
 except Exception as error:d.write(out/'failure.json',dict(status='FAILED',error=repr(error),no_automatic_retry=True));raise
