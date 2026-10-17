"""Existing full-source default head;conditional gradient-noise diagnostic only."""
import os,sys,time
from collections import Counter
import numpy as np
import torch
from . import design as d
from .coupled import three,variance
def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES') in ('',None):raise ValueError('One reserved shared GPU required')
 torch.set_num_threads(2);out=d.resolve(d.BASE);started=time.perf_counter();NN=2;backwards=0;physical_calls=0
 from experiments.cvs_fullsource_concat20 import design as ref,proof,model as api
 from experiments.cvs_sigtime_rdrop_legacy.model import terms
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=ref.config(next(r for r in ref.rows() if r['row_id']==d.REFERENCE_ROW));p=proof.source(c);contract=d.read(d.resolve(d.SOURCE));trained=d.read(p/'source_contract.json')
 if any(trained.get(k)!=v for k,v in contract.items()) or trained.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Matching fullsource parent required')
 prep=d.resolve(d.PREPARATION);ready=d.read(prep/'preparation_complete.json');packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False)
 if ready['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or ready['target_access'] or any(any(a.get(k)!=v for k,v in contract.items()) for a in (packed['contract'],ready['source_contract'])):raise ValueError('Existing unchanged source handles differ')
 L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
 y,rx,day=(L[k].tolist() for k in ('y','rx','day'));ids=L['ids']
 if set(ids)!=set(contract['role_ids']['L_s']) or len(ids)!=6300 or L['x'].shape!=(6300,2,256) or Counter(zip(rx,day,y))!=Counter({(r,z,t):70 for r in d.FIT for z in (1,2,3) for t in range(6)}):raise ValueError('Original full source L6300/all5RX required')
 batches=[]
 for z in (1,2,3):
  for k in range(2):batches.append([sorted((i for i in range(6300) if (y[i],rx[i],day[i])==(t,r,z)),key=lambda i:ids[i])[k] for t in range(6) for r in d.FIT])
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if (cp['epoch'],cp['run_id'],cp['candidate_id'],cp['a1_ema_successful_updates'],cp['checkpoint_selection'])!=(20,d.REFERENCE_RUN,d.REFERENCE_ROW,4440,'final_only'):raise ValueError('Fixed scratch20 parent identity differs')
 m=api.build(c,'cuda:0');m.load_state_dict(cp['model'],strict=True);del cp;m.eval();head=m.id_backbone.encoder.core.id_backbone.cls_head
 if head.weight.shape!=(6,160) or head.scale!=30. or m.id_backbone.encoder.sequence_pool.modes!=('SIG_TIME',)*3 or m.id_backbone.encoder.response.contract()['explicit_CFO_identity_input'] is not False:raise ValueError('Original F/defaulthead/NoExplicitCFO required')
 import leo_practical.channel as ch
 import leo_practical.residual as residual
 before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 parameters=[(k,v) for k,v in m.named_parameters() if v.requires_grad];W=next(i for i,(k,v) in enumerate(parameters) if v is head.weight);offset=sum(v.numel() for _,v in parameters[:W]);size=head.weight.numel();groups=[]
 d.write(out/'resolved_config.json',dict(config=d.config(),checkpoint=dict(source=d.config()['model_source'],epoch=20,selection='final_only',parent_updates=4440,provenance='VERIFIED_EXACT_SOURCE_ROLES'),source_roles={k:len(v) for k,v in trained['role_ids'].items()},selected_source_ids=[[ids[i] for i in b] for b in batches],modules=dict(model=api.__file__,channel=ch.__file__,residual=residual.__file__),numerical_policy=ref.FULL_FP32_POLICY,gradient_parameters=sum(v.numel() for _,v in parameters),all_model_parameters=sum(v.numel() for v in m.parameters()),head_parameter=parameters[W][0],updates=0,target_access=False))
 with numerical_context(ref.FULL_FP32_POLICY):
  with torch.no_grad():sm=m.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
  if sm['tx_logits'].shape!=(2,6) or sm['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(sm['tx_logits']).all()):raise ValueError('Real-checkpoint noquery smoke failed')
  d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,IQ=0,target_access=False));print('SMOKE PASS;continue source gradient diagnostic',flush=True);m.train()
  def grad(x,labels,seed):
   with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(seed);a=m.forward_identity_only(x)['tx_logits'];b=m.forward_identity_only(x)['tx_logits'];loss,*_=terms(a,b,labels,1.);g=torch.autograd.grad(loss,[v for _,v in parameters],allow_unused=True)
   if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite existing RDrop loss')
   v=torch.cat([(torch.zeros_like(p) if a is None else a).detach().reshape(-1).cpu() for a,(_,p) in zip(g,parameters)])
   if not bool(torch.isfinite(v).all()):raise ValueError('Nonfinite gradient')
   return v,float(loss.detach())
  for scene in d.SCENES:
   cfg=ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   for bi,indices in enumerate(batches):
    vectors=[];losses=[];labels=torch.tensor([y[i] for i in indices],device='cuda:0');metadata=Counter()
    for rep in range(d.REPEATS):
     arrays=[[],[],[]]
     for i in indices:
      raw=L['x'][i].numpy();x,_=ch.normalize_input(raw[0].astype(float)+1j*raw[1].astype(float));seed=ch.stable_seed(d.PROBE_SEED,scene,ids[i]);session=f'rx{rx[i]}:day{day[i]}:eq1';hw=ch.receiver_for_session(cfg,d.RECEIVER_SEED,session)
      e1=np.random.default_rng(ch.stable_seed(d.NOISE_SEED,'first',scene,ids[i],rep)).normal(size=(256,2));e2=np.random.default_rng(ch.stable_seed(d.NOISE_SEED,'second',scene,ids[i],rep)).normal(size=(256,2));views=three(ch,residual,cfg,hw,x,seed,e1,e2);physical_calls+=3
      for j,(value,meta) in enumerate(views):arrays[j].append(np.stack((value.real,value.imag)))
      metadata[(views[0][1]['state_end'],views[0][1]['receiver_lock_status'])]+=1
     for j,array in enumerate(arrays):
      x=torch.from_numpy(np.asarray(array,dtype=np.float32)).to('cuda:0');seed=ch.stable_seed(d.DROPOUT_SEED,scene,bi,rep,0 if j==0 else 1);g,loss=grad(x,labels,seed);NN+=60;backwards+=1;vectors.append(g);losses.append(loss)
     d.write(out/'progress.json',dict(scene=scene,batch=bi,repeat=rep+1,NN=NN,backward_calls=backwards,channel_calls=physical_calls,updates=0,target_access=False))
    A=torch.stack(vectors).double();gram=(A@A.T).tolist();B=A[:,offset:offset+size];hgram=(B@B.T).tolist();vf=variance(gram,d.REPEATS);vh=variance(hgram,d.REPEATS);groups.append(dict(scene=scene,batch=bi,count=30,full_gram=gram,head_gram=hgram,losses=losses,full=vf,head=vh,state_lock_counts=[dict(state=k[0],lock=k[1],count=n) for k,n in sorted(metadata.items())]));del A,B,vectors;print('GROUP_COMPLETE '+scene+' '+str(bi),flush=True)
 if NN!=d.config()['expected_NN'] or backwards!=432 or physical_calls!=12960 or any(not torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items()):raise ValueError('Frozen model/budget changed')
 summary={}
 for scene in [*d.SCENES,'all']:
  rows=[x for x in groups if scene=='all' or x['scene']==scene];summary[scene]={}
  for block in ('full','head'):
   nom=sum(x[block]['independent'] for x in rows);anti=sum(x[block]['antithetic'] for x in rows);summary[scene][block]=dict(independent=nom/len(rows),antithetic=anti/len(rows),ratio=anti/nom)
 passed=summary['all']['full']['ratio']<=.8 and summary['all']['head']['ratio']<=.8 and all(summary[s]['full']['ratio']<1. for s in d.SCENES[1:]);value=dict(status='ANALYZED_SOURCE_ESTIMATOR_ONLY',config=d.config(),source_criteria_passed=passed,decision='SOURCE_VARIANCE_SUPPORTED_NOT_IDENTITY_GAIN' if passed else 'FAIL_FIXED_ANTITHETIC_COUPLING_CLOSED',summary=summary,groups=groups,cost=dict(NN=NN,backward_calls=backwards,physical_channel_calls=physical_calls,CP=1,cache=1,source_packets_selected=180,optimizer_updates=0,teacher_NN=0,head_fit_updates=0,wall_seconds=time.perf_counter()-started),gradient_parameters=sum(v.numel() for _,v in parameters),all_model_parameters=sum(v.numel() for v in m.parameters()),head_parameter=parameters[W][0],model_state_unchanged=True,source_parent_roles={k:len(v) for k,v in trained['role_ids'].items()},source_RX_holdout=None,target_access=False,Ag_gain_proved=False,As_identity_gain_proved=False,joint_goal_complete=False)
 d.write(out/'result.json',value);d.write(out/'completion.json',{k:v for k,v in value.items() if k!='groups'});print('SOURCE_ESTIMATOR_COMPLETE '+value['decision'],flush=True)
