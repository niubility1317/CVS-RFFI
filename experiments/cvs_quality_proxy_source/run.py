import gzip,json,os,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from . import design as d
from .physics import measure
from .metrics import group_summary,criteria

def engine():
 sys.path.insert(0,d.ENGINE)
 import leo_practical.channel as ch
 import leo_practical.residual as residual
 from leo_practical.equalization import linear_filter
 return ch,residual,linear_filter

def run():
 c=d.config(d.rows()[0]);d.validate(c)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source physics diagnostic')
 if d.BASE.exists():raise FileExistsError('Reconcile immutable output;no duplicate launch')
 d.BASE.mkdir();out=Path(c['output_root']);out.mkdir();started=time.time();d.write(d.BASE/'execution.json',dict(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,GPU='',actual_code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),read_at=started));d.write(out/'resolved_config.json',c)
 try:
  ch,residual,filt=engine();configs={s:ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False) for s in d.SCENES};d.write(out/'actual_engine.json',dict(channel=ch.__file__,residual=residual.__file__,configs={s:asdict(v) for s,v in configs.items()},zero_model=True))
  # First-step real-native no-query smoke. No model/checkpoint machinery is consumed.
  sx=np.exp(2j*np.pi*.123*np.arange(256));sm=[]
  for j in range(2):
   cfg=configs[d.PRIMARY[j]];rx=ch.receiver_for_session(cfg,d.RECEIVER_SEED,'synthetic-smoke')
   sm.append(measure(ch,residual,filt,cfg,sx,d.SEED+j,rx))
  d.write(d.BASE/'native_noquery_smoke.json',dict(status='PASS',synthetic_packets=2,native_process_calls=sum(z['native_process_calls'] for z in sm),source_IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,target_access=False));print('NATIVE_SMOKE PASS 2synthetic/10physicalcalls/0NN',flush=True)
  from .source import source_observations
  x,tx,rx,day,ids,roles=source_observations()
  if len(x)!=5040 or len(set(ids))!=5040 or x.shape!=(5040,2,256) or {int(v) for v in tx}!=set(d.TX):raise ValueError('Exact fitL5040 required')
  from collections import Counter
  if Counter((int(t),int(r),int(dd)) for t,r,dd in zip(tx,rx,day))!=Counter({(t,r,dd):70 for t in d.TX for r in d.FIT for dd in d.DAYS}):raise ValueError('Complete72sourcecells required')
  d.write(out/'data_roles.json',dict(source_contract=d.SOURCE,reference_contract=d.REFERENCE+'/source_contract.json',effective_L_IDs=ids,roles=roles,source_RX=list(d.FIT),source_TX=list(d.TX),source_days=list(d.DAYS),U_values_used=False,V_values_used=False,held_RX1_values_used=False,target_access=False))
  source=[ch.normalize_input(z[0]+1j*z[1])[0] for z in x];groups={};cells=[];receivers={};calls=0;count=0;minimum=256
  with gzip.open(out/'quality_records.jsonl.gz','xt',encoding='utf8',compresslevel=3) as stream:
   for draw in range(2):
    for scene,cfg in configs.items():
     records=[]
     for i,z in enumerate(source):
      session='rx%d:day%d:eq1'%(rx[i],day[i]);receiver=ch.receiver_for_session(cfg,d.RECEIVER_SEED,session);rmeta=asdict(receiver)
      if receivers.setdefault(session,rmeta)!=rmeta:raise ValueError('RX hardware must be stable acrossscene/draw/TX')
      seed=ch.stable_seed(d.SEED,'source_fit_quality/draw'+str(draw),scene,str(ids[i]));record=measure(ch,residual,filt,cfg,z,seed,receiver);record.update(draw=draw,scene=scene,physical_id=ids[i],physical_seed_key=str(ids[i]),TX=int(tx[i]),RX=int(rx[i]),day=int(day[i]),seed=int(seed),session=session)
      stream.write(json.dumps(record,allow_nan=False)+'\n');records.append(record);calls+=record['native_process_calls'];count+=1;minimum=min(minimum,record['interior_count'])
     key=str(draw)+'/'+scene;groups[key]=group_summary(records)
     for t in d.TX:
      for r in d.FIT:
       for dd in d.DAYS:cells.append(dict(draw=draw,scene=scene,TX=t,RX=r,day=dd,**group_summary([z for z in records if (z['TX'],z['RX'],z['day'])==(t,r,dd)])))
     stream.flush();d.write(d.BASE/'progress.json',dict(complete_scene_draws=len(groups),physical_records=count,native_process_calls=calls,latest=key,source_only=True,target_access=False));print('SCENE_COMPLETE '+key+' records='+str(count),flush=True)
  checks=criteria(groups,minimum);summary=dict(status='PASS_SOURCE_QUALITY_SWITCH_PREMISE_ONLY' if all(checks.values()) else 'FAIL_FIXED_QUALITY_PROXY_PREMISE_CLOSED',checks=checks,groups=groups,source_cells=cells,roles=roles,physical_records=count,native_process_calls=calls,synthetic_native_process_calls=10,unique_source_L=5040,source_physical_counts=dict(L=5040,U=0,V=0,held_RX1=0,target=0),receiver_sessions=receivers,new_models=0,checkpoint_loads=0,model_forwards=0,optimizer_updates=0,teacher_forwards=0,head_fit_updates=0,target_access=False,formal_test_changed=False,automatic_training=False,goal_complete=False,wall_seconds=time.time()-started)
  if count!=60480 or calls!=302400 or len(receivers)!=12:raise ValueError('Complete registered source physics matrix required')
  d.write(out/'summary.json',summary);d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',scientific_status=summary['status'],source_only=True,target_access=False,physical_records=count,native_process_calls=calls,checkpoint_loads=0,model_forwards=0,optimizer_updates=0,automatic_training=False,goal_complete=False,wall_seconds=summary['wall_seconds']));print('COMPLETE '+summary['status'],flush=True)
 except BaseException as e:d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),preserve_outputs=True,no_automatic_retry=True));raise
