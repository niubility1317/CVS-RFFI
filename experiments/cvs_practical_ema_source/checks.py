"""Source-only protocol negatives; synthetic arrays only, no Torch/data/CP."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import check_roles,check_ids,check_state_pair,check_panel,check_panel_membership,check_observation_schema,decision
from .metrics import metric,paired,summarize

def check_panel_join():
 """Actual upstream formats: opaque SHA24 IDs plus separate native five-tuples."""
 import hashlib
 n=27000;tx=np.arange(n)%6;rx=np.asarray(d.RX)[(np.arange(n)//6)%5];day=np.asarray(d.DAY)[(np.arange(n)//30)%3];sig=np.arange(n)//90
 full=dict(ids=np.asarray([hashlib.sha256(f'{t}/{r}/{v}/1/{s}'.encode()).hexdigest()[:24] for t,r,v,s in zip(tx,rx,day,sig)]),TX=tx,RX=rx,day=day,
  physical_ids=np.asarray([(1,int(r),int(v),int(s),i) for i,(r,v,s) in enumerate(zip(rx,day,sig))]))
 chosen=np.arange(1799,-1,-1);panel={k:full[k][chosen].copy() for k in ('ids','TX','RX','day')}
 check_panel_membership(panel,full)
 assert panel['ids'].shape==(1800,) and full['physical_ids'].shape==(27000,5)
 negatives=0
 def bad(p):
  nonlocal negatives
  try:check_panel_membership(p,full)
  except ValueError:negatives+=1
  else:raise AssertionError('Wrong physical-ID representation or alignment accepted')
 p={k:v.copy() for k,v in panel.items()};p['ids']=full['physical_ids'][chosen];bad(p)
 p={k:v.copy() for k,v in panel.items()};p['ids'][0]='f'*24;bad(p)
 p={k:v.copy() for k,v in panel.items()};p['ids'][0]=p['ids'][1];bad(p)
 for key in ('TX','RX','day'):
  p={k:v.copy() for k,v in panel.items()};p[key][0]+=1;bad(p)
 return dict(status='PASS',canonical_opaque_join=True,native_tuple_kept_separate=True,reordered_panel_preserved=True,targeted_negatives=negatives,CP=0,IQ=0,NN=0)

def check_observation_input():
 """Actual eleven-condition header and index0 full; no IQ payload allocation."""
 import ast,io
 meta=dict(status='SOURCE_OBSERVATIONS_COMPLETE',draws=2,unique_V=1800,scenes=list(d.SCENES),conditions=list(d.OBS_CONDITIONS),target_access=False,training_updates=0)
 stream=io.BytesIO();np.lib.format.write_array_header_1_0(stream,dict(shape=d.OBS_SHAPE,fortran_order=False,descr=np.dtype('float32').str));stream.seek(0)
 assert np.lib.format.read_magic(stream)==(1,0)
 shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
 index=check_observation_schema(meta,shape,dtype,order)
 view=np.broadcast_to(np.arange(11).reshape(1,1,11,1,1,1),d.OBS_SHAPE)
 assert index==0 and np.all(view[1,2,index,:90]==0) and np.all(view[1,2,1,:90]==1)
 negative=0
 def bad(m,s=shape,t=dtype,f=order):
  nonlocal negative
  try:check_observation_schema(m,s,t,f)
  except ValueError:negative+=1
  else:raise AssertionError('Wrong original source cache schema accepted')
 bad(meta,(2,3,1,1800,2,256));bad(meta,(2,3,11,1800,2,255));bad(meta,t='float64');bad(meta,f=True)
 for key,value in [('conditions',list(d.OBS_CONDITIONS[1:])+['full']),('scenes',list(reversed(d.SCENES))),('status','INCOMPLETE'),('target_access',True)]:
  wrong=dict(meta);wrong[key]=value;bad(wrong)
 tree=ast.parse(Path(__file__).with_name('evaluate.py').read_text(encoding='utf8'))
 body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 calls={name:min(n.lineno for n in ast.walk(body) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name) for name in ('prepare_inputs','load_models','full_v')}
 assert calls['prepare_inputs']<calls['load_models']<calls['full_v']
 return dict(status='PASS',actual_header_shape=list(shape),full_index=index,condition_axes=11,early_metadata_before_CP_NN=True,targeted_negatives=negative,CP=0,IQ=0,NN=0)

def run():
 negatives=0
 def bad(f):
  nonlocal negatives
  try:f()
  except (ValueError,KeyError,TypeError):negatives+=1
  else:raise AssertionError('Invalid source-only scope accepted')
 counts=dict(L_s=6300,U_s=56700,V=27000)
 roles={r:[r+'-'+str(i) for i in range(n)] for r,n in counts.items()}
 check_roles(roles,roles,{},counts)
 bad(lambda:check_roles(roles,roles,{'target':1},counts))
 for r in roles:
  z=copy.deepcopy(roles);z[r][0]='another'
  bad(lambda z=z:check_roles(z,roles,{},counts))
  z=copy.deepcopy(roles);z[r][0]=z[r][1]
  bad(lambda z=z:check_roles(z,z,{},counts))
 bad(lambda:check_roles(roles,roles,{},dict(counts,V=1800)))
 template={str(i):np.zeros((2,),np.float32) for i in range(205)};template[d.HEAD]=np.zeros((6,160),np.float32)
 finite=lambda v:bool(np.isfinite(v).all())
 check_state_pair(template,template,template,finite)
 for stateindex in (0,1):
  for change in ('keys','shape','dtype','finite','head'):
   z=copy.deepcopy(template)
   if change=='keys':z.pop('0')
   elif change=='shape':z['0']=np.zeros((3,),np.float32)
   elif change=='dtype':z['0']=np.zeros((2,),np.float64)
   elif change=='finite':z['0'][0]=np.nan
   else:z[d.HEAD]=np.zeros((6,159),np.float32)
   a,b=(z,template) if stateindex==0 else (template,z)
   bad(lambda a=a,b=b:check_state_pair(a,b,template,finite))
 ids=list(range(30));rx=list(d.RX)*6;days=list(d.DAY)*10;eq=[1]*30
 check_ids(ids,rx,days,eq,ids,30)
 for a,b,c,e,n in [(ids[:-1],rx,days,eq,30),([0]*30,rx,days,eq,30),(ids,[0]*30,days,eq,30),(ids,rx,[0]*30,eq,30),(ids,rx,days,[0]*30,30)]:bad(lambda a=a,b=b,c=c,e=e,n=n:check_ids(a,b,c,e,ids,n))
 panel=dict(ids=np.arange(9000).reshape(1800,5),TX=np.arange(1800)%6,RX=np.tile(d.RX,360),day=np.tile(d.DAY,600),clean=np.zeros((1800,6)),satellite=np.zeros((2,3,1800,6)))
 check_panel(panel,panel)
 for key in ('ids','TX','RX','day'):
  z={k:v.copy() for k,v in panel.items()};z[key].flat[0]+=1;bad(lambda z=z:check_panel(z,panel))
 for key in ('clean','satellite'):
  z={k:v.copy() for k,v in panel.items()};z[key].flat[0]=np.inf;bad(lambda z=z:check_panel(z,panel))
 with tempfile.TemporaryDirectory() as directory:
  allowed=Path(directory)/'source'
  assert d.assert_source_path(allowed/'final_ssdg.pth',[allowed])==allowed.resolve()/'final_ssdg.pth'
  for name in ('target','prediction','scoring','scores','formal','truth','analysis_auto_r01'):
   bad(lambda name=name:d.assert_source_path(allowed/name/'data.json',[allowed]))
  bad(lambda:d.assert_source_path(Path(directory)/'outside',[allowed]))
  original_root=d.ROOT;d.ROOT=Path(directory)
  try:
   runtime=dict(run_id=d.RUN,project=directory,parent_release=str(Path(directory)/'releases'/d.PARENT_RELEASE),parent_commit=d.PARENT_COMMIT)
   d.write(d.ROOT/'runtime.json',runtime);d.runtime()
   for key,value in [('run_id','other'),('parent_commit','unknown'),('parent_release',str(Path(directory)/'unverified')),('project','relative')]:
    z=dict(runtime);z[key]=value;d.write(d.ROOT/'runtime.json',z);bad(d.runtime)
  finally:d.ROOT=original_root
 y=np.arange(1800)%6;correct=np.eye(6,dtype=np.float32)[y];wrong=np.eye(6,dtype=np.float32)[(y+1)%6]
 assert metric(y,y)['accuracy']==1 and metric(y,y)['macro_f1']==1
 assert paired(y,wrong.argmax(1),y)==dict(count=1800,repairs=1800,new_errors=0,net_repairs=1800,accuracy_delta_pp=100.)
 f={label:dict(accuracy=1.,hits=27000) for label in d.LABELS};p={label:{s:dict(hits=3600) for s in d.SCENES} for label in d.LABELS}
 assert not decision(f,p)['passed']
 p['AS_EMA'][d.SCENES[1]]['hits']+=1;assert decision(f,p)['passed']
 f['AS_EMA']['hits']-=1;assert not decision(f,p)['passed']
 f['AS_EMA']['hits']+=1;f['AS_EMA']['accuracy']-=.00101;assert not decision(f,p)['passed']
 with tempfile.TemporaryDirectory() as directory:
  full=dict(TX=y,RX=panel['RX'],day=panel['day'],**{label:correct for label in d.LABELS})
  pv=dict(TX=y,RX=panel['RX'],day=panel['day'],**{label:dict(clean=correct,satellite=np.tile(correct,(2,3,1,1))) for label in d.LABELS})
  result=summarize(full,pv,Path(directory))
  assert result['fullV']['AS_EMA']['worst_RX_accuracy']==1 and result['paired_fullV']['AS_EMA_minus_AS_STUDENT']['net_repairs']==0
 return dict(status='PASS',protocol_negatives=negatives,panel_join=check_panel_join(),observation_input=check_observation_input(),synthetic_metric_recount=True,actual_CVS_NN=0,actual_IQ_reads=0,actual_checkpoint_reads=0,updates=0,target_access=False)

if __name__=='__main__':print(json.dumps(run()))
