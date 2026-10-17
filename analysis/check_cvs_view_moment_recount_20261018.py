"""Synthetic complete moment-log recurrence and fixed source threshold checks;0 IQ/CP/NN."""
import copy,gzip,json,math,struct
from pathlib import Path
from analysis.cvs_view_moment_recount_20261018 import moment_records
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def run(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=False);negatives=0;cases=[]
 def bad(f):
  nonlocal negatives
  try:f()
  except (ValueError,KeyError,TypeError):negatives+=1
  else:raise AssertionError('Bad scalar source evidence accepted')
 names=['id_backbone.encoder.core.id_backbone.sinc.low_coordinate','id_backbone.encoder.core.id_backbone.sinc.band_coordinate','id_backbone.encoder.core.id_backbone.cls_head.weight']
 for mode in ('NOAS_MEASURE','AS_VIEW'):
  kind='VIEW_SECOND' if mode.endswith('_VIEW') else 'MEASURE';state={n:(0.,0.) for n in names};rows=[]
  for step in range(1,2221):
   norm=2+step/700;fac=min(1.,5/(norm+1e-6));wit=[]
   for i,n in enumerate(names):
    oldm,oldv=state[n];g=f32(math.sin(step*.03+i)*(.01+i));variance=(.0001+i*.001)*(1+math.cos(step*.07)) *fac**2;m=f32(.9*oldm+.1*g);v=f32(.999*oldv+.001*(g*g+(variance if kind=='VIEW_SECOND' else 0.)));state[n]=(m,v);wit.append(dict(parameter=n,gradient=g,view_variance_postclip=variance,old_first=oldm,old_second=oldv,beta1=.9,beta2=.999,actual_first=m,actual_second=v,optimizer_state_step=step))
   rows.append(dict(step=step,cumulative_step=4440+step,mode=kind,extra_VJP_calls=2,physical_global_norm=norm,clip_factor=fac,weighted_view_geometry=dict(ordinary_norm=.3,satellite_norm=.4,dot=.05,cosine=.05/(.3*.4)),variance_physical_trace_postclip=.15*fac**2,witness=wit,capture_host_enqueue_seconds=0.))
  cost=dict(moment_mode=kind,extra_supervised_VJP_calls=4440,view_capture_calls=2220,view_second_moment_updates=2220 if kind=='VIEW_SECOND' else 0,L_forward_packets=1136640);S=output/mode;S.mkdir();file=S/'view_moment_steps.jsonl.gz'
  def write(rr):
   with gzip.open(file,'wt',encoding='utf8') as f:
    for r in rr:f.write(json.dumps(r)+'\n')
  write(rows);value=moment_records(S,mode,cost);cases.append(dict(mode=kind,records=value['records'],coordinate_witnesses=value['coordinate_witnesses'],max_float32_error_units=value['max_float32_error_units']))
  corrupt=copy.deepcopy(rows);corrupt[10]['witness'][0]['actual_second']+=.01;write(corrupt);bad(lambda:moment_records(S,mode,cost))
  corrupt=copy.deepcopy(rows);corrupt[20]['extra_VJP_calls']=0;write(corrupt);bad(lambda:moment_records(S,mode,cost))
  write(rows[:-1]);bad(lambda:moment_records(S,mode,cost))
 # Deployed qualification against an independent integer-count threshold fixture.
 from experiments.cvs_view_moment30 import qualify as q
 modes=('NOAS_MEASURE','AS_MEASURE','NOAS_VIEW','AS_VIEW');views=q.d.VIEWS;data={}
 for mode in modes:
  rr=[]
  for view in views:
   for draw in range(1 if view=='clean' else 2):
    correct=1700 if view=='clean' else 1000 if view=='practical_low_urban' else 1200
    if mode=='AS_VIEW' and view=='practical_low_urban':correct+=9
    rr.append(dict(view=view,draw=draw,correct=correct))
  data[mode]=dict(records=rr)
 def refresh():
  for mode in modes:
   metrics={}
   for view in views:
    group=[r for r in data[mode]['records'] if r['view']==view];metrics[view]=dict(accuracy=sum(r['correct'] for r in group)/(1800*len(group)))
   data[mode]['metrics']=metrics
 oldread,oldwrite=q.d.read,q.d.write;written=[]
 q.d.read=lambda path:data[Path(path).parent.parent.name.split('-s')[0]];q.d.write=lambda path,value:written.append(value)
 original=copy.deepcopy(data);qualification_cases=0
 try:
  refresh();assert q.matrix()['source_pass'];qualification_cases+=1
  for mode,view,draw,delta,key in [('AS_VIEW','clean',0,-1,'clean'),('AS_VIEW','practical_mid_urban',1,-1,'midurban'),('AS_VIEW','practical_low_urban',0,-1,'lowurban'),('AS_VIEW','practical_high',1,-5,'suburban'),('NOAS_VIEW','clean',0,-5,'NOAS_clean')]:
   data=copy.deepcopy(original);next(r for r in data[mode]['records'] if r['view']==view and r['draw']==draw)['correct']+=delta;refresh();value=q.matrix();assert not value['source_pass'] and not value['conditions'][key];qualification_cases+=1
  data=copy.deepcopy(original);next(r for r in data['AS_VIEW']['records'] if r['view']=='practical_high' and r['draw']==0)['correct']-=4;next(r for r in data['NOAS_VIEW']['records'] if r['view']=='clean')['correct']-=4;refresh();assert q.matrix()['source_pass'];qualification_cases+=1
 finally:q.d.read=oldread;q.d.write=oldwrite
 return dict(status='PASS',complete_scalar_fixture_records=4440,coordinate_witnesses=13320,moment_negative_cases=negatives,source_threshold_cases=qualification_cases,scalar_cases=cases,IQ=0,CP=0,NN=0,optimizer_updates=0,U_truth_read=False,target_access=False)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.out)))
