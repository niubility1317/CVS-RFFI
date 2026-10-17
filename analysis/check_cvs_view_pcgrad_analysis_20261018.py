"""Deterministic scalar cases distinguish theoretical correction from actual updates."""
import importlib.util,tempfile,gzip,json,math,copy
from pathlib import Path
P=Path(__file__).with_name('cvs_view_pcgrad_source_logs_20261018.py');spec=importlib.util.spec_from_file_location('source_reader',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def run():
 negative=0
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)
  def write(v):
   with gzip.open(p/'view_gradient_steps.jsonl.gz','wt',encoding='utf8') as f:f.write(json.dumps(v)+'\n')
  row=dict(step=1,cumulative_step=4441,mode='PROJECT',extra_VJP_calls=2,applied=True,applied_coefficient=2.,dot=-2.,ordinary_norm=1.,satellite_norm=math.sqrt(13.),cosine=-2./math.sqrt(13.),coefficient=2.,conflict=True,projected_satellite_dot=0.,projection_delta_norm=2.,capture_host_enqueue_seconds=.01)
  write(row);a=m.gradient_logs(p,'PROJECT',False)
  assert a['actual_projection_updates']==1 and a['extra_VJP_calls']==2 and a['extra_L_backward_packet_derivatives']==512
  assert abs(a['two_supervised_view_gradient_RMS_norm_before']-math.sqrt(10))<1e-12 and abs(a['two_supervised_view_gradient_RMS_norm_after']-math.sqrt(10))<1e-12
  assert a['effective_ordinary_gradient_multiplier']['mean']==3.
  measured=dict(row,mode='MEASURE',applied=False,applied_coefficient=0.);write(measured);b=m.gradient_logs(p,'MEASURE',False)
  assert b['actual_projection_updates']==0 and b['actual_projection_L2_RMS_fraction_of_As_gradient']==0. and b['effective_ordinary_gradient_multiplier']['mean']==1.
  original=dict(step=1,cumulative_step=4441,mode='ORIGINAL',extra_VJP_calls=0,applied=False,coefficient=0.);write(original);c=m.gradient_logs(p,'ORIGINAL',False);assert c['actual_projection_updates']==0 and c['extra_VJP_calls']==0 and c['two_supervised_view_gradient_RMS_norm_before'] is None
  for key,value in [('applied',False),('extra_VJP_calls',0),('coefficient',1.),('cosine',.5),('mode','MEASURE'),('cumulative_step',1),('satellite_norm',-1.)]:
   write(dict(row,**{key:value}))
   try:m.gradient_logs(p,'PROJECT',False)
   except ValueError:negative+=1
   else:raise AssertionError('Wrong scalar/counter accepted: '+key)
  write(row)
  try:m.gradient_logs(p,'PROJECT',True)
  except ValueError:negative+=1
  else:raise AssertionError('One record accepted as complete2220')
 return dict(status='PASS',closed_form_modes=3,negative_cases=negative,IQ=0,CP=0,NN=0,updates=0,raw_truth=0)
if __name__=='__main__':print(json.dumps(run()))
