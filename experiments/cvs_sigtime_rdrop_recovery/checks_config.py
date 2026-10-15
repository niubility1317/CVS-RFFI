import copy,json
from . import design as d
def main():
 original={r['row_id']:d.original.config(r) for r in d.rows()};request=d.request();d.validate(request)
 assert len(request['rows'])==4 and {r['row_id'] for r in request['rows']}==set(d.REMAINING)
 for row in d.rows():
  c=d.config(row);old=original[row['row_id']]
  assert {k:v for k,v in c.items() if k not in ('run_id','output_root','launch_owner')}=={k:v for k,v in old.items() if k not in ('run_id','output_root','launch_owner')}
  assert c['run_id']==d.RUN and c['launch_owner']==d.OWNER and c['output_root']==(d.BASE/row['row_id']/'source').as_posix() and c['checkpoint_sources']==[]
  a=d.original.make_args(c,'cpu')
  assert (a.epochs,a.label_epochs,a.pseudo_epochs)==(50,33,17) and a.label_smoothing==.01 and a.use_aug is True and a.batch_size==128 and a.use_ema_teacher is True
  assert not a.baseline_ckpt and not a.teacher_ckpt and not a.use_concat_sat_channel_aug
 negative=0
 for kind in ('extra_row','missing_row','wrong_seed','wrong_epochs','changed_loss','changed_output','changed_head','checkpoint'):
  bad=copy.deepcopy(request)
  if kind=='extra_row':bad['rows'].append(copy.deepcopy(bad['rows'][0]))
  elif kind=='missing_row':bad['rows'].pop()
  elif kind=='wrong_seed':bad['rows'][0]['model_seed']=2026092701
  elif kind=='wrong_epochs':bad['rows'][0]['epochs']=20
  elif kind=='changed_loss':bad['rows'][0]['training_mechanism']['alpha']=.5
  elif kind=='changed_output':bad['rows'][0]['output_root']=d.ORIGINAL_BASE.as_posix()
  elif kind=='changed_head':bad['rows'][0]['separate_head_training']=True
  else:bad['rows'][0]['checkpoint_sources']=['old.pth']
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Altered recovery contract accepted '+kind)
 assert d.totals()['optimizer_steps']==44400 and d.totals()['epochs']==200
 print(json.dumps(dict(status='PASS',negative_cases=negative,exact_unstarted_rows=4,scientific_recipe_fields_unchanged=True,metadata_only_changes=['run_id','output_root','launch_owner'],checkpoint_loads=0,IQ_reads=0,optimizer_steps=0)))
if __name__=='__main__':main()
