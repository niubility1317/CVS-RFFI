import argparse,json
from pathlib import Path
from copy import deepcopy
from . import design as d
from experiments.cvs_ensemble_final30.contracts import verified_component_contract
def run():
 reference=dict(role_ids={'L_s':['synthetic-L'],'V':['synthetic-V']});runtime=dict(reference,native_role_comparison='EXACT_MATCH')
 if verified_component_contract([runtime,runtime],reference)!=runtime:raise AssertionError('Use actual runtime evidence')
 negative=0
 for bad in (dict(reference),dict(runtime,role_ids={'L_s':['wrong-L'],'V':['synthetic-V']})):
  try:verified_component_contract([bad,runtime],reference)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid source must remain blocked')
 for key,value in [('run_id','wrong'),('recovery',dict(new_model_updates=1)),('source_config',dict(invalid=True))]:
  c=deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Recovery recipe/data/reference mutation')
 return dict(status='PASS',negative_cases=negative,no_training=True,source_permission_checks_preserved=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))
