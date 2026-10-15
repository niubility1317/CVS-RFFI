"""Read the completed fixed Sinc seed extension; never launch or train a model.

The remote Python is supplied by the operator's existing runtime configuration.
The immutable release and output identity come from the registered design.
"""
import argparse
import json
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh
from experiments.cvs_sinc_units_seeds import design as local_design

PAYLOAD = '''import sys,json,torch
from pathlib import Path
torch.set_num_threads(2)
release=Path(RELEASE_PATH)
sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_sinc_units_seeds import design as d,model as api
from experiments.cvs_sigtime_pair50.audit import run as full_audit
from experiments.cvs_sinc_units_seeds.evidence import recount
done=d.read(d.BASE/'completion.json')
if (done.get('status'),done.get('models'),done.get('prediction_count'))!=('EXPERIMENT_COMPLETE',6,1008000):
 raise ValueError('The complete six-row seed extension is required; no relaunch or partial collection')
value=dict(status='COLLECTED',run_id=d.RUN,completion=done,
 analysis=d.read(d.BASE/'analysis.json'),combined_analysis=d.read(d.BASE/'combined_three_seed_analysis.json'),
 source_qualification=d.require_source(),source_features=d.read(d.BASE/'source_features.json'),
 saved_source_observations=d.read(d.BASE/'source_observations.json'))
value['audit']=full_audit(design=d,build_fn=api.build)
value['source_updates']=[recount(row) for row in d.training_rows()]
value['source_curves']=[dict(row_id=row['row_id'],arm=row['arm'],baseline_reused=row['source_reused'],records=row['source_epoch_curve']) for row in value['audit']['audit']['rows']]
print(json.dumps(value,allow_nan=False))
'''

def collect(output, remote_python):
 output=Path(output)
 if output.exists():
  raise FileExistsError('Preserve the existing completed collection: '+str(output))
 release=Path(local_design.PROJECT)/'releases'/local_design.RELEASE
 payload=PAYLOAD.replace('RELEASE_PATH',repr(release.as_posix()))
 wrapper=("import subprocess,os\nresult=subprocess.run(["+repr(remote_python)+",'-'],input="+repr(payload)+
  ",text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))\n"
  "if result.returncode: raise RuntimeError(result.stderr)\nprint(result.stdout)\n")
 value=json.loads(ssh(wrapper))
 if value.get('status')!='COLLECTED' or value['audit'].get('status')!='VERIFIED':
  raise ValueError('The completed collection or full independent recount failed')
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x',encoding='utf-8',newline='\n') as handle:
  handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
 return dict(status=value['status'],models=value['analysis']['models'],predictions=value['analysis']['prediction_count'],
  full_audit=value['audit']['status'],new_source_steps=value['audit']['audit']['new_step_records_parsed'],
  source_gradient_layers=value['audit']['gradient_layer_records_recounted'])

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('output');p.add_argument('--remote-python',required=True)
 args=p.parse_args();print(json.dumps(collect(args.output,args.remote_python)))
