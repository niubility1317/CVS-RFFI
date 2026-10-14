"""Recover completed artifacts without changing a release or training process."""
import json,subprocess
from pathlib import Path
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh
ROOT=Path(__file__).resolve().parents[2]
RELEASE=d.PROJECT+'/releases/'+d.RELEASE
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
OUTPUT=ROOT/'local_artifacts/identity_mixstyle_seeds_delivery_r01'

def fetch(kind,commit):
 parent="experiments.cvs_identity_mixstyle50."+kind
 child="experiments.cvs_identity_mixstyle_seeds."+kind
 sources=[(parent,(ROOT/"experiments/cvs_identity_mixstyle50"/(kind+".py")).read_text(encoding="utf-8")),(child,(ROOT/"experiments/cvs_identity_mixstyle_seeds"/(kind+".py")).read_text(encoding="utf-8"))]
 payload="\n".join([
  'import json,sys,types,torch',
  'from pathlib import Path',
  'torch.set_num_threads(2)',
  'sys.path[:0]='+repr([RELEASE,RELEASE+'/code']),
  'base=Path('+repr(str(d.BASE))+')',
  'done=json.loads((base/"completion.json").read_text())',
  'assert done["status"]=="EXPERIMENT_COMPLETE" and done["models"]==6 and done["new_trained_models"]==4 and done["prediction_count"]==1008000 and done["truth_last"]',
  'p=base/'+repr('source_style_mechanism.json' if kind=='source_moments' else 'frozen_artifact_recount.json'),
  'if p.is_file():',
  ' result=json.loads(p.read_text())',
  'else:',
  ' for name,code in '+repr(sources)+':',
  '  module=types.ModuleType(name);module.__package__=name.rsplit(".",1)[0];sys.modules[name]=module;exec(code,module.__dict__)',
  ' result=sys.modules['+repr(child)+'].'+('complete' if kind=='source_moments' else 'run')+'()',
  ' result["analysis_commit"]='+repr(commit),
  ' p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")',
  'assert result["status"]=="VERIFIED" and result["run_id"]=='+repr(d.RUN),
  'print(json.dumps(result))',
 ])
 wrapper='\n'.join(['import subprocess,os','r=subprocess.run(['+repr(PYTHON)+',"-"],input='+repr(payload)+',text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))','if r.returncode:raise RuntimeError(r.stderr)','print(r.stdout)'])
 value=json.loads(ssh(wrapper));OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/(kind+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 fields=('models','new_step_records_parsed','epoch_records_parsed','prediction_count','metric_records_independently_recounted') if kind=='audit' else ('models','packet_forwards')
 print(json.dumps(dict(kind=kind,status=value['status'],**{k:value[k] for k in fields})),flush=True)

def main():
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 for kind in ('audit','source_moments'):fetch(kind,commit)
if __name__=='__main__':main()
