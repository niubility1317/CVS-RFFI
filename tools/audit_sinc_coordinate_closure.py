"""Run an existing frozen-matrix read-only audit; preserve results locally."""
import argparse,json
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh

def run(script,output):
 source=script.read_text(encoding='utf-8')
 # Native epoch pseudo counts are sums, not per-step averages. The original
 # baseline helper mistakenly multiplied both sums by steps_per_epoch=222.
 source=source.replace("round(e.get('train_pseudo_total',0)*222)","int(e.get('train_pseudo_total',0))").replace("round(e.get('train_pseudo_selected',0)*222)","int(e.get('train_pseudo_selected',0))")
 wrapper="import subprocess,os\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(source)+",text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)"
 value=json.loads(ssh(wrapper));output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:value[k] for k in ('status','run_id','executing_code','new_step_records_parsed','epoch_records_parsed','prediction_count','metric_records_independently_recounted','all_workers_terminal','head_updates')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit-script',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.audit_script,a.output)
