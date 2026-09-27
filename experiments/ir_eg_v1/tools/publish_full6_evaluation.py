"""Publish the fixed completed-row evaluator; never modify the training release."""
import hashlib,json,subprocess
from pathlib import Path

RUN='phase1_adv3b02_xuc_full_s392005_20260913_r1'
EVAL='phase1_adv3b02_xuc_full6_eval_s392005_20260914_r1'
REMOTE=r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
c=CONFIG
p=Path('/home/szu2070436088/2510044040/CV-SincNet');script=Path(c['remote'])
if hashlib.sha256(script.read_bytes()).hexdigest()!=c['sha']:raise ValueError('transfer mismatch')
compile(script.read_bytes(),str(script),'exec')
state=json.loads((p/'runs'/c['run']/'pipeline_state.json').read_text());release=Path(state['release'])
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
if (p/'runs'/c['evaluation']).exists():raise FileExistsError('existing evaluation; reconcile')
# One actual frozen checkpoint load and zero-IQ forward, no target access.
check="import torch;from cvsrffi.xuc_fusion.checkpoints import load_model;torch.set_num_threads(2);m,_=load_model(__import__('pathlib').Path("+repr(str(p/'runs'/c['run']/'F-M12/final_ssdg.pth'))+"),torch.device('cpu'));m.eval();o=m(torch.zeros(2,2,256),return_aux=True);assert o['tx_logits'].shape==(2,6);print('FROZEN_CHECKPOINT_SMOKE_PASS')"
subprocess.run([python,'-c',check],cwd=release,env=env,check=True)
cmd=[python,'-u',str(script),'--project',str(p),'--run-id',c['run'],'--evaluation-id',c['evaluation']]
log=p/'logs'/(c['evaluation']+'.dispatcher.log')
with log.open('x') as h:
 child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='LAUNCH_SUBMITTED',pid=child.pid,argv=cmd,commit=c['commit'],script=str(script),log=str(log),time=time.time())))
'''

def main():
 repo=Path(__file__).resolve().parents[3]
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
 assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]==commit
 assert not subprocess.check_output(['git','status','--porcelain','--','experiments/adv3b02_xuc/tools/test_full_completed_rows.py'],cwd=repo,text=True)
 local=Path(__file__).with_name('test_full_completed_rows.py');sha=hashlib.sha256(local.read_bytes()).hexdigest()
 remote='/home/szu2070436088/2510044040/CV-SincNet/releases/full6_eval_'+commit[:10]+'.py'
 conn=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
 # Refuse overwriting even an uncertain prior landing.
 subprocess.run(['ssh',*conn,'-T','N607','test ! -e '+remote],check=True)
 subprocess.run(['scp',*conn,str(local),'N607:'+remote],check=True)
 config=dict(remote=remote,sha=sha,run=RUN,evaluation=EVAL,commit=commit)
 payload=REMOTE.replace('CONFIG',repr(config));compile(payload,'remote_publish','exec')
 r=subprocess.run(['ssh',*conn,'-T','N607','python3 -'],input=payload.encode(),capture_output=True)
 out=Path('E:/type10-7/automation_reports/CV-SincNet')/EVAL
 (out/'launch.stdout').write_bytes(r.stdout);(out/'launch.stderr').write_bytes(r.stderr)
 print(r.stdout.decode());print(r.stderr.decode());r.check_returncode()

if __name__=='__main__':main()
