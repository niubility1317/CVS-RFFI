"""Launch fixed historical checkpoints from the already verified ratio release."""
import argparse,json,subprocess
from pathlib import Path
REMOTE=r'''
import json,os,subprocess,sys,time
from pathlib import Path
c=CONFIG;project=Path('/home/szu2070436088/2510044040/CV-SincNet');release=Path(c['release'])
if release.parent!=project/'releases' or not release.is_dir():raise ValueError('Invalid verified release')
matrix=json.loads((release/'configs/practical_historical_eval_matrix_20260920.json').read_text())
run=project/'runs'/matrix['run_id'];logs=project/'logs'/matrix['run_id']
if run.exists() or logs.exists():raise FileExistsError('Existing evaluation; reconcile')
run.mkdir();logs.mkdir();sys.path.insert(0,str(release/'code'))
from scripts.dispatch_xuc_full import available_gpu,occupancy
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
target=project/'runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4'
active={};receipts=[]
for item in matrix['rows']:
    gpu=available_gpu(active)
    if gpu is None:raise RuntimeError('No legal GPU slot; reconcile existing receipts')
    checkpoint=Path(item['checkpoint']);initial=json.loads((checkpoint.parent/'initialization.json').read_text())
    if not checkpoint.is_file() or not initial['scratch_only'] or initial['checkpoint_sources'] or initial['target_training_contact'] or initial['source_roles']!='EXACT_MATCH':
        raise ValueError('Historical checkpoint source contract/provenance mismatch')
    recipe=run/(item['id']+'.recipe.json')
    model_args=json.loads((checkpoint.parent/'resolved_config.json').read_text())
    model_args.update(matrix['execution_options'])
    recipe.write_text(json.dumps({'baseline_args':model_args},indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(release/'code'),PYTHONUNBUFFERED='1')
    command=[python,'-u',str(release/'code/scripts/evaluate_practical_checkpoint.py'),
        '--checkpoint',str(checkpoint),'--recipe',str(recipe),'--input-package',str(target/'target_inputs'),
        '--truth',str(target/'target_truth/truth_sidecar.json'),'--output',str(run/item['id']),
        '--run-id',matrix['run_id'],'--row-id',item['id'],'--epoch',str(item['epoch']),'--scenarios',','.join(matrix['scenes'])]
    logpath=logs/(item['id']+'.eval.log')
    with logpath.open('x') as log:
        child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active[item['id']]={'gpu':gpu,'process':child}
    receipt=dict(pid=child.pid,gpu=gpu,argv=command,cwd=str(release),row_id=item['id'],output=str(run/item['id']),log=str(logpath),checkpoint=str(checkpoint),epoch=item['epoch'],commit=c['commit'],time=time.time())
    (run/('launch_'+item['id']+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt)
print(json.dumps(dict(run_id=matrix['run_id'],rows=receipts,release=str(release),commit=c['commit'])))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch-file',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();training=json.loads(Path(a.launch_file).read_text(encoding='utf-8'))
    c={'release':training['rows'][0]['cwd'],'commit':training['commit']}
    script=REMOTE.replace('CONFIG',repr(c));compile(script,'remote','exec')
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','N607','python3 -'],input=script.encode(),capture_output=True)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    (output/'landing.stdout').write_bytes(result.stdout);(output/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()
    with (output/'launch.json').open('x',encoding='utf-8') as f:json.dump(json.loads(result.stdout),f,indent=2)

if __name__=='__main__':main()
