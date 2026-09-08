"""Collect the fixed pair24 terminal state and completed-row test evidence."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = '/home/szu2070436088/2510044040/CV-SincNet'
SSH = ['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607']
ROOT = Path('E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_completed23_test_20260908_v1')
RUN = ROOT.name
ROOT.mkdir(parents=True, exist_ok=True)

def remote(code):
    return subprocess.run(SSH+['python3 -'],input=code,text=True,encoding='utf-8',capture_output=True,check=True,timeout=60).stdout

def save(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def inventory():
    previous = json.loads(Path('E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_completed19_test_20260907_v1/current_training_status.json').read_text(encoding='utf-8'))
    observations=[]
    for name in ['phase1_adv3b02_pair24_manifest.json','phase1_adv3b02_pair12_recovery_manifest.json','phase1_adv3b02_safe3_recovery_manifest.json']:
        cmd=[sys.executable,'-X','utf8','tools/pair_matrix_remote.py','observe']
        if 'pair24' not in name:cmd+=['--manifest','configs/'+name]
        raw=subprocess.run(cmd,text=True,encoding='utf-8',capture_output=True,check=True,timeout=45).stdout
        obj=json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
        observations.append(obj)
    save('observations.json',observations)
    old=previous['rows']
    rows=json.loads(remote('''import pathlib,json
rows=json.loads('''+repr(json.dumps(old))+''')
out=[]
for row in rows:
 p=pathlib.Path(row['checkpoint']).parent
 records=[json.loads(x) for x in (p/'metrics_epoch.jsonl').read_text().splitlines() if x.strip()]
 r=dict(row);r['epoch']=records[-1]['epoch'];r['checkpoint_exists']=(p/'final_ssdg.pth').is_file()
 r['resources']=json.loads((p/'phase1_resource_summary.json').read_text()) if (p/'phase1_resource_summary.json').is_file() else None
 r['metrics_last']=records[-1]
 r['source_validation_all_epochs']=[{k:e.get(k) for k in ['epoch','val_tx_acc','stage_source_val_sat_mean_tx','epoch_time_s','train_skipped_nonfinite_grad','train_skipped_nonfinite_loss']} for e in records]
 out.append(r)
print(json.dumps(out))
'''))
    for row in rows:
        which=2 if 'safe3' in row['run'] else 1 if 'pair12' in row['run'] else 0
        observed=next(x for x in observations[which]['rows'] if x['row_id']==row['row_id'])
        row.update(status=observed['status'],alive=observed['alive'])
    checked={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows}
    save('current_training_status.json',checked)
    save('training_evidence.json',[r for r in rows if r['checkpoint_exists'] and r['epoch']==200 and not r['alive']])
    print(json.dumps({'checked_at_utc':checked['checked_at_utc'],'rows':[{k:r[k] for k in ['row_id','epoch','status','alive','checkpoint_exists']} for r in rows]}))
    print(remote('''import subprocess,pathlib,json
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader'],text=True))
'''))

def collect():
    code='''import pathlib,json
p=pathlib.Path('''+repr(BASE+'/runs/'+RUN)+''')
print(json.dumps({name:(p/name).read_text() for name in ['manifest.json','done.json','results.json'] if (p/name).is_file()}))
'''
    files=json.loads(remote(code))
    for name,txt in files.items():(ROOT/name).write_text(txt,encoding='utf-8')
    if 'done.json' not in files:
        print(remote('import pathlib; p=pathlib.Path('+repr(BASE+'/logs/'+RUN+'/queue.log')+');print(p.read_text()[-3500:])'))
        return
    evidence=json.loads(remote('''import pathlib,json
p=pathlib.Path('''+repr(BASE+'/runs/'+RUN)+''')
m=json.loads((p/'manifest.json').read_text());result=json.loads((p/'results.json').read_text())
preds=[json.loads((p/'predictions'/r['row_id']/'complete.json').read_text()) for r in m['rows']]
print(json.dumps({'predictions':preds,'all_sealed_before_scoring':all(r['sealed_at_unix']<result['scored_at_unix'] for r in preds),'input_resolved':str((p/'inputs').resolve()),'scored_at_unix':result['scored_at_unix']}))
'''))
    save('execution_evidence.json',evidence)
    (ROOT/'queue.log').write_text(remote('import pathlib;print(pathlib.Path('+repr(BASE+'/logs/'+RUN+'/queue.log')+').read_text())'),encoding='utf-8')
    print('VERIFIED SCORED',len(evidence['predictions']),evidence['all_sealed_before_scoring'])

def deploy():
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote_head=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/adv3b02-daot-stn-v1-20260901'],text=True).split()[0]
    assert head==remote_head
    release=BASE+'/releases/'+RUN
    output=BASE+'/runs/'+RUN
    logdir=BASE+'/logs/'+RUN
    print(remote('''import pathlib,shutil
p=pathlib.Path('''+repr(release)+''');r=pathlib.Path('''+repr(output)+''');l=pathlib.Path('''+repr(logdir)+''')
assert not p.exists() and not r.exists() and not l.exists(), 'existing output: reconcile before retry'
assert shutil.disk_usage(p.parent).free>2*1024**3
p.mkdir();l.mkdir()
print('NEW_RELEASE_CREATED')
'''))
    files=[Path('tools/pair_completed_test.py'),Path('configs/phase1_adv3b02_completed23_test_manifest.json')]
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        subprocess.run(['scp','-F','E:/type10-7/tools/n607_ssh_config',str(p),'N607:'+release+'/'+p.name],check=True,timeout=45)
    compiled=remote('''import pathlib,hashlib,json,py_compile
p=pathlib.Path('''+repr(release)+''');expected=json.loads('''+repr(json.dumps(expected))+''')
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in expected.items())
py_compile.compile(str(p/'pair_completed_test.py'),doraise=True)
print('TRANSFER_AND_COMPILE_VERIFIED')
''')
    launched=json.loads(remote('''import pathlib,subprocess,os,json
release='''+repr(release)+'''
output='''+repr(output)+'''
logdir='''+repr(logdir)+'''
assert not pathlib.Path(output).exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
cmd=['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u',release+'/pair_completed_test.py','--manifest',release+'/phase1_adv3b02_completed23_test_manifest.json','--root',output,'--mode','queue']
with open(logdir+'/queue.log','x') as log:
 p=subprocess.Popen(cmd,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'cmd':cmd,'cwd':release,'gpu':1,'output':output,'logdir':logdir}))
'''))
    launched.update(runtime_commit=head,transfer_sha256=expected,compile_verification=compiled.strip())
    save('launch_evidence.json',launched)
    print(json.dumps(launched))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['inventory','collect','deploy']);args=ap.parse_args()
    {'inventory':inventory,'collect':collect,'deploy':deploy}[args.mode]()
