"""Short native SSH connections, one archive verification, independent process reads."""
import argparse,json,subprocess,sys
from pathlib import Path

SSH=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']

def remote(script,timeout=60):
    compile(script,'remote_inspection','exec')
    result=subprocess.run(['ssh',*SSH,'-T','N607','python3 -'],input=script.encode('utf-8'),
        capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+'\n'+result.stdout.decode('utf-8',errors='replace'))
    return json.loads(result.stdout)

def inputs():
    return remote(r'''
import json,os,subprocess
from pathlib import Path
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
source=project/'runs/20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01/SIG_TIME-s2026092701/source'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
done=read(source/'completion.json');resolved=read(source/'resolved_config.json')
contract=read(source/'source_contract.json')
fields=('wisig_pkl','wisig_equalized','wisig_normalize','normalize','wisig_out_len','label_smoothing','label_epochs','pseudo_epochs','epochs','lr','weight_decay','use_unlabeled','use_ema_teacher','sat_training_mode','use_concat_sat_channel_aug','pseudo_temporal_mode')
args=read(source/'actual_native_args.json');args=args.get('args',args)
views=project/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01'
manifests=[]
for p in views.glob('*/manifest.json'):
 v=read(p)
 if v.get('scenes'):manifests.append(dict(path=str(p),value=v))
print(json.dumps(dict(source_root=str(source),resolved_commit=resolved.get('commit'),
 original_done={k:done.get(k) for k in ('status','epoch','optimizer_steps','logged_steps','target_access','checkpoint_sources','initialization','feature_extractor','identity_input_policy','elapsed_seconds')},
 initialization=read(source/'initialization.json'),args={k:args.get(k) for k in fields},
 contract={k:v for k,v in contract.items() if k!='role_ids'},role_counts={k:len(v) for k,v in contract['role_ids'].items()},
 formal_views=manifests,source_cache_exists=(project/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701/source_cache.pt').is_file(),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True))))
''')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',action='store_true');p.add_argument('--script');p.add_argument('--output',required=True)
    a=p.parse_args();v=inputs() if a.inputs else remote(Path(a.script).read_text(encoding='utf-8'))
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(v,ensure_ascii=False,indent=2))
