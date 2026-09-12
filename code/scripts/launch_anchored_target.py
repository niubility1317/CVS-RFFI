"""One exclusive coordinator for four scenes, then independent scoring."""
import json,os,subprocess,sys,time
from pathlib import Path
BASE=Path('/home/szu2070436088/2510044040/CV-SincNet')
ROOT=BASE/'runs/core90_anchored_target_all_s392005_20260912_r1'
SOURCE=BASE/'runs/core90_anchored_s392005_20260911_r1'
OLD=BASE/'runs/core90_evidence_frozen_392005_20260911'
SCRIPT=Path(__file__).with_name('test_anchored_target.py')
def main():
    ROOT.mkdir(exist_ok=False)
    logs=BASE/'logs'/ROOT.name;logs.mkdir(exist_ok=False)
    base=[sys.executable,'-u',str(SCRIPT),'--source',str(SOURCE),'--ground',str(OLD/'H0/ground_checkpoint.pt'),'--contract',str(OLD/'data_contract.json'),'--output',str(ROOT)]
    jobs=[]
    for gpu,scene in enumerate(('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')):
        active=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
        if len(active)>=2:raise RuntimeError('GPU capacity changed; no dispatch to occupied slot')
        with (logs/(scene+'.log')).open('x') as f:
            command=base+['--stage','predict','--scene',scene]
            p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),cwd=SCRIPT.parents[2])
        jobs.append((p,scene))
        print(json.dumps(dict(scene=scene,pid=p.pid,gpu=gpu,command=command)),flush=True)
    codes=[dict(scene=s,exit_code=p.wait()) for p,s in jobs]
    (ROOT/'prediction_exits.json').write_text(json.dumps(codes),encoding='utf-8')
    if any(x['exit_code'] for x in codes):raise RuntimeError('prediction failed; preserve outputs, no scoring or automatic retry')
    with (logs/'score.log').open('x') as f:subprocess.run(base+['--stage','score'],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=SCRIPT.parents[2])
    (ROOT/'completed.json').write_text(json.dumps(dict(status='SCORED',time=time.time(),candidates=14,scenes=4,confirmation=False)),encoding='utf-8')
    print('COMPLETED',flush=True)
if __name__=='__main__':main()
