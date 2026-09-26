"""Collect read-only launch evidence and mirror this batch's local records."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=Path('E:/type10-7')
SOURCE='20260927-phase1-baselines-practical-manysig-m5-r01'
DATA='20260927-phase2-practical-data-manytx-s2026092705-r01'
PHASE2='20260927-phase2-baselines-practical-manytx-m5-r01'
REMOTE='/home/szu2070436088/2510044040/CV-SincNet'


def main():
    def remote(cmd):
        return subprocess.check_output(['ssh','-F',str(WORKSPACE/'tools/n607_ssh_config'),'-T',
            '-o','BatchMode=yes','-o','ConnectTimeout=10','N607',cmd],text=True,encoding='utf-8',timeout=30)
    launch=json.loads(remote(f'cat {REMOTE}/runs/{PHASE2}/launch.json'))
    state=json.loads(remote(f'cat {REMOTE}/runs/{PHASE2}/state.json'))
    evidence=dict(time=datetime.now(timezone.utc).isoformat(),launch=launch,state=state,
        processes=remote(f"ps -p {launch['pid']} -o pid,etime,args"),cwd=remote(f"readlink /proc/{launch['pid']}/cwd"),
        counts=dict(Counter(row['status'] for row in state.values())))
    record_dir=ROOT/'automation_reports/CV-SincNet'/PHASE2
    (record_dir/'evidence').mkdir(exist_ok=True)
    with (record_dir/'evidence/launch_readback.json').open('x',encoding='utf-8') as f:
        json.dump(evidence,f,indent=2)
    commits={SOURCE:'d8d0f333d968a7ec4df3b85636c474c634c6c8cb',DATA:'7dd3e79a62c8f7e8487ceeff2dd9ae85ca1e74ee',PHASE2:launch['commit']}
    messages={SOURCE:'VERIFIED：8个源域训练进程持续运行，40行中其余32行排队；最新证据见evidence/progress_readback_03.json。',
        DATA:'VERIFIED：36036条received IQ、2100个split已生成；capsule_id=residual-noeq-ba667eee4fb061055e4c08b5。远端builder_report已读回。',
        PHASE2:f"VERIFIED：dispatcher PID={launch['pid']}，工作目录和进程已读回。状态计数{evidence['counts']}。等待各源模型固定200轮完成后自动预测，所有40行预测固定后独立评分。"}
    for run in (SOURCE,DATA,PHASE2):
        folder=ROOT/'automation_reports/CV-SincNet'/run
        spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
        spec['code']['commit']=commits[run]
        if run==PHASE2:
            spec['parent_run_ids']=[SOURCE,DATA]
        (folder/'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
        with (folder/'report.md').open('a',encoding='utf-8') as f:
            f.write('\n## 2026-09-27发布读回\n\n'+messages[run]+'\n\n代码版本：`'+commits[run]+'`。\n')
    subprocess.run([str(Path('F:/App/miniconda3/python.exe')),'-X','utf8','tools/experiment_registry.py','record',PHASE2,
        '--status','QUEUED','--evidence',f'automation_reports/CV-SincNet/{PHASE2}/evidence/launch_readback.json',
        '--note',messages[PHASE2]],cwd=ROOT,check=True)
    for run in (SOURCE,DATA,PHASE2):
        folder=ROOT/'automation_reports/CV-SincNet'/run
        destination=WORKSPACE/'automation_reports/CV-SincNet'/run
        for path in folder.rglob('*'):
            if path.is_file():
                target=destination/path.relative_to(folder)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
    shutil.copy2(ROOT/'docs/CVS_PRACTICAL_PHASE2_EXECUTION_20260927.md',WORKSPACE/'docs/CVS_PRACTICAL_PHASE2_EXECUTION_20260927.md')
    print(json.dumps(evidence['counts']))


if __name__=='__main__':
    main()
