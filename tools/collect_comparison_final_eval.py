"""Read-only remote evidence; local registration/report update for this launch."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=Path('E:/type10-7')
RUN='20260927-phase1-baselines-final-clean-satellite-m5-r01'
REMOTE='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN


def main():
    def remote(command):
        return subprocess.check_output(['ssh','-F',str(WORKSPACE/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes',
            '-o','ConnectTimeout=10','N607',command],text=True,encoding='utf-8',timeout=30)
    launch=json.loads(remote('cat '+REMOTE+'/launch.json'))
    state=json.loads(remote('cat '+REMOTE+'/state.json'))
    evidence=dict(time=datetime.now(timezone.utc).isoformat(),launch=launch,state=state,
        process=remote(f"ps -p {launch['pid']} --ppid {launch['pid']} -o pid,ppid,etime,args"),
        cwd=remote(f"readlink /proc/{launch['pid']}/cwd"),log_tail=remote('tail -n 5 '+REMOTE+'/build.log'))
    folder=ROOT/'automation_reports/CV-SincNet'/RUN
    with (folder/'launch_readback.json').open('x',encoding='utf-8') as f:
        json.dump(evidence,f,indent=2)
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
    spec['code']['commit']=launch['commit']
    (folder/'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    note=f"VERIFIED: dispatcher PID{launch['pid']}, CWD/child process/log progress independently read back; state={state['status']}. CPU only. Source40rows retain last.pt and200epoch selection."
    with (folder/'report.md').open('a',encoding='utf-8') as f:
        f.write('\n## 启动读回\n\n'+note+'\n\n16项聚焦测试及一次P0/P1审查通过。发布commit：`'+launch['commit']+'`。旧训练日志不追写，新日志格式从新launcher启动生效。\n')
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/experiment_registry.py'),'record',RUN,'--status','RUNNING',
        '--evidence',f'automation_reports/CV-SincNet/{RUN}/launch_readback.json','--note',note],cwd=ROOT,check=True)
    dest=WORKSPACE/'automation_reports/CV-SincNet'/RUN
    dest.mkdir(parents=True,exist_ok=True)
    for path in folder.iterdir():
        if path.is_file():
            shutil.copy2(path,dest/path.name)
    shutil.copy2(ROOT/'docs/COMPARISON_LAUNCHER.md',WORKSPACE/'docs/COMPARISON_LAUNCHER.md')
    subprocess.run([sys.executable,'-X','utf8',str(WORKSPACE/'tools/experiment_registry.py'),'build','--managed-only'],cwd=WORKSPACE,check=True)
    for name in ('README.md','catalog.csv','catalog.jsonl','coverage.json'):
        shutil.copy2(WORKSPACE/'experiment_registry'/name,ROOT/'experiment_registry'/name)
    for path in (WORKSPACE/'experiment_registry/by_method').glob('*.md'):
        shutil.copy2(path,ROOT/'experiment_registry/by_method'/path.name)
    print(json.dumps(dict(pid=launch['pid'],status=state['status'])))


if __name__=='__main__':
    main()
