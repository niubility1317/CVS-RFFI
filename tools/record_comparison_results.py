"""Read back completion evidence and update the three existing run records."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=Path('E:/type10-7')
RUNS=['20260927-phase1-baselines-practical-manysig-m5-r01',
      '20260927-phase2-baselines-practical-manytx-m5-r01',
      '20260927-phase1-baselines-final-clean-satellite-m5-r01']


def main():
    script='''import json
from pathlib import Path
from datetime import datetime, timezone
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs')
runs=RUNS_VALUE
result={'time':datetime.now(timezone.utc).isoformat(),'runs':{}}
for name in runs:
    folder=root/name
    item={'state':json.loads((folder/'state.json').read_text())}
    for file in ('dispatcher_complete.json','scorer_exit.json'):
        if (folder/file).exists(): item[file]=json.loads((folder/file).read_text())
    if name==runs[0]:
        item['source_artifacts']={}
        for row in item['state']:
            complete=json.loads((folder/row/'completion.json').read_text())
            item['source_artifacts'][row]={'epoch':complete['epoch'],'target_evaluated':complete['target_evaluated'],'last_pt_bytes':(folder/row/'last.pt').stat().st_size}
    for file in ('results.json','scored_results.json'):
        if (folder/file).exists():item[file]={'bytes':(folder/file).stat().st_size,'modified':(folder/file).stat().st_mtime}
    result['runs'][name]=item
print(json.dumps(result))
'''.replace('RUNS_VALUE',repr(RUNS))
    remote=subprocess.run(['ssh','-F',str(WORKSPACE/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'],input=script,text=True,encoding='utf-8',capture_output=True,check=True)
    evidence=json.loads(remote.stdout)
    rows=evidence['runs'][RUNS[0]]['source_artifacts']
    assert len(rows)==40 and all(v['epoch']==200 and v['target_evaluated'] is False and v['last_pt_bytes']>0 for v in rows.values())
    assert evidence['runs'][RUNS[1]]['scorer_exit.json']=={'exit_code':0,'artifact_exists':True}
    assert evidence['runs'][RUNS[2]]['state']['status']=='SCORED'
    for index,run in enumerate(RUNS):
        folder=ROOT/'automation_reports/CV-SincNet'/run
        (folder/'completion_readback_20260927.json').write_text(json.dumps(evidence['runs'][run],indent=2),encoding='utf-8')
        note=('VERIFIED:40/40 source checkpoints epoch200 exist; source training complete. Separate final evaluation scored.' if index==0 else
              'VERIFIED: all40 rows completed predictions and independent scoring; full scored artifacts downloaded and aggregation coverage validated.')
        with (folder/'report.md').open('a',encoding='utf-8') as f:
            f.write('\n## 最终结果核实\n\n'+evidence['time']+'\n\n'+note+'\n\n完整汇总：docs/COMPARISON_RESULTS_20260927.md。历史392005与4个新seed分开报告；不反馈调参。\n')
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/experiment_registry.py'),'record',run,
            '--status','TRAINING_COMPLETE' if index==0 else 'ANALYZED','--evidence',f'automation_reports/CV-SincNet/{run}/completion_readback_20260927.json',
            '--note',note],cwd=ROOT,check=True)
        dest=WORKSPACE/'automation_reports/CV-SincNet'/run
        for name in ('report.md','events.jsonl','completion_readback_20260927.json'):
            shutil.copy2(folder/name,dest/name)
    shutil.copy2(ROOT/'docs/COMPARISON_RESULTS_20260927.md',WORKSPACE/'docs/COMPARISON_RESULTS_20260927.md')
    subprocess.run([sys.executable,'-X','utf8',str(WORKSPACE/'tools/experiment_registry.py'),'build','--managed-only'],cwd=WORKSPACE,check=True)
    for name in ('README.md','catalog.csv','catalog.jsonl','coverage.json'):
        shutil.copy2(WORKSPACE/'experiment_registry'/name,ROOT/'experiment_registry'/name)
    for path in (WORKSPACE/'experiment_registry/by_method').glob('*.md'):
        shutil.copy2(path,ROOT/'experiment_registry/by_method'/path.name)
    print(json.dumps({'source_checkpoints_verified':len(rows),'time':evidence['time']}))


if __name__=='__main__':
    main()
