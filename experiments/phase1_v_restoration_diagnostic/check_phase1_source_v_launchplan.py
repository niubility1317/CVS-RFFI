import copy,json
from pathlib import Path
from phase1_source_v_launchplan import validate

spec=json.loads(Path('E:/type10-7/automation_reports/CV-SincNet/20261102-diagnostic-phase1-source-v-restoration-s2026092701-r01/experiment.json').read_text(encoding='utf8'))
plan=validate(spec)
assert plan['cwd'].startswith('/home/') and plan['argv'][0].startswith('/home/')
for field in ('cwd','executable'):
    bad=copy.deepcopy(spec)
    if field=='cwd':bad['code']['cwd']='E:/type10-7/local_envs/phase1-cu128/Scripts/python.exe'
    else:
        old='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';new='E:/type10-7/local_envs/phase1-cu128/Scripts/python.exe'
        bad['execution']['launch_command']=bad['execution']['launch_command'].replace(old,new)
        bad['rows'][0]['command']=bad['execution']['launch_command']
    try:validate(bad)
    except ValueError:pass
    else:raise AssertionError('Windows path must be rejected at remote boundary')
print('VERIFIED registered N607 command and rejection of Windows cwd/executable')
