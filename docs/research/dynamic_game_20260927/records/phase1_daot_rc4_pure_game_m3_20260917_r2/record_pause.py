"""Persist independently observed pause state and a non-executed continuation plan."""
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SUPPORT = ROOT / 'pause_support'
state = json.loads((SUPPORT / 'post_pause_readback.json').read_text(encoding='utf-8'))
manifest = json.loads((SUPPORT / 'manifest.json').read_text(encoding='utf-8'))
checks = json.loads((SUPPORT / 'restore_validation.json').read_text(encoding='utf-8'))
assert manifest['phase'] == 'VERIFIED_PAUSED' and checks['status'] == 'PASS'
assert state['dispatcher'].get('absent') is True
assert all(v['process'].get('absent') is True for k,v in state['workers'].items() if k.startswith('PURE_'))
release = '/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_pause_77d6e6b67b'
run = manifest['run']
project = str(Path(run).parent.parent).replace('\\','/')
python = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
plan = dict(status='PREPARED_NOT_LAUNCHED', automatic_resume=False, owner='unassigned_until_explicit_resume',
            parent_run=run, recovery_code_commit='77d6e6b67b352a82aa3048574cc5a1f8dd43ac4a', rows=[])
spec = json.loads((ROOT / 'experiment.json').read_text(encoding='utf-8'))
for row in spec['rows']:
    rid = row['row_id']
    current = state['state']['rows'][rid]
    row['status'] = current['status']
    if rid in manifest['rows']:
        saved = manifest['rows'][rid]
        row['pause_recovery'] = dict(checkpoint=saved['checkpoint'], epoch=saved['epoch'], step=saved['step'],
            stopped_step=saved['observed_steps_at_stop'], steps_to_replay=saved['steps_to_replay'],
            validation='CPU_STRICT_RESTORE_PASS_AND_LOCAL_ACTIVE_UPDATE_PARITY_PASS', parent_run_id=spec['run_id'])
    if rid.startswith('PURE_'):
        command = [python,'-u',release+'/code/scripts/train_response_games.py','--config',
            release+'/configs/separate_controls/'+rid+'.json','--dataset',project+'/Dataset_WigSig/ManySig.pkl',
            '--output',run+'_resume01/'+rid,'--source-contract',run+'/source_contract.json','--device','cuda:0','--execute',
            '--pause-request',run+'_resume01/PAUSE_REQUEST']
        if rid in manifest['rows']:
            command += ['--resume',manifest['rows'][rid]['checkpoint'],'--allow-legacy-ticket-resume']
        plan['rows'].append(dict(row_id=rid, mode='resume' if rid in manifest['rows'] else 'not_started',
                                command=command, gpu='assign_only_after_live_occupancy_check'))
spec['execution']['status'] = 'PARTIALLY_PAUSED_NATIVE_DRAINING'
spec['execution']['pause_evidence'] = str(SUPPORT / 'post_pause_readback.json')
spec['execution']['resume_plan'] = str(SUPPORT / 'resume_plan.json')
spec['notes'].append('2026-09-17用户要求释放GPU：13行纯博弈断点恢复验证后暂停，5行未启动保留，原生3行无安全中间恢复接口而继续至E200；GPU3–7已核实空闲。最多重算239步。')
(ROOT / 'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(SUPPORT / 'resume_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'events.jsonl').open('a',encoding='utf-8') as handle:
    handle.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),event='USER_GPU_RELEASE_VERIFIED',
        status='PARTIALLY_PAUSED_NATIVE_DRAINING',paused=13,held=5,native_running=3,
        evidence=str(SUPPORT/'post_pause_readback.json')),ensure_ascii=False)+'\n')
lines=['\n## 实际执行结果：VERIFIED\n',
    '13行PAUSED_RECOVERABLE、5行HELD_USER_PAUSE、原生3行RUNNING。本批dispatcher已退出。独立/proc及nvidia-smi读回：仅原生PID2153677/2153690/2153703在GPU0/1/2运行；GPU3/4/5/6/7利用率0%、各约1MiB，已释放。\n',
    '13个远端真实快照逐项恢复PASS。验证期间原进程仍运行，所以实际需重算33–239步；最大239步约1.08个epoch，此值取代早期“最多不足一个epoch”的估计。所有原日志、checkpoint和未提交片段保留，不把重算步计为额外有效训练预算。\n',
    '旧断点缺少累计耗时与forward计数，后续成本统计需连接父目录日志；不会把恢复后的计时当作总成本。恢复入口及18行待执行命令保存在pause_support/resume_plan.json，未自动恢复、未启动新训练或测试。\n',
    '|纯博弈row|保存epoch|恢复步|需重算步|\n|---|---:|---:|---:|']
for rid, saved in manifest['rows'].items():
    lines.append(f"|{rid}|{saved['epoch']}|{saved['step']}|{saved['steps_to_replay']}|")
with (ROOT/'pause_restore.md').open('a',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')
with (ROOT/'report.md').open('a',encoding='utf-8') as handle:
    handle.write('\n## 用户请求释放GPU：VERIFIED\n\n13行纯博弈已保存并验证恢复断点后暂停，5行排队保持未启动；原生3行继续运行。GPU3–7已释放。详情及恢复计划见pause_restore.md和pause_support。当前状态取代上文启动时RUNNING计数，不表示整批训练完成。\n')
worktree = Path('E:/type10-7/code/snapshots/native_dr_eg_prepare_20260914_wt')
target = worktree / 'automation_reports/CV-SincNet' / ROOT.name
for path in ROOT.rglob('*'):
    if path.is_file() and path.suffix not in ('.tar','.pyc'):
        out=target/path.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
print(json.dumps(dict(status='RECORDED',paused=13,held=5,native_running=3,free_gpus=[3,4,5,6,7])))
