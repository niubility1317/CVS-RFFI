from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,re,shutil
root=Path(__file__).resolve().parent
wt=Path('E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-xuc-fusion-design-20260913')
mirror=wt/'docs/research'/root.name
snapshot=json.loads((root/'remote_snapshot_3.json').read_text(encoding='utf-8'))
previous=json.loads((root/'remote_snapshot_2.json').read_text(encoding='utf-8'))
state=snapshot['state'];stamp=datetime.fromtimestamp(snapshot['time'],timezone(timedelta(hours=8))).isoformat()
growth={}
for rid,entry in state['rows'].items():
    assert entry['status']=='RUNNING'
    before=previous['artifacts'][rid];after=snapshot['artifacts'][rid]
    if rid not in ('M09','M10'):
        delta=after['actions.jsonl']['last']['step']-before['actions.jsonl']['last']['step']
        assert delta>0;growth[rid]={'accepted_step_delta':delta,'epoch':after['logs.jsonl']['last']['epoch']}
    else:
        assert '[EPOCH-END] E001/200' in after['train.log']['tail']
        assert after['train.log']['bytes']>before['train.log']['bytes']
        growth[rid]={'epoch':1,'epoch_end_observed':True,'log_byte_growth':after['train.log']['bytes']-before['train.log']['bytes']}
(root/'progress_growth_verification.json').write_text(json.dumps({'status':'VERIFIED','time':stamp,'rows':growth},indent=2),encoding='utf-8')
automation=Path('E:/codex/home/automations/xuc15/automation.toml').read_text(encoding='utf-8')
assert 'status = "ACTIVE"' in automation and 'rrule = "FREQ=HOURLY;INTERVAL=1"' in automation
assert state['commit'] in automation and state['run_id'] in automation
automation_evidence={'status':'VERIFIED','id':'xuc15','active':True,'interval_hours':1,'target_thread_id':'01a096a4-aa1c-7ac3-8a58-414b9ba959cc','run_id':state['run_id'],'commit':state['commit'],'readback':'E:/codex/home/automations/xuc15/automation.toml','time':stamp}
(root/'automation_verification.json').write_text(json.dumps(automation_evidence,indent=2),encoding='utf-8')
text=(root/'report.md').read_text(encoding='utf-8')
text=text.replace('状态：LOCAL_VERIFIED，正在固定Git版本并发布。当前任务已授权15组实验、每小时监控、技术故障修复后重发；正式训练尚待远端读回确认。','状态：RUNNING / VERIFIED。15/15行已在N607启动，实际PID/PPID/CWD/argv/GPU/scratch全部核对，连续日志增长已验证。每小时监控ACTIVE。正式结果未完成。')
text=text.replace('实现及接口测试通过；远端实际source contract待发布读回','实现、接口与远端实际source contract通过；native两行EXACT_MATCH')
text=text.replace('待固定提交、落地、PID/CWD/argv/GPU/log增长读回','VERIFIED：15行PID/PPID/CWD/argv/GPU/scratch匹配且日志增长')
text+='\n## 发布结果与当前交接\n\n'
text+=f'最后读回时间：{stamp}。执行提交：`{state["commit"]}`，GitHub分支已独立ls-remote核对一致，无ahead/behind。release：`{state["release"]}`。dispatcher PID：{state["pid"]}，PPID=1，实际CWD及argv与发布记录匹配。\n\n'
text+='正式15行均RUNNING，13个CORE90行已完成E2—E6范围内的epoch，两条native均完成E1；所有行连续进度检查通过。GPU0—7各2个compute PID，其中GPU4的原任务PID612456保留，本任务使用其余15个名额。没有确认的训练故障，无需重发。\n\n'
text+='|行|PID|GPU|已完成epoch|\n|---|---:|---:|---:|\n'
for rid,entry in state['rows'].items():text+=f'|{rid}|{entry["pid"]}|{entry["gpu"]}|{growth[rid]["epoch"]}|\n'
text+='\n实际source contract已读回L6300/U56700/V27000、6TX、15个RX/day域；M09/M10初始化记录source_roles=EXACT_MATCH。CUDA smoke已从自生成checkpoint strict重建为4×6 logits，source-only。X开启行128个合法锚点，U开启行4个有效块、0个不可用块。该计数只证明路径在真实训练中可用，不能证明性能提升。C*尚处早期校准，未自然触发动作不属于技术故障；不为强制触发调整阈值。\n\n'
text+='独立P0/P1审查及恢复定点复核全部通过，当前无未解决P0/P1。16项本地测试及3行真实入口/自生成checkpoint检查通过；已完成一份发布归档SHA比较和一次远端编译。\n\n'
text+='后续唯一监控：heartbeat xuc15，每小时一次，已更新到实际run/commit并读回。主发布已结束，后续由该heartbeat沿本报告执行健康检查和授权的技术恢复；先核实旧owner及所有关联进程，禁止重复提交。原生A1使用final_only，在E200以前没有latest checkpoint属于预期；现有监控helper对超过20KB的native单行telemetry可能标partial_write，此时用完整metrics_epoch.jsonl或EPOCH-END日志确认，不能据此判训练失败。\n\n'
text+='证据：launch_identity_verification.json、progress_growth_verification.json、remote_source_smoke.json、remote_snapshot_1/2/3.json、automation_verification.json、delivery/landing.stdout。训练尚未E200，prediction/scoring尚未执行；最终评分由dispatcher按已登记链路接续，全部闭合后暂停heartbeat。\n'
(root/'report.md').write_text(text,encoding='utf-8',newline='\n')
for file in root.iterdir():
    if file.is_file() and file.suffix in ('.md','.py','.json','.xml'):shutil.copy2(file,mirror/file.name)
print(json.dumps({'status':'VERIFIED','running':15,'growth_verified':len(growth),'automation':'ACTIVE','report':str(root/'report.md')},ensure_ascii=False))
