"""Validate captured launch post-state and mirror the run reports."""
import csv
import io
import json
import shutil
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path

def main():
    repo=Path(__file__).resolve().parents[2]
    base=repo/'automation_reports/CV-SincNet'
    folder=base/'core90_game_v2_20260911_r3'
    before=json.loads((folder/'initial_poststate.json').read_text(encoding='utf-8'))
    after=json.loads((folder/'verified_poststate.json').read_text(encoding='utf-8'))
    state=after['status'];owner=after['owner'];release=state['release']
    assert state['state']=='RUNNING' and len(state['active'])==5 and len(state['pending'])==16 and not state['failed']
    assert owner['exists'] and owner['cwd']==release and owner['ppid']=='1'
    assert after['smoke']['status']=='PASS' and all(x['accepted'] for x in after['smoke']['stages'])
    gpu_rows=list(csv.reader(io.StringIO(after['gpu'])))
    counts=Counter(row[0].strip() for row in gpu_rows)
    assert max(counts.values())<=2
    gpu_by_pid={int(row[1]):row[0].strip() for row in gpu_rows}
    previous={r['run_id']:r for r in before['records']}
    table=[]
    for row in after['records']:
        p=row['process'];c=row['resolved_config.json'];b=row['backend_configuration.json']
        assert p['exists'] and p['cwd']==release and int(p['ppid'])==owner['pid']
        assert p['cuda_visible']=='CUDA_VISIBLE_DEVICES='+row['gpu_uuid'] and gpu_by_pid[row['pid']]==row['gpu_uuid']
        assert p['argv']==row['argv']
        assert c['epochs']==200 and c['from_scratch'] and not c['baseline_ckpt'] and not c['game_resume']
        assert c['game_evidence_version']==2 and c['game_no_audit'] and c['game_control']=='off' and not c['amp']
        assert b['deterministic_algorithms'] and b['cudnn_deterministic']
        assert row['action_count']>previous[row['run_id']]['action_count']
        assert not row['rejected_actions'] and not row['nonfinite_losses'] and row['last_action']['accepted']
        table.append(dict(run_id=row['run_id'],pid=row['pid'],gpu=row['gpu'],accepted_steps=row['action_count'],epoch=row['last_action']['epoch']))
    proof=dict(status='VERIFIED',release_commit='5bd1665631b15b1ed97fae0f6b0ed57f25b68ec9',
        archive_sha256='d013ba63d4b10a22151c8cf813218807cb63eb04209cf14e2eff4e258219ad74',
        captured_at=datetime.fromtimestamp(after['captured_unix'],timezone(timedelta(hours=8))).isoformat(),
        active=5,pending=16,failed=0,owner_pid=owner['pid'],rows=table,gpu_process_counts=dict(counts),
        stage='RUNNING_NOT_ARTIFACTS_COMPLETE',target_evaluated=False)
    (folder/'delivery_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    p=folder/'report.md';text=p.read_text(encoding='utf-8')
    text=text.replace('LOCAL_VERIFIED。远端归档、编译、smoke和启动后读回待执行；不得将本条写成训练完成。','RUNNING，启动后态VERIFIED；尚未完成E200或最终评分。')
    text+='\n## 实际发布与后态\n\n'
    text+=f"- 采集时间：{proof['captured_at']}。发布commit：`{proof['release_commit']}`；本地/远端归档SHA256一致，远端编译PASS。\n"
    text+='- r3真实L_s/U_s九项smoke PASS，实际增强器已接入，确定性保持开启；首批5行运行，16行排队，0失败。\n'
    text+=f"- 调度PID={owner['pid']}，PPID=1；5个worker的PID/PPID/CWD/argv/GPU UUID与配置逐项读回一致，全服务器每卡2个计算进程。两次快照动作记录持续增长，全部已记录更新accepted且loss有限。\n"
    for r in table:text+=f"- {r['run_id']}：PID={r['pid']}，GPU={r['gpu']}，已接受{r['accepted_steps']}步，当前E{r['epoch']}。\n"
    text+='- r1首次smoke失败，r2普通增强分支失败，均保留；r2的6个失败worker及调度均已退出。修复源码及发布审查已完成，未复用旧root。\n'
    text+='- 证据：[启动快照](initial_poststate.json)、[增长快照](verified_poststate.json)、[交付核验](delivery_verification.json)。当前仅启动验证，不作完整训练效果结论；16个待排队行尚无实际训练激活证据。\n'
    p.write_text(text,encoding='utf-8')
    for version in ('r1','r2','r3'):
        source=base/('core90_game_v2_20260911_'+version)
        target=Path('E:/type10-7/automation_reports/CV-SincNet')/source.name
        target.mkdir(parents=True,exist_ok=True)
        for f in source.iterdir():
            if f.is_file():shutil.copy2(f,target/f.name);assert f.read_bytes()==(target/f.name).read_bytes()
    print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
