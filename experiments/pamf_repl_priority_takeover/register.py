"""Record the dispatcher-only takeover; no new scientific rows are created."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from .publish import RUN,ORIGINAL


def make(template):
    value=deepcopy(template)
    value.update(run_id=RUN,display_name='现有浅层与 KL 队列扩卡接管',
        description='只替换本轮调度控制器；继续原 20 行，不重启训练或重复预测。',
        parent_run_ids=[ORIGINAL],replaces_run_id=None,
        authorization='2026-11-17 用户要求实验不排队、直接启动；保留现有训练并均匀使用显卡。')
    value['code'].update(commit='takeover runtime.json code_commit; existing workers retain original release commit',
        cwd='PROJECT/releases/'+RUN+'-release')
    value['execution'].update(launch_owner='codex-'+RUN,
        gpu_policy='GPU 0,1,2,3,5,7, disjoint from active low-weight owner GPU 4,6; all-owner count <=5, <=2 priority workers/GPU, >=12 GiB free before admission',
        remote_run_root='PROJECT/runs/'+RUN,remote_log_root='original priority run/logs',
        local_artifact_root='local_artifacts/pamf_priority_takeover_20261117',
        launch_command='python -m experiments.pamf_repl_priority_takeover.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG',
        stop_rule='Stop new dispatch on protocol, duplicate job, missing completion, nonfinite or technical failure; preserve all original workers and artifacts.')
    value['expected_artifacts']=['takeover.json with adopted worker identities','updated original priority pipeline_state.json',
        'original priority full source, frozen U and clean prediction/score artifacts','takeover complete.json']
    value['notes']=['No new candidates or model changes; use the exact original immutable worker release, configs and output roots.',
        'Before takeover stop only the exact original dispatcher after confirming its process, children, state and queued logs; do not signal workers.',
        'Adopted worker exit status is unavailable after reparenting; require its original expected completion artifact.',
        'One new dispatcher owner writes original state; old dispatcher must be absent before adoption.',
        'Previously exposed target remains exploratory. U truth only by frozen independent scorer.']
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();data=json.loads(Path(a.template).read_text(encoding='utf-8-sig'))
    Path(a.output).write_text(json.dumps(make(data),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
