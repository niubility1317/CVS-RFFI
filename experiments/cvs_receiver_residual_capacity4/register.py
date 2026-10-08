"""Record the explicit user capacity override in the two existing runs."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ('20261008-phase1-receiver-residual-manysig-m16-r01', '20261008-phase1-receiver-residual-manysig-m16-r02')


def main():
    now = datetime.now(timezone.utc).isoformat()
    for run in RUNS:
        folder = ROOT/'automation_reports/CV-SincNet'/run
        path = folder/'experiment.json';record=json.loads(path.read_text(encoding='utf-8'))
        execution=record['execution']
        if 'gpu_policy_original' in execution: raise ValueError('Override already registered')
        execution['gpu_policy_original']=execution['gpu_policy']
        execution['gpu_policy']='user override: <=4 total experiment processes/GPU including CUDA and pre-CUDA reservations; max16 own active rows; >=12GB free; preserve live training workers'
        execution['capacity_override']=dict(authorization='用户：每张卡4个实验进程',at=now,per_gpu_limit=4,
            controller_release='cvs_receiver_residual_capacity4_20261008_r01',
            handoff='replace only owned dispatcher parent; adopt exact PID/start_ticks/cwd/argv; do not signal training workers',
            worker_release_immutable=True,algorithm_or_budget_change=False)
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        with (folder/'events.jsonl').open('a',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(dict(event='CAPACITY_OVERRIDE_REGISTERED',status='RUNNING',at=now,run_id=run,
                authorization='用户：每张卡4个实验进程',per_gpu_limit=4,scope='scheduler only; preserve trained/completed/live rows'),ensure_ascii=False)+'\n')
        with (folder/'report.md').open('a',encoding='utf-8',newline='\n') as f:
            f.write('\n## 每卡4进程调度覆盖\n\n用户新增指示“每张卡4个实验进程”，将本轮R1/R2总任务上限从每卡2个覆盖为每卡4个，包括预CUDA预约。原启动记录和策略保留为历史。新控制release只替换本轮调度父进程，按PID/start_ticks/CWD/argv接管训练子进程；已完成行跳过，失败行保留且不自动重跑。每run单launch owner，两个调度器共享GPU预约锁。训练模型/config及预算继续使用原不可变release。source矩阵全部冻结后再预测，全部预测完成后truth-last独立评分。此处为预登记，实际接管和占用须由独立读回确认。\n')
    print('Registered two existing runs; no new algorithm rows')


if __name__ == '__main__':main()
