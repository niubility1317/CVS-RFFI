"""Record independently verified capacity handoff in existing reports and registry."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from experiments.cvs_receiver_residual_capacity4.register import ROOT


def main():
    evidence=ROOT/'local_artifacts/cvs_receiver_residual_capacity4_20261008_r02/verified_readback.json'
    value=json.loads(evidence.read_text(encoding='utf-8'))
    if value['status']!='VERIFIED':raise ValueError('No verified capacity post-state')
    now=datetime.now(timezone.utc).isoformat()
    for run in value['runs']:
        rid=run['run_id'];folder=ROOT/'automation_reports/CV-SincNet'/rid
        local=folder/'evidence';local.mkdir(exist_ok=True)
        shutil.copyfile(evidence,local/'capacity4_verified.json')
        path=folder/'experiment.json';spec=json.loads(path.read_text(encoding='utf-8'))
        spec['status']='RUNNING';spec['execution']['active_dispatcher']=run['owner']
        spec['execution']['capacity_evidence_ref']='evidence/capacity4_verified.json'
        if rid.endswith('r02'):
            spec['execution']['completion_sync']=dict(pid=60084,owner_ref='local_artifacts/cvs_receiver_residual_v2_20261008_r02/completion_sync_owner_capacity4.json',
                previous_pid=62900,previous_exit_verified=True,status='RUNNING_VERIFIED',no_duplicate_observer=True)
        path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        text=(folder/'report.md').read_text(encoding='utf-8').replace('当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）',
            '当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）')
        text+='\n\n## 容量接管VERIFIED\n\n'
        text+=f"独立/proc身份与nvidia-smi、完成产物核实：新dispatcher PID{run['owner']['pid']}，控制commit`{run['owner']['control_commit']}`；模型运行commit仍为`{run['owner']['worker_release_commit']}`。保留{len(run['preserved_workers'])}个活跃训练PID/start_ticks/CWD/argv；首seed4行已自然完成E200，未重跑。目前12行在训、0行排队，无失败。\n\n"
        text+='GPU0至3各3个总实验进程，GPU4至7各4个；其中本轮R1/R2共24个，另4个既有任务。每卡已设最多4个，含预CUDA预约；显存余量约19至21GB。固定矩阵剩余任务已全部启动，因此不补开额外实验。证据：[capacity4_verified.json](evidence/capacity4_verified.json)。\n\n'
        text+='新调度器使用专用handoff锁及全生命周期owner锁；跨run共享GPU预约锁。原训练PID不signal，不热改模型，不改变预算/损失/数据角色。完成16源模型后继续原冻结→112预测→独立truth-last评分。当前未完成目标测试，尚无新增机制收益结论。\n'
        if rid.endswith('r02'):
            text+='\n本地旧observer PID62900退出已独立核实；唯一新observer PID60084已读回正确的新dispatcher，日志增长且stderr为空。其任务仍为结果回收、报告/登记/索引更新及Git交付，不调参、不重跑。\n'
        (folder/'report.md').write_text(text,encoding='utf-8',newline='\n')
        subprocess.run([sys.executable,'-X','utf8','E:/type10-7/tools/experiment_registry.py','--root',str(ROOT),
            'record',rid,'--status','RUNNING','--evidence','evidence/capacity4_verified.json',
            '--note','VERIFIED capacity4 scheduling handoff; 12 source rows active,4 E200 complete,no pending/no failure; worker releases fixed; test not complete'],check=True)
    subprocess.run([sys.executable,'-X','utf8','E:/type10-7/tools/experiment_registry.py','--root',str(ROOT),'build','--managed-only'],check=True)
    from experiments.cvs_receiver_residual_v2 import finalize_local as sync
    from experiments.cvs_receiver_residual import design as r1
    sync.mirror_root()
    sync.d=r1;sync.REPORT=ROOT/'automation_reports/CV-SincNet'/r1.RUN
    sync.mirror_root()
    print('Both reports/registry/root mirrors updated; training and test completion remain pending')


if __name__=='__main__':main()
