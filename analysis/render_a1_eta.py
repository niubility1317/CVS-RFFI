import csv,json,statistics,subprocess
from datetime import datetime,timedelta
from pathlib import Path

base=Path(__file__).parent
source=base/'a1_eta_latest.json'
snapshot=base/'a1_eta_20260910_1618.json'
if source.exists() and not snapshot.exists():source.rename(snapshot)
d=json.loads(snapshot.read_text(encoding='utf-8'))
now=datetime.strptime(d['time'].rsplit(' ',1)[0],'%Y-%m-%d %H:%M:%S')
rows=[]
for run_name,run in d['runs'].items():
    for name,row in run['rows'].items():
        last=row['last'] or {};epoch=int(last.get('epoch',0));total=row['total']
        result={'row':name,'run':run_name,'epoch':epoch,'total':total,'gpu':row['gpu'],'status':row['state']['status']}
        if row['active']:
            recent=row['recent']
            seconds=statistics.median(x['epoch_time_s']-x.get('train_time_periodic_target_seconds',0) for x in recent)
            argv=row['argv'];opts={k:argv[i+1] for i,k in enumerate(argv[:-1]) if k.startswith('--')}
            start=int(opts.get('--a1_periodic_target_start',0));interval=int(opts.get('--a1_periodic_target_interval',10))
            plan=[e for e in range(start,total+1,interval)] if start else []
            if start and total not in plan:plan.append(total)
            scored={x['epoch'] for x in row['scores']}
            remaining_tests=len(set(plan)-scored)
            observed=[x['scope']['seconds'] for x in row['scores']]
            # Same four scenarios; target coverage is 168000 vs source V 27000 per scenario.
            evaluation=(statistics.median(observed[-3:]) if observed else statistics.median(
                x.get('train_muse_time_base_validation_s',0)+x.get('train_muse_time_heavy_source_validation_s',0)
                for x in recent)*168000/27000)
            train_seconds=max(0,(total-epoch)*seconds-min(row['log_age_seconds'],seconds))
            hours=(train_seconds+remaining_tests*evaluation)/3600
            lo=hours*.8;hi=hours*(1.4 if name=='G1_FISHER_GATE' else 1.25)
            fmt=lambda h:(now+timedelta(hours=h)).strftime('%m-%d %H:%M')
            result.update(base_epoch_seconds=seconds,remaining_tests=remaining_tests,
                test_seconds=evaluation,test_estimated=not bool(observed),remaining_hours=hours,
                estimated_end=fmt(hours),earliest=fmt(lo),latest=fmt(hi))
        rows.append(result)

remote="""import json,datetime
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/tweak_config2_portability_20260910_v7/official_config2_full/results.json')
d=json.loads(p.read_text())
print(json.dumps({'exists':p.exists(),'modified':datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),'keys':list(d),'status':d.get('status'),'pid_alive':Path('/proc/852891').exists()}))
"""
extra=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=remote,text=True,encoding='utf-8',timeout=20))
result={'captured_at':d['time'],'rows':rows,'external_lora':extra}
(base/'a1_eta_20260910_1618_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 全部活动实验预计结束时间','',f"快照：{d['time']}，全部时间为香港时间UTC+8。",'',
'当前三行延长实验全部运行，原16行中13行运行、3行失败，无排队。以下预计包含剩余固定周期测试。','']
for title,selected in [('延长训练',[r for r in rows if 'extended' in r['run']]),('200轮机制矩阵',[r for r in rows if 'periodic' in r['run']])]:
    lines.extend(['## '+title,'','|实验|已完成/总轮数|剩余小时（中心）|预计结束（中心）|估计结束范围|','|---|---:|---:|---|---|'])
    for r in selected:
        if 'remaining_hours' in r:
            lines.append(f"|{r['row']}|{r['epoch']}/{r['total']}|{r['remaining_hours']:.1f}|{r['estimated_end']}|{r['earliest']}—{r['latest']}|")
        else:lines.append(f"|{r['row']}|{r['epoch']}/{r['total']}|失败|无有效ETA|未重启|")
    lines.append('')
lines += ['## 计算与边界','',
'每行采用最近10轮的epoch_time_s中位数，扣除周期测试耗时，再乘剩余轮数；扣除当前轮已过去时间（最多一轮）。剩余测试次数来自实际argv计划与已完成evaluation_scope逐项差集。已有测试用最近3次中位耗时；尚无目标测试的3行延长实验及G1，以源域四场景验证耗时按168000/27000样本量比例估算，不假称已实测目标测试耗时。',
'范围一般为中心剩余时间的0.8—1.25倍；G1因阶段较早及重评估成本不确定，采用0.8—1.4倍。属于工程预估，不是保证或统计置信区间；共享GPU、阶段变化和健康任务陆续完成都会改变速度。训练完成后固定目标评分也需闭合，3个失败实验没有结束时间。','',
f"其他GPU任务LoRa v7：原PID852891已不存在，results.json已产生，mtime为{extra['modified']}，status={extra['status']}。这是已完成产物时间，不是未来ETA；本次未审计它的全部评分正确性。",'',
'只读刷新所有当前row的完整训练stdout和metrics_epoch.jsonl用于状态、错误及近期耗时；未读取目标truth或修改运行。原始证据a1_eta_20260910_1618.json，估计明细a1_eta_20260910_1618_summary.json。']
text='\n'.join(lines)+'\n'
report=base/'A1_ALL_ETA_20260910_1618.md';report.write_text(text,encoding='utf-8')
dest=Path('E:/type10-7/automation_reports/CV-SincNet/a1_extended_s392005_20260910_r1/eta_20260910_1618.md');dest.write_bytes(report.read_bytes())
print(text)
