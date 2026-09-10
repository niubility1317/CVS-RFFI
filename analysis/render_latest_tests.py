import csv,json,math
from pathlib import Path

base=Path(__file__).parent
d=json.loads((base/'a1_latest_tests_20260910_1623.json').read_text(encoding='utf-8'))
health=json.loads((base/'a1_latest_tests_health.json').read_text(encoding='utf-8'))
scenes=('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
rows=[];classes=[]
for name,run in d['runs'].items():
    for entry in run['scores']:
        score=entry['score'];scope=entry['scope']
        assert entry['predictions_exist'] and score['record_count']==672000
        assert set(score['metrics'])==set(scenes) and scope['feeds_training'] is False
        item={'run':name,'row':entry['row'],'epoch':entry['epoch'],'records':score['record_count']}
        for scene in scenes:
            v=score['metrics'][scene]
            assert v['total']==168000 and 0<=v['correct']<=v['total']
            assert math.isfinite(v['accuracy']) and abs(v['accuracy']-v['correct']/v['total'])<1e-12
            item[scene]=v['accuracy']*100
            for cls,value in v['per_class_accuracy'].items():
                assert math.isfinite(value) and 0<=value<=1
                classes.append({'run':name,'row':entry['row'],'epoch':entry['epoch'],'scene':scene,'class':cls,'accuracy_pct':value*100})
        item['leo_mean']=sum(item[s] for s in scenes[1:])/3
        rows.append(item)
latest={}
for row in sorted(rows,key=lambda x:x['epoch']):latest[(row['run'],row['row'])]=row
latest_rows=sorted(latest.values(),key=lambda x:x['row'])
lora=[]
for section in ('single_domain_calibration_4x4','multiple_configuration_calibration'):
    for r in d['lora_v7'][section]:
        assert r['decisions']==39060 and abs(r['accuracy']-r['correct_decisions']/r['decisions'])<1e-7
        lora.append({'calibration':r.get('calibration_configuration','multiple'),'test':r['test_configuration'],
            'accuracy_pct':r['accuracy']*100,'correct':r['correct_decisions'],'decisions':r['decisions']})
assert len(lora)==20
folder=base/'latest_tests_20260910_1623';folder.mkdir(exist_ok=True)
for name,data in [('latest_target_scores.csv',latest_rows),('all_target_scores.csv',rows),('all_target_per_class.csv',classes),('lora_v7_all_20_tests.csv',lora)]:
    with (folder/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
def table(data):
    out=['|实验|测试轮次|clean|clear|low-elev|rain|LEO均值|','|---|---:|---:|---:|---:|---:|---:|']
    for r in data:out.append('|'+r['row']+'|E'+str(r['epoch'])+'|'+'|'.join(f'{r[k]:.2f}' for k in (*scenes,'leo_mean'))+'|')
    return '\n'.join(out)
text=f'''# 最新目标测试结果

采集时间：{d['captured_at']}，香港时间UTC+8。准确率均为百分比。

## 每行最新完成的测试

{table(latest_rows)}

每次测试四场景各168000条，共672000条。全部{len(rows)}份已完成评分的场景、记录数、correct/total计算和prediction文件存在性检查PASS；{len(classes)}条逐类准确率已导出。只展示固定测试点的最新值，未选择各自最高分；各行训练预算尚未完成、轮次不同，不据此确定方法排名或晋级。全部scope声明feeds_training=false，本次未据目标分数改变训练。

## 暂无目标测试

|实验|已完成训练轮次|原因|
|---|---:|---|
'''
for name,run in health['runs'].items():
    for key,r in run['rows'].items():
        if (name,key) in latest:continue
        epoch=int((r['last'] or {}).get('epoch',0))
        if r['active']:
            argv=r['argv'];start=argv[argv.index('--a1_periodic_target_start')+1]
            why='尚未到首测E'+start
        else:why='训练失败，未到首测点'
        text+=f'|{key}|{epoch}/{r["total"]}|{why}|\n'
text+='''
## 同轮次E160参考

只列已完成E160的行，避免把不同轮次混为同训练预算。目标观察不反馈选模。

'''+table([r for r in rows if r['epoch']==160])+'''

## 独立LoRa v7测试

以下为独立LoRa数据集，保留其校准配置，不与WiSig/LEO横向比较。状态PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS；本次核对20项评分的计数与准确率，不重新声称完整论文复现。

|校准配置|测试配置|准确率%|正确/总数|
|---|---|---:|---:|
'''
for r in lora:text+=f'|{r["calibration"]}|{r["test"]}|{r["accuracy_pct"]:.2f}|{r["correct"]}/{r["decisions"]}|\n'
text+=f'\n证据范围：当前两批共19行的所有已完成固定期次评分，覆盖{len(rows)}份；完整读取各训练stdout和metrics_epoch.jsonl用于最新进度与异常，未将尾部数据当作全程收敛分析。本次未读取目标truth或修改远端。原始评分见a1_latest_tests_20260910_1623.json。\n'
(folder/'report.md').write_text(text,encoding='utf-8')
dest=Path('E:/type10-7/automation_reports/CV-SincNet/latest_tests_20260910_1623');dest.mkdir(exist_ok=True)
for p in folder.iterdir():
    (dest/p.name).write_bytes(p.read_bytes())
    assert (dest/p.name).read_bytes()==p.read_bytes()
print(table(latest_rows))
print('PASS',len(rows),'scores',len(classes),'class rows',len(lora),'LoRa scores')
print('LoRa concise',[(r['calibration'],r['test'],round(r['accuracy_pct'],2)) for r in lora if r['calibration']==r['test'] or r['calibration']=='multiple'])
