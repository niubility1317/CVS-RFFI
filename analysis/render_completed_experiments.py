import csv,json,math
from pathlib import Path
base=Path(__file__).parent
d=json.loads((base/'completed_experiments_20260910_2122.json').read_text(encoding='utf-8'))
folder=base/'completed_tests_20260910_2122';folder.mkdir(exist_ok=True)
scenes=('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
all_scores=[];all_classes=[];curves=[];final=[];resources=[]
for name,r in d['rows'].items():
    assert r['checkpoint_epoch']==200 and r['e200_snapshot_model_equal_final']
    assert r['checkpoint_args']['from_scratch'] and r['checkpoint_args']['a1_scratch_only']
    assert r['checkpoint_args']['seed']==392005 and not r['errors']
    for e in r['scores']:
        s=e['score']; assert s['record_count']==672000 and set(s['metrics'])==set(scenes)
        assert e['scope']['feeds_training'] is False and e['prediction_bytes']>0
        row={'row':name,'epoch':e['epoch']}
        for scene in scenes:
            m=s['metrics'][scene]
            assert m['total']==168000 and abs(m['accuracy']-m['correct']/m['total'])<1e-12
            row[scene]=m['accuracy']*100
            for cls,v in m['per_class_accuracy'].items():
                assert 0<=v<=1 and math.isfinite(v)
                all_classes.append({'row':name,'epoch':e['epoch'],'scene':scene,'class_index':cls,'accuracy_pct':100*v})
        row['leo_mean']=sum(row[s] for s in scenes[1:])/3
        row['leo_worst_scene']=min(row[s] for s in scenes[1:])
        all_scores.append(row)
        if e['epoch']==200:final.append(dict(row))
    curves.extend({'row':name,**v} for v in r['curve'])
    resources.append({'row':name,'completed_at':r['completed_at'],'epoch_hours':r['epoch_hours'],
        'target_test_hours':r['target_test_hours'],'peak_allocated_MiB':r['peak_mb'],
        'final_source_clean':r['final_source_clean'],'final_source_leo_mean':r['final_source_leo'],
        'final_train_loss':r['final_loss']})
baseline=next(r for r in final if r['row']=='B0_FIXED')
for row in final:
    row['delta_clean_vs_B0_pp']=row['clean']-baseline['clean']
    row['delta_leo_mean_vs_B0_pp']=row['leo_mean']-baseline['leo_mean']
def dump(name,rows):
    with (folder/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
dump('final_e200_scores.csv',final);dump('all_fixed_epoch_scores.csv',all_scores)
dump('final_e200_per_class.csv',[r for r in all_classes if r['epoch']==200]);dump('all_fixed_epoch_per_class.csv',all_classes)
dump('all_training_curves.csv',curves);dump('resources_and_source_validation.csv',resources)
class_mean=[]
for name in d['rows']:
    row={'row':name}
    for i in range(6):
        values=[r['accuracy_pct'] for r in all_classes if r['row']==name and r['epoch']==200 and r['scene'] in scenes[1:] and r['class_index']==str(i)]
        assert len(values)==3
        row['class_'+str(i)]=sum(values)/3
    class_mean.append(row)
dump('final_e200_class_leo_mean.csv',class_mean)
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join('---' for _ in headers)+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
num=lambda v:f'{v:.2f}'
metric_table=table(['实验','clean','clear','low-elev','rain','LEO均值','相对B0的LEO变化/pp'],[
    [r['row']]+[num(r[k]) for k in (*scenes,'leo_mean','delta_leo_mean_vs_B0_pp')] for r in final])
text=f'''# 已完成训练实验的详细测试数据

采集：{d['captured_at']}，香港时间UTC+8。本报告主体为当前a1_mechanism_periodic_s392005_20260910_r1矩阵已经完成的{len(final)}行，不混入仍运行的中途结果或历史不同run的最高分。准确率单位%。

## 完成与来源核验

{len(final)}行均完成E200，训练JSONL与CSV分别完整解析{len(curves)}条记录，每行E1—E200无缺失或重复；全stdout扫描无Traceback、OOM或系统性非有限退出。每行E80、90…200的13次测试均完整，合计{len(all_scores)}份四场景评分。

独立CPU加载最终final_ssdg.pth和固定E200快照，epoch均为200，所有model张量逐项相等；from_scratch/a1_scratch_only为真，baseline/teacher来源为空，seed392005。因此下面固定E200测试对应本行最终模型。全部prediction文件存在，四场景各168000条，总计672000条；评分correct/total一致，scope为不反馈训练的固定epoch观察。未重新读取target truth或重评分。

## 最终E200目标测试

{metric_table}

这些是单seed、固定E200的描述性结果，不能由本次测试反向选模型、调参或提出统计显著性/科学晋级结论。B/D/F行的具体机制与对照关系见原矩阵预登记。

## 六个类别的E200准确率

类别0—5是scorer中的类别索引。当前score.json不提供逐接收机指标，本报告不由汇总值推算接收机结果。
'''
for scene in scenes:
    text+='\n### '+scene+'\n\n'
    text+=table(['实验']+[f'类{i}' for i in range(6)],[
        [name]+[num(next(r['accuracy_pct'] for r in all_classes if r['row']==name and r['epoch']==200 and r['scene']==scene and r['class_index']==str(i))) for i in range(6)] for name in d['rows']])+'\n'
text+='\n## 训练耗时、资源与源域验证\n\n'
text+='各类别的三种LEO准确率算术均值另见final_e200_class_leo_mean.csv。\n\n'
text+=table(['实验','完成时间','累计epoch/小时','其中目标测试/小时','峰值MiB','源V clean','源V LEO均值','最终训练loss'],[
    [r['row'],r['completed_at'][11:19],num(r['epoch_hours']),num(r['target_test_hours']),num(r['peak_allocated_MiB']),
     num(r['final_source_clean']),num(r['final_source_leo_mean']),num(r['final_train_loss'])] for r in resources])
text+='\n\n累计epoch耗时包含训练、验证及同PID周期目标测试，不等于纯GPU计算时间。峰值为日志记录的CUDA内存遥测，可能包含周期预测重建模型的瞬时分配，不用于直接断言某方法训练显存更省。源域V与目标测试分开列示；不同机制的loss量纲/组成可能不同，不按loss数值横向排名。\n'
text+='\n## 全部固定测试点\n\n'
for name in d['rows']:
    text+='\n### '+name+'\n\n'+table(['epoch','clean','clear','low-elev','rain','LEO均值'],[
        [r['epoch']]+[num(r[k]) for k in (*scenes,'leo_mean')] for r in all_scores if r['row']==name])+'\n'
text+='\n## 未纳入完成组\n\n'
for name,r in d['state']['rows'].items():
    if name not in d['rows']:text+=f'- {name}：{r["status"]}。\n'
text+='\n三行400/600轮延长实验也仍在运行，不纳入E200完成组。历史已完成实验与独立LoRa v7属于不同run/数据集，见之前的全量报告；本报告不重复混排。\n'
(folder/'report.md').write_text(text,encoding='utf-8')
dest=Path('E:/type10-7/automation_reports/CV-SincNet/completed_tests_20260910_2122');dest.mkdir(exist_ok=True)
for p in folder.iterdir():(dest/p.name).write_bytes(p.read_bytes())
print(metric_table)
print('VERIFIED rows',len(final),'scores',len(all_scores),'per_class',len(all_classes),'epochs',len(curves))
