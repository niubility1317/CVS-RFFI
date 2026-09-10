import csv,json
from pathlib import Path
base=Path(__file__).parent
d=json.loads((base/'source_classes_results.json').read_text(encoding='utf-8'))
target=json.loads((base/'completed_experiments_20260910_2122.json').read_text(encoding='utf-8'))
rows=[];all_rows=[];deltas=[]
for name,r in d['rows'].items():
    if name!='B1_TAIL_LR':continue
    assert r['model_unchanged']
    out={'row':name}
    for scene,stats in r['scenarios'].items():
        assert stats['aggregate']['tx_total']==27000
        deltas.append({'row':name,'scene':scene,'delta_correct_vs_saved':stats['delta_correct_vs_saved']})
        for cls,v in stats['per_class'].items():
            assert v['total']==4500 and abs(v['accuracy_pct']-100*v['correct']/v['total'])<1e-10
            all_rows.append({'row':name,'scene':scene,'class_index':cls,**v})
    for cls in ('1','3'):
        out['class'+cls+'_source_clean']=r['scenarios']['clean']['per_class'][cls]['accuracy_pct']
        out['class'+cls+'_source_leo_mean']=sum(r['scenarios'][s]['per_class'][cls]['accuracy_pct'] for s in ('leo_clear_weak','leo_low_elev_weak','leo_rain_weak'))/3
        out['class'+cls+'_target_leo_mean']=sum(target['rows'][name]['scores'][-1]['score']['metrics'][s]['per_class_accuracy'][cls]*100 for s in ('leo_clear_weak','leo_low_elev_weak','leo_rain_weak'))/3
    rows.append(out)
folder=base/'source_classes_20260910';folder.mkdir(exist_ok=True)
for filename,data in [('class1_class3_summary.csv',rows),('all_source_class_counts.csv',all_rows),('replay_deltas.csv',deltas)]:
    with (folder/filename).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
table=['|实验|类1 source clean|类1 source LEO均值|类3 source clean|类3 source LEO均值|','|---|---:|---:|---:|---:|']
for r in rows:table.append('|'+r['row']+'|'+'|'.join(f'{r[k]:.2f}' for k in ('class1_source_clean','class1_source_leo_mean','class3_source_clean','class3_source_leo_mean'))+'|')
max_delta=max(abs(x['delta_correct_vs_saved']) for x in deltas)
detail=['|场景|类1准确率%|类3准确率%|','|---|---:|---:|']
for scene,stats in d['rows']['B1_TAIL_LR']['scenarios'].items():
    detail.append('|'+scene+'|'+f"{stats['per_class']['1']['accuracy_pct']:.2f}|{stats['per_class']['3']['accuracy_pct']:.2f}|" )
text='''# 类1和类3的source准确率

本报告按用户最新要求仅选取B1_TAIL_LR。source指原源域验证集V，不是L_s训练集。冻结E200模型，在原source RX/day、划分与评估batch配置下重新评估；每场景27000条、每类4500条，采用原评估器、原LEO随机seed。准确率单位%。

'''+ '\n'.join(table)+'\n\n'+'\n'.join(detail)+f'''

全部模型评估前后state_dict逐张量一致，未创建optimizer或更新参数，未构造目标loader。所有逐类correct/total与总体计数一致。原日志只保留总体指标，故本表为新增冻结评估数据。与原存档总体correct最大差异为{max_delta}条/27000（{100*max_delta/27000:.5f}个百分点）；所有差异逐场景保存在replay_deltas.csv。观察到小量重算差异，未将其强行归零，也未断言唯一数值原因。

第一次检查要求总体correct完全相同，在B0 LEO-clear相差1条时退出；第二次仍要求clean完全一致，在B1 clean相差1条时退出。两次只读评估日志保留。最终评估将新旧统计差异完整记录，以实际计数给出逐类结果；不把逐bit分数一致增设为读取源域准确率的条件。

最终脚本提交e85e8018ef6df174fb45f11aff4d9de9bb93a7d6，发布releases/a1_source_classes_e85e8018，单文件SHA256=0b9b1f905c3bfc1b302cda5fb91509867f8753879b64fdbafb748b49459ede97；使用空闲GPU2顺序完成8行，未停止任何原训练任务。

完整四场景×六类计数见all_source_class_counts.csv；source与target的类1/类3均值并列见class1_class3_summary.csv。数据为source V准确率，不代表target泛化性能。
'''
(folder/'report.md').write_text(text,encoding='utf-8')
dest=Path('E:/type10-7/automation_reports/CV-SincNet/source_classes_20260910');dest.mkdir(exist_ok=True)
for p in folder.iterdir():(dest/p.name).write_bytes(p.read_bytes())
print('\n'.join(table));print('maximum replay delta',max_delta);print('rows',len(rows))
