"""Validate every saved score and render all long-budget test details."""
import csv,gzip,json,math,shutil
from pathlib import Path
OUT=Path(__file__).parent/'long_training_tests_20260913'
d=json.load(gzip.open(OUT/'evidence.json.gz','rt',encoding='utf-8'))
scenes=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
all_scores=[];classes=[];curves=[];status=[]
for run,v in d['runs'].items():
 for row,r in v['rows'].items():
  epochs=[x['epoch'] for x in r['records']]
  assert epochs==list(range(1,len(epochs)+1))
  status.append({'run':run,'row':row,'completed_epochs':len(epochs),'budget':int(r['configured']['options']['--epochs']),
   'alive':r['alive'],'final_checkpoint':r['final_checkpoint'],'test_points':len(r['scores'])})
  for x in r['records']:
   curves.append({'run':run,'row':row,'epoch':x['epoch'],'source_clean':x.get('val_tx_acc'),
    'source_leo':x.get('stage_source_val_sat_mean_tx'),'train_loss':x.get('train_loss')})
  for e in r['scores']:
   s=e['score'];scope=e['scope'];m=s['metrics']
   assert s['record_count']==672000 and e['prediction_bytes']>0
   assert scope and scope['record_count']==672000 and scope['feeds_training'] is False
   assert set(m)==set(scenes)
   item={'run':run,'row':row,'epoch':e['epoch']}
   for scene in scenes:
    z=m[scene];a=z['per_class_accuracy']
    assert z['total']==168000 and set(a)==set(map(str,range(6)))
    assert math.isclose(z['accuracy'],z['correct']/z['total'],abs_tol=1e-12)
    assert math.isclose(z['accuracy'],sum(a.values())/6,abs_tol=1e-12)
    assert all(math.isfinite(x) and 0<=x<=1 for x in a.values())
    item[scene]=z['accuracy']*100;item[scene+'_correct']=z['correct']
    for label,acc in a.items():classes.append({'run':run,'row':row,'epoch':e['epoch'],'scene':scene,'class':label,'accuracy_pct':acc*100})
   item['leo_mean']=sum(item[s] for s in scenes[1:])/3
   item['leo_scene_floor']=min(item[s] for s in scenes[1:])
   all_scores.append(item)
  if r['final_checkpoint']:
   assert len(epochs)==400 and [e['epoch'] for e in r['scores']]==list(range(200,401,20))
assert len(all_scores)==22 and len(classes)==528 and len(curves)==1059
for name,data in [('all_test_points.csv',all_scores),('per_class_all_test_points.csv',classes),('source_training_curves.csv',curves),('run_status.csv',status)]:
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def table(rows,columns):
 lines=['|'+'|'.join(label for key,label in columns)+'|','|'+'|'.join('---' for _ in columns)+'|']
 for r in rows:lines.append('|'+'|'.join(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]) for k,label in columns)+'|')
 return '\n'.join(lines)
columns=[('epoch','测试epoch')]+list(zip(scenes,['Clean','LEO-clear','LEO-low','LEO-rain']))+[('leo_mean','LEO均值')]
finals=[x for x in all_scores if x['epoch']==400]
text=f'''# 长训练实验完整测试结果

远端采集时间：{d['at']}。准确率为百分比，差值为百分点。范围是本任务两行E400、旧X2_E600和两行FP32 E600；只读采集，没有重训、重新评分或读取truth。

## 最终测试

两行E400均完成400轮；最终checkpoint存在，各有11个完整评分点。批级state为FAILED是因为同批X2_E600失败，不应把两行已完成的E400也标为失败。

'''+table(finals,[('row','实验')]+columns)+'''

X2使用跨接收机clean+LEO ECRS，R3_CLEAN_RX使用R3和clean跨接收机ECRS。两者不是仅一个开关的消融，差值不能单独归因于R3或LEO-ECRS。

## 全部固定测试点

E200为400轮预算中的中间点，参考课程时钟约在100；它不是独立200轮预算的训练终点。下列E200→E400变化同时伴随训练日程推进，不能当作唯一“多训练200轮”的因果增益。

'''
summary={}
for final in finals:
 row=final['row'];rs=[x for x in all_scores if x['row']==row];start=rs[0];best=max(rs,key=lambda x:x['leo_mean'])
 summary[row]={'final':final,'start':start,'observed_best':best,'clean_change':final['clean']-start['clean'],'leo_change':final['leo_mean']-start['leo_mean']}
 text+='### '+row+'\n\n'+table(rs,columns)+'\n\n'
 text+=f"E200→E400：Clean变化{final['clean']-start['clean']:+.4f}个百分点，LEO均值变化{final['leo_mean']-start['leo_mean']:+.4f}个百分点。观测LEO最高点为E{best['epoch']}，{best['leo_mean']:.4f}%；仅描述曲线，不以此重新选择checkpoint。\n\n"
text+='## 最终逐类别结果\n\n类别编号使用scorer的0–5标签，不擅自映射为物理TX编号。\n\n'
finalclasses=[]
for final in finals:
 row=final['row'];rs=[]
 for label in map(str,range(6)):
  r={'row':row,'class':label}
  for s in scenes:r[s]=next(c['accuracy_pct'] for c in classes if c['row']==row and c['epoch']==400 and c['scene']==s and c['class']==label)
  r['leo_mean']=sum(r[s] for s in scenes[1:])/3;rs.append(r);finalclasses.append(r)
 text+='### '+row+'\n\n'+table(rs,[('class','类别')]+columns[1:])+'\n\n'
text+='## 同轮次最终差异\n\n'
a,b=finals
text+=f"{a['row']}减{b['row']}："+'，'.join(f'{s}={a[s]-b[s]:+.4f}个百分点' for s in scenes+['leo_mean'])+'。\n\n'
for row in summary:
 fc=[c for c in finalclasses if c['row']==row];weak=min(fc,key=lambda c:c['leo_mean']);strong=max(fc,key=lambda c:c['leo_mean'])
 text+=f"{row}最弱类别是{weak['class']}，LEO均值{weak['leo_mean']:.4f}%；最强类别是{strong['class']}，{strong['leo_mean']:.4f}%。\n\n"
text+='## 源域与目标域差距\n\n'
for final in finals:
 r=d['runs'][final['run']]['rows'][final['row']]['records'][-1]
 text+=f"{final['row']}最终source Clean={r['val_tx_acc']:.4f}%，source LEO={r['stage_source_val_sat_mean_tx']:.4f}%；对应target Clean={final['clean']:.4f}%，target LEO={final['leo_mean']:.4f}%，差距分别为{r['val_tx_acc']-final['clean']:.4f}和{r['stage_source_val_sat_mean_tx']-final['leo_mean']:.4f}个百分点。延长训练仍未消除跨接收机泛化差距。\n\n"
text+='''## E600失败与缺项

- 旧X2_E600完成259轮，E260/batch133触发RC4_SYSTEMIC_NONFINITE_BATCH_GUARD；无最终checkpoint，未到预定首测E300，没有目标测试分数。E259的16.6667%是source指标，不能填写到测试结果表。
- B1_FP32_E600和B1_CLEAN_ECRS_FP32_E600均未开始正式训练；远端入口检查因MKL_THREADING_LAYER=INTEL与libgomp冲突失败。没有目标测试结果，不能据此判断FP32长期效果。

## 验证与解释边界

完整解析1059条训练epoch记录，并扫描三行完整训练stdout；两行E400没有匹配到Traceback/RuntimeError/OOM/Killed，旧E600有系统性非有限梯度保护错误。两行新E600的失败位于dispatcher/入口检查，未产生正式训练日志。

全部22份score逐项核对：四场景集合、每场景168000条、合计672000条、correct/total准确率、六类均值与总体一致、prediction非空、scope的feeds_training=false及完整预定测试点。它们是同一测试集的重复测量，14784000次累计决策不代表14784000个独立样本。本次没有重新逐条解析约GB级prediction或重新读取truth；验证范围是既有scorer计数、评分内部一致性及prediction存在性，不声称重新独立评分。

现有score只有场景总分和逐类准确率，未提供逐接收机、逐日期分解或混淆矩阵，本报告不推算这些字段。两行均为单seed392005、从零初始化的既有探索实验；多次target观察不构成独立确认，不用最高测试点替代预定最终E400，不据这些分数触发调参、选模或重跑。

## 可下载数据

- [22个测试点及正确数量](all_test_points.csv)
- [全部528条逐类测试准确率](per_class_all_test_points.csv)
- [1059轮源域训练曲线](source_training_curves.csv)
- [五行状态](run_status.csv)
- [只读采集证据](evidence.json.gz)
'''
(OUT/'report.md').write_text(text,encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps({'at':d['at'],'summary':summary,'final_classes':finalclasses,'validation':'PASS'},indent=2),encoding='utf-8')
dest=Path('E:/type10-7/automation_reports/CV-SincNet/long_training_tests_20260913');dest.mkdir(exist_ok=False)
for p in OUT.iterdir():
 shutil.copy2(p,dest/p.name)
 assert p.read_bytes()==(dest/p.name).read_bytes()
print(table(finals,[('row','实验')]+columns))
print(json.dumps(summary,ensure_ascii=False))
print(table(finalclasses,[('row','实验'),('class','类别')]+columns[1:]))
print('VERIFIED 22 scores, 528 class scores, 1059 epochs; mirrored',dest)
