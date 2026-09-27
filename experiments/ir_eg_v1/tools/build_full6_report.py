"""Build source-backed tables from independent recount and complete training audit."""
import csv,gzip,json,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta

BASE=Path('E:/type10-7/automation_reports/CV-SincNet')
EVAL='phase1_adv3b02_xuc_full6_eval_s392005_20260914_r1'
RUN='phase1_adv3b02_xuc_full_s392005_20260913_r1'
P=BASE/EVAL
D=json.loads((P/'detailed_recount.json').read_text(encoding='utf-8'))
A=json.loads((BASE/RUN/'full_run_audit.json').read_text(encoding='utf-8'))
OLD=json.loads((BASE/'phase1_adv3b02_xuc_dr_s392005_20260913_r1/full_run_audit.json').read_text(encoding='utf-8'))
LIVE=json.loads((P/'evaluation_readback.json').read_text(encoding='utf-8'))
assert LIVE['state']['status']=='COMPLETE' and all(v['state']['status']=='SCORED' for v in LIVE['rows'].values())
ROWS=['F-A1','F-M14','F-M11','F-M05','F-M08','F-M12']
SC=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
def table(h,rows):return '|'+ '|'.join(h)+'|\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('|'+ '|'.join(map(str,r))+'|\n' for r in rows)
def putcsv(name,rows):
 with (P/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def m(n):return D['rows'][n]['score']['metrics']
def leo(n):return sum(m(n)[s]['accuracy'] for s in SC[1:])*100/3
def group(n,scope,k,scenes):
 v=[r for r in D['rows'][n]['details'] if r['scope']==scope and r['group']==str(k) and r['scene'] in scenes]
 return sum(r['correct'] for r in v)/sum(r['total'] for r in v)*100
def hours(n):return A['full']['rows'][n]['completion']['elapsed_seconds']/3600
summary=[];scenes=[];details=[];pairs=[];classes=[];conf=[];curves=[];mechanism=[]
for n in D['rows']:
 summary.append(dict(row=n,kind='new_completed' if n.startswith('F-') else 'historical_reference',
  **{s:100*m(n)[s]['accuracy'] for s in SC},leo_mean=leo(n),leo_scene_floor=min(m(n)[s]['accuracy'] for s in SC[1:])*100,
  leo_rx_floor=min(group(n,'rx',r,SC[1:]) for r in [0,2,5,7,9,10,11]),leo_tx_floor=min(group(n,'tx',c,SC[1:]) for c in range(6)),
  delta_vs_F_A1_pp=leo(n)-leo('F-A1'),delta_vs_historical_M00_pp=leo(n)-leo('M00')))
 for s in SC:
  v=m(n)[s];mcr=D['rows'][n]['macro'][s]
  scenes.append(dict(row=n,scene=s,correct=v['correct'],total=v['total'],accuracy_percent=v['accuracy']*100,macro_f1_percent=mcr['macro_f1']*100,
   macro_precision_percent=mcr['macro_precision']*100,macro_recall_percent=mcr['macro_recall']*100))
  for c in range(6):classes.append(dict(row=n,scene=s,tx=c,total=28000,recall_percent=mcr['per_class_recall'][c]*100,precision_percent=mcr['per_class_precision'][c]*100,f1_percent=mcr['per_class_f1'][c]*100))
  for i,line in enumerate(D['rows'][n]['confusion'][s]):
   for j,count in enumerate(line):conf.append(dict(row=n,scene=s,true_tx=i,predicted_tx=j,count=count))
 details.extend(dict(row=n,**r) for r in D['rows'][n]['details'])
 pairs.extend(dict(row=n,**r) for r in D['rows'][n]['paired'])
for n,r in A['full']['rows'].items():
 for v in r['logs.jsonl']['records']:
  curves.append(dict(row=n,complete=n in ROWS,epoch=v['epoch'],accepted=v['accepted'],total_step=v['total_step'],mean_loss=v['mean_loss'],elapsed_seconds=v['elapsed_seconds'],source_val_accuracy=v['source_validation']['accuracy']))
 if n not in ROWS:continue
 aa=r['action_audit'];c=aa['counts']
 assert c['records']==c['accepted']==44400 and not aa['bad'] and not aa['gaps']
 assert not any(v for k,v in c.items() if k.startswith('nonfinite') or 'mismatch' in k or 'incomplete' in k)
 assert not any(x['errors'] for x in r['stdout_scan'])
 assert aa['unique_L']==6300 and aa['unique_U']==56700 and c['U_exposures']==11340000
 mechanism.append(dict(row=n,training_hours=hours(n),actions=aa['actions'],field_evaluations=aa['field_evaluations'],counts=c,
  components=(r['activation'] or {}).get('components',{}),anchor=r['anchor'],cstar=r['completion']['cstar']))
putcsv('test_summary.csv',summary);putcsv('scene_metrics.csv',scenes);putcsv('rx_day_tx_metrics.csv',details)
putcsv('per_class_metrics.csv',classes);putcsv('confusion_matrices.csv',conf);putcsv('paired_rescue_harm.csv',pairs);putcsv('training_curves.csv',curves)
(P/'mechanism_evidence.json').write_text(json.dumps(mechanism,ensure_ascii=False,indent=2),encoding='utf-8')
historical=[]
for family in ['old','dr']:
 for n,r in OLD[family]['rows'].items():
  mm=r['score']['metrics'];historical.append(dict(row=n,family=family,**{s:mm[s]['accuracy']*100 for s in SC},leo_mean=sum(mm[s]['accuracy'] for s in SC[1:])*100/3))
historical.extend(dict(row=n,family='full_new',**{s:m(n)[s]['accuracy']*100 for s in SC},leo_mean=leo(n)) for n in ROWS)
putcsv('all_28_scored_comparison.csv',historical)
stamp=datetime.fromtimestamp(LIVE['time'],timezone(timedelta(hours=8))).isoformat()
rank=sorted(ROWS,key=leo,reverse=True)
text='# FULL9已完成六组：详细测试集结果\n\n'
text+=f'评估状态：VERIFIED / ARTIFACTS_COMPLETE。截至{stamp}，六组均完成独立测试；新预测4032000条。原FULL9尚有F-M00/F-M07/F-M13训练中，此处不报告其中间checkpoint测试值。\n\n'
text+=f'已完成六组中，F-A1的LEO平均最高，为{leo("F-A1"):.4f}%；五组完整扩展中F-M11最高，为{leo("F-M11"):.4f}%。五组完整扩展都低于F-A1，降幅{min(leo("F-A1")-leo(n) for n in ROWS[1:]):.4f}–{max(leo("F-A1")-leo(n) for n in ROWS[1:]):.4f}pp。本次尚不支持“扩展全部加入后进一步提高整体LEO泛化”。\n\n'
text+='## 测试口径与完整性\n\n'
text+='ManySig equalized=1、seed392005；源L/U/V=6300/56700/27000，源RX1/3/4/6/8、day1/2/3；目标RX0/2/5/7/9/10/11、day0/1/2/3、TX0–5。每场景168000条，每类每场景28000条，每RX每场景24000条，每组四场景672000条。LEO平均是三场景等权平均，也等于504000条LEO记录合并准确率。所有新评估均加载各自scratch训练的最终E200/44400步checkpoint，未选择源V最好epoch或目标最好epoch。\n\n'
text+='全部六组预测先固定，再由独立scorer连接truth；另逐条重读六组及历史M00/M09/M10共6048000条预测，四场景无重复或缺失，独立重算正确数与原score完全一致。RX/day/TX映射按构建器物理ID重建，168000个opaque ID集合与truth逐ID精确一致。未重新读取目标IQ来分组，也未使用结果改变其余训练。\n\n'
text+='本评估为既有研究benchmark的描述性测试，单seed；不据样本条数夸大跨seed或跨接收机统计显著性。输出为六类闭集预测，未定义拒识阈值；unknown/rejection coverage/accepted-only accuracy不适用，不伪造NLL/ECE等未保存完整概率所需指标。\n\n'
text+='## 四场景总表（%）\n\n'
text+=table(['行','Clean','LEO晴空','LEO低仰角','LEO雨衰','LEO平均','相对F-A1(pp)'],[[n,*[f'{m(n)[s]["accuracy"]*100:.4f}' for s in SC],f'{leo(n):.4f}',f'{leo(n)-leo("F-A1"):+.4f}'] for n in rank])
text+='\nF-A1为修正GRL后的CORE90承载A1目标参考；F-M11为完整DR+C2；F-M08为完整DR+X+Ux+C2；F-M05为完整DR+X+Ux；F-M14为完整DR+X+Ux+曝光匹配课程；F-M12为完整DR+X+Ux+C*+曝光匹配课程。\n\n'
text+='历史同数据对照如下。M00是旧49步/轮CORE90，不能代替尚未完成的匹配222步F-M00；M09/M10是原生训练器及原生A1启用项，不能称作本次完整扩展。\n\n'
text+=table(['历史行','Clean','晴空','低仰角','雨衰','LEO平均'],[[n,*[f'{m(n)[s]["accuracy"]*100:.4f}' for s in SC],f'{leo(n):.4f}'] for n in ['M00','M09','M10']])
text+='\n## 正确数、Macro-F1及低端表现\n\n'
text+=table(['行','Clean正确/168000','LEO正确/504000','Clean Macro-F1','LEO Macro-F1场景均值','最差LEO场景','最差RX的LEO均值','最差TX的LEO均值'],[[n,m(n)['clean']['correct'],sum(m(n)[s]['correct'] for s in SC[1:]),f'{D["rows"][n]["macro"]["clean"]["macro_f1"]*100:.4f}',f'{sum(D["rows"][n]["macro"][s]["macro_f1"] for s in SC[1:])*100/3:.4f}',f'{min(m(n)[s]["accuracy"] for s in SC[1:])*100:.4f}',f'{min(group(n,"rx",r,SC[1:]) for r in [0,2,5,7,9,10,11]):.4f}',f'{min(group(n,"tx",c,SC[1:]) for c in range(6)):.4f}'] for n in rank])
text+='\nMacro-F1由6×6混淆矩阵逐类precision/recall计算；LEO列为三个场景Macro-F1均值。由于各类等量，macro-recall与accuracy相等。完整每场景正确数、precision、recall、F1见scene_metrics.csv及per_class_metrics.csv。\n\n'
for label,ss in [('Clean',SC[:1]),('LEO三场景平均',SC[1:])]:
 text+=f'## TX细分：{label}（%）\n\n'
 text+=table(['行',*[f'TX{i}' for i in range(6)]],[[n,*[f'{group(n,"tx",c,ss):.3f}' for c in range(6)]] for n in rank])
 text+=f'\n## RX细分：{label}（%）\n\n'
 text+=table(['行',*[f'RX{i}' for i in [0,2,5,7,9,10,11]]],[[n,*[f'{group(n,"rx",c,ss):.3f}' for c in [0,2,5,7,9,10,11]]] for n in rank])
text+='\n## 日期细分：LEO三场景平均（%）\n\n'
text+=table(['行','day0','day1','day2','day3'],[[n,*[f'{group(n,"day",c,SC[1:]):.3f}' for c in range(4)]] for n in rank])
text+='\n所有RX×TX、RX×day的四场景正确数/分母/准确率保存在rx_day_tx_metrics.csv，全部混淆矩阵保存在confusion_matrices.csv。TX编号是数据集索引，不代表物理难度排序。\n\n'
text+='## 相对同批F-A1的错误转移\n\n'
def paired_total(n,b,key):return sum(v[key] for v in D['rows'][n]['paired'] if v['baseline']==b and v['scene'] in SC[1:])
text+=table(['完整扩展行','救回F-A1错误','损坏F-A1正确','净增正确','LEO变化(pp)'],[[n,paired_total(n,'F-A1','rescue'),paired_total(n,'F-A1','harm'),paired_total(n,'F-A1','net_correct'),f'{leo(n)-leo("F-A1"):+.4f}'] for n in rank if n!='F-A1'])
text+='\n分母为504000条LEO记录；救回与损坏均按同一(scene,opaque ID)逐样本配对。全部场景及相对历史M00的配对计数见paired_rescue_harm.csv。各扩展组即使能救回部分错误，损坏数量仍更多。\n\n'
text+='## 历史父行与旧DR7对照\n\n'
def oldleo(f,n):return sum(OLD[f]['rows'][n]['score']['metrics'][s]['accuracy'] for s in SC[1:])*100/3
text+=table(['新行','原XUC父行LEO','旧DR7 LE0','当前LEO','比旧DR7(pp)','比原父行(pp)'],[[n,f'{oldleo("old",n[2:]):.4f}',f'{oldleo("dr","DR-"+n[2:]):.4f}',f'{leo(n):.4f}',f'{leo(n)-oldleo("dr","DR-"+n[2:]):+.4f}',f'{leo(n)-oldleo("old",n[2:]):+.4f}'] for n in ROWS[1:]])
text+='\n旧DR7有U GRL偏移且每组9800步；本次修正GRL、增加到44400步并加入扩展，因此不能把与旧DR7的大幅差值单独归因DAOT扩展、预算或某个修复。完整原15+旧7+新6共28个已评分配置见all_28_scored_comparison.csv；未完成三行不会被伪装成缺失或零分。\n\n'
text+='## 真实机制证据与训练成本\n\n'
text+='完整解析六组共266400条接受更新、1200轮结构化日志及完整stdout，另解析其余三组至审计快照的全部已写记录。六组均无step缺口、非有限loss/梯度、scale提交错位或不完整DR目标场；每组覆盖全部6300个L和56700个U，L累计5683200、U累计11340000样本次。五个完整扩展组各有45个梯度探针。\n\n'
text+=table(['行','训练小时','H累计样本次','P累计样本次','N累计样本次','卫星H累计样本次','EG校正步','Head catchup步'],[[n,f'{hours(n):.3f}',*[A['full']['rows'][n]['action_audit']['counts'].get(k,0) for k in ['hard_count_sum','partial_count_sum','negative_count_sum','satellite_samples_sum']],A['full']['rows'][n]['completion']['actions']['CORRECT'],A['full']['rows'][n]['completion']['head_steps']] for n in ROWS])
text+='\n这些为共享GPU现场耗时，不是隔离效率基准；含校准/源验证/审计摊销，不包括本轮测试。全部扩展不等于每个条件损失始终非零：\n\n'
text+='- orbit_z/logit/proto/relation及nuisance：五组均在40/45个探针观察非零梯度；tangent在31/45个探针非零，与E21/E61日程一致。\n- fingerprint：五组均观察过非零损失，但45个采样梯度探针均为0，因此只能报告“非零损失已出现，非零梯度未证实”，不能宣称该项在正式训练已有有效梯度。\n- F-M05的P累计选择为0，P-set/P-conditional未取得非零损失或梯度证据；其余完整组P有实际选择及非零梯度。\n- 五组均有N、anchor及卫星Hard真实选择/非零梯度证据；启用X/U的行也有相应非零梯度。\n- F-M11/F-M08的C2分别执行1056/1076次EG校正。F-M12虽配置C*，实际CORRECT/CATCHUP均为0；其成绩不能归因于C*控制动作。F-M14/F-M05按设计无控制动作。\n\n'
text+='完整计数、分项探针、anchor来源及接受更新证据见mechanism_evidence.json；完整训练审计gzip可解压重读。原报告中的合成“分支可执行”证据与这里正式训练“是否实际触发”严格区分。\n\n'
text+='## 未完成三组及结论边界\n\n'
text+=table(['行','审计时已完成epoch','接受更新','目标测试结果'],[[n,A['full']['rows'][n]['logs.jsonl']['records'][-1]['epoch'],A['full']['rows'][n]['action_audit']['counts']['accepted'],'尚无最终E200测试'] for n in ['F-M00','F-M07','F-M13']])
text+='\nF-M00未完成，不能给出新预算CORE90的正式对照；F-M07未完成，不能完成匹配C2条件下单独X的对比；F-M13未完成，不能完成固定课程下完整联合与课程对照。本次保留全部健康训练，不基于上述分数重训、选模或改参数。\n\n'
text+='当前仅能确认：相对旧短预算DR7表现大幅恢复；但在已完成同预算行中，完整扩展联合尚未超过F-A1，且最弱TX仍是明显短板。不能把“更多损失已接入”解释为“已证明融合增益”。\n\n'
text+=f'训练执行commit=e7dbe643f5b85062e8f4e2de6d711a5447c7a422；提前评估脚本commit=1fa3477d11cbf0288419429ab5639a371c68060c。评估run={EVAL}；原FULL9 run={RUN}。\n'
(P/'detailed_results.md').write_text(text,encoding='utf-8')
report=(P/'report.md').read_text(encoding='utf-8').replace('当前状态：LOCAL_VERIFIED，待提交、发布和评分。',f'当前状态：VERIFIED / ARTIFACTS_COMPLETE，六组4032000条预测已固定并独立评分，{stamp}完成读回。详细结果见[detailed_results.md](detailed_results.md)。')
(P/'report.md').write_text(report,encoding='utf-8')
with gzip.open(P/'full_run_audit.json.gz','wb') as f:f.write((BASE/RUN/'full_run_audit.json').read_bytes())
mirror=Path(__file__).resolve().parents[3]/'docs/research'/EVAL;mirror.mkdir(parents=True,exist_ok=True)
for name in ['report.md','detailed_results.md','test_summary.csv','scene_metrics.csv','per_class_metrics.csv','rx_day_tx_metrics.csv','confusion_matrices.csv','paired_rescue_harm.csv','all_28_scored_comparison.csv','training_curves.csv','mechanism_evidence.json','detailed_recount.json','evaluation_readback.json','launch_readback.json','full_run_audit.json.gz']:
 shutil.copy2(P/name,mirror/name)
print(table(['row','clean','LEO mean','LEO delta vs F-A1','LEO TX floor','LEO RX floor'],[[r['row'],f'{r["clean"]:.4f}',f'{r["leo_mean"]:.4f}',f'{r["delta_vs_F_A1_pp"]:+.4f}',f'{r["leo_tx_floor"]:.3f}',f'{r["leo_rx_floor"]:.3f}'] for r in summary]))
print('prediction_fields',D['rows']['F-A1']['prediction_record_fields'])
print('CSV counts',dict(summary=len(summary),scene=len(scenes),classes=len(classes),rx_day_tx=len(details),confusion=len(conf),paired=len(pairs),curves=len(curves),all_scored=len(historical)))
