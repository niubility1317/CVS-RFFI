import csv,gzip,json,statistics,shutil
from pathlib import Path
P=Path('E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_xuc_dr_s392005_20260913_r1')
D=json.loads((P/'full_run_audit.json').read_text(encoding='utf-8'))
R=json.loads((P/'receiver_class_details.json').read_text(encoding='utf-8'))
SC=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
ROWS={**D['old']['rows'],**D['dr']['rows']}
def metrics(n):return ROWS[n]['score']['metrics']
def leo(n):return sum(metrics(n)[s]['accuracy'] for s in SC[1:])*100/3
def hours(n):
 r=ROWS[n]
 return (r['completion'].get('elapsed_seconds') or r['resource']['wall_time_seconds'])/3600
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('|'+ '|'.join(map(str,r))+'|\n' for r in rows)
def csvwrite(name,rows):
 with (P/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
score_rows=[];class_rows=[];receiver_rows=[];mechanism=[];curves=[]
for n,r in ROWS.items():
 m=metrics(n)
 score_rows.append(dict(row=n,**{s:100*m[s]['accuracy'] for s in SC},leo_mean=leo(n),leo_scene_floor=min(m[s]['accuracy'] for s in SC[1:])*100,train_hours=hours(n),parent_delta_leo_pp=leo(n)-leo(n[3:]) if n.startswith('DR-') else '',core90_delta_leo_pp=leo(n)-leo('M00')))
 for s in SC:
  for c,v in m[s]['per_class_accuracy'].items():class_rows.append(dict(row=n,scene=s,tx=int(c),accuracy_percent=v*100,scene_correct=m[s]['correct'],scene_total=m[s]['total']))
 for v in R['rows'][n]['details']:receiver_rows.append(dict(row=n,**v))
 records=r.get('logs.jsonl',r.get('metrics_epoch.jsonl',{})).get('records',[])
 previous=0
 for a in records:
  if 'elapsed_seconds' in a:sec=a['elapsed_seconds']-previous;previous=a['elapsed_seconds']
  else:sec=a['epoch_time_s']
  curves.append(dict(row=n,epoch=a['epoch'],epoch_seconds=sec,source_val_accuracy_percent=100*a['source_validation']['accuracy'] if 'source_validation' in a else a.get('val_tx_acc'),loss=a.get('mean_loss',a.get('train_loss')),daot_labeled=a.get('terms',{}).get('daot_labeled'),daot_unlabeled=a.get('terms',{}).get('daot_unlabeled'),native_daot_total=a.get('train_loss_daot_total')))
 if n.startswith('DR-'):
  aa=r['action_audit'];probes=aa['gradient_probes'];cnt=aa['counts']
  assert cnt['accepted']==cnt['records']==9800 and not aa['gaps'] and not aa['bad']
  assert not any(v for k,v in cnt.items() if k.startswith('nonfinite') or k in ['scale_commit_mismatch','incomplete_dr_fields'])
  assert all(not s['errors'] for s in r['stdout_scan'])
  mechanism.append(dict(row=n,**cnt,gradient_probes=len(probes),positive_daot_gradient_probes=sum(v['daot_grad_norm']>0 for v in probes),positive_rc4_gradient_probes=sum(v['rc4_grad_norm']>0 for v in probes),correct_steps=aa['actions'].get('CORRECT',0),head_catchup_steps=r['completion']['head_steps'],unique_L=aa['unique_L'],unique_U=aa['unique_U'],daot_gradient_last=probes[-1]['daot_grad_norm'],rc4_gradient_last=probes[-1]['rc4_grad_norm']))
csvwrite('detailed_test_metrics.csv',score_rows);csvwrite('per_class_test_metrics.csv',class_rows);csvwrite('receiver_day_class_metrics.csv',receiver_rows);csvwrite('training_curves.csv',curves)
(P/'mechanism_activation_summary.json').write_text(json.dumps(mechanism,ensure_ascii=False,indent=2),encoding='utf-8')
text='# DR7完整测试与训练耗时审计\n\n'
text+='结论：DAOT和RC4确有真实执行及非零梯度；训练快主要因为CORE90载体每轮49步，只有原生A1每轮222步预算的22.07%。但新7组LEO均低于父行及CORE90基线，当前实验不支持融合增益。另发现RC4 U分支GRL从原生0变为1的未声明梯度语义差异，撤回此前“完整原生接入且无差异”的表述；保留“产物闭合、机制路径实际执行”的验证结论。\n\n'
text+='执行commit=4d3880208e40e0e8a875284e2dc31e41ad0d54f4。只读审计覆盖新7组和原15组：20组完整actions各9800条（共196000条）、22组完整200轮日志（共4400轮）、全部训练/预测/评分stdout；原生M09/M10无actions.jsonl，使用完整metrics_epoch.jsonl及已固定原生迭代器语义。额外读完22组14784000条冻结预测，按独立truth重新核对正确数及receiver/day/TX分组，未重新训练或推理。\n\n'
text+='数据：ManySig equalized=1，source RX1/3/4/6/8、day1/2/3，L6300/U56700/V27000；target RX0/2/5/7/9/10/11、day0/1/2/3、TX0–5。每场景168000条，每组672000条。所有方法相同模型/数据seed392005，从零训练，最终固定E200checkpoint。LEO平均为三场景算术平均；因样本量相同也等于三场景合并accuracy。单位为%，差值为百分点。公开过的研究benchmark不称新盲确认集。\n\n'
text+='## 为什么不到4小时训练结束\n\n'
text+=table(['项目','新DR7','原生M09/M10'],[['每轮主更新','49','222'],['200轮主更新','9800','44400'],['DAOT有效阶段主更新','8820','39960（按E21–200及迭代器预算）'],['U累计暴露','2502992','11340000（200次完整U遍历）'],['U全池等效遍历','44.14次','200次'],['单组训练耗时','3.65–3.87小时',f"{hours('M09'):.2f}/{hours('M10'):.2f}小时"],['参数量','1130809','1130809']])
text+='\n222步由U56700、batch256、不drop最后批次及原生_muse_epoch_pairs完整U遍历确定；原生日志平均U batch=255.405405与56700/222相符。新行用L6300//128=49步，但旋转消费U，累计覆盖全部56700个唯一U样本，并非只使用固定22%子集。U批次上限256，轮转到末批时存在短批，所以实际累计2502992而非机械计算的2508800。新方案的200轮只相当于原生约44.14轮的主更新量；相同轮号下DAOT预热、LR和尾部日程也在更少更新后进入，不能称等预算原生复现。\n\n'
text+='新组单次主更新平均约1.34–1.42秒，原生约1.05–1.08秒（总墙钟/主更新，包含校准、验证、审计等摊销），因此不能将较短总时长解释为单步更快。7组分别在7张GPU并行，不是依次训练。实际11:11:57启动，15:04:07全部训练完成，15:16:21全部评分闭合；总墙钟4小时4分25秒。\n\n'
text+=table(['行','父行训练小时','DR训练小时','墙钟变化'],[[n,f'{hours(n[3:]):.3f}',f'{hours(n):.3f}',f'{(hours(n)/hours(n[3:])-1)*100:+.1f}%'] for n in D['dr']['rows']])
text+='\n上表为共享GPU现场观察，非隔离效率消融。相对同49步预算父行，加入DR后耗时增加6.3%–26.2%。原7组约3小时，新增组约3.7小时，时间量级并不反常。C*行零控制动作、其余C2行仅约3.1%步使用额外EG场，也是没有巨大额外开销的直接记录。\n\n'
text+='## 四场景详细结果\n\n'
display=['M00','M09','M10',*D['dr']['rows']]
text+=table(['行','Clean','LEO晴空','LEO低仰角','LEO雨衰','LEO平均','最差LEO场景'],[[n,*[f"{metrics(n)[s]['accuracy']*100:.4f}" for s in SC],f'{leo(n):.4f}',f"{min(metrics(n)[s]['accuracy'] for s in SC[1:])*100:.4f}"] for n in display])
text+='\nM00为本次同契约ADV3B02 CORE90对比基线。M09为原生DAOT+FastTrust-RC4，M10为原生DAOT+FastTrust-RC4+X。M09/M10也已完成四场景评分；它们步数更高，不作为同预算纯机制消融。\n\n'
text+=table(['父行→新增行','原Clean','新Clean','Clean变化','原LEO平均','新LEO平均','LEO变化'],[[n[3:]+'→'+n,f"{metrics(n[3:])['clean']['accuracy']*100:.4f}",f"{metrics(n)['clean']['accuracy']*100:.4f}",f"{(metrics(n)['clean']['accuracy']-metrics(n[3:])['clean']['accuracy'])*100:+.4f}",f'{leo(n[3:]):.4f}',f'{leo(n):.4f}',f'{leo(n)-leo(n[3:]):+.4f}'] for n in D['dr']['rows']])
text+='\n新7组相对父行LEO全部下降，幅度2.0196–9.3887pp；相对CORE90下降7.2222–10.4415pp。DR-M12的55.2825%只是这7个已完成配置中观察值最高，不构成候选晋级；它低于M00的62.5048%及M09的68.3260%。不能把独立模块历史提升相加推断融合有效；本次整体结果为负。\n\n'
text+='## 实际机制激活\n\n'
text+='7组每组9800/9800主更新接受、无step缺口、无非有限loss/梯度、无scale提交计数错位；实际EG两场均包含DAOT labeled/U和RC4总目标。每组40个梯度探针中，预热后的36个DAOT梯度全部非零，40个RC4总梯度全部非零。每行完整覆盖6300个L与56700个U唯一物理样本。source-V校准5次，E1/21/41/91/161各27000条。\n\n'
text+=table(['行','DAOT执行步','H选中累计样本次','P选中累计样本次','额外EG步','最终DAOT梯度探针','最终RC4梯度探针'],[[v['row'],v['daot_executed'],v['hard_count_sum'],v['partial_count_sum'],v['correct_steps'],f"{v['daot_gradient_last']:.4f}",f"{v['rc4_gradient_last']:.4f}"] for v in mechanism])
text+='\nH/P为跨更新累计样本次，不是去重样本数；梯度列为step9750的总范数，不代表分支收益。只有DR-M05产生partial样本，其余6行虽启用P分支但自然选择为0。DR-M12/M13各40次C*审计均RECOVERY_UNRELIABLE，未触发CORRECT/CATCHUP；实际不能宣称C*控制动作参与并产生效果。DR-M11/M08/M07的CORRECT分别307/307/317次，其他行均0，所有行head catchup均0。X与Ux按父行开关执行；启用行9800步均有合法anchor/4个有效normalized blocks。\n\n'
text+='DAOT为原生A1已启用的orbit-z/logit目标：权重0.5/0.2，2-view mean、identity_sequential、teacher EMA0.999；不是DAOT所有研究分支全部打开。tangent/nuisance/fingerprint权重0，relation开关false；prototype_matrix=None，因此prototype项未生效。RC4 negative/anchor/satellite-hard-only关闭；satellite权重虽然配置0.1，但开关false，实际无卫星H监督，与本参考A1一致。应分别理解代码存在、开关打开、样本选中、非零梯度和测试增益。\n\n'
text+='## 新发现的移植语义差异\n\n'
text+='原生train_ssdg.py中fasttrust_rc4使muse_state.sat_anchor_ssl=true；该分支调用_forward_sat_anchor_student_views(grl_lambda=0.0)。新的dr_objective.py对同一U strong输入调用model(...,grl_lambda=1.)。两种模型的adv_dom_logits均由adv_head(grad_reverse(z_id,grl_lambda))得到。在当前identity_domain_objective_mode=grl_ce与rc4_lambda_domain=0.16下，这使原生U域对抗项对z_id的GRL梯度为0，而新增接线向身份骨干传入反向域梯度。损失函数名称和系数一致不能证明梯度语义一致。\n\n'
text+='这是未在最初配置确认中声明的实现偏移，此前审查漏掉了这一点。现有结果应标为“CORE90短步数+DAOT+有GRL偏移的RC4移植”，不能作为原生DAOT+FastTrust-RC4无差异完整联合的验收通过证据。本次没有改变训练代码、覆盖结果或发布重跑。该偏移可能影响泛化，但没有匹配消融，不能量化其对2–9pp下降的因果贡献；训练步数、U暴露、普通U目标替换、课程及随机轨迹也共同变化。\n\n'
text+='## 发射机与接收机细分\n\n'
text+='以下TX为数据集索引0–5，每类每场景28000条，LEO平均每类84000条。\n\n'
text+=table(['行',*[f'TX{i} LEO平均' for i in range(6)]],[[n,*[f"{sum(metrics(n)[s]['per_class_accuracy'][str(c)] for s in SC[1:])*100/3:.3f}" for c in range(6)]] for n in ['M00','M09',*D['dr']['rows']]])
text+='\n接收机细分来自冻结prediction与truth评分后关联物理ID。按正式构建器的TX0–5/RX列表/day0–3/equalized1/sig0–999生成ID，168000个ID集合与原封存truth完全一致，且TX标签逐ID一致，才生成以下统计；没有读取target IQ或重新推理。每RX每场景24000条。\n\n'
def rxmean(n,rx):return sum(v['accuracy'] for v in R['rows'][n]['details'] if v['scope']=='rx' and v['group']==str(rx) and v['scene']!='clean')*100/3
text+=table(['行',*[f'RX{rx} LEO平均' for rx in [0,2,5,7,9,10,11]]],[[n,*[f'{rxmean(n,rx):.3f}' for rx in [0,2,5,7,9,10,11]]] for n in ['M00','M09',*D['dr']['rows']]])
text+='\n全部22组四场景逐TX、逐RX、逐day、RX×day、RX×TX准确率及6×6混淆矩阵已保存。分组低值不能直接被当作根因；本次单seed结果没有提供跨seed不确定性。\n\n'
text+='## 完整产物\n\n- detailed_test_metrics.csv：22组四场景、LEO平均/下限、耗时及差值。\n- per_class_test_metrics.csv：22×4×6逐类准确率。\n- receiver_day_class_metrics.csv：22组完整接收机/日期/发射机分组及分母、正确数。\n- receiver_class_details.json：逐组混淆矩阵、分组及覆盖验证。\n- training_curves.csv：22组4400轮耗时、source-V、loss和DAOT记录。\n- mechanism_activation_summary.json：7组完整动作与梯度探针摘要。\n- full_run_audit.json：完整结构化日志及全stdout扫描；Git镜像使用gzip保存。\n\n以上只给出现有冻结结果与复核发现，不从本次target评分发起改参、选模或选择性重跑。\n'
(P/'detailed_results.md').write_text(text,encoding='utf-8')
mirror=Path('docs/research')/P.name
for name in ['detailed_results.md','detailed_test_metrics.csv','per_class_test_metrics.csv','receiver_day_class_metrics.csv','receiver_class_details.json','training_curves.csv','mechanism_activation_summary.json']:
 shutil.copyfile(P/name,mirror/name)
with gzip.open(mirror/'full_run_audit.json.gz','wb') as f:f.write((P/'full_run_audit.json').read_bytes())
report=mirror/'report.md';t=report.read_text(encoding='utf-8')+'\n\n## 完整结果与移植复核更新\n\n详见[detailed_results.md](detailed_results.md)。22组评分、全量训练日志与14784000条冻结预测已复核。新增7组DAOT/RC4非零梯度确有证据，但LEO全部低于父行。发现新接线RC4 U分支GRL=1而原生=0的未声明语义偏移；撤回此前原生无差异完整接入的验收结论，不撤回实际运行和评分闭合事实。DR-M12/M13的C*全程无控制动作，不得宣称完整联合机制均生效。当前代码和结果保持，不做结果驱动重跑。\n';report.write_text(t,encoding='utf-8');(P/'report.md').write_text(t,encoding='utf-8')
print(text)
