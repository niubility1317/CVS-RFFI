"""Render evidence-backed same-row delivery only after independent analysis completes."""
import argparse,json,shutil
from pathlib import Path
RUN='20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pp(a,b):return 100*(a-b)
def fmt(x):return f'{x:.6f}'
def formal_rows(summary):
 names=('C0','AS50','R25');views=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');rows=summary['rows']
 if tuple(rows)!=names or any(set(rows[a]['views'])!=set(views) for a in names):raise ValueError('Exact three declared rows and seven views required')
 return ['|环境|C0 无As|AS50|R25|','|---|---:|---:|---:|']+['|'+v+'|'+'|'.join(fmt(100*rows[a]['views'][v]['accuracy']) for a in names)+'|' for v in views]
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args();root=a.artifacts;analysis=root/'analysis_r01';done=read(analysis/'completion.json');s=read(analysis/'analysis_summary.json');q=read(analysis/'independent_source_decision.json');pair=read(root/'physics_pair_r01/complete.json');spec=read(a.record/'experiment.json')
 if done['status']!='CLEANRETAIN_ANALYSIS_COMPLETE' or done['formal_confusion_matrices']!=378 or done['NN'] or done['IQ_reads'] or done['CP_reads'] or done['training_updates'] or spec['run_id']!=RUN or s['model_seed_count']!=1 or s['training_parent_epochs']!=0 or s['new_model_epochs']!=30 or s['new_updates']!=6660:raise ValueError('Actual completed fullsource scratch30 delivery required')
 figures=[analysis/Path(f).name for f in read(analysis/'figures.json')['files']]
 if any(not f.is_file() for f in figures) or not (analysis/'per_row_seed.csv').is_file():raise ValueError('Do not create report with missing figure or metrics paths')
 rows=s['rows'];R=rows['R25']['views'];C=rows['C0']['views'];A=rows['AS50']['views'];clean=pp(R['clean']['accuracy'],C['clean']['accuracy']);low=pp(R['practical_low_urban']['accuracy'],A['practical_low_urban']['accuracy']);mid=pp(R['practical_mid_urban']['accuracy'],A['practical_mid_urban']['accuracy']);urban=pp(s['urban_mean_accuracy']['R25'],s['urban_mean_accuracy']['AS50']);clean_as=pp(R['clean']['accuracy'],A['clean']['accuracy']);rule='通过' if q['source_protection_evidence'] else '未通过'
 lines=[f'# 全源 F：保留 clean 监督的 E30 配对结果',f'状态：ANALYZED／VERIFIED。新 R25 与两个复用控制完成原七环境预测及独立 truth-last；整体联合目标仍 ACTIVE。',f'**本轮固定源保护条件{rule}。** R25 相对同预算 C0 的 clean 为 {clean:+.6f} 个百分点；相对 AS50，clean {clean_as:+.6f}、mid urban {mid:+.6f}、low urban {low:+.6f}、urban 均值 {urban:+.6f} 个百分点。',f'这是一个 model seed2026092701、全部五源 RX、scratch30/20L10U 的结果。75%普通视图／25%完整 practical 是用户明确要求参考的训练组织，本轮没有新增物理 Ag，也不能把源保护通过当作域泛化创新或联合目标完成。','## 默认头七环境准确率','单位为%。所有模型固定既有默认6×160 cosine头/scale30，骨干与头联合训练；没有换头、另训诊断头或目标拟合。']
 lines+=formal_rows(s)
 lines+=['## 收益、代价与配对错误',f"R25 urban 均值 {100*s['urban_mean_accuracy']['R25']:.6f}%；low urban {100*R['practical_low_urban']['accuracy']:.6f}%；clean {100*R['clean']['accuracy']:.6f}%。low>60%、clean正净收益、每个suburban距自身clean≤2pp仍按原探索目标报告，不能靠clean下降缩小差距。",'|模型|clean宏F1 %|low宏F1 %|low最差RX %|','|---|---:|---:|---:|']
 for name in ('C0','AS50','R25'):
  v=rows[name]['views'];lines.append(f"|{name}|{100*v['clean']['macro_f1']:.6f}|{100*v['practical_low_urban']['macro_f1']:.6f}|{100*v['practical_low_urban']['worst_RX_accuracy']:.6f}|")
 lines+=['|R25差值参照／视图|准确率差pp|修复|新增错误|clean与该视图同时正确净差pp|','|---|---:|---:|---:|---:|']
 for ref in ('C0','AS50'):
  for view in ('clean','practical_mid_urban','practical_low_urban'):
   v=s['paired_comparisons']['R25-'+ref][view];lines.append(f"|{ref}/{view}|{v['accuracy_gain_pp']:+.6f}|{v['repairs']}|{v['breaks']}|{v['clean_and_view_joint_gain_pp']:+.6f}|")
 lines+=['|模型|high suburban距clean pp|mid suburban距clean pp|low suburban距clean pp|','|---|---:|---:|---:|']
 for name in ('C0','AS50','R25'):lines.append('|'+name+'|'+'|'.join(fmt(s['suburban_clean_gaps_pp'][name][v]) for v in ('practical_high','practical_mid','practical_low_suburban'))+'|')
 lines+=[f"固定C0 clean正确集合共{s['fixed_C0_clean_correct_count']}包；R25 low保持率{100*s['fixed_C0_clean_correct_retention']['R25']['practical_low_urban']:.6f}%，AS50为{100*s['fixed_C0_clean_correct_retention']['AS50']['practical_low_urban']:.6f}%。固定集合只是诊断参照，两个控制均非新训练父模型。",'## 源判据、分层和因果边界',f"正式结果读取前已落盘源判据。fullV clean R25−C0={q['clean_net_pp']:+.6f}pp；两个独立draw×mid/low相对AS50的差={[[round(v,6) for v in row] for row in q['urban_delta_vs_AS50_pp']]}pp。固定要求clean≥−.25pp、四个urban差均≥−2pp。任何失败即本轮保护假设失败，不换份额、扰动、seed、阈值或预算救跑；不自动扩种子。",f"全部物理实现比较：{pair['pairs']}个选中As包、{pair['steps']}步，完整记录一致{pair['identical_records']}，差异{pair['record_mismatches']}。首次批次吻合不能代替该全量统计；若有差异，应同时报告对应字段，不将效果单独归因为监督份额。",'源V1800使用同一真实IQ和两组固定完整practical观测，按LOS/PARTIAL/BLOCKED、locked/degraded/unlocked、RX/day/TX及新增噪声比分层。新增噪声比不是录音已有噪声与新增噪声合计的总SNR。正式IQ没有逐包状态/质量旁表，所以不能把源分层冒称目标urban的状态因果效果。']
 source=read(analysis/'source_summary.json');lines+=['|源low urban新增噪声比|样本数|C0 %|AS50 %|R25 %|','|---|---:|---:|---:|---:|']
 for label in ('<3','3-8','8-15','>=15'):
  cells={arm:next(r for r in source['rows'] if (r['arm'],r['scene'],r['draw'],r['dimension'],r['stratum'])==(arm,'practical_low_urban',-1,'added_noise_ratio',label)) for arm in ('C0','AS50','R25')};lines.append('|'+label+' dB|'+str(cells['C0']['count'])+'|'+'|'.join(fmt(100*cells[arm]['accuracy']) for arm in ('C0','AS50','R25'))+'|')
 lines+=['## 完整训练、数据与预算','全部RX1/3/4/6/8、day1/2/3、六TX、L6300/U56700隐藏TX/V27000、7/63/30物理ID保持；不留训练RX，不合并V，不解封U TX。F为SIG_TIME_X3、Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、真实joint160；900722参数/206状态；保留原必要包内补偿、旧U门控、自身EMA、SupCon和RDrop。', '每个L128，原ordinary128作为第一半；第二半交替64个ordinary和64个原practical(raw)，全部生成As监督。每对RDrop共享同一生成IQ，无增强重抽。CE/KL两半归一化再.5/.5，实际ordinary.75/satellite.25；SupCon只用原第一半。4L NN/物理样本，不增加部署模块或损失。', '实际导入为原不可变release中experiments/adv3b02_xuc/code/leo_practical，不是顶层code。25MHz/IQ256、dataset原RMS，As前原unitRMS、As后无额外RMS；完整residual/post_sync/EQ关闭即G1/referenceSNR30/fc2.45GHz/reflect，质量适配、相位跟踪、RXIQ补偿、AGC及相互依赖保持；62tap核和24点数值延迟不变。没有改变正式IQ、SNR、边界或困难样本。','本轮与C0、AS50的全部训练序列比较：'+json.dumps(s['training_sequences'],ensure_ascii=False)+'。完整30轮/6660step、原SupCon5550有效步、Sinc6660更新、全量physics及活动损失均由独立分析读取。RDrop是原随机子网络正则，不记作新Ag。',f"监督分量的独立尺度复算最大误差{s['source_classification_decomposition_max_error']:.9g}。共有ordinary完整日程与波形统计比较：{json.dumps(s['ordinary_comparison'],ensure_ascii=False)}。默认头/Sinc参数运动只证明联合更新，不证明分支或门控导致收益；没有保存逐包门控/分支梯度，不补造该机制证据。",'|模型|训练源NN|训练秒|正式NN|预测秒|新训练/复用|','|---|---:|---:|---:|---:|---|']
 for name in ('C0','AS50','R25'):
  c=s['costs'][name];lines.append(f"|{name}|{c['source_identity_NN']}|{c['source_wall_seconds']:.6f}|{c['prediction_identity_NN']}|{c['prediction_wall_seconds']:.6f}|{'新30E/6660更新' if name=='R25' else '复用历史30E/6660更新'}|")
 lines += [f"新R25父训练0、30轮/6660更新、3409920L网络前向、426240真实As训练视图；U/学生/教师/评估分别由实际counter计入，teacher训练0。兼容smoke2、随机模型合成512、新C0源读出12602、新R25源读出12602、formal1176002分别保存；本轮新增总NN={s['actual_new_NN']}，历史两个控制NN={s['historical_control_NN']}单列，不能记作新增或父继承。调度到终态墙钟{s['run_wall_seconds']:.6f}秒；独立分析CPU墙钟{done['analysis_wall_seconds']:.6f}秒，不包含等待训练。预算结束不等于收敛，不延长至E50/E200。",'## 代码、验证和可检查产物',f"训练代码{s['training_code_commit']}；独立结果分析{s['code_commit']}；物理实现比较{pair['code_commit']}。训练实现experiments/cvs_cleanretain37；原匹配控制P25 B00 ordinary-repeat和P32 B01 rawpractical；E20＋10的旧B01只作路线背景，不混作scratch30控制，也没有加载旧目标选中权重。", '本地8项训练聚焦检查、4项独立分析检查、3项物理配对反例通过。一次独立P0/P1审查发现合成验证输出冲突，已隔离且定向复核；首步真实CP2包、原生64As/512NN随机模型验证通过，无query、0optimizer；848个native args仅运行标识/输出位置不同。一次发布传输比较/编译，实际PID/CWD/argv/GPU/log增长已核实。所有其他健康任务保持。',f"训练命令见[experiment.json]({(a.record/'experiment.json').as_posix()})；环境/CWD/GPU/停止规则及source权限在原预登记中保持。所有正式预测固定后独立CPU scorer，378个混淆矩阵独立复算；模型默认头无额外拟合。",f"[逐row/seed/RX/day/TX]({(analysis/'per_row_seed.csv').as_posix()})；[完整训练曲线]({(analysis/'training_curves.csv').as_posix()})；[源状态质量分层]({(analysis/'source_strata.csv').as_posix()})；[独立完整统计与成本]({(analysis/'analysis_summary.json').as_posix()})。"]
 for name in ('formal_accuracy','training_curves','source_added_noise_accuracy','source_state_lock_accuracy','R25_supervision_components'):
  f=analysis/(name+'.png')
  if f.is_file():lines.append(f'![{name}]({f.as_posix()})')
 lines+=['## 未完成项', '当前只有一个模型seed，不能回答跨seed稳定复现。源两draw不是两个模型seed。没有独立有效Ag，不能虚构Ag×As四格交互。low>60%、clean净收益、所有suburban接近clean和跨seed联合验证仍按实际结果分别审查，整体goal不标complete。', f"固定CVS和固定practical条件下，本轮可核实的是完整组合practical的训练组织效应：默认头相对AS50的low变化{low:+.6f}pp、mid{mid:+.6f}pp；clean相对C0变化{clean:+.6f}pp、相对AS50变化{clean_as:+.6f}pp。单个多径/频偏/噪声因素的训练因果增益及跨seed复现尚不能由这一行证明。"]
 old=a.record/'report.md';history=a.record/'progress_history.md'
 if old.exists() and not history.exists():shutil.copy2(old,history)
 text=''
 for index,line in enumerate(lines):
  new_table=line.startswith('|') and index+1<len(lines) and lines[index+1].startswith('|---')
  separator='\n' if text and text.splitlines()[-1].startswith('|') and line.startswith('|') and not new_table else '\n\n'
  text+=('' if not text else separator)+line
 old.write_text(text+'\n',encoding='utf8');print(json.dumps(dict(status='EVIDENCE_REPORT_RENDERED',path=str(old),figures_checked_existing=len(figures),goal_complete=False)))
if __name__=='__main__':main()
