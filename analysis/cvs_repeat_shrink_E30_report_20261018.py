"""Render only preserved aggregate evidence;never load IQ/CP/logits/rawtruth."""
import argparse,json
from pathlib import Path

def cost(v):
 keys=('actual_steps','parent_steps','cumulative_steps','parent_epochs','new_epochs','cumulative_epochs','L_forward_packets','student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets','SupCon_applied','SupCon_layer_forward_terms','As_training_observations','Ag_training_observations','elapsed_seconds','head_fit_updates')
 return {k:v[k] for k in keys if k in v}
def source_public(v):
 src=v['source'];q=v['source_qualification'];rows={}
 for mode,r in src['rows'].items():
  rows[mode]={k:r[k] for k in ('step_log','epoch_log','sequence_log','reference_log','unit_log','error_markers','observed_L_packets','observed_U_packets','observed_unique_L','observed_unique_U','source_RX','source_days','source_RX_holdout','native_telemetry','operator_recount','reference_epoch_curves')}
  rows[mode]['training_cost']=cost(r['training_cost'])
 return dict(status='VERIFIED_SOURCE_ONLY',run_id=src['run_id'],source_qualification=q,rows=rows,matched_comparisons=src['matched_comparisons'],parent_training_cost=cost(src['parent_training_cost']),IQ=0,CP=0,NN=0,updates=0,target_scores_read=False,new_Ag_efficacy=False,joint_goal_complete=False)
def public(v):
 p=source_public(v);keys=('status','run_id','summary','paired','strata','formal_metrics','suburban_gaps_pp','CM_recounts','partition_recounts','paired_comparisons','max_metric_error','IQ','CP','NN','updates','raw_truth_reads','new_Ag','Ag_As_interaction','single_seed_only','joint_goal_complete','target_feedback','wall_seconds')
 out={k:v[k] for k in keys};out['source']=p;out['cost']={}
 for mode,c in v['cost'].items():
  pred={k:c['prediction'][k] for k in ('count','views','NN_forward_packets','inference_seconds','wall_seconds','fixed_received_IQ','head_fit_updates')};out['cost'][mode]=dict(training=cost(c['training']),prediction=pred,scorer=c['scorer'],source_qualification=c['source_qualification'])
 return out

def source_markdown(v):
 q=v['source_qualification'];s=v['source'];r=['# 完整源 V 固定判据结果','', '状态：VERIFIED_SOURCE_ONLY。全部训练记录与原末轮默认头读出已复算，尚未读取正式 target 数值。','', '|模型|完整源 V clean|正确数／27000|','|---|---:|---:|']
 for mode in ('B00','B10'):
  m=q['rows'][mode]['metrics'];r.append(f"|{mode}|{m['accuracy']*100:.6f}%|{m['correct']}|")
 r+=['',f"B10−B00：{q['clean_delta_pp']:+.6f} 个百分点；预登记要求至少 +0.5。每个 RX 要求至少 −0.25 个百分点。",'', '|源 RX|准确率差（百分点）|','|---|---:|']
 r += [f'|{rx}|{delta:+.6f}|' for rx,delta in q['RX_delta_pp'].items()]
 r+=['', '判据：'+json.dumps(q['conditions'],ensure_ascii=False)+'。', '', '源结论：'+q['decision']+'。', '', '源条件通过也不等于跨 RX 泛化成立；若失败则按固定规则关闭，不按随后 target 分数调整强度、窗口、阈值、seed、父权重或预算。B01/B11 与交互项当前尚未定义。','', '两行完整日志均为 2220 步、10 新轮、4440 条 L/U 序列；全部 L6300/U56700、5 RX、3 day 不变。每行 1704960 次有标签前向，原 U/teacher 计数完全匹配；284160 个私有 lambda 抽样跨行相同。归一化 .5/.25/.25 总尺度1，原始 raw 锚点 .25。全部 native CSV/JSONL 与 observer 逐值相符，完整 stdout 无技术异常。','', '本独立复算 0 IQ/CP/NN/更新，仅读取保存日志和允许的源聚合。']
 return '\n'.join(r)+'\n'

def markdown(v):
 q=v['source_qualification'];summary=v['summary'];pairs=v['paired']['B10__B00'];r=['# 重复残差 IQ 增强：E30 B00/B10 完整结果','', '状态：ANALYZED／VERIFIED。两行均完成固定 E20＋10、默认头 clean＋六种 practical 预测、逐 row 独立 truth-last 评分与全部复算。联合目标仍 ACTIVE。','', '|固定视图|B00 准确率|B10 准确率|B10−B00（百分点）|','|---|---:|---:|---:|']
 views=v['summary']['B00'].keys()
 for view in views:r.append(f"|{view}|{summary['B00'][view]['accuracy']*100:.6f}%|{summary['B10'][view]['accuracy']*100:.6f}%|{pairs[view]['delta_accuracy_pp']:+.6f}|")
 r+=['',f"源 V 判据已先冻结：整体差 {q['clean_delta_pp']:+.6f}pp，条件 {json.dumps(q['conditions'],ensure_ascii=False)}，结论 {q['decision']}。正式结果不回流改写这一决定。",'', '|视图|修复原错误|新增错误|clean 与该场景同时正确的净变化（包）|','|---|---:|---:|---:|']
 for view,p in pairs.items():r.append(f"|{view}|{p['repairs']}|{p['new_errors']}|{round(p['joint_clean_and_scene_delta_pp']*1680)}|")
 r+=['', '该同时正确量比较的是两个仅研究 Ag 的模型，不是 Ag＋As 联合结果。所有源与正式 RX/day/TX 结果、macro-F1、最差 RX 和共同 clean 正确集合上的修复/新错均保留在聚合结果中。','', '|模型|high suburban 与自身 clean 差|mid suburban 差|low suburban 差|','|---|---:|---:|---:|']
 for mode,g in v['suburban_gaps_pp'].items():r.append('|'+mode+'|'+'|'.join(f'{value:.6f}pp' for value in g.values())+'|')
 r+=['', '2pp 仅为事先固定的探索目标。需同时检查 suburban 绝对准确率与 clean，不能把 clean 下降造成的差距缩小当作保护。','', '本矩阵没有 As 训练，也没有 B01/B11，Ag–As 交互项未定义。仅一个模型 seed，不宣称跨 seed 重现。不能将本矩阵的 clean 差与另一个 As 实验的收益相加预测联合成绩。','', '两行来自同一个合法 fullsource REPEAT final20 学生及 own EMA，各重置 AdamW/cos10、续训 2220 次，累计 E30/6660 更新。F/SIG_TIME 三路、统一 Sinc、joint160、默认 cosine 头与骨干共同训练；NoExplicitCFO/原补偿和 U 规则保持。全部源 RX 参与训练，未使用 L5580/dev720 或长 IQ。','', '监督为 .5RD(ordinary)＋.25RD(raw 锚点)＋.25RD(raw 或 Ag(raw))，总 L 尺度1，原 SupCon/U 另按原配方。每包6次有标签前向且每个 RDrop 对复用同一 IQ；0新损失/部署参数/头拟合/新教师/VJP/As 训练。固定 Ag 仅将四周期偏离自身均值部分按私有 Uniform[0,.1] 收缩，不能视为真实接收机替换或纯 RX 噪声去除。','', '成本：共享 E20/4440 父成本单列，新增两行各 E10/2220，唯一累计40 model-epoch，非两个 scratch E30。每行有标签 NN1704960（比旧4前向多568320），原 U/教师/验证及正式预测另计。全部精确前向和 CPU bridge/operator 时间见结果 cost；sourceguard 先前14404 NN/0更新另列。','', '独立复算全部252正式混淆矩阵、42组分区求和、7组同物理包配对，最大度量误差 '+str(v['max_metric_error'])+'。源全部步骤/序列/lambda/损失尺度和 native CSV/JSONL 逐值核对；仅读已保存日志/源聚合/正式计分与正确性向量，0 IQ/CP/NN/更新/raw truth。','', '测试 received IQ、信道/SNR、困难样本与全部168000物理包保持，全部7视图固定后才独立计分。代码执行身份与资源核实见原 report.md；完整私有日志/预测/物理ID/权重保留，不提交 Git。','', '联合目标尚未完成：有效 Ag 跨 seed 证据、同契约完整四格与交互项、clean 净正、low urban>60%、suburban 保持仍需按完整证据判断。']
 return '\n'.join(r)+'\n'
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('input');a.add_argument('output');a.add_argument('--source-only',action='store_true');args=a.parse_args();v=json.loads(Path(args.input).read_text(encoding='utf8'));o=Path(args.output);o.mkdir(parents=True,exist_ok=False)
 result=source_public(v) if args.source_only else public(v);(o/('source_analysis.json' if args.source_only else 'final_analysis.json')).write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8');(o/('source_decision.md' if args.source_only else 'final_analysis.md')).write_text(source_markdown(v) if args.source_only else markdown(v),encoding='utf8');print(json.dumps(dict(status='RENDERED',source_only=args.source_only,IQ=0,CP=0,NN=0,updates=0)))
