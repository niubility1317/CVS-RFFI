"""Record the complete source falsification without escalating a failed route."""
from pathlib import Path
import json,sys,csv,io,importlib.util
W=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W),str(W/'code')]
from experiments.cvs_rx_teacher20 import design as d
A=W/'automation_reports/CV-SincNet'/d.RUN;P=W/'local_artifacts'/d.RELEASE
if (A/'results_analysis.json').is_file():
 print('ALREADY_RECORDED');sys.exit(0)
x=d.read(P/'independent_analysis.json');live=d.read(P/'independent_readback.json')
assert x['status']=='VERIFIED' and x['complete_steps']==8880 and x['complete_epochs']==40
assert live['completion']['status']=='SOURCE_MATRIX_COMPLETE' and not live['dispatcher'] and not live['compare_process'] and not live['failure'] and all(not r['process'] for r in live['rows'])
assert not x['qualification']['qualified'] and not x['target_access'] and not x['teacher_bank_or_student_launched']
def table(name,rows):
 stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 (A/name).write_text(stream.getvalue(),encoding='utf-8',newline='\n')
overview=[];curves=[]
for r in x['rows'].values():
 overview.append(dict(arm=r['arm'],seed=r['model_seed'],epochs=r['epochs'],optimizer_steps=r['optimizer_steps'],train_L=r['train_L_unique'],RX1_train_L=r['RX1_train_L'],final_L_CE=r['final_training_L_CE'],fit_V_accuracy=r['source_V']['source_val_accuracy'],fit_V_worst_RX=r['source_V']['source_val_worst_rx'],held_RX1_accuracy=r['held_RX1']['accuracy'],held_RX1_errors=r['held_RX1']['errors'],held_RX1_CE=r['held_RX1']['mean_CE'],held_RX1_true_probability=r['held_RX1']['mean_true_probability'],held_RX1_wrong_confidence=r['held_RX1']['mean_max_probability_on_errors'],training_seconds=r['elapsed_seconds'],feature_L2_delta=r['feature_parameter_L2_delta'],default_W_L2_delta=r['default_W_L2_delta']))
 curves.extend(dict(arm=r['arm'],seed=r['model_seed'],**e) for e in r['complete_epoch_curves'])
table('summary.csv',overview);table('training_complete_curves.csv',curves);table('paired_source_strata.csv',x['records']);table('class_pair_errors.csv',x['class_pair_records']);d.write(A/'results_analysis.json',x)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
for r,color in zip(x['rows'].values(),('#3274a1','#c45f39')):
 e=r['complete_epoch_curves'];grid=[v['epoch'] for v in e]
 for ax,key in zip(axes,('train/loss_tx_labeled','source_V_accuracy_percent','epoch_time_s')):
  ax.plot(grid,[v[key] for v in e],label=r['arm'],color=color,linewidth=1.6)
for ax,title in zip(axes,('Pure labeled CE','Fit-only source V accuracy (%)','Epoch wall time (s)')):
 ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('Matched source E20 teachers | seed 2026092701 | held RX1 evaluated after freeze',fontsize=10)
fig.savefig(A/'training_curves.png',dpi=170);plt.close(fig)
a,b=overview;q=x['qualification'];proxy=x['A0_source_proxy']
lines=['# 跨 RX 教师：等样本 E20 源域证伪结果','',
'状态 **ANALYZED / 完整训练、冻结源 V 预测和独立产物分析 VERIFIED；预登记科学资格 FAIL**。固定 RX1 排除教师没有产生预期错误互补，因此关闭这一固定配方，不扩展五教师库、U 蒸馏或学生目标测试。完整研究目标仍 active。','',
'| 教师 | 唯一训练 L | 训练 RX1 的 L | 末轮训练 CE | 拟合 RX 的 V 准确率 | 留出 RX1 V 准确率 | 留出错误 | 留出 CE |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in overview:
 lines.append(f"| {r['arm']} | {r['train_L']} | {r['RX1_train_L']} | {r['final_L_CE']:.6f} | {r['fit_V_accuracy']*100:.6f}% | {r['held_RX1_accuracy']*100:.6f}% | {r['held_RX1_errors']} | {r['held_RX1_CE']:.6f} |")
lines+=['',
f"两教师均从相同 seed 的完全一致 200 个初始 state 从零训练，L 总量均为 5,040，每个 TX×day 均为 280。SAMPLE_80 在 5 个源 RX 的每个 TX×RX×day 使用 56 个 L；EXCLUDE_RX1 在其余 4 个源 RX 使用原有 70 个 L。前者共使用 1,008 个 RX1 L，后者全部训练、校准、缓存和祖先均排除 RX1。两者共同的拟合 V 只含 RX3/4/6/8 共 21,600 个样本；原源 V_RX1 的 5,400 个包在固定 final20 冻结后才首次前向。L 与 V 物理角色分离，没有扩大有标签权限。",'',
f"EXCLUDE_RX1 比 SAMPLE_80 净下降 {(b['held_RX1_accuracy']-a['held_RX1_accuracy'])*100:.6f} 个百分点，纠正 {q['overall_corrected']} 个、引入 {q['overall_introduced']} 个错误；平均 CE 也明显升高。拟合 V 的准确率反而增加 {(b['fit_V_accuracy']-a['fit_V_accuracy'])*100:.6f} 个百分点，两者末轮训练 CE 均低于 0.5。这排除了本次简单训练失败的解释，并说明较好的已见 RX 拟合表现不能代替跨 RX 证据。只对这两个固定 E20、单 seed 教师作此判断，不推广为所有交叉拟合方法无效。",'',
'| 固定 A0 门控区域 | 数量 | SAMPLE_80 准确率 | EXCLUDE_RX1 准确率 | 原 A0 EMA 准确率 | 纠正 / 引入 |','|---|---:|---:|---:|---:|---:|']
for r in x['records']:
 if r['dimension']=='gate':
  lines.append(f"| {r['stratum']} | {r['count']} | {r['SAMPLE_80_accuracy']*100:.6f}% | {r['EXCLUDE_RX1_accuracy']*100:.6f}% | {r['A0_EMA_accuracy']*100:.6f}% | {r['corrected']} / {r['introduced']} |")
lines+=['',
f"原完整 BASE_SSL final20 的 ownEMA/强视图 student 在这批源 V 上重放原门控，固定拒绝 {proxy['rejected']} 个、接受 {proxy['accepted']} 个。拒绝区原 EMA 准确率仍为 99.423248%；排除 RX1 教师在此区仅纠正 {q['rejected_corrected']} 个、引入 {q['rejected_introduced']} 个错误。‘门控拒绝’在该源代理上主要不等于‘教师预测错误’，因此不能把拒绝数量当作等待跨 RX 教师补全的错误标签数量。该结论只适用于固定 A0 末态源 V 代理，不是实际 U 的正确率，也不代表训练早期状态。A0 权重只用于冻结重放，不是两个教师的祖先。",'',
'3 个源 day 均下降，6 个 TX 中 5 个下降、TX4 持平。TX5 从 99.888889% 降至 13.666667%，主要新增 TX5→TX4 共 567 个和 TX5→TX2 共 189 个错误；TX2→TX0 新增 436 个错误。完整 30 个真类/错类对以及 gate、day、TX 配对见 CSV，不选择局部正项挽救资格。','',
'资格判定保持发布前条件：两者 L 健康，EXCLUDE_RX1 错误更少且 CE 不升，固定拒绝区非空且纠正大于引入。实际三个效果条件均未通过。没有阈值扫描、替换 RX、换 seed、延长轮次、换头、叠加 RDrop 或重新预测来挽救本次结果。已确认的 RDrop、CENTER25K、NONE 等正收益证据仍独立保留；本次负结果没有回写或停改任何旧训练。','',
f"独立 CPU 分析完整解析全部 8,880 条 step、40 条 epoch、两份训练 stdout 和比较 stdout，核对每步纯 CE、0 个非有限跳步、原生计数及实际角色、曝光、固定 final20 及空祖先。日志的 `a1_ema_successful_updates` 是原生成功更新计数名；实际 EMA 开关关闭、checkpoint 的 ema_model=None，不能由该计数名推断存在教师更新。主干和既有默认 W 均发生实际更新；16 个合成包前向确认部署 joint160 改变、feature-only 与 joint160 逐位相同、扰动 W 不改变 joint160。完整源 V 决定重算为 10,800 个教师预测和 5,400 个既有 A0 门控决定；本次分析新增真实 IQ 读取、优化、拟合头、checkpoint 改写和 target 访问均为 0。",'',
f"两行实际成本为 40 model-epoch、8,880 次原生更新、1,136,640 次 L 训练前向；原始实验另有 10,800 次教师留出 V 前向和 10,800 次 A0 门控前向，拟合 V 的每轮前向已计入实测 epoch 时间。两行训练耗时分别 {a['training_seconds']:.2f} s 和 {b['training_seconds']:.2f} s，矩阵从首次准入至源比较完成 {x['matrix_elapsed_seconds']:.2f} s。首轮各约 29.3 s，GPU0/1 每行约 2,928 MiB；全部原进程自然结束，资源已释放，未干预其他 owner。",'',
'本次是预登记的 source-only 证伪实验，未进行目标、卫星或 Phase2 测试。没有用源域结果宣称完整泛化目标完成。没有删除旧产物或重新登记同一行。后续新机制需要独立源证据，不能将本次已被否定的教师直接用于学生训练。','',
'![完整源训练曲线](training_curves.png)','',
'执行代码和固定配置提交：`'+x['execution_commit']+'`。完整预登记、命令、路径、一次 P0/P1 检查和启动证据见 experiment.json、review_evidence.json、launch_verification.json；独立完整复算见 results_analysis.json。']
(A/'report.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8',newline='\n')
spec=d.read(A/'experiment.json');spec['result']=dict(status='ANALYZED',scientific_qualification='FAIL',evidence_ref='results_analysis.json',actual_rows=2,actual_epochs=40,actual_optimizer_steps=8880,source_only=True,target_access=False,teacher_bank_or_student_launched=False,goal_completed=False);d.write(A/'experiment.json',spec)
mod=importlib.util.spec_from_file_location('registry',W.parents[2]/'tools/experiment_registry.py');reg=importlib.util.module_from_spec(mod);mod.loader.exec_module(reg)
reg.event(W,d.RUN,'ANALYZED',['results_analysis.json','summary.csv','paired_source_strata.csv','class_pair_errors.csv'],'Both scratch source E20 teachers complete; full8880steps/40epochs/10800teacher predictions/5400fixedA0 proxy independently recounted. Scientific qualification FAIL; fixed recipe closed, no bank/student/target.');reg.build(W,managed_only=True)
goal=W/'docs/CVS_PHASE1_FEATURE_EXTRACTION_GOAL_20261013.md';marker='<!-- RX_TEACHER20_SOURCE_COMPLETE_20261015 -->';text=goal.read_text(encoding='utf-8')
if marker not in text:
 text+=f'\n\n{marker}\n2026-10-15 `{d.RUN}` source-only 两行 E20 完整 ANALYZED / VERIFIED，8,880 更新和 40 epoch；无 U/EMA/换头/target。固定 RX1 排除教师 60.685185% 对等量 SAMPLE_80 的 99.444444%，纠正3/引入2096，CE升高；固定 A0 拒绝区纠正3/引入1787，科学资格 FAIL，关闭此配方，不扩库/学生/选择性救跑。原已见 RX 拟合 V 反而 +0.222222pp，不能用作跨域互补替代证据。固定源 V A0拒绝不代表预测错（拒绝区 EMA 99.423248%），不是实际 U 精度。RDrop 两预算三seed正收益独立保留，84%目标仍 active。[完整报告](../automation_reports/CV-SincNet/{d.RUN}/report.md)。\n'
 goal.write_text(text,encoding='utf-8',newline='\n')
names=('report.md','experiment.json','events.jsonl','results_analysis.json','summary.csv','paired_source_strata.csv','class_pair_errors.csv','training_complete_curves.csv','training_curves.png')
paths=[(A/n).relative_to(W).as_posix() for n in names]+['experiments/cvs_rx_teacher20/analyze.py','experiments/cvs_rx_teacher20/report.py',goal.relative_to(W).as_posix()]
d.write(W/'.codex_tmp/rx_teacher20_analysis_plan.json',dict(paths=paths,run_id=d.RUN,status='ANALYZED',message='Close source E20 cross RX teacher falsification with full evidence',evidence=str(P/'analysis_delivery.json')))
print(json.dumps(dict(status='ANALYZED',scientific_qualification='FAIL',source_summary=overview,qualification=q,goal_completed=False),ensure_ascii=False))
