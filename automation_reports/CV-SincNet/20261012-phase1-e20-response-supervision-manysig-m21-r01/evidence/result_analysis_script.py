from pathlib import Path
import sys,json,statistics as st,math,shutil
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_e20_response import design as d
p=R/'automation_reports/CV-SincNet'/d.RUN
src=d.read(p/'evidence/source_full_analysis.json');endpoint=d.read(p/'evidence/source_endpoint_audit.json');summary={r['arm']:r for r in d.read(p/'summary.json')['results']};scores=d.read(p/'scores.json')['results'];cost=d.read(p/'resources.json')['rows'];goal=d.read(p/'goal_assessment.json')
assert len(src['rows'])==len(endpoint['rows'])==len(cost)==21
fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
lines=['# E20+20有限响应身份监督：完整实验结果与源证据','','状态：VERIFIED / ANALYZED，21/21行完整完成。执行版本 `acdaf7e47908d8b27fc996d6b1b49787b23f6639`。每行对应纯CE旧门控cos、无显式CFO E20父权重续训20epoch/4440更新；累计E40。三种子固定，identity全可训练、无EMA、无星地增强。冻结D输入有梯度，D参数无更新。','','本批clean三种子均值没有达到84%。主组合RESP_LU为80.3940%，比同预算CONT_SSL增加0.0792个百分点；比额外clean eval对照低0.1474个百分点。三个响应arm都未一致超过SSL_EVAL，不能把相对纯CE的收益全部归因于物理响应监督。所有方案保留，下面的测试结果只作固定实验结论，不选择下一训练教师、超参数或重跑。','','## 完整clean成绩','','seed列按2026092701/02/03。每行168000个固定物理query；3528000个自身预测先固定后独立truth-last评分，294个整体/RX/TX和84个day分层。E20父参照复用原固定预测，不重新测试或选教师。','','|方案|seed01%|seed02%|seed03%|均值%|SD百分点|Macro-F1%|最差RX均值%|','|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in ['CE',*d.ARMS]:
 r=summary[arm];lines.append('|'+('原CE E20' if arm=='CE' else arm)+'|'+'|'.join(f'{100*r["seed_accuracy"][str(s)]:.4f}' for s in d.SEEDS)+f'|{100*r["accuracy_mean"]:.4f}|{100*r["accuracy_sd"]:.4f}|{100*r["macro_f1_mean"]:.4f}|{100*r["worst_rx_mean"]:.4f}|')
lines+=['','|方案|相对E20父CE百分点|相对CONT_CE百分点|相对CONT_SSL百分点|相对SSL_EVAL百分点|父错误修复/seed均值|父正确破坏/seed均值|净修复/seed均值|','|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in d.ARMS:
 r=summary[arm];a=[x for x in scores if x['arm']==arm and x['dimension']=='overall'];lines.append('|'+arm+'|'+'|'.join(f'{100*(r["accuracy_mean"]-summary[b]["accuracy_mean"]):.4f}' for b in ['CE','CONT_CE','CONT_SSL','SSL_EVAL'])+'|'+'|'.join(f'{st.mean(x[k] for x in a):.2f}' for k in ['parent_fix_count','parent_break_count','net_correct_gain'])+'|')
lines+=['','三seed配对和SD仅为描述统计，不宣称显著性。CONT_SSL比CONT_CE增加0.5272个百分点，SSL_EVAL在CONT_SSL上再增加0.2266个百分点。RESP_U相对CONT_SSL增加0.1825个百分点；主组合L/U叠加没有出现额外净收益。ACTION_LU不高于CONT_SSL。上述结果检验的是这组固定幅度、读出、权重和E20续训日程，不证明所有物理增强或多解耦都无效。','','## 所有目标RX分层','','表中为对应方案三种子等权准确率均值%；逐seed/day/TX、NLL、混淆矩阵、父修复/破坏均保存在scores.json、day_scores.json和CSV。','','|RX|'+'|'.join(d.ARMS)+'|','|---|'+'|'.join(['---:']*7)+'|']
for rx in sorted({r['stratum'] for r in scores if r['dimension']=='receiver'}):lines.append('|'+rx+'|'+'|'.join(f'{100*st.mean(r["accuracy"] for r in scores if r["arm"]==a and r["dimension"]=="receiver" and r["stratum"]==rx):.4f}' for a in d.ARMS)+'|')
lines+=['','## 完整源日志与成本','','实际analyzer消费全部93240步gzip记录、420个epoch JSONL/CSV，并闭合每行访问计数、20epoch预算、详细stdout和固定D参数/buffer状态。各半监督行完整覆盖56700合法U物理ID，没有U TX真值、live教师前向、EMA更新或星地生成。源V只用于E5/10/15/20健康记录，不选模。','','|arm|三seed训练总耗时均值s|主循环均值s|峰值CUDA均值MiB|峰值RSS均值MiB|clean推断均值s|E1 CE→E20 CE|E1额外损失→E20|','|---|---:|---:|---:|---:|---:|---|---|']
diagnostics=[]
for arm in d.ARMS:
 rows=[x for x in src['rows'] if x['arm']==arm];res=[x for x in cost if x['row_id'].startswith(arm+'-')];f=st.mean(x['epochs'][0]['ce'] for x in rows);z=st.mean(x['epochs'][-1]['ce'] for x in rows);ef=st.mean(x['epochs'][0]['extra'] for x in rows);ez=st.mean(x['epochs'][-1]['extra'] for x in rows)
 vals=[st.mean(x['source']['measured_training_resources'][k] for x in res) for k in ['total_seconds','main_seconds','peak_cuda_memory_bytes','peak_process_rss_bytes']];vals[2]/=1024**2;vals[3]/=1024**2;inf=st.mean(x['prediction']['inference_seconds']['clean'] for x in res)
 lines.append('|'+arm+'|'+'|'.join(f'{v:.2f}' for v in vals)+f'|{inf:.2f}|{f:.5f}→{z:.5f}|{ef:.5f}→{ez:.5f}|')
 endpoint_rows=[x for x in endpoint['rows'] if x['arm']==arm];cex=[x['audit']['encoder']['cosines']['ce:extra'] for x in endpoint_rows];ratios=[x['audit']['encoder']['weighted_gradient_norms']['extra']/max(x['audit']['encoder']['weighted_gradient_norms']['ce'],1e-12) for x in endpoint_rows]
 history_ratios=[]
 for row in rows:
  for audit in row['gradient_audits']:
   norms=audit['weighted_dense_gradient_norms'];den=norms['ce']['encoder'];history_ratios.append(norms['extra']['encoder']/den if den>1e-12 else None)
 modules=['core.time_fuse','core.t1','core.t2','core.t3','core.f1','core.f2','core.f3','core.pa_b1','core.pa_b2','core.pa_b3','response.response_projection']
 diagnostics.append(dict(arm=arm,endpoint_CE_extra_cosines=cex,endpoint_extra_CE_norm_ratios=ratios,training_audit_extra_CE_norm_ratios=history_ratios,source_CE_E1=f,source_CE_E20=z,extra_E1=ef,extra_E20=ez,module_relative_movement={k:[x['module_parameter_movement'][k]['relative_l2'] for x in endpoint_rows] for k in modules}))
lines+=['','硬件：NVIDIA GeForce RTX3090，各卡包含其他合法任务且总进程≤3；不同row共用资源的负载不同，耗时不作为受控加速比。训练可更新身份参数177025，各row资源原值见resources.json；外部双骨干/域模块并未进入身份优化。推断仅自身E40身份网络，不调用D、参考模型或peer。完整全源码的total_parameters是模型容器尺寸，不能等同实际可更新参数。','','三份E20父训练、其他两seed peer概率缓存生成，以及finite3的351拟合/351540辅助更新/7020epoch/1474200源包读出为既有共享成本，不包含在上述新增20epoch计时。RESP缓存原文件906082898bytes/seed，重复引用不是零成本；U weak前向是原半监督诊断的一部分，额外响应只在对应arm执行。完整每row访问/前向/作用次数见source completion。训练FLOPs未测量，记N/A。','','## 梯度与原生模块移动：可证明的边界','','发现记录缺口并保留：训练期间每444步记录梯度范数，但没有保存CE/extra余弦；训练移动仅encoder/fusion_head汇总，不包含细原生模块历史。不能把终点诊断写成完整训练中曲线。补充固定所有21行E40，各seed取相同私有DomainStream首批L128/U128；仅求梯度、不创建optimizer、不更新任何身份/D参数或buffer，无target/V前向。源诊断脚本在读取本批全局目标成绩前固定，全row执行。端点不是全源分布或更新因果估计。','','|arm|终点CE/extra骨干cos均值|终点额外/CE梯度范数均值%|训练10点/seed额外/CE范数均值%|','|---|---:|---:|---:|']
for x in diagnostics:
 vals=x['endpoint_CE_extra_cosines'];mean=None if any(v is None for v in vals) else st.mean(vals);hist=x['training_audit_extra_CE_norm_ratios'];hm=None if any(v is None for v in hist) else st.mean(hist);lines.append('|'+x['arm']+'|'+fmt(mean)+f'|{100*st.mean(x["endpoint_extra_CE_norm_ratios"]):.4f}|'+fmt(None if hm is None else 100*hm)+'|')
lines+=['','|arm|time_fuse移动%|t1移动%|t2移动%|t3移动%|f1移动%|f2移动%|f3移动%|PA_b1移动%|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in diagnostics:lines.append('|'+x['arm']+'|'+'|'.join(f'{100*st.mean(x["module_relative_movement"][k]):.4f}' for k in ['core.time_fuse','core.t1','core.t2','core.t3','core.f1','core.f2','core.f3','core.pa_b1'])+'|')
lines+=['','移动量为同一坐标E40−对应E20的L2范数/父L2范数；不是输出几何距离或因果性能贡献。完整模块包含Hz频率参数，梯度夹角/范数排除low_hz_/band_hz_，沿用记录的单位可比口径；余弦量不能跨不同参数化当绝对物理意义。RESP的extra只依浅中响应，因此fusion_head额外梯度为0是结构上的预期；主CE/KL仍更新身份头。','','源证据说明：新增响应项确实进入身份优化，四读出保持冻结，浅中原生模块确实改变；不能再把本批负结果简单解释为EMA使模型不动。响应预测损失与身份CE是不同任务，冻结D可能约束对操作敏感的坐标，却不保证这些坐标在未知RX下保持身份可分。小净效用与新增错误抵消只是本批事实；端点梯度大小/夹角和源拟合下降不能单独解释目标误差原因，也不授权据目标调高λ或挑某seed。','','随机R未进入身份去噪/置信门控，前导重复区间仍未由协议确认，均明确保留。浅中仅真实mean/std坐标，不宣称恢复完整时序图、全部RX逆映射或实现因果解耦。接下来的科学设计须从合法源表示/梯度/可辨识性证据独立预登记；本报告不发布目标反馈驱动的新配置。','','## 交付与目标','','本轮完整实现并测试的固定21行已闭合，84%目标NOT_MET。N09历史诊断缺口保留partial_diagnostics，补充终点证据不追溯修改健康run。所有负结果、各seed和成本保留；原checkpoint、预测和完整步日志仍存远端原路径，本地/Git保存小型完整指标和来源元数据。']
text='\n'.join(lines)+'\n';(p/'result_analysis.md').write_text(text,encoding='utf-8');(p/'final_analysis.md').write_text(text,encoding='utf-8');d.write(p/'evidence/source_diagnostic_summary.json',dict(status='VERIFIED',diagnostics=diagnostics,training_cosine_history_available=False,target_used_only_for_fixed_reporting=True))
with (p/'report.md').open('a',encoding='utf-8') as stream:stream.write('\n完整成绩与源诊断解释见[result_analysis.md](result_analysis.md)。训练余弦/细模块历史缺口明确保留，终点诊断不冒充历史。21行完整成绩，84%未达到。\n')
shutil.copy2(__file__,p/'evidence/result_analysis_script.py')
print(json.dumps(dict(status='VERIFIED',primary_mean=summary[d.PRIMARY]['accuracy_mean'],fixed_best_descriptive_mean=max(x['accuracy_mean'] for x in summary.values()),source_diagnostics=diagnostics),ensure_ascii=False))
