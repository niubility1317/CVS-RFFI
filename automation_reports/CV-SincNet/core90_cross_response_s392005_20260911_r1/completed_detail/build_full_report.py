"""Build a complete, source-backed research report without running experiments."""
import csv
import base64
import html
import json
import re
from pathlib import Path
import statistics
import sys

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
R = D.parent
W = Path('E:/type10-7/code/snapshots/core90_cross_response_20260911_wt')
NAME = 'CORE90_CROSS_RESPONSE_FULL_REPORT_20260911'
S = json.loads((D/'all_completed_summary.json').read_text(encoding='utf-8'))
V = list(S)
SCENES = ('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
figdir = D/'full_report_figures'
figdir.mkdir(exist_ok=True)
E, audit, longrows = {}, {}, []
for v in V:
    epochs = [json.loads(line) for line in (D/v/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert [r['epoch'] for r in epochs] == list(range(1,201)), v
    log = (D/v/f'{v}.log').read_text(encoding='utf-8')
    assert not any(t in log for t in ('Traceback (most recent call last)','CUDA out of memory')), v
    assert all(r['seed']==392005 and r['from_scratch'] and not r['baseline_ckpt'] for r in epochs), v
    E[v] = epochs
    audit[v] = {'epochs':len(epochs),'stdout_lines':len(log.splitlines()),'successful_steps':S[v]['activation']['counts']['successful_steps']}
    for e in epochs:
        keys = ['epoch','phase','val_tx_acc','train_tx_acc','epoch_time_s']
        keys += [k for k in e if k.startswith('train_cross_response_')]
        longrows.append({'variant':v,**{k:e.get(k) for k in keys}})
with (D/'full_report_epoch_curves.csv').open('w',encoding='utf-8',newline='') as f:
    fields = list(dict.fromkeys(k for row in longrows for k in row))
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(longrows)
with (D/'all_scores_long.csv').open(encoding='utf-8') as f:
    score_rows=list(csv.DictReader(f))
assert len(score_rows)==1480
assert len(longrows)==2000

colors=dict(zip(V,plt.get_cmap('tab10').colors))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for ax, key,title,ys in ((axs[0,0],'val_tx_acc','Source validation accuracy (%)',V),
    (axs[0,1],'train_tx_acc','Augmented training accuracy (%)',V),
    (axs[1,0],'train_cross_response_loss_response','Training response MSE (unweighted)',[v for v in V if S[v]['activation']['config']['response_enabled']]),
    (axs[1,1],'train_cross_response_decision_gap','Training decision gap (dynamic references)',[v for v in V if S[v]['activation']['config']['decision_enabled']])):
    for v in ys:
        ax.plot(range(1,201),[e.get(key,0) for e in E[v]],label=v,color=colors[v],linewidth=1.1,alpha=.85)
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.18)
    ax.axvline(80,color='gray',ls=':',lw=.8); ax.axvline(131,color='gray',ls='--',lw=.8)
    ax.legend(fontsize=7,ncol=2)
fig.savefig(figdir/'training_curves.png',dpi=170); plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for v in V:
    clean=S[v]['groups']['clean']['aggregate']['tx_acc']
    leo=statistics.mean(S[v]['groups'][s]['aggregate']['tx_acc'] for s in SCENES[1:])
    hrs=S[v]['resource']['wall_time_seconds']/3600
    axs[0].scatter(clean,leo,color=colors[v],s=55)
    axs[0].annotate(v,(clean,leo),xytext=(4,5 if v!='U5' else -12),textcoords='offset points',fontsize=8)
    axs[1].scatter(hrs,leo,color=colors[v],s=55)
    axs[1].annotate(v,(hrs,leo),xytext=(4,5 if v!='U5' else -12),textcoords='offset points',fontsize=8)
axs[0].set_xlabel('Clean accuracy (%)'); axs[1].set_xlabel('Observed wall time (h; concurrent workload)')
for ax in axs:
    ax.set_ylabel('Mean LEO accuracy (%)'); ax.grid(alpha=.18); ax.margins(.2)
axs[0].set_title('Identity performance trade-off'); axs[1].set_title('Observed performance / cost')
fig.savefig(figdir/'performance_cost.png',dpi=170); plt.close(fig)

rows=[(D/'full_report_narrative.md').read_text(encoding='utf-8'),
'## 8.身份性能与成本概览\n',
'![clean、LEO与墙钟成本](completed_detail/full_report_figures/performance_cost.png)\n',
'图中为固定E200终态，同机并发墙钟不用于隔离效率结论；图和后续全表来自同一份已复核评分。\n']
tables=(D/'all_results.md').read_text(encoding='utf-8')
tables=tables[tables.index('## 总体准确率'):tables.index('## 科学解释')]
# Preserve every receiver, class, F1 and cost table, but nest under the full report.
tables=re.sub(r'^(#{2,3}) ',lambda m:m[1]+' ',tables,flags=re.M)
rows.append(tables)
rows += ['## 9.全程训练与源验证曲线\n',
'![2000轮完整曲线](completed_detail/full_report_figures/training_curves.png)\n',
'图中每条曲线使用全部200轮，无抽样替代；E80标出卫星辅助CE开始计入，E131为pseudo阶段开始。训练准确率受增强和训练分类头语义影响，不能与clean源验证直接相减解释泛化差距。训练响应MSE和决策缺口的参照/样本随训练变化，不是固定目标测试曲线。\n',
'|变体|最终源V准确率|最佳源V准确率（epoch，仅遥测）|label小时|pseudo小时|最低epoch更新比例|\n|---|---:|---:|---:|---:|---:|']
for v in V:
    best=max(E[v],key=lambda e:e['val_tx_acc']); p=S[v]['phase_stats']
    rows.append(f"|{v}|{E[v][-1]['val_tx_acc']:.4f}|{best['val_tx_acc']:.4f}（E{best['epoch']}）|{p['label']['epoch_seconds']/3600:.4f}|{p['pseudo']['epoch_seconds']/3600:.4f}|{min(x['minimum_update_fraction'] for x in p.values()):.6f}|")
rows += ['\n最佳源V仅描述已有曲线，实际评价始终使用E200，未重新选checkpoint。label/pseudo epoch时间不含全部终态评估/调度开销，不能与端到端墙钟混为同一口径。\n',
'Ux有一次明显短暂波动：E145/E146/E147的源V准确率为97.5519/81.5037/97.3111%，训练准确率为89.5408/77.1524/54.7991%，训练loss为6.8021/9.5102/12.3814。三轮非有限loss/梯度跳步均0；源V下一轮恢复，不能写成持续技术故障或全程无波动。身份尺度同期3.5813→3.2437→3.1831，存在共同变化，但不足以把该波动归因于特征收缩；没有为此重训或选另一checkpoint。通用nonfinite_metric_count含未评估指标占位，不能直接视为非有限优化事件。\n',
'### 9.1分阶段机制量\n',
'|变体|epoch区间|响应MSE|决策缺口|有效决策参照比例|身份尺度|TX间展开|身份交互量|\n|---|---|---:|---:|---:|---:|---:|---:|']
for v in V[1:]:
    for lo,hi in ((1,20),(21,79),(80,130),(131,200)):
        a=S[v]['activation']['config']
        def avg(key):return statistics.mean(e.get('train_cross_response_'+key,0) for e in E[v][lo-1:hi])
        vals=[f'{avg("loss_response"):.5f}' if a['response_enabled'] else 'N/A',
              f'{avg("decision_gap"):.5f}' if a['decision_enabled'] else 'N/A',
              f'{avg("decision_valid_fraction"):.5f}' if a['decision_enabled'] else 'N/A',
              *(f'{avg(k):.5f}' for k in ('identity_scale','identity_between_tx','identity_interaction'))]
        rows.append(f'|{v}|E{lo}—{hi}|'+'|'.join(vals)+'|')
rows += ['\n这些量为区间内epoch均值的平均，N/A表示任务关闭。交互量小不代表TX/RX解耦，身份尺度变化也不能单独替代分类收益。\n',
'## 10.响应机制检验：实际使用域描述，但拟合不等于识别\n',
'下表为E200固定源验证16块、四种角色共64个查询任务。512次记录读取对应505个不同物理记录，块间允许重复，不应称为512个互异样本。全部行常数MSE=0.538725；统计已按源训练尺度标准化。\n',
'|变体|辅助参数|响应MSE|响应/常数|打乱RX描述MSE|更换供体TX MSE|交互MSE|首次实际身份响应梯度epoch|\n|---|---:|---:|---:|---:|---:|---:|---|']
for v in V:
    a=S[v]['activation']; se=a['source_evaluations']
    if not se: continue
    d=se[-1]
    rows.append(f"|{v}|{a['auxiliary_parameters']}|{d['response_mse']:.6f}|{d['gate_ratio']:.6f}|{d['shuffled_rx_mse']:.6f}|{d['changed_donor_tx_mse']:.6f}|{d['interaction_mse']:.6f}|{S[v]['first_joint_epoch'] if S[v]['first_joint_epoch'] is not None else 'N/A：按定义关闭'}|")
rows += ['\n打乱RX描述后误差明显增大，支持预测器实际使用域描述；更换供体TX后误差小幅变差，提供有限源条件下的复用证据。U4_bilinear比U4_additive的响应/交互MSE低，但身份clean/LEO却更低，表明辅助统计拟合与身份收益并不等价。head_only在不向主干施加响应梯度时也达到接近的MSE，因此不能仅用拟合改善证明主干更好。\n',
'head_only/permanent_detach可以记录源验证门达标，但joint_open还受对照开关约束；实际身份响应梯度必须保持0。U3未做响应源验证，其gate关闭为N/A，而不是未激活故障。门只比较常数参照，当前证据不支持“门同时要求超过加性模型”的说法。\n',
'## 11.覆盖反馈的实际执行与限制\n',
'|变体|候选数|历史项|联合矩形|定向迁移|唯一L_s记录曝光|epoch平均选择概率最小—最大|\n|---|---:|---:|---:|---:|---:|---|']
for v in V[1:]:
    a=S[v]['activation']; c=a['sampling_evidence']
    probabilities=[e['train_cross_response_selection_probability'] for e in E[v]]
    rows.append(f"|{v}|{a['candidate_count']}|{c['feedback_history_entries']}|{c['joint_rectangles']}|{c['directed_transfers']}|{c['unique_exposed_physical']}|{min(probabilities):.8f}—{max(probabilities):.8f}|")
rows += ['\n选择概率为已选块概率的epoch均值范围，不能解释为全部候选概率的全局最小/最大。均匀行约为1/128；U5按历史改变分布。最终覆盖同为450个联合矩形、626个定向迁移和6300条L_s记录，只证明所记录覆盖达到同一末态，不证明U5更早覆盖或更均衡。当前报告未计算覆盖达成时间，不能添加该收益。\n',
'## 12.逐变体科学解释\n',
'|变体|本次支持的事实|不能据此声称|\n|---|---|---|',
'|U0|当前契约下的CORE90方法参照，完整E200闭合|历史checkpoint逐位复现|',
'|U1|总体三LEO均值最高；组织/隔离本身带来明显差异|全部改善仅由单个采样因素造成|',
'|U2|响应域/身份路径实际激活，源拟合优于常数；总体四场景低于U1|响应监督改善了身份泛化|',
'|U3|clean最高；比U1的未知日期弱RX下限更高；LEO均值略降|全面优于U1或已证明多seed稳健|',
'|U4_additive|纯双因素差值clean与LEO均值小幅为正，但性能未超过U1/U3|统计显著的协同或联合胜出|',
'|U4_bilinear|源响应及交互拟合优于加性，身份表现低于加性及detach对照|保留双线性交互必然利于身份|',
'|U5|较U4_bilinear恢复clean/LEO；仍低于U1总体表现|反馈调度让完整方案成为最优|',
'|Ux|身份交互约束实际反传，但总体低于U1|交互压零等于解耦|',
'|head_only|无需响应主干梯度即可达到较好统计拟合|身份主干因响应任务得到改进|',
'|permanent_detach|响应身份梯度为0，clean高于U4_bilinear|永远detach在所有条件下最优|',
'\n### 12.1U3的弱侧收益\n',
'相对U1，U3总体clean+0.5879个百分点，未知日期/未知RX整体clean由73.6476%到75.0738%；该子集最弱RX2的clean由52.1167%到61.8167%，提高9.7个百分点。三种LEO最弱RX2分别提高3.4667、3.2、3.3667个百分点，尽管LEO总体均值下降0.2803个百分点。平均表现与下限存在取舍，不能只按一个总体数决定路线。详细机制和完整600轮重读见[U3专项报告](completed_detail/u3_analysis.md)。\n',
'### 12.2当前可识别的缺口\n',
'完整矩阵已完成，没有因性能较差停止、重训或调门。未完成的是科学证据：独立训练seed、不依赖当前target反馈的确认、目标未知类与Phase2/Phase3、隔离成本profile、真实卫星验证。模型选择与新实验必须遵守冻结确认集边界；本报告不授权或执行后续训练。\n',
'若继续研发，可在合法源侧研究响应拟合与身份梯度方向是否冲突、决策保护在弱RX上的稳定性及循环实现开销，并用独立预登记证据验证。当前数据不支持直接扩大双线性容量、降低梯度门槛或按target结果选择性重跑。\n',
'## 13.实现追踪与验证证据\n']
verification=(W/'analysis/CORE90_CROSS_RESPONSE_IMPLEMENTATION_VERIFICATION_20260911.md').read_text(encoding='utf-8')
trace=verification[verification.index('|ID|已实现'):verification.index('## 防止静默失活')]
rows.append(trace.replace('正式E200结果尚不存在','当时为本地验证；正式结果现已完成，见本报告'))
rows += ['\nCR01—CR20的本地实现追踪为20项已验证、0项待实现；其中可选能力正式关闭，不计为本次运行已启用。没有给未知多seed/部署科学结论填写“已验证”。\n',
'### 13.1代码模块索引\n',
'|文件|职责|\n|---|---|']
modules={'schema.py':'物理记录与完整块契约','sampler.py':'有界候选、完整块和固定步数','roles.py':'供体/查询角色与MixStyle mask','statistics.py':'固定FFT/自相关/IQ/事件统计与源尺度','readouts.py':'共享TX/RX描述读出','predictor.py':'加性/低秩双线性预测','tensor_ops.py':'双重中心化、交互与响应损失','decision.py':'单侧跨RX分类裕量保护','training.py':'源门与逐损失梯度路由','integration.py':'真实训练连接、状态与激活','coverage.py':'联合/定向关系覆盖','scheduler.py':'历史可靠性、探索与成本概率','source_eval.py':'固定源验证、供体置换检验','final_scoring.py':'最终预测完整性与独立评分','config.py':'参数/开关契约','baseline_compat.py':'历史损失局部适配','legacy_losses.py':'历史B02公式保存','unlabeled.py':'U_s真标签隐藏','analysis.py':'匹配变体与配对差分解释'}
for file,desc in modules.items():
    rows.append(f'|[{file}](../../../code/cvsrffi/cross_response/{file})|{desc}|')
rows += ['\n集成入口：[train_ssdg.py](../../../code/SSDG/train_ssdg.py)、[model_dual_cvsincnet.py](../../../code/model_dual_cvsincnet.py)、[model.py](../../../code/model.py)；矩阵与发布：[matrix](../../../code/scripts/core90_cross_response_matrix.py)、[dispatcher](../../../code/scripts/run_core90_cross_response_experiment.py)、[inspector](../../../code/scripts/inspect_core90_cross_response_experiment.py)。\n',
'### 13.2可复查验证文件\n']
for label,path in [('设计实施计划','analysis/ADV3B02_CORE90_CROSS_RESPONSE_IMPLEMENTATION_PLAN_20260911.md'),('基线审计','analysis/CORE90_CROSS_RESPONSE_BASELINE_AUDIT.md'),('实现验证汇总','analysis/CORE90_CROSS_RESPONSE_IMPLEMENTATION_VERIFICATION_20260911.md'),('JUnit','analysis/core90_cross_response_verification_20260911/unit_results.xml'),('十变体CUDA','analysis/core90_cross_response_verification_20260911/matrix.json'),('AMP','analysis/core90_cross_response_verification_20260911/amp.json'),('U0关闭回归','analysis/core90_cross_response_verification_20260911/u0_off.json'),('U5续跑','analysis/core90_cross_response_verification_20260911/resume.json'),('部署输出','analysis/core90_cross_response_verification_20260911/deployment.json')]:
    rows.append(f'- [{label}](../../../{path})')
rows += ['\n## 14.运行版本、命令与checkpoint定位\n']
pub=json.loads((R/'publication.json').read_text(encoding='utf-8'))
rows += [f"训练release：`{pub['remote_release']}`。普通用户环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；N607为RTX3090计算节点。本地验证为ssr-gpu CUDA环境。具体版本日志以原始发布/训练文件为准，本报告不把本地PyTorch版本当远端版本。\n",
'正式发布命令（历史记录，不应再次执行同run）：\n','```text\n'+pub['launch_command']+'\n```\n',
'|变体|E200 checkpoint SHA256|远端相对run路径|\n|---|---|---|']
for v in V:
    rows.append(f"|{v}|`{S[v]['checkpoint_sha256']}`|`runs/{pub['run_id']}/{v}/final_ssdg.pth`|")
rows += ['\n远端以上路径相对`/home/szu2070436088/2510044040/CV-SincNet`。大checkpoint和逐条预测保留在原run，报告不复制或覆盖它们。单个文件hash用于定位已有产物，不构建额外审批链。\n',
'## 15.完整冻结配置\n',
'以下为正式发布配置，包括历史基线有效值、交叉任务默认值及变体覆盖。使用时必须通过variant解析，不能仅看到非零lambda就判断某任务启用。\n',
'```json\n'+(W/'code/configs/phase1_core90_cross_response_s392005_20260911.json').read_text(encoding='utf-8').strip()+'\n```\n',
'## 16.结果文件与复核方式\n',
'- [全部1480组评分CSV](completed_detail/all_scores_long.csv)：计数、总数和准确率，含命名别名。\n- [十项完整解析摘要](completed_detail/all_completed_summary.json)：评分、激活、资源和源划分。\n- [2000轮曲线CSV](completed_detail/full_report_epoch_curves.csv)：逐epoch指标，未平滑覆盖原值。\n- [原三项预测审计](completed_detail/prediction_audit.json)与[其余七项审计](completed_detail/prediction_audit_new7.json)：全量预测复算及混淆矩阵。\n- [所有配对差值](completed_detail/all_contrasts.json)。\n- [发布记录](publication.json)、[source预检](source_preflight.json)、[启动核对](startup_verification.json)。\n- [历史阶段报告](report.md)：保留启动、监控和分批完成快照。\n- [报告生成脚本](completed_detail/build_full_report.py)与[报告正文源](completed_detail/full_report_narrative.md)。\n',
'本次生成重新完整解析2000轮和全部stdout，检查连续epoch、seed/scratch及评分行数；图、机制表和源曲线均由同一解析数据生成。没有重新运行GPU测试、目标推理或训练。历史数学/运行验证引用已有证据，不冒充本次重跑。\n']
md='\n\n'.join(rows)
# Keep consecutive Markdown table rows adjacent; section prose stays separated.
md=re.sub(r'(?<=\|)\n\s*\n(?=\|)', '\n', md)
md=md.replace('](../../../', ']('+W.as_posix()+'/')
# All headings become a navigable table of contents in the HTML export.
mdpath=R/(NAME+'.md'); mdpath.write_text(md,encoding='utf-8')
assert '\ufffd' not in mdpath.read_text(encoding='utf-8')
broken=[]
for target in re.findall(r'\]\(([^)]+)\)',md):
    if target.startswith(('http:', 'https:','#')):continue
    # Root report lives outside worktree; ../../../code & analysis resolve in W.
    resolved=W/target[len('../../../'):] if target.startswith('../../../') else R/target
    if not resolved.exists():broken.append(str(resolved))
assert not broken, broken
body=markdown.markdown('[TOC]\n\n'+md,extensions=['tables','fenced_code','toc','sane_lists'])
body=body.replace('href="../../../','href="'+W.as_posix()+'/')
for png in figdir.glob('*.png'):
    body=body.replace('src="completed_detail/full_report_figures/'+png.name+'"', 'src="data:image/png;base64,'+base64.b64encode(png.read_bytes()).decode('ascii')+'"')
style='''body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#202b3c;background:#f5f7fb;margin:0}main{max-width:1180px;margin:auto;background:white;padding:48px 64px}h1{font-size:32px;line-height:1.4}h2{margin-top:56px;border-bottom:2px solid #dce5ed;padding-bottom:10px}h3{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto;margin:20px 0}th,td{border:1px solid #dbe3ed;padding:8px 10px;text-align:left;white-space:nowrap}th{background:#eaf1f8}tr:nth-child(even){background:#f8fafc}code{font-family:Consolas,monospace;font-size:.9em}pre{overflow:auto;background:#f0f3f7;padding:18px}p code{overflow-wrap:anywhere}a{color:#175ea4}img{max-width:100%;height:auto}.toc{font-size:14px;border-left:3px solid #8aaac6;padding:12px 20px}.toc ul{list-style:none;padding-left:18px}@media(max-width:750px){main{padding:20px}h1{font-size:26px}}@media print{body{background:white}main{padding:0;max-width:none}h2{break-after:avoid}tr{break-inside:avoid}a{color:inherit}.toc{break-after:page}pre{white-space:pre-wrap}table{display:table;font-size:9px}td,th{white-space:normal;padding:4px}}'''
(R/(NAME+'.html')).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CORE90交叉响应完整报告</title><style>'+style+'</style><main>'+body+'</main></html>',encoding='utf-8')
(D/'full_report_validation.json').write_text(json.dumps({'status':'VERIFIED_LOCAL_REPORT','variants':len(V),'epochs':len(longrows),'score_groups':len(score_rows),'logs':audit,'broken_local_links':broken,'report_characters':len(md),'report_lines':len(md.splitlines()),'evidence_cutoff':'2026-09-11T14:44:00+08:00','new_training_or_target_inference':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report':str(mdpath),'characters':len(md),'lines':len(md.splitlines()),'audit':audit},ensure_ascii=False))
