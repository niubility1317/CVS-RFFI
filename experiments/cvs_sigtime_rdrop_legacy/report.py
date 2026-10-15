"""Publish the complete predeclared comparison, including source-failed skips."""
from pathlib import Path
import sys,json,csv,io,importlib.util,statistics
W=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W),str(W/'code')]
from experiments.cvs_sigtime_rdrop_legacy import design as d
from experiments.cvs_sigtime_rdrop_legacy.publish import PYTHON
A=W/'automation_reports/CV-SincNet'/d.RUN;P=W/'local_artifacts'/d.RELEASE
def public(value):
 if isinstance(value,str):return value.replace(d.PROJECT,'PROJECT').replace(PYTHON.rsplit('/bin/python',1)[0],'PYTHON_ENV')
 if isinstance(value,list):return [public(a) for a in value]
 if isinstance(value,dict):return {k:public(a) for k,a in value.items()}
 return value
def table(name,rows):
 if not rows:raise ValueError('Expected nonempty audit table '+name)
 buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows);(A/name).write_text(buf.getvalue(),encoding='utf-8',newline='\n')
def number(x,scale=1.,suffix='',signed=False):return '缺项' if x is None else format(x*scale,'+.6f' if signed else '.6f')+suffix
def main():
 if (A/'results_analysis.json').is_file():print('ALREADY_RECORDED');return
 v=d.read(P/'independent_readback.json');x=d.read(P/'independent_analysis.json')
 assert x['status']=='VERIFIED' and v['completion']['status']==x['terminal_status'] and not v['dispatcher'] and not any(r['process'] for r in v['rows']) and not v['failure']
 actual=x['actual_totals'];assert len(x['rows'])==actual['models'] and len(x['records'])==actual['metric_records']==18*actual['models']
 x=public(x);overview=[];curves=[];cost=[]
 for rid,r in x['rows'].items():
  overall=next(a for a in x['records'] if a['row_id']==rid and a['dimension']=='overall');source=r['source_V'];last=r['complete_epoch_curves'][-1]
  overview.append(dict(row_id=rid,arm=r['arm'],seed=r['model_seed'],epochs=r['epochs'],accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],source_V_accuracy=source['source_val_accuracy'],source_V_worst_RX=source['source_val_worst_rx'],final_train_CE1=r['final_training_L_CE'],final_pairedCE=last.get('legacy_rdrop/paired_CE'),final_KL=last.get('legacy_rdrop/symmetric_KL'),selected_U=r['pseudo_selected_observations'],U_observations=r['pseudo_observations'],training_seconds=r['elapsed_seconds'],prediction_seconds=r['prediction_seconds'],feature_L2_delta=r['feature_parameter_L2_delta'],default_W_L2_delta=r['default_W_L2_delta'],source_health=r['source_health']))
  for epoch in r['complete_epoch_curves']:curves.append(dict(row_id=rid,arm=r['arm'],seed=r['model_seed'],**epoch))
  for stage,name in ((True,'L_only'),(False,'L_plus_U')):
   eps=[a for a in r['complete_epoch_curves'] if a['label_stage']==stage]
   cost.append(dict(row_id=rid,arm=r['arm'],seed=r['model_seed'],stage=name,epochs=len(eps),mean_epoch_seconds=statistics.mean(a['epoch_time_s'] for a in eps),total_epoch_seconds=sum(a['epoch_time_s'] for a in eps)))
 table('summary_by_seed.csv',overview)
 # ORIGINAL has no paired hook fields. Preserve absent values instead of
 # fabricating observed zeros, and use a stable union of CSV columns.
 curvekeys=list(dict.fromkeys(k for r in curves for k in r));table('training_complete_curves.csv',[{k:r.get(k) for k in curvekeys} for r in curves])
 table('paired_errors.csv',x['paired_errors']);table('paired_strata.csv',x['stratum_pairs']);table('training_stage_cost.csv',cost)
 table('metrics_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in x['records']]);d.write(A/'results_analysis.json',x)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
 colors={'E20_ORIGINAL':'#575e68','E20_TWO_CE':'#24549b','E20_RDROP':'#c34e36','E50_ORIGINAL':'#989fa8','E50_TWO_CE':'#459eac','E50_RDROP':'#8d4e9d'}
 keys=('train/loss_tx_labeled','legacy_rdrop/symmetric_KL','train/loss','source_V_accuracy_percent','observed/U_selected','epoch_time_s')
 titles=('Native smoothed labeled CE1','Symmetric KL (paired arms only)','Actual training objective','Source V accuracy (%)','Accepted U / update','Epoch wall time (s)')
 for arm in d.ARMS:
  rs=[r for r in x['rows'].values() if r['arm']==arm]
  if not rs:continue
  count=rs[0]['epochs'];grid=range(1,count+1)
  for ax,key in zip(axes.flat,keys):
   if any(key not in r['complete_epoch_curves'][0] for r in rs):continue
   values=[[r['complete_epoch_curves'][i][key] for r in rs] for i in range(count)];means=[statistics.mean(a) for a in values]
   ax.plot(grid,means,label=arm+' n='+str(len(rs)),color=colors[arm],linewidth=1.5)
   if len(rs)>1:
    sds=[statistics.stdev(a) for a in values];ax.fill_between(grid,[a-b for a,b in zip(means,sds)],[a+b for a,b in zip(means,sds)],color=colors[arm],alpha=.09)
 for ax,title in zip(axes.flat,titles):ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.suptitle('Original SIG_TIME | RAW_HZ / normal augmentation / LS .01 | mean +/- sample SD when n>1',fontsize=11)
 fig.savefig(A/'training_curves.png',dpi=170);plt.close(fig)
 title=(A/'report.md').read_text(encoding='utf-8').split('\n',1)[0];complete=x['terminal_status']=='ARTIFACTS_COMPLETE'
 lines=[title,'',f"状态 **ANALYZED / 实际 {actual['models']}/18 行训练、clean 预测、独立 truth-last 评分与完整产物分析 VERIFIED**。"+('18 行完整，两个预算各三个条件、三个种子。' if complete else '源资格未通过的预算按固定规则关闭其未启动种子，缺项明确保留，未根据测试成绩救跑。')+'全研究目标保持 active。','',
 '| 条件 | seed01 | seed02 | seed03 | 实际种子数 | 均值 ± 样本 SD | 最差 RX 均值 |','|---|---:|---:|---:|---:|---:|---:|']
 for arm in d.ARMS:
  s=x['summary'][arm];ps={r['model_seed']:r for r in s['per_seed']};values=[number(ps[seed]['accuracy'],100,'%') if seed in ps else '缺项' for seed in d.SEEDS]
  lines.append('| '+arm+' | '+' | '.join(values)+f" | {s['seed_count']}/3 | {number(s['accuracy_mean'],100,'%')} ± {number(s['accuracy_sample_SD'],100,' pp')} | {number(s['worst_RX_accuracy_mean'],100,'%')} |")
 lines+=['','| 固定配对 | 平均差 | 配对样本 SD | 正收益种子 |','|---|---:|---:|---:|']
 for name,s in x['paired_summary'].items():lines.append(f"| {name} | {number(s['mean_pp'],suffix=' pp',signed=True)} | {number(s['sample_SD_pp'],suffix=' pp')} | {s['positive_seeds']}/{s['seed_count']} |")
 lines+=['',f"预登记主条件 E50_RDROP 的完整三种子 84% 检查为 {x['E50_RDROP_goal84_check']}；E20_RDROP 为 {x['E20_RDROP_goal84_check']}。种子不足三时不作完整目标达成判断。该研究沿用已暴露的固定 ManySig clean 基准，不声明新的独立盲测、未见 TX、星地或 Phase2 能力。",'',
 '本次直接保留旧 SIG_TIME 的 RAW_HZ、普通训练增强、0.01 标签平滑、NoExplicitCFO、默认 6×160 cosine 头联合训练以及原 U/ownEMA 配方。E20 为13L+7U，E50 为33L+17U，分别从零训练并固定 final。ORIGINAL 保留原单 CE 路径，无辅助 hook；TWO_CE 与 RDROP 都在同一增强后 IQ 上做两次随机前向，CE 取均值，只有 RDROP 添加固定 alpha=1 的对称 KL。没有第二次增强重抽、另训头、新参数或新 U 一致性。','',
 '两类比较分别回答不同问题：RDROP−ORIGINAL 是整个组合的净效果，RDROP−TWO_CE 隔离一致性项，TWO_CE−ORIGINAL 反映增加随机前向与平均 CE 的影响。E50−E20 同时改变总轮次、cosine 日程终点与 U 阶段轮次，不能解释为同一 E20 轨迹的单纯续训。三条件每 seed 的初始 state 相同，ORIGINAL 的后续随机数消耗因单前向而不同，不声称训练轨迹逐位相同。','',
 '历史 SIG_TIME 83.0393% ±1.4557 pp 是保留的非配对背景。本次的新 ORIGINAL 才是同配方对照；此前 RDrop E50 82.1030% 属于 CENTER25K/cleanL/LS0 配方，不能据此判断对旧 SIG_TIME 的因果增益。没有继承任何历史高分权重，也没有按测试成绩选择父模型、轮次、主条件、种子或重跑。','',
 '| 预算源资格 | 判定 | L 健康 | KL 正值调用比例 |','|---|---|---|---:|']
 for e,g in x['source_gates'].items():lines.append(f"| E{e} | {g['status']} | {g['all_final_L_healthy']} | {g['positive_KL_call_fraction']*100:.6f}% |")
 lines+=['','每预算首 seed 的三个条件先完成 source freeze，固定源规则只读取最终 L CE、源 V 均值/最差 RX 和 KL 信号，两个预算独立决定其余六行。完整分析重算判定并核对扩展 launch 时间晚于所需 source freeze；target_metrics_read=False。实际跳过行：'+(', '.join(r['row_id'] for r in x['source_skipped']) or '无')+'。','',
 f"独立 CPU 分析解析全部 {actual['optimizer_steps']} 条 step、{actual['epochs']} 条 epoch、实际各行训练和评分 stdout，核对 {2*actual['models']} 份初末 checkpoint、{actual['clean_decisions']} 个既有决定及 {actual['metric_records']} 个分层混淆矩阵。每行物理角色为 6300 L、56700 隐藏 U、27000 V，空祖先、ownEMA、固定 final、headfit=0；各 seed 初始 state 核对为 {x['initial_state_equal_count_by_seed']}。",'',
 f"实际损失逐步重构的最大绝对误差为 {max(r['objective_max_abs_error'] for r in x['rows'].values()):.3g}。paired 条件只在 native CE1 上补 .5(CE2−CE1)+alphaKL，标签平滑保持0.01；ORIGINAL 无额外项。所有相关损失有限。原生熵未独立记录，`derived/original_weighted_U_entropy` 是 closed loss 扣除 CE1 和加权 U 损失的派生量。U 接受数不是 U 真实正确率，U 真值读取为0。",'',
 f"全部实际模型的特征参数和默认 W 均更新。额外 {x['separate_synthetic_boundary_forwards']} 个合成 packet 前向核对初末 joint160 改变、feature-only 与部署 joint160 逐位相同，以及扰动 W 不改变 joint160。该分析新增真实 IQ 读取、新预测、优化、拟合头和 checkpoint 改写均为0；全行 prediction 完整验证后才独立读 truth 复算。特征路径检查本身不代替泛化效果证据。",'',
 f"预登记完整预算为630 model-epoch/139860更新；实际完成 {actual['epochs']} model-epoch/{actual['optimizer_steps']} 更新、{actual['extra_L_identity_forwards']} 次额外 L 前向、{actual['U_observations']} 次 U 重复观察。矩阵首次准入至评分汇总墙钟 {x['matrix_elapsed_seconds']:.2f} s，实测吞吐 {x['aggregate_completed_rows_per_hour']:.3f} 模型/小时，受共享资源负载影响。原普通增强、每轮源 V 和预测成本保留于实测时间，分阶段见 CSV。所有自有进程自然结束，未干预其他任务。",'',
 '旧 RDrop、CENTER25K、NONE 等正证据和所有负结果独立保留；未发布的残差交换原型没有叠加本轮。RDrop 是已有方法，本次验证与旧 SIG_TIME 的实际结合效果，不声明全球方法创新。研究目标不会仅凭一个最高分自动关闭。','',
 '![完整训练曲线](training_curves.png)','',
 '执行代码与冻结配置提交：`'+x['execution_commit']+'`。完整预登记、一次 P0/P1、本地验证与启动证据见 experiment.json、review_evidence.json、local_verification.json、launch_verification.json；完整结果与配对分层见 results_analysis.json 和 CSV。']
 (A/'report.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8',newline='\n')
 spec=d.read(A/'experiment.json');spec['result']=dict(status='ANALYZED',terminal_status=x['terminal_status'],evidence_ref='results_analysis.json',actual_totals=actual,source_skipped=x['source_skipped'],primary='E50_RDROP',E50_RDROP_goal84_check=x['E50_RDROP_goal84_check'],E20_RDROP_goal84_check=x['E20_RDROP_goal84_check'],goal_completed=False);d.write(A/'experiment.json',spec)
 mod=importlib.util.spec_from_file_location('registry',W.parents[2]/'tools/experiment_registry.py');reg=importlib.util.module_from_spec(mod);mod.loader.exec_module(reg)
 reg.event(W,d.RUN,'ANALYZED',['results_analysis.json','summary_by_seed.csv','paired_errors.csv','paired_strata.csv','training_complete_curves.csv'],f"Original SIG_TIME RDrop completed {actual['models']}/18; full {actual['optimizer_steps']} steps/{actual['epochs']} epochs/{actual['clean_decisions']} existing decisions/{actual['metric_records']} CM independently recounted; explicit source skips preserved; no new predictions or fitting");reg.build(W,managed_only=True)
 goal=W/'docs/CVS_PHASE1_FEATURE_EXTRACTION_GOAL_20261013.md';marker='<!-- ORIGINAL_SIGTIME_RDROP_COMPLETE_20261015 -->';text=goal.read_text(encoding='utf-8')
 if marker not in text:
  summary='；'.join(arm+' '+number(s['accuracy_mean'],100,'%')+f" n={s['seed_count']}" for arm,s in x['summary'].items())
  contrasts='；'.join(k+' '+number(s['mean_pp'],suffix='pp',signed=True)+f" 正{s['positive_seeds']}/{s['seed_count']}" for k,s in x['paired_summary'].items())
  text+=f'\n\n{marker}\n2026-10-15 `{d.RUN}` 实际{actual["models"]}/18行已ANALYZED/VERIFIED，{actual["optimizer_steps"]}更新/{actual["epochs"]}epoch，保留旧RAW_HZ/LS.01/普通增强/默认头joint配方。{summary}。{contrasts}。E50_RDROP/E20_RDROP完整三seed84%检查分别{x["E50_RDROP_goal84_check"]}/{x["E20_RDROP_goal84_check"]}；不足三seed不冒充完整矩阵。固定源资格及缺项保留，无target反馈或选择性救跑。既有正负证据和未发布原型保留，RDrop不作为全球新方法；goal保持active。[完整报告](../automation_reports/CV-SincNet/{d.RUN}/report.md)。\n'
  goal.write_text(text,encoding='utf-8',newline='\n')
 names=('experiment.json','events.jsonl','report.md','results_analysis.json','summary_by_seed.csv','training_complete_curves.csv','paired_errors.csv','paired_strata.csv','training_stage_cost.csv','metrics_strata.csv','training_curves.png')
 paths=[(A/n).relative_to(W).as_posix() for n in names]+['experiments/cvs_sigtime_rdrop_legacy/analyze.py','experiments/cvs_sigtime_rdrop_legacy/report.py',goal.relative_to(W).as_posix()]
 d.write(W/'.codex_tmp/original_SIGTIME_rdrop_analysis_plan.json',dict(paths=paths,run_id=d.RUN,status='ANALYZED',message='Analyze original SIG_TIME RDrop E20 E50 default head comparison',evidence=str(P/'analysis_delivery.json')))
 print(json.dumps(dict(status='ANALYZED',actual_totals=actual,summary=x['summary'],paired=x['paired_summary'],goal_completed=False),ensure_ascii=False))
if __name__=='__main__':main()
