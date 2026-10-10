"""Actual fixed matrix registration, canonical mirror and report evidence."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json,shutil
from . import design as d
REPORT=d.ROOT/'automation_reports/CV-SincNet'/d.RUN

def register():
 if REPORT.exists():raise FileExistsError('Existing registration: reconcile')
 prior=d.read(d.ROOT/'automation_reports/CV-SincNet'/d.parent.RUN/'experiment.json')
 s={k:deepcopy(prior[k]) for k in ('schema','kind','stage','data','test_completion_plan')}
 s.update(run_id=d.RUN,group_id='cvs-multi-action-risk',display_name='真实动作提案＋R判别分布多解耦：纯CE三seed全对照',
  description='16种预登记配置×3seed，scratch E200。先三seed冻结源机制诊断，再固定完整身份矩阵，逐行自动clean+六practical独立评分。',
  aliases=['multi_action_risk'],tags=['phase1','pure_ce','multi_disentanglement','three_seed','action_proposals','margin_distribution'],
  comparison_group_id=d.RUN,parent_run_ids=[],replaces_run_id=None,
  authorization='用户要求严格落实新报告、实现下一版多解耦并发布三seed探索；随后明确纯CE底座、无半监督和星地增强。报告要求的全部必要对照每种三seed。',
  code=dict(commit='release_commit.txt at actual launch',checkout=str(d.ROOT),environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',cwd=d.PROJECT+'/releases/'+d.RELEASE),
  permissions=dict(regime='source_only_CVS',query_use='fixed row E200 per-packet inference, separate truth-last scoring',external_method_exception=None,
   claim_scope='Previously exposed ground proxy benchmark; no new blind or physical satellite claim'),
  checkpoint=dict(initialization='all48 identity models scratch; no inherited state',sources=[],
   provenance_verdict='SCRATCH_NO_INHERITANCE; auxiliary own periodically frozen student; diagnostic-only sources separately registered',
   contract_check_ref=d.SOURCE,selection_rule='Fixed ownE200 for every preregistered control; source ranking descriptive only; no target feedback'),
  execution=dict(host='N607',launch_owner=d.OWNER,gpu_policy='<=2 total experiment processes/GPU with preCUDA reservations; max12 active rows; >=12GB free; protect other healthy runs',
   remote_run_root=d.BASE.as_posix(),remote_log_root=(d.BASE/'logs').as_posix(),local_artifact_root=str(REPORT),
   launch_command='python -m experiments.cvs_multi_action_risk.publish --output local_artifacts/'+d.RELEASE,
   stop_rule='Affected technical failure only; stop launching new rows and preserve healthy active rows/artifacts; no low-performance stop, retry or target-feedback tuning'),
  expected_artifacts=['diagnostic-s*/completion.json','<row>/source/completion.json','<row>/source/source_stress.json','<row>/source_frozen.json',
   '<row>/prediction/predictions.npz','row_scoring/<row>/scoring_complete.json','source_ranking.json','analysis.md','completion.json'],
  metrics_plan=dict(metric_names=['accuracy','macro_f1','worst_rx','source_V_pressure','margin_tail','action_endpoint_error','proposal_real_risk',
   'R_covariance','R_tail','unique_physical_ids','weighted_gradient_cosine','weighted_gradient_ratio','actual_optimizer_update','elapsed_seconds','peak_cuda_bytes'],
   dimensions=['arm','model_seed','view','receiver','transmitter','day'],prediction_ref=(d.BASE/'<row>/prediction/predictions.npz').as_posix(),
   scorer_ref='experiments/cvs_multi_action_risk/evaluate.py',primary='Three-seed sample mean/SD and paired differences, six-practical mean and worstRX; incomplete rows explicit',
   source_selection=d.STRESS,resource_limits='shared hardware measured time/memory; no isolated speed claim; Phase2 K/adaptation/new classes N/A'),
  rows=[],mechanism_recipe=d.RECIPE,diagnostic_recipe=d.DIAGNOSTIC,
  diagnostics=[],status='PLANNED',results_ref='analysis.md',notes=[
   'Native is clean labeled-source CE with label_smoothing=0, no EMA/pseudo/U iteration/satellite simulation.',
   'Only LT_label_free_U reads source U IQ for label-free action fitting; metadata is hidden; no identity pseudo target.',
   'Contribution arms preserve fixed L/T/joint/R coefficients; absent contributions stay empty.',
   'L_budget/LT_budget/LR_budget replace absent digital slots with random real endpoints and absent R with exact clean G-only CE.',
   'random_ce/random_cons/exact_g match .05 digital coefficient and three slots; consistency is its own .01 addition.',
   'All48 rows are fixed scientific comparisons and tested; source pressure ranking neither grants nor removes target access.',
   'No fourth action network: explicitly conditional on future source evidence, not triggered in this preregistration.'])
 s['data']['roles'].update(U_use='No identity U loss. Only LT_label_free_U: IQ-only known-action fit; no TX/RX/day labels')
 s['data']['leo_config_ref']='Satellite training disabled in all48 rows; mechanism bounded digital source actions only. Test existing six practical observations.'
 s['test_completion_plan'].update(rows=len(d.rows()),weight_selection='Fixed ownE200 each row',trigger='After source diagnosis: each trained row freezes and immediately tests seven views; independent scorer after all seven predictions',
  own_source_row_frozen_before_query=True,all_predictions_before_truth='same row all seven views before scorer; full matrix before aggregate',weak_views=None,weak_view_seeds=None)
 s['test_completion_plan'].pop('all_source_rows_frozen_before_query',None)
 from .diagnostics import reference_config
 for seed in d.SEEDS:
  c=reference_config(seed)
  s['diagnostics'].append(dict(row_id='diagnostic-s'+str(seed),seed=seed,checkpoint=(Path(c['output_root'])/'final_ssdg.pth').as_posix(),
   parent_config=c,selection='All three native fixed E200 source references; no target scores used',
   input_contract=d.SOURCE,initializes_identity_training=False,source_fit_role='L_s',identity_frozen=True,target_access=False))
 for row in d.rows():
  c=d.config(row);rid=row['row_id'];source=c['output_root'];d.write(d.ROOT/'experiments/cvs_multi_action_risk/configs'/(rid+'.json'),c)
  s['rows'].append(dict(row_id=rid,method='multi_action_risk_'+row['arm'],purpose='fixed_preregistered_comparison',gpu=None,
   config=c,config_ref='experiments/cvs_multi_action_risk/configs/'+rid+'.json',resolved_config_ref=source+'/resolved_config.json',data_overrides={},
   seeds=dict(model=row['model_seed'],split=392005,data=None,augmentation=row['model_seed']+49071,support=None,evaluation=2026101007),
   seed_notes='Source split392005 fixed; action private RNG model+49071; source pressure2026101007; target existing frozen observations; support N/A',
   k=None,scenario=','.join(s['test_completion_plan']['views']),optimizer='AdamW',lr=.0002,epochs=200,fl_rounds=None,
   budget_ref='44400 identity updates,222/epoch; auxiliary every4steps afterE20, <=32L; fixedbranchweights',
   output_root=source,prediction_root=(d.BASE/rid/'prediction').as_posix(),log_path=(d.BASE/'logs'/(rid+'.log')).as_posix(),
   command='python experiments/cvs_multi_action_risk/train_worker.py --kind identity --row '+rid,
   expected_artifacts=['source/final_ssdg.pth','source/initialization.json','source/resolved_native_args.json','source/epoch_metrics.csv',
    'source/step_metrics.jsonl.gz','source/auxiliary_final.pth','source/mechanism_execution.json','source/source_stress.json','prediction/complete.json','scores.json']))
 REPORT.mkdir(parents=True);d.write(REPORT/'experiment.json',s);d.write(d.ROOT/'experiments/cvs_multi_action_risk/experiment.json',s)
 (REPORT/'events.jsonl').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),status='PLANNED',rows=len(d.rows()),seeds=list(d.SEEDS)))+'\n',encoding='utf-8')
 shutil.copy2(d.ROOT/'analysis/multi_action_risk_traceability.md',REPORT/'traceability.md')
 shutil.copy2(Path('E:/codex/home/attachments/70624102-bbe2-49c5-b74c-3e38240426a7/已粘贴的文本.txt'),REPORT/'design_report.md')
 recovered=d.ROOT/'local_artifacts/multi_action_risk_historical_r01'
 shutil.copy2(recovered/'recovery.json',REPORT/'historical_recovery.json')
 lines=['# 下一版多解耦：真实动作提案与R判别分布风险','','本轮纯CE底座，无半监督、EMA教师和星地增强。身份网络结构与cosine日程保留；作用模型使用本run周期冻结学生快照。所有身份模型从零训练E200/44400步。',
  '', '先完成三seed源机制诊断，随后自动运行以下16种配置，每种2026092701/02/03，共48个身份模型。源诊断使用历史固定native E200合规权重，仅用于诊断，不初始化新模型。',
  '', '| 配置 | 实际计划 |','|---|---|']
 for arm in d.ARMS:lines.append('| '+arm+' | `'+json.dumps(d.plan(arm),ensure_ascii=False)+'` |')
 lines+=['','L/T提案经过真实IQ核验，身份E/G/C沿真实端点回传；R拟合5维竞争margin分布，身份目标为CE与CVaR，仅更新G/C。保持固定分支系数，不因缺分支重分配。',
  '', '每个row完成源V压力验证并冻结后，自动测试clean与六种practical；本row预测完整固定后独立评分。完整矩阵汇总实际seed数、配对差、RX/TX/day分层。源排名只读取source_stress，不读取目标成绩。',
  '', '旧结果完整恢复316份结构化源文件，位于`'+str(recovered)+'`；压缩原始证据约143MB，保留本地产物，不入Git大文件。完整文件清单见[恢复记录](historical_recovery.json)。历史稀疏梯度epoch归一化错误保留说明，不把旧值解释为实际余弦。',
  '', '逐项要求见[设计追溯](traceability.md)，原文见[设计报告](design_report.md)。第四个交互/随机作用网络遵照条件触发条款，本轮不添加。当前状态以experiment.json与远端读回为准。']
 (REPORT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return s

def mirror():
 from experiments.cvs_multi_action_audit import register as r
 r.d=d;r.REPORT=REPORT;r.mirror()

if __name__=='__main__':register()
