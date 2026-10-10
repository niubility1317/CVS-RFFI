"""Source selection, all training rows and automatic test scope in one record."""
from copy import deepcopy
from datetime import datetime,timezone
import argparse,json,shutil
from pathlib import Path
from . import design as d
REPORT=d.ROOT/'automation_reports/CV-SincNet'/d.RUN

def register():
 if REPORT.exists():raise FileExistsError('Existing registration; reconcile')
 previous=d.read(d.ROOT/'automation_reports/CV-SincNet'/d.parent.RUN/'experiment.json')
 s={k:deepcopy(previous[k]) for k in ('schema','kind','stage','data','permissions','test_completion_plan')}
 s.update(run_id=d.RUN,group_id='cvs-feature-disentangle',display_name='特征增强多解耦：B48关系采样、弱MixStyle与方向Fishr三seed',
  description='48次scratch E200训练；源V三seed选择Fishr强度，36行自动clean+六practical测试，12未选候选仅source。',
  aliases=['feature_disentangle'],tags=['phase1','pure_ce','multi_disentanglement','three_seed','relation_sampling','weak_mixstyle','directional_fishr'],
  comparison_group_id=d.RUN,parent_run_ids=[],replaces_run_id=None,status='PLANNED',
  authorization='用户要求严格实现新报告并发布三seed特征增强版多解耦，继续纯CE无半监督无星地训练增强。',
  code=dict(commit='release_commit.txt at launch',checkout=str(d.ROOT),environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',cwd=d.PROJECT+'/releases/'+d.RELEASE),
  checkpoint=dict(initialization='all48 identity and auxiliary models scratch; no inherited state',sources=[],
   provenance_verdict='SCRATCH_NO_INHERITANCE',contract_check_ref=d.SOURCE,
   selection_rule='Own fixed E200 every row; ratio selected on full three-seed sourceV before selected-candidate target access'),
  execution=dict(host='N607',launch_owner=d.OWNER,gpu_policy='<=2 total experiment processes/GPU, pre-CUDA reservations, max12 active, >=12GB free',
   remote_run_root=d.BASE.as_posix(),remote_log_root=(d.BASE/'logs').as_posix(),local_artifact_root=str(REPORT),
   launch_command='python -m experiments.cvs_feature_disentangle.publish --output local_artifacts/'+d.RELEASE,
   stop_rule='Technical failure: stop new dispatch, preserve active healthy tasks/artifacts. No low-score halt/retry or target feedback.'),
  expected_artifacts=['<row>/source/resolved_config.json','<row>/source/final_ssdg.pth','<row>/source/auxiliary_final.pth',
   '<row>/source/epoch_metrics.jsonl','<row>/source/epoch_metrics.csv','<row>/source/source_stress.json',
   'source_selection.json','<selected-row>/source_frozen.json','<selected-row>/prediction/complete.json',
   'row_scoring/<selected-row>/scoring_complete.json','analysis.md','completion.json'],
  metrics_plan=dict(metric_names=['accuracy','macro_f1','worst_rx','margin_q10','tail_risk','quality_strata',
   'relation_coverage','unique_pid','repeat_rate','maximum_wait','style_coverage','style_delta_norm','branch_gradients',
   'time_dependency_probe','conditional_fishr_variance','bucket_age','weight_row_norm','weighted_gradient_ratio',
   'gradient_cosine','actual_optimizer_update','action_holdout_error','R_actual_relations','elapsed_seconds','peak_cuda_bytes'],
   dimensions=['arm','model_seed','view','receiver','transmitter','day'],prediction_ref=(d.BASE/'<selected-row>/prediction/predictions.npz').as_posix(),
   scorer_ref='experiments/cvs_feature_disentangle/evaluate.py',primary='Actual three-seed means/sample SD and paired deltas; six-practical mean and worst RX',
   resource_limits='shared hardware measured wall time/peak memory; no isolated speed claim; Phase2 K/adaptation/new classes N/A'),
  rows=[],source_selection_rule=deepcopy(d.SELECTION_RULE),mechanism_recipe=deepcopy(d.RECIPE),feature_recipe=deepcopy(d.FEATURE_RECIPE),
  results_ref='analysis.md',notes=[
   'PureCE labeled clean identity baseline. No U iteration, pseudo-label loss, EMA teacher, satellite generator or concat.',
   'Fishr EMA stores gradient statistics, not model weights. Only clean condition, coefficient1; no duplicate LEO observation.',
   'B48 every8 native steps from sourceL only. Same .1 CE weight in all relation controls. 5550calls/266400packet appearances, not independent samples.',
   'Weak style changes only nativeL time_down; representation regularization, not physical RX replacement. No duplicated styleCE/KL.',
   'Direction Fishr adapted to cosine scale30, all960 coordinates, no per-gradient normalization; raw counterpart uses same EMA/schedule.',
   'Fishr3/5/10% active-step gradient calibration on mature E20 sourceL, weights frozen then E21-40 ramp; no Adam reset.',
   '12 unselected source candidates never access target. 36 selected/fixed comparison rows automatically test after required freeze.',
   'joint_direct replaces learned digital L/T proposals with same-budget real random IQ; retains R to isolate proposal value.',
   'Independent hidden48 parameters1193635 vs shared hidden50 parameters1195559 (+.1612%); tolerance.2%, no unused padding.',
   'Hard resampling, DSU and fourth network remain conditional future alternatives as specified in report.'])
 s['data']['roles'].update(U_use='No U use in any arm; B48 and Fishr use sourceL true labels only')
 s['data']['leo_config_ref']='All training satellite paths disabled; fixed existing six practical observations only after source freeze'
 s['test_completion_plan'].update(rows=36,trained_rows=48,source_only_rows=12,weight_selection='OwnE200, selected SF/AF ratio from sourceV three seeds',
  trigger='Fixed controls test per-row immediately. SF/AF candidates wait source-only selection; only chosen pair tests. Deferred controls use selected ratio and test per-row.',
  all_predictions_before_truth='All seven views fixed per row before separate scorer',own_source_row_frozen_before_query=True)
 template=deepcopy(previous['rows'][0])
 for row in d.rows():
  c=d.config(row);rid=row['row_id'];r=deepcopy(template)
  for key in ('pid','launch_command','status'):r.pop(key,None)
  r.update(row_id=rid,method='feature_disentangle_'+row['arm'],purpose='source_ratio_candidate' if row['arm'] in d.SEARCH_ARMS else 'fixed_comparison',
   gpu=None,config=c,config_ref='experiments/cvs_feature_disentangle/configs/'+rid+'.json',resolved_config_ref=c['output_root']+'/resolved_config.json',
   seeds=dict(model=row['model_seed'],split=392005,data=None,augmentation=row['model_seed']+49071,support=None,evaluation=2026101007),
   seed_notes='Model seed controls scratch model; native split392005 fixed. Private relation/style/action streams derived and logged; support N/A. Source stress seed2026101007; fixed target observations.',
   budget_ref='E200×222=44400 identity updates; B48 every8steps; D≤32L every4 afterE20, R cached audit≤32 separately counted',
   output_root=c['output_root'],prediction_root=(d.BASE/rid/'prediction').as_posix(),log_path=(d.BASE/'logs'/(rid+'-train.log')).as_posix(),
   command='python experiments/cvs_feature_disentangle/train_worker.py --kind train --row '+rid,
   expected_artifacts=['source/final_ssdg.pth','source/completion.json','source/source_stress.json','source/auxiliary_final.pth','source_frozen.json'],
   target_access='Source selection required' if row['arm'] in d.SEARCH_ARMS else 'After ownsource freeze',status='PLANNED')
  s['rows'].append(r);d.write(d.ROOT/'experiments/cvs_feature_disentangle/configs'/(rid+'.json'),c)
 REPORT.mkdir(parents=True);d.write(REPORT/'experiment.json',s);d.write(d.ROOT/'experiments/cvs_feature_disentangle/experiment.json',s)
 shutil.copy2(Path('E:/codex/home/attachments/8f00af9d-39ef-43d2-9af5-ab77f7803c39/已粘贴的文本.txt'),REPORT/'design_report.md')
 shutil.copy2(d.ROOT/'analysis/feature_disentangle_traceability.md',REPORT/'traceability.md')
 (REPORT/'events.jsonl').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),status='PLANNED',trained=48,tested=36,source_only=12))+'\n',encoding='utf-8')
 lines=['# 特征增强版多解耦三种子探索','','纯 CE、无半监督、无星地训练增强；保持当前身份结构、cosine分类头与学习率日程。48行均从零训练E200/44400步，不加载旧权重。',
  '', '## 实验矩阵','','| 配置 | 关系流 | 弱Style | Fishr | 机制 |','|---|---|---|---|---|']
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));p=c['feature_plan']
  lines.append(f"| {arm} | {p['relation']} | {p['style']} | {p['fishr']} / {p['fishr_ratio'] if p['fishr_ratio'] is not None else '源选强度'} | {','.join(c['arm_plan']['paths']) or '无作用网络'} |")
 lines += ['', '每配置seed为2026092701、2026092702、2026092703。先完成13种固定配置39行；以SAF3/5/10完整源V三seed选择强度，并选择同强度SF，未选4配置12行只保留源结果。再从零训练noR/shared/direct的9行。最终测试36行，每行clean及六practical。',
  '', '固定控制逐行冻结即测试；需要源选的候选等待必要源选择，未选候选不读target。所有预测固定后独立truth-last评分，目标结果不回流任何健康训练、排序或参数。',
  '', 'B48为6TX×2RX×4不同物理PID，同一天；30block公平轮换。每8个原生step一次，5550关系调用、266400clean包次，不等于独立样本数。所有关系对照CE权重0.1。Style只复用nativeL原有CE。',
  '', 'Fishr使用当前cosine头6×160参数方向梯度；raw梯度作同流程对照。每桶RX/day/clean独立计数去偏EMA，E1–20积累，E20成熟源L定标，E21–40升权，之后固定。EMA只保存梯度统计，不是教师。',
  '', 'L/T预测用于筛选并在真实IQ训练；合法困难IQ不因代理预测失准被丢弃。R保留独立角色分袋、判别margin分布和尾部风险，noR对照仅关闭身份约束。共享作用核心hidden50与独立hidden48参数差约0.1612%，保持候选和真实视图预算。',
  '', '最高尚未验证的科学问题是弱Style与方向Fishr是否在不损害身份间隔的情况下改善困难RX；不能用辅助loss下降替代识别收益。分支关闭仅依赖性探针，非严格因果结论。',
  '', '报告中旧U/EMA/LEO配方按用户明确要求覆盖为pureCE和clean-only Fishr；DSU、困难重采样和第四网络按首轮条件延后。完整要求见[设计追溯](traceability.md)和[原报告](design_report.md)。']
 (REPORT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return s

def mirror():
 from experiments.cvs_multi_action_audit import register as r
 r.d=d;r.REPORT=REPORT;r.mirror()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mirror',action='store_true');a=p.parse_args();mirror() if a.mirror else register()
