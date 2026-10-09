"""One preregistration covers every report stage and fixed control row."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
from experiments.cvs_multi_state_action import design as d
from experiments.cvs_multi_state_action.evaluate import VIEWS,VIEWS_ROOT,HELDOUT_RX,TARGET_DAYS
from experiments.cvs_multi_action_audit import register as registry

REPORT=d.ROOT/'automation_reports/CV-SincNet'/d.RUN

def mirror():
 registry.d=d;registry.REPORT=REPORT;registry.mirror()

def register():
 if REPORT.exists():raise FileExistsError('Existing report; reconcile')
 old=d.read(d.ROOT/'automation_reports/CV-SincNet'/d.parent.RUN/'experiment.json')
 spec=deepcopy(old)
 spec.update(run_id=d.RUN,display_name='多解耦状态条件作用：参数/前端/分布修正、联合条件、36行身份验证',
  description='逐项执行用户新报告：四seed源作用匹配对照、联合clean/LEO和六折辅助TX留出，随后9身份对照×4seed从零E200，全冻结后10视图独立truth-last测试。',
  group_id='cvs-multi-state-action',aliases=['multi_state_action','状态条件多解耦'],
  comparison_group_id='manysig-state-action-fixed-controls-v1',tags=['phase1','multi_state_action','source_diagnostic','scratch','legacy_cosine'],
  parent_run_ids=[d.parent.RUN],replaces_run_id=None,
  authorization='用户2026-10-09：根据设计报告继续完成下一步优化探索，不能遗漏，要一一完成。沿用每卡4个实验进程；全三阶段及冻结后的独立测试均在本次范围。',
  rows=[],status='PLANNED',state_recipe=d.RECIPE,audit_recipe=d.AUDIT,
  notes=['参考报告逐条追溯analysis/multi_state_action_traceability.md；LT是报告明确保留的诊断，不新增LT网络。',
   '只有冻结源诊断加载旧multi E200四个已核实来源；所有新身份/EMA/辅助参数从零，无旧checkpoint继承。',
   '固定对照全测，源指标只记录偏好，不删减测试行。负收益与零可靠性保留，不触发低性能停机。',
   '状态和增量分开；接受作用样本状态允许含身份信息。L/T为数字受控后处理，不声明纯硬件因子。',
   'R先逐包非线性再聚合，独立分袋、排除被评估TX、匹配day/condition；无跨RX逐包反事实。',
   '原实测6场景沿用；额外3正式LEO弱场景只在冻结后用固定物理ID顺序/seed/batch生成，所有row共享。Phase1压力测试并非Phase2数据胶囊。',
   'K、适应前后、新类注册、H为N/A；本次是Phase1身份域泛化，无目标适应。',
   'sharedGPU耗时仅共享负载实测，不声称隔离加速；未测FLOPs记N/A。'])
 spec['code'].update(checkout=str(d.ROOT),cwd=d.PROJECT+'/releases/'+d.RELEASE,commit='Actual committed/pushed release_commit.txt')
 spec['checkpoint'].update(initialization='Identity: scratch, no ancestors. Diagnostic: all4 fixed parentmultiE200.',
  sources=[d.audit_config(r)['checkpoint'] for r in d.audit_rows()],provenance_verdict='Diagnostic parent source_provenance+checkpoint_provenance rechecked before load; identity SCRATCH_NO_INHERITANCE',
  selection_rule='All36 fixed ownE200 controls; source preference mean4(.5V+.5worstRX), exact tie fewerparameters. No target selection.')
 spec['execution'].update(launch_owner=d.OWNER,remote_run_root=d.BASE.as_posix(),remote_log_root=d.PROJECT+'/logs/'+d.RUN,
  local_artifact_root='automation_reports/CV-SincNet/'+d.RUN,
  launch_command='python -m experiments.cvs_multi_state_action.publish --output local_artifacts/'+d.RELEASE)
 spec['permissions'].update(claim_scope='Frozen-source auxiliary diagnostics and fixed scratch identity comparison; previously exposed proxy benchmark, not new blind/physical satellite evidence')
 spec['metrics_plan'].update(scorer_ref='experiments/cvs_multi_state_action/evaluate.py score; all360 predictions complete before truth; independent recount',
  prediction_ref=(d.BASE/'<row>/prediction/predictions.npz').as_posix(),
  metric_names=spec['metrics_plan']['metric_names']+['normalized_z_skill','margin_skill','parameter_error','donor_transfer','R_directed_mean_baseline','reference_drift','full_native_gradient_cosine'])
 spec['test_completion_plan'].update(rows=36,views=list(VIEWS),weight_selection='all fixed ownE200',
  weak_views=(d.BASE/'weak_views').as_posix(),weak_view_seeds=[2026100912,2026100913,2026100914])
 spec['data']['leo_config_ref']='Unchanged native practical3 concat; source diagnostic clean/practical_mid; final frozen clean+six practical+three LEO_WEAK views'
 for row in d.rows()+d.audit_rows():
  audit=row.get('kind')=='audit';c=d.audit_config(row) if audit else d.config(row);rid=row['row_id'];root=c['output_root']
  d.write(d.ROOT/'experiments/cvs_multi_state_action/configs'/(rid+'.json'),c)
  spec['rows'].append(dict(row_id=rid,method='source_action_audit' if audit else row['arm'],purpose='source_diagnostic' if audit else 'fixed_identity_control',
   gpu=None,config=c,config_ref='experiments/cvs_multi_state_action/configs/'+rid+'.json',resolved_config_ref=root+'/resolved_config.json',
   data_overrides=dict(only_source_L=True) if audit else {},
   seeds=dict(model=row['model_seed'],split=20261009 if audit else 392005,data=None,augmentation=2026100901 if audit else row['model_seed'],support=None,evaluation=2026100911 if audit else 2026100912),
   seed_notes='Identity roles fixed392005. Auxiliary physicalsplit20261009; fixed3increments/privategenerators. Weak finalseeds2026100912/13/14, fixedbatch256; existing6views reused. No support.',
   k=None,scenario='clean,source_practical_mid,joint' if audit else ','.join(VIEWS),optimizer='AdamW auxiliary' if audit else 'NativeAdamW cosine plus separateauxAdamW',
   lr=.0002,epochs=None if audit else 200,fl_rounds=None,
   budget_ref='6modes×2branches×3exposures×200 plus6TX×2modes×2branches×200; Rnew/legacy3×2×200' if audit else '44400successfulnativeupdates; every4steps max32sourceL; fixedtotalvirtualbudget',
   output_root=root,prediction_root=None if audit else (d.BASE/rid/'prediction').as_posix(),
   log_path=d.PROJECT+'/logs/'+d.RUN+'/'+('audit-' if audit else 'source-')+rid+'.log',
   command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u '+d.PROJECT+'/releases/'+d.RELEASE+'/experiments/cvs_multi_state_action/train_worker.py --kind '+('audit' if audit else 'source')+' --row '+rid,
   expected_artifacts=['physical_roles.json','resolved_config.json','*_actions.json','*_steps.jsonl','*_receiver.json','*_composition.json','completion.json'] if audit else spec['expected_artifacts']))
 REPORT.mkdir(parents=True)
 d.write(REPORT/'experiment.json',spec)
 (REPORT/'events.jsonl').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),status='PLANNED',note='All report stages registered before execution'))+'\n',encoding='utf-8')
 (REPORT/'report.md').write_text('''# 多解耦状态条件作用：三阶段完整验证

本次按新设计报告逐项落实，不以辅助诊断替代身份性能验证。设计追溯见[32项验收](../../../analysis/multi_state_action_traceability.md)。实际配置以experiment.json逐行config及远端resolved_config为准。

第一阶段比较真实/解析/神经增量、h与h+信号状态、一阶与二阶、全349维与精确29维前端；保留零增量和偶部、5类donor、固定多增量、曲线和组合诊断。第二阶段使用同一套网络联合clean与源practical_mid，并与分别拟合对照；6折TX仅从辅助拟合留出。四seed共享1920/960源L物理划分，全部衍生视图保持角色；原身份骨干见过这些包。

R执行逐包作用后非线性聚合，描述/接受作用/目的统计独立分袋，匹配day/condition，其他TX共享描述；与零作用、有向RX对条件均值及重新拟合的旧质心模型比较。固定目的源RX锚点实际进入训练。重复分袋用于划分稳定性，不当作独立样本重复。

第三阶段固定9组×4seed从零训练E200：native、real_views、unified、L、LT、LTR及L_EG/LT_EG/LTR_EG。默认虚拟动作仅更新G/分类器，EG对照允许更新前段；真实IQ正常更新。旧伪标签门控、cosine、源物理角色和原生clean+LEO拼接保持。真实增强CE权重独立，学习作用按源留出判别误差校准；所有虚拟分支共享总预算。记录完整native参照梯度、漂移、版本隔离、步骤/曝光与参数/资源。

全部身份行完成并冻结后统一预测，每行168000包×10视图：clean、原6场景及3个LEO弱场景。所有预测固定后独立truth-last评分，分seed/RX/TX/day报告accuracy、Macro-F1及配对差。目标分数不用于候选选择、调参或重跑。不存在目标support适应，K/新类/H均N/A。

源作用科学判定同时观察归一化身份特征与竞争类margin，并比较强基线；正h-skill不能单独证明约束有效。固定对照不因源负结果被删除，在线可靠性可为零且如实报告。只有技术失败停止所属任务，保留所有产物。当前状态以原登记事件及独立进程/产物读回为准。
''',encoding='utf-8')
 mirror()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mirror',action='store_true');a=p.parse_args();mirror() if a.mirror else register()
