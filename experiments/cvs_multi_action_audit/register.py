"""Maintain the existing experiment registry and canonical report mirror."""
import argparse
from collections import Counter
import csv
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import json
import importlib.util
import subprocess
import sys
from experiments.cvs_multi_action_audit import design as d

REPORT=d.ROOT/'automation_reports/CV-SincNet'/d.RUN
CANONICAL=Path('E:/type10-7')

def mirror():
    dest=CANONICAL/'automation_reports/CV-SincNet'/d.RUN
    shutil.copytree(REPORT,dest,dirs_exist_ok=True)
    subprocess.run([sys.executable,'-X','utf8',str(CANONICAL/'tools/experiment_registry.py'),'build','--managed-only'],cwd=CANONICAL,check=True)
    # Merge only this managed run into the Git index; do not import unrelated
    # canonical history/snapshots into this task's checkout.
    dest=d.ROOT/'experiment_registry'
    records=[json.loads(s) for s in (dest/'catalog.jsonl').read_text(encoding='utf-8').splitlines()]
    entry=next(json.loads(s) for s in (CANONICAL/'experiment_registry/catalog.jsonl').read_text(encoding='utf-8').splitlines()
               if json.loads(s)['id']==d.RUN)
    records=[r for r in records if r['id']!=d.RUN]+[entry]
    records.sort(key=lambda r:(r['record_type']!='managed_run',-r.get('latest_file_mtime',0),r['id']))
    (dest/'catalog.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    columns=['id','record_type','name','title','kind','stage','status','methods','tags','seeds','report','path','evidence_state','fact_count']
    with (dest/'catalog.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader()
        for r in records:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(list,dict)) else r.get(k,'') for k in columns})
    coverage=d.read(dest/'coverage.json');coverage.update(indexed_at=datetime.now(timezone.utc).isoformat(),record_counts=dict(Counter(r['record_type'] for r in records)))
    d.write(dest/'coverage.json',coverage)
    module_spec=importlib.util.spec_from_file_location('task_registry',CANONICAL/'tools/experiment_registry.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.render_index(dest,records,coverage)

def register():
    if REPORT.exists():raise FileExistsError('Existing report; reconcile')
    old=d.read(d.ROOT/'automation_reports/CV-SincNet'/d.parent.RUN/'experiment.json')
    spec={k:deepcopy(old[k]) for k in ('schema','group_id','kind','stage','data')}
    spec.update(run_id=d.RUN,display_name='多解耦下一步：留出源包作用迁移、G后误差与R统计/锚定诊断',
        description='按用户报告第1步执行。固定4个multi E200身份骨干，在L_s独立物理包上比较辅助拟合目标和跨TX迁移，修正R统计并审查类别锚定。身份不更新；无新目标域评分。',
        aliases=['multi_action_audit','多解耦源端作用诊断'],tags=['source_only','multi_disentanglement','heldout_physical_packets','cross_tx_action'],
        comparison_group_id='manysig-fixed-multi-E200-source-action-audit-v1',
        parent_run_ids=[d.parent.RUN],replaces_run_id=None,
        authorization='用户2026-10-09：按照报告进行下一步优化验证，发布实验验证。报告第10节先源端诊断，再开展身份训练；沿用每GPU最多4进程授权。',
        code=dict(commit='Recorded from release_commit.txt after code commit',checkout=str(d.ROOT),
            environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',cwd=d.PROJECT+'/releases/'+d.RELEASE),
        permissions=dict(regime='source_only_diagnostic',query_use='NONE; no target prediction or truth reads',
            external_method_exception=None,claim_scope='Auxiliary action generalization to held-out source-L packets only; identity has seen all L_s originally.'),
        checkpoint=dict(initialization='fixed own-parent multi E200 student; fresh auxiliary models',
            sources=[d.config(r)['checkpoint'] for r in d.rows()],
            provenance_verdict='Parent scratch EXACT_MATCH evidence verified; runtime rechecks each actual checkpoint before use',
            contract_check_ref=d.SOURCE,selection_rule='All four registered multi rows at E200; no ranking or target metric consumption'),
        execution=dict(host='N607',launch_owner=d.OWNER,gpu_policy='<=4 total processes/GPU including reservations; existing shared lock; >=12GB free',
            remote_run_root=str(d.BASE),remote_log_root=d.PROJECT+'/logs/'+d.RUN,
            local_artifact_root='automation_reports/CV-SincNet/'+d.RUN,
            launch_command='python -m experiments.cvs_multi_action_audit.publish --output local_artifacts/'+d.RELEASE,
            stop_rule='Technical failure only; retain outputs, no automatic retry or low-skill stop; preserve unrelated healthy tasks'),
        expected_artifacts=['physical_roles.json','provenance.json','resolved_config.json','*_actions.json','*_actions_state_dicts.pth',
            '*_actions_steps.jsonl','*_actions_steps.csv','*_receiver.json','*_receiver_state_dict.pth','*_receiver_steps.jsonl',
            '*_receiver_steps.csv','*_composition.json','*_gradients.json','completion.json'],
        metrics_plan=dict(metric_names=['heldout_zero_skill','heldout_training_mean_skill','G_mse','margin_delta_error','action_energy',
            'cross_TX_transfer','shuffled_q','oracle_parameters','route_changed_error','singular_spectrum','RX_group_coverage',
            'contribution_age','relation_vs_anchor','gradient_norm_ratio','gradient_cosine','elapsed_seconds','peak_cuda_bytes'],
            dimensions=['model_seed','source_exposure','branch','fit_mode','action_code_source'],
            prediction_ref='Heldout source action estimates evaluated after each fixed auxiliary fit; no target artifacts',
            scorer_ref='actions.py / receiver.py source-only diagnostics',
            primary='All four seeds and both exposures reported. No early stopping; zero/mean skill and G/margin errors jointly assessed.',
            decision='Source evidence will inform a separately registered E200 study; no automatic target-driven candidate or weight selection.'),
        rows=[],notes=['This is report step1 and bounded step2 source comparisons, not a new identity-training efficacy experiment.',
            'Fresh auxiliary audit packets were excluded from auxiliary fit, but not from parent identity training.',
            'No stronger identity weights, larger rank or learned LT in this first diagnostic.',
            'Gradient reference is clean labeled CE; full native pseudo/LEO gradient is N/A in this frozen-model diagnostic.'],
        status='PLANNED',mechanism_recipe=d.RECIPE,
        test_completion_plan=dict(scope='Explicit source diagnostic prescribed by report step1',
            test='Disjoint physical source-L audit packets, clean and source practical_mid; not R_t',
            target_testing='No new identity exists; preserve historical tests and do not rerun unchanged checkpoints',
            subsequent_identity_training='Separately register source-selected design and full truth-last tests after source evidence'))
    spec['data'].update(physical_ids_ref=d.SOURCE+'#role_ids.L_s; each row physical_roles.json',
        target_receivers=[],target_days=[],capsule_id=None,split_id=None,validation_ref=d.SOURCE,
        support_query_ref=None,leo_config_ref='source practical_mid residual/post_sync fs25MHz fc2.462GHz; deterministic physical IDs',
        roles=dict(original_L_s=.07,original_U_s=.63,original_V=.30,access='Only L_s sampled; U/V not consumed',
            fit_per_TX_RX=64,audit_per_TX_RX=32,physical_partition_seed=20261009))
    for row in d.rows():
        c=d.config(row);root=c['output_root']
        spec['rows'].append(dict(row_id=row['row_id'],method='source_action_audit',purpose='fixed_source_diagnostic',gpu=None,
            config=c,config_ref='experiments/cvs_multi_action_audit/configs/'+row['row_id']+'.json',
            resolved_config_ref=root+'/resolved_config.json',data_overrides={},
            seeds=dict(model=row['model_seed'],split=20261009,data=None,augmentation=2026100901,support=None,evaluation=2026100902),
            seed_notes='Original source roles fixed at392005. Auxiliary partition20261009. Model is parent seed; auxiliary init/sample schedule derived from evaluation generator. Support N/A.',
            k=None,scenario=','.join(d.EXPOSURES),optimizer='auxiliary AdamW only; fixed reference identity',lr=.0002,
            epochs=None,fl_rounds=None,budget_ref='200 optimizer steps per L/T fit mode and R; <=32 samples/step; no identity updates',
            output_root=root,prediction_root=None,log_path=d.PROJECT+'/logs/'+d.RUN+'/'+row['row_id']+'.log',
            command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u '+d.PROJECT+'/releases/'+d.RELEASE+
                '/experiments/cvs_multi_action_audit/train_worker.py --row '+row['row_id'],expected_artifacts=spec['expected_artifacts']))
        d.write(d.ROOT/'experiments/cvs_multi_action_audit/configs'/(row['row_id']+'.json'),c)
    REPORT.mkdir(parents=True)
    d.write(REPORT/'experiment.json',spec)
    (REPORT/'events.jsonl').write_text(__import__('json').dumps(dict(at=datetime.now(timezone.utc).isoformat(),status='PLANNED',note='Report step1 source-only audit'))+'\n',encoding='utf-8')
    (REPORT/'report.md').write_text('''# 多解耦源端作用验证

本轮严格执行用户报告的第一步：先验证作用网络，再决定身份训练和LT强化。4个模型seed复用各自合规multi E200固定身份骨干；所有辅助模型重新拟合，不更新身份参数。

每个TX×源RX按物理ID确定64个辅助拟合包、32个独立审查包，全部视图继承同一角色。共有1920个拟合包与960个审查包。原U标签及V不进入诊断。审查包只对新辅助拟合独立，不能声称身份骨干从未见过。

L/T依次比较原拟合、分块定标+G后误差、再加入跨TX作用迁移；初始化、样本曝光、rank8和200步预算匹配。各自审查自身q、另一TX同增量q、打乱q、零预测、拟合均值及已知参数诊断参考。独立R网络采用先逐包统计后聚合、真实贡献年龄与参考版本、关系均衡采样，建模整体源RX条件差，不将L/T外推残差称为纯硬件因素。关系保持与目的源RX类别锚定在同一动作上比较。

两种接收条件固定为clean与合法源包的practical_mid代理视图。记录G后误差、margin、目标能量、路由切换、时间窗口条件数、边界、奇异谱与组合剩余误差。梯度参考仅为clean有标签CE，不冒充完整原生伪标签+LEO训练损失；身份优化和更强权重留待源端证据。

这次源端审查不生成新的目标测试成绩。没有新的身份模型，因此不重复测试旧checkpoint。下一阶段如发布身份训练，仍须预登记完整冻结后预测与独立truth-last评分。

当前状态以experiment.json、events.jsonl及独立远端读回为准。完整设计追溯见[traceability](../../../analysis/multi_action_audit_traceability.md)。实际脚本和配置见[新包](../../../experiments/cvs_multi_action_audit/)。
''',encoding='utf-8')
    mirror()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mirror',action='store_true');a=p.parse_args()
    mirror() if a.mirror else register()
