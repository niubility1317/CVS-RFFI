"""Register the fixed non-PAMF matrix without launching or reading target scores."""
import json
from pathlib import Path
import subprocess
import sys

from .plan import RUN, PARENT, make_plan

ROOT = Path(__file__).resolve().parents[2]


def register():
    plan = make_plan()
    out = ROOT / 'local_artifacts/phase1_external_dg_20261019'
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / 'experiment_spec.json'
    tool = ROOT / 'tools/experiment_registry.py'
    subprocess.run([sys.executable, '-X', 'utf8', str(tool), 'template', '--output', str(temporary)], check=True)
    spec = json.loads(temporary.read_text(encoding='utf-8-sig'))
    spec.update(run_id=RUN, group_id='phase1-nonpamf-complete-dg-comparisons',
        display_name='Phase1非PAMF完整对比：三种子与两GPU依赖队列',
        description='四领域方法跨RX与日期／ManyRx；ResNet通用DG、源数量、标签量；六个冻结历史模型补测。162新训练＋6补测，接续既有24行。',
        kind='comparison', stage='Phase1', parent_run_ids=[PARENT],
        tags=['non-PAMF', 'DG', 'three-seed', 'two-GPU', 'native-budget', 'truth-last'],
        authorization='用户明确继续优化完成设计并完成全部非PAMF对比；保留两GPU、三seed和各方法原生预算，RIEI/DRIFT E100。')
    local = 'local_artifacts/phase1_external_dg_20261019/runtime.json'
    spec['code'] = dict(commit='release_commit.txt; publisher records actual Git commit in package.json',
        checkout='github_publish/CVS-RFFI-repo',
        environment='existing SDG-SEI Py3.8/torch1.11 + existing task-local NumPy1.23.5; legacy uses original CVS-RFFI torch2.1; no shared install',
        cwd=local + '#/release_root')
    spec['data'] = dict(dataset='ManySig and ManyRx', version='existing N607 compact datasets and immutable source/received handles',
        representation='equalized1, center256, per-packet float32 unitRMS',
        contract_ref=local + '#/run_root/data_handles.json',
        physical_ids_ref='each source_contract.json#/role_ids and each validated views/index.npz',
        label_map_ref='each source_cache/metadata.json#/classes',
        source_receivers=None, target_receivers=None, source_days=None, target_days=[0, 1, 2, 3],
        varying_roles_note='Explicit per-row protocol overrides below; source-count targets remain fixed F0 seven RX.',
        roles=None, train_ratio=None, capsule_id=None, split_id=None,
        validation_ref='Existing F0 VALIDATED_ONCE reused; new physical-role wrappers validated once by builder',
        support_query_ref=None, leo_config_ref='existing sixscene engine7669a6de, original namespace/seeds, clean plus six practical')
    spec['permissions'] = dict(regime='source-only limited-label closed-set DG',
        query_use='frozen per-sample inference only; all tasks complete before truth scoring',
        external_method_exception=None, claim_scope='exposed terrestrial proxy benchmark; source-RX unseen-date diagnostic separate; no blind/in-orbit claim')
    spec['checkpoint'] = dict(initialization='scratch except six historical frozen evaluation rows',
        sources=['none for new training; IB stage2 only same-run stage1',
                 'six original standard CE scratch E200 default heads, individually checked before legacy prediction'],
        contract_check_ref='source.check_training / source.frozen_model / evaluation.legacy_model',
        provenance_verdict='new models scratch; historical rows require strict per-checkpoint provenance and source-role checks at smoke and load',
        selection_rule='RIEI/DRIFT final E100; IB/SDG minimum source V NLL; ResNet fixed budget; no head refit')
    spec['execution'] = dict(host='N607 ordinary account', launch_owner='codex-root-phase1-external-dg-20261019',
        launch_command='python -u -m experiments.phase1_external_dg.driver dispatch --config runtime.json',
        remote_run_root=local + '#/run_root', remote_log_root=local + '#/log_root',
        local_artifact_root='local_artifacts/phase1_external_dg_20261019', runtime_ref=local,
        gpu_policy='physical0,1 only; one own worker/GPU; CPU prepare/smoke; GPU start only after parent terminal and process gone; wait foreign workloads',
        stop_rule='output collision, role/contract/nonfinite/no legal prediction or wrong truth closure are technical failures; two matching row failures pause unstarted; healthy rows continue; low scores never stop')
    spec['metrics_plan'] = dict(metric_names=['accuracy', 'macro_f1_all_registered_classes', 'worst_rx_accuracy',
        'TX_recall_precision', 'paired_full_minus_ce', 'training_cost'],
        dimensions=['group', 'family', 'variant', 'seed', 'test', 'view', 'RX', 'date', 'TX'],
        prediction_ref='ROW/prediction/TEST/predictions.npz', scorer_ref='separate CPU evaluation.score after all-task preflight')
    spec['expected_artifacts'] = ['dispatcher.json', 'data_handles.json', 'smoke/result.json', 'legacy_smoke/result.json',
        'pipeline_state.json', 'ROW/source/freeze.json (training rows)', 'ROW/prediction/complete.json',
        'ROW/scores/scores.json', 'ROW/complete.json', 'summary.json', 'audited_rows.json', 'summary.csv',
        'final_report.md', 'completion.json']
    records = []
    for i, row in enumerate(plan['rows']):
        p = plan['protocols'][row['protocol_id']]
        if p['dataset'] == 'ManyRx':
            counts = dict(L_s=6533, U_s=58828, V=28023)
        else:
            pool = 6 * len(p['source_rxs']) * len(p['source_days']) * 1000
            labeled = p['fixed_labels'] if p['fixed_labels'] is not None else pool * p['label_per_thousand'] // 1000
            counts = dict(L_s=labeled, U_s=pool - labeled - pool * 3 // 10, V=pool * 3 // 10)
        records.append(dict(row_id=row['row_id'], method=row['family'], variant=row.get('route', row['variant']),
            purpose='fixed_comparison', config_ref='experiments/phase1_external_dg/matrix.json#/rows/' + str(i),
            resolved_config_ref=(local + '#/legacy_run_root/' + row['family'] + '-s' + str(row['model_seed']) + '/source/resolved_config.json'
                if row['budget_kind'] == 'frozen_eval' else local + '#/run_root/' + row['row_id'] + '/source/resolved_config.json'),
            data_overrides=dict(dataset=p['dataset'], source_receivers=p['source_rxs'], target_receivers=p['target_rxs'],
                source_days=p['source_days'], target_days=p['target_days'], roles=counts,
                protocol_ref='matrix.json#/protocols/' + row['protocol_id'], diagnostic_day=p['diagnostic_day']),
            seeds=dict(model=row['model_seed'], split=392005, data=None, augmentation=row['model_seed']
                if row['family'] == 'sdg' or row.get('route', '').startswith('mixstyle') else None,
                support=None, evaluation=392005),
            seed_notes='model controls initialization/loader; augmentation shares model RNG where active; physical data fixed; no support; shared observations seed392005/receiver2027',
            epochs=row.get('epochs'), optimizer=(None if row['budget_kind'] == 'frozen_eval' else
                row.get('optimizer', 'SGD momentum0' if row['family'] == 'riei' else 'Adam')),
            lr=row.get('lr'), budget_ref='matrix.json row: native stage caps or exact update/epoch control; frozen_eval means zero training',
            output_root=local + '#/run_root/' + row['row_id'],
            log_path=local + '#/log_root/' + row['row_id'] + '.log',
            command='python -u -m experiments.phase1_external_dg.driver worker --config RUN_ROOT/' + row['row_id'] + '.json',
            expected_artifacts=['prediction/complete.json', 'scores/scores.json', 'complete.json']))
    spec['rows'] = records
    temporary.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, '-X', 'utf8', str(tool), 'validate', str(temporary), '--launch-ready'], check=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(tool), '--root', str(ROOT), 'new', '--spec', str(temporary)], check=True)
    report = ROOT / 'automation_reports/CV-SincNet' / RUN / 'report.md'
    report.write_text('''# Phase1非PAMF完整对比预登记

162个新增训练、6个历史冻结模型补测，接续原24行；不训练或评估PAMF。三seed2026092701/02/03，实际矩阵在experiment.json，运行配置由同一plan.py生成matrix.json。详见docs/PHASE1_EXTERNAL_DG_EXECUTION_PLAN_20261019.md。

## 数据与方法

17个源数据契约；F1/F2各5源→7目标RX，chrono源D0/D1/D2、D3未见；ManyRx为10TX、16源→16目标RX。ManyRx实际L/U/V=6533/58828/28023，目标123021包。源数量及标签量控制、默认头、预算和全部seed角色逐row列明。

RIEI/DRIFT按用户E100，恒定LR无需移动milestone；IB两阶段各最多200及原早停；SDG200；ResNet通用DG5000更新、源数量10000更新、标签量E200。完整方法与自身CE匹配停止规则，实际成本单列，不声称严格等GPU时间。所有新训练从零，U标签隐藏不用；IB仅继承自身同契约stage1。历史6模型只加载既有联合训练默认头，不拟合新头。

原F0源缓存和七视图复用。新物理ID的六practical由原7669a6de发布内的冻结引擎生成，原有168000 ID观测直接复制；不使用当前工作区不同版本的引擎。新wrapper一次验证物理源目标互斥，后续复用VALIDATED_ONCE，不追加hash链。

## 生命周期与资源

当前PLANNED，尚未声明发布成功。代码commit由publisher写入release_commit.txt和本地package.json，实际CWD、命令、解释器和私有输入输出路径引用本地runtime.json。

源角色和观测构建在CPU完成；数学库2线程、最多4个生成worker。真实L_s checkpoint无query smoke后直接继续。历史checkpoint使用原torch2.1解释器，其余方法使用已验证作者torch1.11环境，无共享安装。

仅物理GPU0/1，每卡一个本任务worker；原dispatcher必须实际结束且24行终结后才开始新GPU行，其他owner占用时等待。绝不借用2至7或停止健康任务。输出root不可覆盖、一个launch owner；同指纹两行技术失败暂停未启动行，健康行完成。低分不停止、不调参、不改变后续顺序。

每row冻结后自动预测clean与六practical；chrono两个任务全完成后独立CPU scorer才读truth。历史补测只补六practical，旧clean保留并在最终报告独立复算后引用。逐RX/date/TX及actual seed数保留，TX报告recall/precision；Macro-F1为全部注册类别口径。

## 验证和完成

发布前聚焦负测、一次独立P0/P1审查；归档只做一次传输SHA比较、一次remote compile。post-launch以实际PID/CWD/argv/GPU/log增长判定VERIFIED，提交回执单独不足为证。

最终审计读取原24行及新增168行实际预测、源checkpoint及预算、选模和评分，复算总体CM/Accuracy/F1、检查目标覆盖及actual seed。只有186个训练实例及6个补测均闭合才完成整体目标。所有负结果与失败产物保留；三seed不冒充独立接收机总体显著性。

## 当前证据

本地基础与训练适配14项检查已通过，输入120组与不可变数据代码逐元素一致；新增预测/评分/调度检查结果在发布前追加。原IB-RFF六行已独立复算完成，原SDG健康训练。所有目标性能不反馈到此固定矩阵。
''', encoding='utf-8')
    print('REGISTERED', RUN, len(records))


if __name__ == '__main__':
    register()
