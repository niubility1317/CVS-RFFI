"""Register the prospective matrix and the completed frozen-score aggregation only."""
import json
from pathlib import Path
from .build import ROOT, collect
from .plan import RUN, ANALYSIS_RUN, F_RUN, G_RUN, make_plan
from tools import experiment_registry as registry


def base(run, analysis=False):
    spec = registry.template()
    spec.update(run_id=run, group_id='phase1-cric-dura-complete-system',
        display_name='CRIC+DURA五源冻结结果并表' if analysis else '完整CRIC+DURA跨配置与匹配监督/PL对比设计',
        description='仅从已有评分复算18种配方、54模型读出、378格，无新训练/目标访问。' if analysis else
            '17数据契约×3训练设置×3Fseed；原生CRIC与同增强监督/普通PL分列。153F实例，3复用/150新训练；36G来源，1复用/35新谱系。当前只完成设计，未发布训练。',
        kind='analysis' if analysis else 'cvs', stage='Phase1',
        tags=['CRIC', 'DURA', 'complete-system', 'three-F-seeds', 'latency-ablation-design'],
        parent_run_ids=['20261019-phase1-external-dg-concat-pl-m219-r03', F_RUN, G_RUN,
            '20261021-phase1-SARD-NoAs-DURA-e50-m6-r02'],
        authorization='用户2026-10-21指定CRIC+DURA完整方法，与当前对比配置比较，并详细设计机制时延与消融；五源数量扫点不新跑。')
    spec['code'] = dict(commit='Design/aggregation delivery Git commit; no new training code released',
        checkout='github_publish/CVS-RFFI-repo', environment='Local verified Python, standard library only',
        cwd='E:/type10-7', entry='experiments.cric_dura_comparison')
    spec['data'].update(dataset='WiSig ManySig/ManyRx', representation='equalized1 center256 float32 unitRMS',
        contract_ref='docs/goals/CRIC_DURA_COMPARISON_AND_LATENCY_20261021.md',
        physical_ids_ref='Existing source/target data-contract records; no IQ or truth opened for this design/aggregation',
        label_map_ref='Existing native source-contract metadata; no posthoc head fitting',
        validation_ref='Reuse existing VALIDATED_ONCE handles; new contracts bound before actual launch')
    spec['permissions'].update(regime='Frozen-score-only analysis' if analysis else 'Source-only training planned; target frozen inference only',
        query_use='No query IQ/truth reads by aggregation', claim_scope='Previously exposed engineering benchmark; not independent blind or in-orbit validation')
    spec['checkpoint'].update(initialization='No models loaded for this local delivery',
        sources=[F_RUN, G_RUN], contract_check_ref='Existing exact-contract F0 evidence; every new F/G/teacher must match its own contract',
        provenance_verdict='F0 reuse references retained; prospective cross-contract sources prohibited',
        selection_rule='Frozen existing F0 scores; prospective fixed-final source schedule, no target selection')
    spec['execution'].update(host='Local aggregation; prospective N607 training not launched',
        launch_owner='codex-root-cric-dura-design-20261021',
        gpu_policy='No GPU used by aggregation/design; new train queues independent of active 360-row comparison; official timing exclusive GPU',
        local_artifact_root='local_artifacts/cric_dura_comparison_20261021',
        launch_command='python -X utf8 -m experiments.cric_dura_comparison.build' if analysis else None,
        stop_rule='Design has no worker; future technical failures preserve outputs/healthy peers; no performance stopping')
    spec['metrics_plan'].update(metric_names=['Accuracy', 'Macro-F1', 'worst RX', 'paired deltas', 'training and deployment cost'],
        dimensions=['data contract', 'regime', 'F seed', 'G seed', 'view', 'RX', 'TX', 'date'],
        scorer_ref='No new scoring: aggregate immutable existing scores' if analysis else 'Independent truth-last after all seven predictions per row')
    return spec


def install(spec, status, evidence):
    errors = registry.validate_spec(spec)
    if errors:
        raise ValueError(errors)
    folder = ROOT / 'automation_reports/CV-SincNet' / spec['run_id']
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'experiment.json').exists():
        raise FileExistsError('Registered run already exists; do not overwrite: ' + str(folder))
    spec['registered_at'] = registry.now()
    registry.write_text(folder / 'experiment.json', registry.json_text(spec), exclusive=True)
    registry.write_text(folder / 'report.md', registry.report_text(spec), exclusive=True)
    registry.write_text(folder / 'events.jsonl', json.dumps(dict(at=registry.now(), status=status,
        evidence=evidence, note='Existing-score aggregation complete' if status == 'ANALYZED' else 'Design complete; new training NOT launched', row_id=None), ensure_ascii=False) + '\n', exclusive=True)
    return folder


def main():
    plan = make_plan()
    spec = base(RUN)
    spec['research_plan'] = plan
    records = []
    for i, r in enumerate(plan['rows']):
        records.append(dict(row_id=r['row_id'], method=r['display_method'], purpose=r['regime'],
            config_ref='experiment.json#/research_plan/rows/%d' % i,
            data_overrides=plan['protocols'][r['protocol_id']],
            seeds=dict(model=r['model_seed'], split=392005, data=None, augmentation=None, support=None, evaluation=2027),
            seed_notes='Only F model seed varies; deterministic stored role split; support N/A; native augmentation RNG must be bound in actual launcher, not guessed here.',
            epochs=50, optimizer='AdamW', lr=2e-4, budget_ref='11100 updates; source-only fixed final; original Sinc unit-aware groups',
            output_root='runs/' + RUN + '/' + r['row_id'],
            log_path='logs/' + RUN + '/' + r['row_id'] + '.log', command=None,
            launch_status='NOT_LAUNCHED_NO_TRAINING_ADAPTER',
            expected_artifacts=['source/freeze', 'seven-view predictions', 'independent scores', 'same-row report']))
    spec['rows'] = records
    spec['expected_artifacts'] = ['Contract-matched F/G lineage', 'seven-view predictions and scores', 'source curves', 'latency tables']
    folder = install(spec, 'PLANNED', ['docs/goals/CRIC_DURA_COMPARISON_AND_LATENCY_20261021.md'])
    (folder / 'report.md').write_text('# 完整 CRIC＋DURA：跨配置对比及监督／PL控制\n\n'
        '**PLANNED / NOT LAUNCHED**。当前交付为完整设计，尚无新增跨配置训练、预测或时延结果。\n\n'
        '逐行矩阵唯一登记见 [experiment.json](experiment.json)。17契约、153个F实例（3复用/150新训练）；36个G来源（1复用/35条新谱系）。F0已有三Fseed/单G的七景结果由独立分析run复算。\n\n'
        '详细方法、源/目标RX数量、标注控制、预算、权限、15项F消融、7项G新设置、时延5行、ETA及未完成项见 '
        '[设计说明](../../../docs/goals/CRIC_DURA_COMPARISON_AND_LATENCY_20261021.md)。\n\n'
        '当前缺项：新数据契约F/G适配launcher、全部G教师来源和逐阶段预算展开、聚焦验证与一次P0/P1审查、实际发布与逐行运行。原63000硬编码恢复器不能直接套用新契约。不得误标RUNNING；原360行对比任务继续由原owner运行。\n', encoding='utf-8')
    analysis = base(ANALYSIS_RUN, True)
    rows = collect()
    distinct = sorted({(r['method'], r['regime'], r['seed']) for r in rows})
    analysis['rows'] = [dict(row_id=('%s-%s-s%d' % (m, g, s)).replace('+', '-').replace(' ', '-').lower(),
        method=m, purpose='existing_frozen_scores_' + g,
        config_ref='paper/phase1_experiments_ieee_20261020/source_data/cric_f0_per_seed_54x7.csv',
        seeds=dict(model=s, split=392005, data=None, augmentation=None, support=None, evaluation=2027),
        seed_notes='Existing frozen row; no new RNG or support; original per-run augmentation preserved.',
        output_root='local_artifacts/cric_dura_comparison_20261021/' + str(i),
        command='python -X utf8 -m experiments.cric_dura_comparison.build',
        budget_ref='No training or prediction; existing-score aggregation only') for i, (m, g, s) in enumerate(distinct)]
    analysis['expected_artifacts'] = ['final_analysis.md', 'f0_per_seed_scores.csv', 'f0_method_summary.csv', 'IEEE table']
    folder = install(analysis, 'ANALYZED', ['final_analysis.md', 'local_artifacts/cric_dura_comparison_20261021/build_evidence.json'])
    (folder / 'report.md').write_text('# CRIC＋DURA：F0完整方法并表\n\n'
        '**VERIFIED / ANALYZED**。仅复算已有冻结评分，54模型读出×7景＝378格，18个方法／配方；无新增训练、预测、query IQ或truth读取。\n\n'
        '三模型seed全部齐全；CRIC＋DURA只共享一个固定G，其SD仅表示F的条件方差。外部监督/普通PL保留sat拼接；CRIC保持原无As原生SSL，配方权限与预算差异明确保留。\n\n'
        '完整七景均值/SD见 [结果](final_analysis.md)。逐seed与Macro-F1在IEEE工程source_data与本地导出包；可用已提交build模块复算。其他数据契约和统一时延仍PLANNED，不随本分析标完成。\n', encoding='utf-8')
    print(json.dumps(dict(status='REGISTERED', planned=RUN, analyzed=ANALYSIS_RUN, rows=[153, 54])))
    registry.build(ROOT, managed_only=True)


if __name__ == '__main__':
    main()
