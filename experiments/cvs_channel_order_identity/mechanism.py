"""Reconcile complete source-only mechanism evidence without changing selection.

Reads collected source artifacts only. No model construction, checkpoints,
formal IQ, target artifacts, remote calls, or training are used.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics as st


RUN = '20261003-phase1-cvs-channel-order-identity-manysig-m16-r01'
CONTROL_RUN = '20261002-phase1-cvs-neural-residual-identity-manysig-m8-r01'
CONTROL = 'neural_residual_shallow'
VARIANTS = ('channel_capacity', 'channel_compensated', 'channel_dual', 'channel_order')
SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
RXS = ('1', '3', '4', '6', '8')
G_METRICS = ('g_magnitude_mean', 'g_magnitude_min', 'g_magnitude_max',
             'g_operator_delta_bound_max', 'g_coefficient_l1_max', 'g_basis_l1_max',
             'g_context_gradient_norm', 'g_context_exit_gradient_norm', 'g_basis_gradient_norm')
D_METRICS = ('d_relative_output_mean', 'd_relative_output_max', 'd_gradient_norm',
             'd_readout_gradient_norm', 'd_readout_norm')
METRICS = G_METRICS + D_METRICS + ('f_gradient_norm', 'residual_relative_output_mean')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def final_metrics(row):
    epochs = row['epochs']
    require([e['epoch'] for e in epochs] == list(range(1, 201)), 'Incomplete/duplicate epochs')
    end = epochs[-1]
    require(end['source_val_count'] == 27000 and set(end['source_val_rx_accuracy']) == set(RXS),
            'Incomplete source validation')
    require(end['source_val_worst_rx'] == min(end['source_val_rx_accuracy'].values()), 'Worst RX mismatch')
    require(all(close(end[k], value) if isinstance(value, (int, float)) else end[k] == value
                for k, value in row['completion']['final_source_metrics'].items()), 'E200/completion mismatch')
    require(close(st.mean(end['source_val_rx_accuracy'].values()), end['source_val_accuracy']),
            'Equal-size source RX means do not reproduce V')
    return dict(SOURCE_SCORE=.5 * (end['source_val_accuracy'] + end['source_val_worst_rx']),
                ALL_V=end['source_val_accuracy'], WORST_RX=end['source_val_worst_rx'],
                **{'RX_' + rx: end['source_val_rx_accuracy'][rx] for rx in RXS})


def describe(values):
    numeric = [v for v in values if v is not None]
    require(all(math.isfinite(v) and v >= 0 for v in numeric), 'Nonfinite/negative mechanism measurement')
    return dict(records=len(values), measured=len(numeric), missing=len(values) - len(numeric),
                positive=sum(v > 0 for v in numeric), exact_zero=sum(v == 0 for v in numeric),
                minimum=min(numeric) if numeric else None, maximum=max(numeric) if numeric else None,
                mean=st.mean(numeric) if numeric else None, median=st.median(numeric) if numeric else None)


def window_mean(values, start, stop):
    selected = [v for v in values[start - 1:stop] if v is not None]
    return st.mean(selected) if selected else None


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def analyze(root):
    folder = root / 'automation_reports/CV-SincNet' / RUN
    evidence = folder / 'evidence'
    source_path = evidence / 'source_research_complete.json'
    control_path = root / 'automation_reports/CV-SincNet' / CONTROL_RUN / 'evidence/source_research_complete.json'
    source, old = read(source_path), read(control_path)
    require(source['ready'] and source['run_id'] == RUN, 'Source collection not complete')
    rows = source['rows']
    keyed = {(r['resolved']['variant'], r['resolved']['model_seed']): r for r in rows}
    require(len(rows) == 16 and set(keyed) == {(v, s) for v in VARIANTS for s in SEEDS}, 'Source matrix differs')
    old_keyed = {(r['resolved']['variant'], r['resolved']['model_seed']): r for r in old['rows']
                 if r['resolved']['variant'] == CONTROL}
    require(set(old_keyed) == {(CONTROL, s) for s in SEEDS}, 'Missing shallow source controls')
    final = {key: final_metrics(row) for key, row in {**old_keyed, **keyed}.items()}
    controls = {(r['variant'], r['seed']): r for r in source['source_controls']}
    require(set(controls) == set(old_keyed), 'Control seed map differs')
    for key, recorded in controls.items():
        require(close(recorded['accuracy'], final[key]['ALL_V']) and close(recorded['worst_rx'], final[key]['WORST_RX']),
                'Historical source control differs from current frozen input')

    diagnostic_rows = {}
    audit_steps = audit_stdout = 0
    for (variant, seed), row in keyed.items():
        require(not row['completion']['target_access'] and not row['completion']['target_evaluated'], 'Non-source record')
        audit = row['log_audit']
        require(audit['epochs'] == 200 and audit['steps'] == 10000 and audit['csv_epochs'] == 200
                and audit['step_epoch_csv_stdout_reconciled'] and not audit['errors'], 'Incomplete upstream log audit')
        audit_steps += audit['steps']
        audit_stdout += audit['full_stdout_lines']
        for epoch in row['epochs']:
            d = epoch['channel_diagnostics']['channel_order']
            require(d['variant'] == variant and d['packets'] == 28 and d['valid_tokens'] == 246,
                    'Diagnostic scope changed')
            require(d['compensation_active'] == (variant != 'channel_capacity')
                    and d['order_difference_active'] == (variant == 'channel_order'), 'Activation mismatch')
            for metric in METRICS:
                active = not ((metric in G_METRICS and variant == 'channel_capacity')
                              or (metric in D_METRICS and variant != 'channel_order'))
                require((d[metric] is not None) == active, 'Inactive/missing metric mismatch: ' + metric)
            if d['compensation_active']:
                require(d['g_operator_delta_bound_max'] <= .25 + 1e-6
                        and d['g_coefficient_l1_max'] <= 1 + 1e-6 and d['g_basis_l1_max'] <= 1 + 1e-6,
                        'Observed G bound exceeded')
            diagnostic_rows[variant, seed, epoch['epoch']] = d
    require(len(diagnostic_rows) == 3200, 'Incomplete mechanism history')

    # Reconcile all provided mechanism CSV records, not a sampled tail.
    exported = csv_rows(evidence / 'channel_outputs.csv')
    require(len(exported) == 3200, 'Incomplete channel CSV')
    seen = set()
    for item in exported:
        key = item['variant'], int(item['model_seed']), int(item['epoch'])
        require(key not in seen and key in diagnostic_rows, 'Duplicate/foreign channel CSV row')
        seen.add(key)
        actual = diagnostic_rows[key]
        for metric in METRICS:
            require(item[metric] == '' if actual[metric] is None else close(item[metric], actual[metric]),
                    'Channel CSV/source mismatch: ' + metric)

    exported_final = csv_rows(evidence / 'source_final.csv')
    require(len(exported_final) == 20, 'Final source CSV coverage differs')
    seen = set()
    for item in exported_final:
        key = item['variant'], int(item['model_seed'])
        require(key in final and key not in seen, 'Duplicate/foreign final source row')
        seen.add(key)
        for field, metric in [('V', 'ALL_V'), ('worst_RX', 'WORST_RX'), ('score', 'SOURCE_SCORE')]:
            require(close(item[field], final[key][metric]), 'Final CSV/source mismatch')

    cells = csv_rows(evidence / 'source_cells.csv')
    require(len(cells) == 1800, 'Source cell coverage differs')
    cell_keys = {(r['variant'], int(r['model_seed']), int(r['tx']), str(r['receiver']), int(r['day'])) for r in cells}
    require(len(cell_keys) == 1800 and cell_keys == {(v, s, t, rx, day) for v in (CONTROL, *VARIANTS)
            for s in SEEDS for t in range(6) for rx in RXS for day in (1, 2, 3)}, 'Source cell matrix differs')
    for key in final:
        for rx in RXS:
            group = [r for r in cells if (r['variant'], int(r['model_seed'])) == key and r['receiver'] == rx]
            require(sum(int(r['count']) for r in group) == 5400 and all(int(r['count']) == 300 for r in group),
                    'Source cell sample counts differ')
            require(close(st.mean(float(r['accuracy']) for r in group), final[key]['RX_' + rx]),
                    'Source TX/RX/day cells do not reproduce RX accuracy')

    absolute = []
    selection = source['source_selection']
    for variant in (CONTROL, *VARIANTS):
        scores = [final[variant, s]['SOURCE_SCORE'] for s in SEEDS]
        record = dict(variant=variant, score=st.mean(scores), score_seed_sd=st.stdev(scores),
                      source_V=st.mean(final[variant, s]['ALL_V'] for s in SEEDS),
                      worst_RX=st.mean(final[variant, s]['WORST_RX'] for s in SEEDS))
        frozen = selection['source_summaries'][variant]
        require(close(record['score'], frozen['score']) and close(record['source_V'], frozen['source_accuracy'])
                and close(record['worst_RX'], frozen['worst_rx_accuracy']), 'Frozen source score mismatch')
        absolute.append(record)
    require(selection['status'] == 'SOURCE_SELECTION_FROZEN' and selection['selected_variant'] == 'channel_dual'
            and not selection['target_access'] and not selection['target_score_used'], 'Frozen source selection differs')

    paired, paired_summary = [], []
    for baseline in (CONTROL, 'channel_dual'):
        for variant in VARIANTS:
            if variant == baseline:
                continue
            for metric in final[variant, SEEDS[0]]:
                deltas = []
                for seed in SEEDS:
                    candidate_value, baseline_value = final[variant, seed][metric], final[baseline, seed][metric]
                    delta = 100 * (candidate_value - baseline_value)
                    deltas.append(delta)
                    paired.append(dict(variant=variant, baseline=baseline, model_seed=seed, metric=metric,
                        candidate_value=candidate_value, baseline_value=baseline_value, delta_pp=delta,
                        validation_count=27000 if metric == 'ALL_V' else (5400 if metric.startswith('RX_') else None)))
                paired_summary.append(dict(variant=variant, baseline=baseline, metric=metric,
                    delta_pp_mean=st.mean(deltas), delta_pp_seed_sd=st.stdev(deltas),
                    delta_pp_by_seed=deltas, model_seeds=list(SEEDS), positive_seeds=sum(x > 1e-10 for x in deltas),
                    negative_seeds=sum(x < -1e-10 for x in deltas), equal_seeds=sum(abs(x) <= 1e-10 for x in deltas)))

    mechanism_rows, variant_summaries = [], {}
    for variant in VARIANTS:
        variant_summaries[variant] = {}
        for metric in METRICS:
            all_values, late_values = [], []
            by_seed = {}
            for seed in SEEDS:
                values = [diagnostic_rows[variant, seed, epoch][metric] for epoch in range(1, 201)]
                stats = describe(values)
                positions = [index + 1 for index, value in enumerate(values) if value is not None and value > 0]
                record = dict(variant=variant, model_seed=seed, metric=metric, **stats,
                    first_positive_epoch=min(positions) if positions else None,
                    last_positive_epoch=max(positions) if positions else None,
                    E1=values[0], E200=values[-1], E1_20_mean=window_mean(values, 1, 20),
                    E91_110_mean=window_mean(values, 91, 110), E181_200_mean=window_mean(values, 181, 200))
                mechanism_rows.append(record)
                by_seed[str(seed)] = record
                all_values.extend(values)
                late_values.extend(values[-20:])
            variant_summaries[variant][metric] = dict(full_E1_200=describe(all_values),
                late_E181_200=describe(late_values), seeds=by_seed)

    worst_receivers = {v: {str(s): [rx for rx in RXS if close(final[v, s]['RX_' + rx], final[v, s]['WORST_RX'])]
                          for s in SEEDS} for v in (CONTROL, *VARIANTS)}
    summary = dict(status='VERIFIED', run_id=RUN, scope='source-only descriptive mechanism analysis',
        inputs=[str(source_path.relative_to(root)), str(control_path.relative_to(root)),
                *(str((evidence / name).relative_to(root)) for name in ('channel_outputs.csv', 'source_final.csv', 'source_cells.csv'))],
        source_read_at=source['read_at'], target_access=False, target_scores_used=False,
        model_or_checkpoint_loaded=False, selection_modified=False, frozen_selected_variant=selection['selected_variant'],
        coverage=dict(new_models=16, source_control_models=4, complete_epochs_per_model=200,
            new_epoch_records=3200, channel_csv_records=3200, source_final_rows=20, source_cells=1800,
            historical_control_epochs_parsed=800, upstream_audited_steps=audit_steps,
            upstream_audited_stdout_lines=audit_stdout, original_stdout_rescanned_here=False),
        diagnostic_scope=dict(packets=28, source='last training batch of each epoch',
            forward='eval-mode recomputation after last optimizer update and source V; same last training inputs',
            gradients='last CE backward before that final optimizer update, not gradients of diagnostic recomputation',
            missing='inactive G/D kept null; no imputation',
            windows='All 200 epochs included; E1-20/E91-110/E181-200 are descriptive summaries only',
            replication='4 model seeds; 800 diagnostic records per variant are not 800 independent replicates'),
        source_absolute=absolute, source_paired=paired_summary, worst_receiver_by_seed=worst_receivers,
        mechanism=variant_summaries,
        unmeasured=['D cross-order statistic norm before d_project', 'd_project output norm',
                    'separate u_project/v_project/d_project contributions to logits or correctness',
                    'CE loss sensitivity under removal of G or D', 'full V/query distribution of G and D'],
        conclusions=dict(performance='dual wins frozen source score by a very small mean; paired seed/RX evidence required',
            activation='Recorded nonzero D and gradient norms exclude an identically zero or disconnected D branch at logged observations only',
            causality='Activity does not establish useful identity information, causal gain, true channel recovery, or universal RX invariance',
            order='This order implementation does not surpass dual by frozen source score; D mechanism in general is not disproved'))
    text = report(summary)
    outputs = {'source_rx_paired.csv': csv_text(paired), 'channel_mechanism_summary.csv': csv_text(mechanism_rows),
               'mechanism_summary.json': json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n'}
    for name, content in outputs.items():
        (evidence / name).write_text(content, encoding='utf-8', newline='\n')
    (folder / 'mechanism_report.md').write_text(text, encoding='utf-8', newline='\n')
    require(read(evidence / 'mechanism_summary.json') == summary, 'JSON output readback differs')
    require(len(csv_rows(evidence / 'source_rx_paired.csv')) == 224, 'Paired output count differs')
    require(len(csv_rows(evidence / 'channel_mechanism_summary.csv')) == 256, 'Mechanism output count differs')
    for path in [folder / 'mechanism_report.md', *(evidence / name for name in outputs)]:
        raw = path.read_bytes()
        require(not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in raw.decode('utf-8'), 'Output encoding differs')
    print(json.dumps(dict(status='VERIFIED', new_epochs=3200, paired_rows=len(paired), mechanism_rows=len(mechanism_rows),
        report=str(folder / 'mechanism_report.md'), target_access=False, frozen_selection_unchanged=True), ensure_ascii=False))
    return summary


def report(summary):
    paired = {(r['variant'], r['baseline'], r['metric']): r for r in summary['source_paired']}
    absolute = {r['variant']: r for r in summary['source_absolute']}
    mechanism = summary['mechanism']
    dual = paired['channel_dual', CONTROL, 'SOURCE_SCORE']
    order = paired['channel_order', 'channel_dual', 'SOURCE_SCORE']
    def value(v, metric, window='late_E181_200', field='mean'):
        return mechanism[v][metric][window][field]
    def fmt(v, digits=6):
        return 'N/A' if v is None else f'{v:.{digits}f}'
    def scientific(v):
        return 'N/A' if v is None else f'{v:.3e}'
    lines = ['# CVS信道与非线性运算次序：源域机制证据', '',
        f'`channel_dual`按既定规则胜出，但相对原shallow的四seed源综合分仅增加{dual["delta_pp_mean"]:.5f}个百分点，'
        f'同seed差值SD为{dual["delta_pp_seed_sd"]:.5f}个百分点，{dual["positive_seeds"]}/4个seed为正。'
        f'`channel_order`比dual低{abs(order["delta_pp_mean"]):.5f}个百分点。记录显示D及其梯度并未始终为零；'
        '当前证据不能把order未胜出归因于死支路，也不能证明D对识别具有实质贡献。', '',
        '## 证据范围与口径', '',
        '完整读取16个新模型各200个epoch，共3200条epoch诊断，并逐字段核对3200行`channel_outputs.csv`、'
        '20行最终源指标和1800个TX×RX×day单元；四份shallow源控制另从其原源run完整记录取得。'
        '所有最终控制V/最差RX与本轮冻结选择输入一致。原有审计已对160000个训练step、epoch、CSV及stdout进行完整核对；'
        '本分析使用该审计记录，没有再次读取远端原始stdout或IQ。', '',
        '**G、D及辅助输出统计只覆盖每个epoch最后一个28包训练batch。**前向统计在最后一次参数更新及源V评估后，'
        '以eval模式对该训练batch重新计算；梯度范数来自该epoch最后一次CE backward，即最终更新之前。'
        '因此前向值与梯度不是同一参数时刻，也不是整个V集的分布。800条诊断是4个模型的重复观测，不能当作800次独立实验。'
        '下文E181–200是描述性末期汇总；所有200个epoch均进入全程统计，不改变固定E200选择。', '',
        '## 1. 源收益及同seed、同RX差值', '',
        '综合分定义为`0.5×源V准确率+0.5×最差源RX准确率`。百分比是分数×100；差值与SD的单位均为百分点。', '',
        '|模型|源V/%|最差RX/%|源综合分/%|对shallow差值±配对SD/百分点|正差seed|',
        '|---|---:|---:|---:|---:|---:|']
    for variant in (CONTROL, *VARIANTS):
        r = absolute[variant]
        p = paired.get((variant, CONTROL, 'SOURCE_SCORE'))
        diff = '—' if p is None else f'{p["delta_pp_mean"]:+.5f} ± {p["delta_pp_seed_sd"]:.5f}'
        lines.append(f'|{variant}|{100*r["source_V"]:.5f}|{100*r["worst_RX"]:.5f}|{100*r["score"]:.5f}|{diff}|{str(p["positive_seeds"])+"/4" if p else "—"}|')
    lines += ['', '|model seed|dual−shallow综合分|order−dual综合分|', '|---|---:|---:|']
    for i, seed in enumerate(SEEDS):
        lines.append(f'|{seed}|{dual["delta_pp_by_seed"][i]:+.5f}|{order["delta_pp_by_seed"][i]:+.5f}|')
    lines += ['', '|源RX（每模型5400包）|dual−shallow平均差/百分点|正差seed|order−dual平均差/百分点|正差seed|',
              '|---|---:|---:|---:|---:|']
    for rx in RXS:
        d = paired['channel_dual', CONTROL, 'RX_' + rx]
        o = paired['channel_order', 'channel_dual', 'RX_' + rx]
        lines.append(f'|{rx}|{d["delta_pp_mean"]:+.5f}|{d["positive_seeds"]}/4|{o["delta_pp_mean"]:+.5f}|{o["positive_seeds"]}/4|')
    dv, dw = (paired['channel_dual', CONTROL, metric]['delta_pp_mean'] for metric in ('ALL_V', 'WORST_RX'))
    worst_is_three = all(receivers == ['3'] for group in summary['worst_receiver_by_seed'].values() for receivers in group.values())
    lines += ['', f'dual的源V平均差为{dv:+.5f}个百分点，最差RX平均差为{dw:+.5f}个百分点；各乘0.5后形成上述综合分差。'
        + ('全部20个源模型的最差接收机均为RX3。' if worst_is_three else '各模型最差RX明细见JSON。')
        + '源V仍含训练阶段已见RX，不能据此证明未见RX泛化。四个seed覆盖模型/loader随机性，不能支持统计显著、普遍有效或域不变性结论。', '',
        '## 2. 补偿及辅助支路是否实际活动', '',
        'G输入改变量为逐包`||Gₓx−x||₂/||x||₂`的batch均值；它不是估计信道误差。'
        '辅助残差比为全部新增分支合成向量范数相对原主干特征范数，不是G或D的单独贡献。', '',
        '|模型|末期G输入改变量/%|全程G上界最大值|全程系数L1最大值|末期辅助残差比|G basis正梯度记录|G出口正梯度记录|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        g = value(variant, 'g_magnitude_mean')
        basis = mechanism[variant]['g_basis_gradient_norm']['full_E1_200']
        exit_gradient = mechanism[variant]['g_context_exit_gradient_norm']['full_E1_200']
        lines.append(f'|{variant}|{fmt(None if g is None else 100*g,4)}|'
            f'{fmt(value(variant,"g_operator_delta_bound_max","full_E1_200","maximum"))}|'
            f'{fmt(value(variant,"g_coefficient_l1_max","full_E1_200","maximum"))}|'
            f'{fmt(value(variant,"residual_relative_output_mean"),4)}|'
            f'{str(basis["positive"])+"/800" if basis["measured"] else "N/A"}|'
            f'{str(exit_gradient["positive"])+"/800" if exit_gradient["measured"] else "N/A"}|')
    active_records = sum(mechanism[v]['g_magnitude_mean']['full_E1_200']['measured'] for v in VARIANTS)
    g_nonzero = sum(mechanism[v]['g_magnitude_mean']['full_E1_200']['positive'] for v in VARIANTS)
    max_coefficient = max(value(v, 'g_coefficient_l1_max', 'full_E1_200', 'maximum') for v in VARIANTS[1:])
    lines += ['', f'三个补偿候选的{active_records}条记录中，G输入改变量非零的记录为{g_nonzero}条。'
        f'观测到的系数L1最大值为{max_coefficient:.6f}，低于约束1；日志没有显示这些记录中的系数触及上界。'
        '这支持补偿器实际参与了前向和优化，而非始终停在恒等映射。它不能证明G学到了真实信道逆，也不能排除未观测batch中更大的变化。'
        'capacity没有G，相关字段保持N/A，未用0冒充测量。', '',
        '## 3. D交互是否生成及参与优化', '',
        '`D=F(Gₓx)−GₓF(x)`，表中相对幅度为`||D||₂/||u||₂`的batch均值，`u=F(x)`。'
        '正梯度计数仅按记录值严格大于0统计，没有另设训练门槛。', '',
        '|model seed|全程D相对幅度范围/%|末期D相对幅度/%|D激活正梯度|D出口正梯度|E200 D出口权重范数|',
        '|---|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        s = str(seed)
        m = mechanism['channel_order']
        d = m['d_relative_output_mean']['seeds'][s]
        dg, pg = m['d_gradient_norm']['seeds'][s], m['d_readout_gradient_norm']['seeds'][s]
        lines.append(f'|{seed}|{100*d["minimum"]:.5f}至{100*d["maximum"]:.5f}|{100*d["E181_200_mean"]:.5f}|'
            f'{dg["positive"]}/200|{pg["positive"]}/200|{m["d_readout_norm"]["seeds"][s]["E200"]:.6f}|')
    dg = mechanism['channel_order']['d_gradient_norm']['full_E1_200']
    pg = mechanism['channel_order']['d_readout_gradient_norm']['full_E1_200']
    lines += ['', f'全程D激活梯度范数范围为{scientific(dg["minimum"])}至{scientific(dg["maximum"])}，'
        f'D出口梯度范数为{scientific(pg["minimum"])}至{scientific(pg["maximum"])}。'
        '这些记录排除了“在所记录的epoch末观测中D始终为零或梯度完全断开”的解释。它们没有覆盖每个optimizer step，'
        '也不能仅凭非零梯度或权重范数证明D改善了身份判别。', '',
        'D读出是`mean(conj(u)×D)/mean(|u|²)`的实部与虚部，再经16→160线性投影。'
        '**现有日志未测量交叉统计向量范数、D投影输出范数，也未分离u/v/D对logits或正确预测的贡献。**'
        'D能量非零仍可能在交叉统计汇聚中抵消；出口权重增大也不等于最终输出贡献增大。'
        '记录中的合成辅助残差不能补足这一缺项。因此不能判断“D信息无用”“模型忽略D”或“D导致性能下降”。', '',
        '## 可用于论文的结论', '',
        '在固定单CE、相同物理源数据和E200预算下，共享原始/补偿双路取得本轮最高平均源综合分，但差值很小且存在seed间变化。'
        '显式运算次序差确实产生了非零输出并有梯度参与优化，却没有超过双路的冻结源分数；这一结构假设的性能价值尚未得到本轮证据支持。'
        '结论限于当前网络、读出、预处理契约与源接收机集合，不能扩展为真实信道恢复、TX/RX分离、任意接收机不变性或目标域模块消融。', '',
        '本分析不读取目标评分、不更新候选或冻结选择、不启动补充实验。历史基准暴露及独立测试结论由相应报告另行说明。', '',
        '[结构化机制汇总](evidence/mechanism_summary.json) · [全部同seed源差值](evidence/source_rx_paired.csv) · '
        '[全部200epoch机制统计](evidence/channel_mechanism_summary.csv) · [原始完整源证据](evidence/source_research_complete.json)', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    analyze(parser.parse_args().root)
