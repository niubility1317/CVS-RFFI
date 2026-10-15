"""Independent CPU truth-last scoring and complete three-seed paired analysis."""
import os

from . import design as d
from .predict import preflight
from experiments.cvs_feature_sequence50 import score as shared


COMPLETE_MATRIX = dict(rows=12, new_backbones=12, existing_backbone_references=0,
    readouts=12, prediction_count=2016000, reused_native_decisions=0,
    new_prediction_decisions=2016000, metric_records=216,
    backbone_updates_new=53280, historical_baseline_updates=0)


def validate_complete_matrix(result):
    """Require this actual four-arm/three-seed closure, not inherited E50 totals."""
    rows = d.rows()
    expected = {(arm, seed) for arm in d.ARMS for seed in d.SEEDS}
    actual = [(row['arm'], row['model_seed']) for row in rows]
    if (len(d.ARMS) != 4 or len(d.SEEDS) != 3 or len(rows) != 12
            or len(set(actual)) != 12 or set(actual) != expected
            or d.EPOCHS != 20 or d.UPDATES != 4440
            or tuple(d.READOUTS) != ('NATIVE_HEAD',) or tuple(d.VIEWS) != ('clean',)):
        raise ValueError('Complete registered scratch20 four-arm/three-seed matrix required')
    if any(result.get(key) != value for key, value in COMPLETE_MATRIX.items()):
        raise ValueError('Complete12 rows/readouts/2016000decisions/216metrics/53280updates required')
    records = result.get('records', [])
    overall = [record for record in records if record.get('dimension') == 'overall']
    if (len(records) != 216 or len(result.get('source_diagnostics', [])) != 12
            or len(overall) != 12
            or {(record.get('arm'), record.get('model_seed')) for record in overall} != expected
            or any(record.get('readout') != 'NATIVE_HEAD' or record.get('query_count') != 168000
                   for record in overall)):
        raise ValueError('All actual per-row metrics and source freeze records are required')
    pairs = result.get('paired', [])
    if (len(pairs) != len(d.PAIRS)
            or {(pair.get('treatment'), pair.get('control')) for pair in pairs} != set(d.PAIRS)
            or any(pair.get('readout') != 'NATIVE_HEAD' or len(pair.get('seed_differences', [])) != 3
                   for pair in pairs)):
        raise ValueError('Complete registered three-seed paired accuracy records are required')
    return result


def paired_errors(control, treatment):
    """Count changed correctness after both rows have frozen predictions."""
    import numpy as np
    if (not isinstance(control, np.ndarray) or not isinstance(treatment, np.ndarray)
            or control.shape != (168000,) or treatment.shape != control.shape
            or control.dtype != np.bool_ or treatment.dtype != np.bool_):
        raise ValueError('Two complete boolean default-head correctness records required')
    corrected = int((~control & treatment).sum())
    introduced = int((control & ~treatment).sum())
    return dict(count=168000, corrected_errors=corrected, introduced_errors=introduced,
                both_correct=int((control & treatment).sum()),
                both_wrong=int((~control & ~treatment).sum()),
                net_correct_change=corrected - introduced,
                accuracy_difference=(corrected - introduced) / 168000)


def score(row):
    # The dispatcher replaces the prediction process with this fresh CPU process.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent truth-side scoring requires the fresh CPU process')
    import numpy as np
    from experiments.cvs_report_ssl.evaluate import truth_arrays, metric_rows
    c, ids = preflight(row)
    out = d.BASE / row / 'scores'
    if out.exists():
        raise FileExistsError('Existing scoring output; preserve it')
    out.mkdir()
    y, rx, days = truth_arrays(d.read(d.TRUTH), ids)
    records = []
    correctness = {}
    with np.load(d.BASE / row / 'prediction/predictions.npz', allow_pickle=False) as predictions:
        for readout in d.READOUTS:
            decisions = predictions[readout]
            rr, dd = metric_rows(c, decisions, ids, y, rx, days)
            for record in rr + dd:
                record['readout'] = readout
            records.extend(rr + dd)
            correctness[readout] = decisions == y
    if len(records) != 18:
        raise ValueError('The complete row must have all 18 confusion-matrix records')
    np.savez_compressed(out / 'correctness.npz', ids=ids, **correctness)
    d.write(out / 'metrics.json', dict(records=records, truth_last=True,
                                      independent_recount='VERIFIED'))
    d.write(d.BASE / row / 'completion.json', dict(
        status='TRAINED_DEFAULT_PREDICTED_SCORED', config=c, views=['clean'],
        readouts=list(d.READOUTS), count_per_readout=168000, metric_records=18,
        truth_last=True, independent_recount='VERIFIED'))
    print('SCORED ' + row, flush=True)


def finish():
    # Shared finish independently recounts every CM and rejects incomplete seeds.
    result = shared.finish(design=d)
    validate_complete_matrix(result)
    import numpy as np
    correctness = {}
    physical = None
    for row in d.rows():
        with np.load(d.BASE / row['row_id'] / 'scores/correctness.npz', allow_pickle=False) as record:
            if set(record.files) != {'ids', 'NATIVE_HEAD'}:
                raise ValueError('Complete default-head truth-side correctness artifact required')
            if len(record['ids']) != 168000:
                raise ValueError('All168000 frozen physical correctness records are required')
            if physical is None:
                physical = record['ids'].copy()
            elif not np.array_equal(physical, record['ids']):
                raise ValueError('Paired frozen rows have different physical IDs')
            correctness[row['arm'], row['model_seed']] = record['NATIVE_HEAD'].copy()
    pairs = []
    for treatment, control in d.PAIRS:
        rows = [dict(model_seed=seed,
                     **paired_errors(correctness[control, seed], correctness[treatment, seed]))
                for seed in d.SEEDS]
        matching = next(p for p in result['paired']
                        if p['treatment'] == treatment and p['control'] == control)
        if any(abs(r['accuracy_difference'] - expected) > 1e-12
               for r, expected in zip(rows, matching['seed_differences'])):
            raise ValueError('Paired error counts differ from independently recounted accuracy')
        pairs.append(dict(treatment=treatment, control=control, readout='NATIVE_HEAD',
                          seeds=rows, complete_seed_count=len(d.SEEDS),
                          corrected_errors=sum(r['corrected_errors'] for r in rows),
                          introduced_errors=sum(r['introduced_errors'] for r in rows)))
    result['paired_error_changes'] = pairs
    d.write(d.BASE / 'analysis.json', result)
    lines = ['# ' + d.TITLE, '', '|条件|准确率均值|SD|Macro-F1|', '|---|---:|---:|---:|']
    for r in result['summary']:
        lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['macro_f1_mean']:.6f}|")
    lines += ['', '|配对（三种子合计）|更正错误|引入错误|', '|---|---:|---:|']
    for p in pairs:
        lines.append(f"|{p['treatment']} − {p['control']}|{p['corrected_errors']}|{p['introduced_errors']}|")
    lines += ['', '全部 12 个模型均为 scratch20、每模型4440次真实更新，合计53280次；固定原默认头clean决定2016000条、216份指标。BASE_CE 使用20轮纯L CE，实际关闭U和EMA；其余条件为13轮L加7轮原U门控与own EMA。普通L增强及星地训练关闭，原U专用Gaussian强视图保留。所有条件使用SIG_TIME、固定CENTER25K和既有默认6×160分类头联合训练，没有另训或替换头。', '', 'JOINT_GAP与ROUTED_GAP只在前13个L轮每16步辅助一次，各180次；BASE_CE与BASE_SSL辅助次数为0。原主sampler和完整U损失保持；责任路由只改变辅助梯度路径，默认W与三个gain不接收辅助梯度，主CE仍正常共同训练。额外L曝光及四个真实AdamW探针按source机制记录独立披露，不计为新增主更新。', '', '冻结final20后逐行clean预测，再由独立CPU scorer连接truth；配对更正／引入错误来自全部冻结truth-side正确性记录。源V和源机制记录只用于预登记源规则，不拟合参数。source资格通过不等于84%目标达成，结果保留全部三种子和负结果，不回流训练、参数或选模；该基准已经暴露，不称新的独立盲测。']
    (d.BASE / 'final_analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
