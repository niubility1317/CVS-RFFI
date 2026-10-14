"""Independent CPU truth-last scoring and complete three-seed paired analysis."""
import os

from . import design as d
from .predict import preflight
from experiments.cvs_feature_sequence50 import score as shared


def paired_errors(control, treatment):
    """Count changed correctness after both rows have frozen predictions."""
    import numpy as np
    if (control.shape != (168000,) or treatment.shape != control.shape
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
    import numpy as np
    correctness = {}
    physical = None
    for row in d.rows():
        with np.load(d.BASE / row['row_id'] / 'scores/correctness.npz', allow_pickle=False) as record:
            if set(record.files) != {'ids', 'NATIVE_HEAD'}:
                raise ValueError('Complete default-head truth-side correctness artifact required')
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
    lines += ['', '全部 15 个模型均为 scratch20、4440 次真实更新。BASE_CE 使用 20 轮纯 L CE，实际关闭 U 和 EMA；其余条件为 13 轮 L 加 7 轮原 U 门控与 own EMA。所有条件使用 SIG_TIME、固定 CENTER25K 坐标和既有默认 6×160 分类头联合训练，没有另训或替换头。冻结 final20 后逐行 clean 预测，再由独立 CPU scorer 连接 truth；配对更正／引入错误来自冻结的 truth-side 正确性记录。源 V 和源机制记录只用于预登记源规则，不拟合参数。结果保留全部三种子和负结果，不回流训练、参数或选模；该基准已经暴露，不称新的独立盲测。']
    (d.BASE / 'final_analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
