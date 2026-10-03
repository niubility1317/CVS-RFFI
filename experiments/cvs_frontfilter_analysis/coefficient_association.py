"""Balanced orthogonal source coefficient association from existing cell moments.

No model, checkpoint, target data or training module is loaded. Population trace
variances describe eight real coefficient components (real4, imaginary4).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUN = '20261003-phase1-cvs-frontfilter-identity-manysig-m8-r01'
VARIANTS = ('frontfilter_static', 'frontfilter_dynamic')
SEEDS = tuple(range(2026092701, 2026092705))
CLASSES = ('14-10', '14-7', '20-15', '20-19', '6-15', '8-20')
TX = tuple(range(6))
RX = (1, 3, 4, 6, 8)
DAYS = (1, 2, 3)
FACTORS = ('TX', 'RX', 'day', 'TX_RX', 'TX_day', 'RX_day', 'TX_RX_day', 'within_cell')
# Existing source moment validation permits 1e-10 for the static filter's
# cancellation residual. Preserve raw values; suppress unstable ratios only.
NUMERICAL_ZERO_TOLERANCE = 1e-10
SCHEMA = frozenset(('variant', 'model_seed', 'tx', 'receiver', 'day', 'count',
    'coefficient_mean', 'coefficient_trace_variance', 'relative_input_change_mean',
    'relative_input_change_max', 'input_norm_ratio_min', 'input_norm_ratio_max',
    'input_norm_ratio_eligible_count', 'coefficient_l1_mean', 'coefficient_l1_max',
    'kernel_l1_mean', 'kernel_l1_max'))
SCOPE = ('Observed source coefficient association, not causal attribution. These factors do not '
    'establish channel recovery, TX/RX disentanglement or generalization. Never use this diagnostic '
    'for model/epoch selection, ranking, hyperparameters or selective reruns. Four model seeds '
    'share the same source samples, not four independent datasets.')


def number(value, label, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(label+' must be finite numeric data')
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(label+' outside the source measurement range')
    return float(value)


def decompose_coefficients(cell_means, within_cell_variance):
    """Orthogonal 6x5x3 balanced decomposition; all terms are population variance.

    The within-cell input is E||a-E(a|cell)||², not a variance of the cell mean.
    Equal count 300 makes the complete-cell average the sample-weighted average.
    """
    raw = np.asarray(cell_means)
    raw_within = np.asarray(within_cell_variance)
    if raw.shape != (6,5,3,8) or raw_within.shape != (6,5,3):
        raise ValueError('Expected all 6 TX x 5 RX x 3 day cells with eight coefficients')
    if raw.dtype.kind not in 'fiu' or raw_within.dtype.kind not in 'fiu':
        raise ValueError('Coefficient moments must be numeric arrays')
    means = raw.astype(np.float64); within = raw_within.astype(np.float64)
    if not np.isfinite(means).all() or not np.isfinite(within).all() or np.any(within < 0):
        raise ValueError('Nonfinite coefficient means or negative/nonfinite cell variance')
    with np.errstate(over='raise', invalid='raise'):
        try:
            # Translation removes artificial scatter from repeated decimal constants.
            reference = means[0,0,0].copy()
            grid = means-reference
            grand = grid.mean(axis=(0,1,2), keepdims=True)
            tx = grid.mean(axis=(1,2), keepdims=True)-grand
            rx = grid.mean(axis=(0,2), keepdims=True)-grand
            day = grid.mean(axis=(0,1), keepdims=True)-grand
            tx_rx = grid.mean(axis=2, keepdims=True)-grand-tx-rx
            tx_day = grid.mean(axis=1, keepdims=True)-grand-tx-day
            rx_day = grid.mean(axis=0, keepdims=True)-grand-rx-day
            triple = grid-grand-tx-rx-day-tx_rx-tx_day-rx_day
            terms = (tx,rx,day,tx_rx,tx_day,rx_day,triple)
            energies = {name:float(np.square(term).sum(axis=-1).mean())
                        for name,term in zip(FACTORS[:-1],terms)}
            energies['within_cell'] = float(within.mean())
            between = float(np.square(grid-grand).sum(axis=-1).mean())
            total = between+energies['within_cell']
            rebuilt = sum(energies.values())
            residual = float(np.max(np.abs(grid-(grand+tx+rx+day+tx_rx+tx_day+rx_day+triple))))
        except FloatingPointError as exc:
            raise ValueError('Coefficient variance exceeds finite float64 range') from exc
    if not np.isfinite(total) or not np.isfinite(rebuilt):
        raise ValueError('Coefficient variance exceeds finite float64 range')
    if not np.isclose(total, rebuilt, rtol=1e-10, atol=1e-14):
        raise ValueError('Orthogonal components do not reconstruct total coefficient variance')
    near_zero = total <= NUMERICAL_ZERO_TOLERANCE
    return dict(global_mean=(grand.reshape(8)+reference).tolist(),
        components=energies, fractions={k:None if near_zero else v/total for k,v in energies.items()},
        total_trace_variance=total, between_cell_trace_variance=between,
        reconstructed_total_trace_variance=rebuilt,reconstruction_absolute_error=abs(total-rebuilt),
        cell_mean_reconstruction_max_abs_error=residual,
        numerically_zero_total=near_zero,fraction_denominator_tolerance=NUMERICAL_ZERO_TOLERANCE,
        raw_variances_preserved=True)


def validate_cells(rows):
    expected={(v,s,t,r,d) for v in VARIANTS for s in SEEDS for t in TX for r in RX for d in DAYS}
    if len(rows)!=720:raise ValueError('Complete source coefficient association requires 720 cells')
    indexed={}
    for row in rows:
        if set(row)!=SCHEMA:raise ValueError('Source coefficient cell fields differ')
        for key in ('model_seed','tx','receiver','day','count','input_norm_ratio_eligible_count'):
            if type(row[key]) is not int:raise ValueError('Source cell metadata must use integers: '+key)
        key=tuple(row[k] for k in ('variant','model_seed','tx','receiver','day'))
        if key not in expected or key in indexed or row['count']!=300:
            raise ValueError('Unregistered, repeated or unbalanced source cell')
        for field in ('coefficient_trace_variance','relative_input_change_mean','relative_input_change_max',
                      'coefficient_l1_mean','coefficient_l1_max','kernel_l1_mean','kernel_l1_max'):
            number(row[field],field,0)
        mean=row['coefficient_mean']
        if not isinstance(mean,list) or len(mean)!=8:raise ValueError('Eight source coefficient means required')
        for value in mean:number(value,'coefficient_mean',-1-1e-5,1+1e-5)
        for stem in ('coefficient_l1','kernel_l1'):
            if row[stem+'_mean']>row[stem+'_max']+1e-5 or row[stem+'_max']>1+1e-5:
                raise ValueError('Source coefficient/kernel L1 bound differs')
        if (row['kernel_l1_mean']>row['coefficient_l1_mean']+1e-5 or
                row['kernel_l1_max']>row['coefficient_l1_max']+1e-5 or
                row['relative_input_change_mean']>row['relative_input_change_max']+1e-5 or
                row['relative_input_change_max']>.25*row['kernel_l1_max']+1e-5 or
                sum(np.hypot(mean[j],mean[j+4]) for j in range(4))>row['coefficient_l1_mean']+1e-5 or
                sum(v*v for v in mean)+row['coefficient_trace_variance']>1+1e-5):
            raise ValueError('Source coefficient moment or fixed-coefficient bound differs')
        n=row['input_norm_ratio_eligible_count']
        if not 0<=n<=300:raise ValueError('Source norm eligibility differs')
        for field in ('input_norm_ratio_min','input_norm_ratio_max'):
            if (row[field] is None)!=(n==0):raise ValueError('Source norm-ratio N/A differs')
            if n:number(row[field],field,.75-1e-5,1.25+1e-5)
        if n and row['input_norm_ratio_min']>row['input_norm_ratio_max']:
            raise ValueError('Source norm-ratio extrema differ')
        indexed[key]=row
    if set(indexed)!=expected:raise ValueError('Missing source cells')
    for seed in SEEDS:
        static=[r for key,r in indexed.items() if key[:2]==(VARIANTS[0],seed)]
        reference=np.asarray(static[0]['coefficient_mean'])
        if (any(r['coefficient_trace_variance']>1e-10 for r in static) or
                any(np.max(np.abs(np.asarray(r['coefficient_mean'])-reference))>1e-10 for r in static)):
            raise ValueError('Static coefficient moments differ from fixed shared coefficients')
    return indexed


def analyze_cells(rows):
    indexed=validate_cells(rows); models=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            cells=[indexed[variant,seed,t,r,d] for t in TX for r in RX for d in DAYS]
            means=np.asarray([r['coefficient_mean'] for r in cells]).reshape(6,5,3,8)
            within=np.asarray([r['coefficient_trace_variance'] for r in cells]).reshape(6,5,3)
            models.append(dict(variant=variant,model_seed=seed,cells=90,count=27000,
                               **decompose_coefficients(means,within)))
    summary=[]
    for variant in VARIANTS:
        members=[m for m in models if m['variant']==variant]
        for factor in (*FACTORS,'total'):
            values=[m['total_trace_variance'] if factor=='total' else m['components'][factor] for m in members]
            fractions=[] if factor=='total' else [m['fractions'][factor] for m in members if m['fractions'][factor] is not None]
            summary.append(dict(variant=variant,factor=factor,seeds=4,
                trace_variance_mean=float(np.mean(values)),trace_variance_seed_sd=float(np.std(values,ddof=1)),
                fraction_defined_seeds=len(fractions),
                fraction_mean=float(np.mean(fractions)) if fractions else None,
                fraction_seed_sd=float(np.std(fractions,ddof=1)) if len(fractions)>1 else None))
    return dict(analysis='source coefficient association',models=models,four_seed_summary=summary,
        statistics_dtype='float64',classes=list(CLASSES),factor_levels=dict(TX=list(TX),RX=list(RX),day=list(DAYS)),
        coefficient_order='real basis 0..3, imaginary basis 0..3',
        definition='Population trace variance: mean squared Euclidean deviation across eight coefficients. '
                   'Seven balanced orthogonal cell-mean effects plus mean within-cell population trace variance. '
                   'No division of within-cell variance by 300; no fitting, hypothesis test or p-values.',
        numeric_scope='Raw near-zero variances are retained; fractions are null when total <= 1e-10 to avoid amplifying cancellation noise. This is numerical reporting, not a performance gate.',
        source_only=True,model_selection=False,target_access=False,interpretation=SCOPE)


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames)!=len(SCHEMA) or set(reader.fieldnames)!=SCHEMA:
            raise ValueError('Source coefficient CSV header differs')
        rows=[]
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()):raise ValueError('Malformed source coefficient CSV row')
            row=dict(raw)
            try:
                for key in ('model_seed','tx','receiver','day','count','input_norm_ratio_eligible_count'):row[key]=int(raw[key])
                row['coefficient_mean']=json.loads(raw['coefficient_mean'])
                for key in SCHEMA-{'variant','model_seed','tx','receiver','day','count','input_norm_ratio_eligible_count','coefficient_mean'}:
                    row[key]=None if raw[key]=='' and key.startswith('input_norm_ratio_') else float(raw[key])
            except (ValueError,TypeError,json.JSONDecodeError) as exc:
                raise ValueError('Malformed source coefficient cell value') from exc
            rows.append(row)
    return rows


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def completed_source_cells(evidence):
    """Require existing terminal source evidence before consuming its derived CSV."""
    selection=read_json(evidence/'source_selection.json')
    completion=read_json(evidence/'source_completion_validation.json')
    analysis=read_json(evidence/'source_analysis_validation.json')
    universe=['neural_residual_shallow','response_anchor_mean',*VARIANTS]
    if (selection.get('status')!='SOURCE_SELECTION_FROZEN' or selection.get('candidate_universe')!=universe or
            selection.get('selected_variant') not in universe or selection.get('target_access') is not False or
            selection.get('target_score_used') is not False or
            selection.get('new_candidate_selected') is not (selection.get('selected_variant') in VARIANTS)):
        raise ValueError('Complete source-only selection freeze required')
    required=dict(status='VERIFIED',new_rows=8,control_rows=8,new_epochs=1600,new_steps=80000,
        full_stdout_scanned=True,step_epoch_csv_stdout_reconciled=True,full_source_records_reverified=True,
        source_rule_recomputed=True,target_access=False,source_selection=selection)
    if any(completion.get(k)!=v for k,v in required.items()):raise ValueError('Source completion not independently verified')
    required=dict(status='VERIFIED',models=8,source_epochs=1600,control_epochs=1600,source_frontfilter_cells=720,
                  target_results_read=False,selection=selection)
    if any(analysis.get(k)!=v for k,v in required.items()):raise ValueError('Source analysis incomplete or freeze differs')
    source=read_json(evidence/'source_research_complete.json')
    terminal=('SOURCE_RESEARCH_COMPLETE_AWAITING_FROZEN_CLEAN_TEST','SOURCE_RESEARCH_COMPLETE_BASELINE_RETAINED')
    if (source.get('ready') is not True or source.get('run_id')!=RUN or
            source.get('pipeline',{}).get('status') not in terminal or source.get('source_selection')!=selection):
        raise ValueError('Only terminal source artifacts with a consistent freeze may be analyzed')
    models=source.get('rows',[])
    if len(models)!=8 or {(r['resolved']['variant'],r['resolved']['model_seed']) for r in models}!={(v,s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('Complete eight-model source matrix required')
    raw={}; expected={(t,r,d) for t in TX for r in RX for d in DAYS}
    for model in models:
        v=model['resolved']['variant'];s=model['resolved']['model_seed'];done=model['completion'];diag=model['source_diagnostics']
        if any(done.get(k)!=x for k,x in dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False).items()):
            raise ValueError('Nonterminal or contaminated source model')
        if (diag.get('role')!='V' or diag.get('count')!=27000 or
                any(diag.get(k) is not False for k in ('target_access','used_for_training','used_for_selection')) or
                len(diag.get('groups',[]))!=90 or {(g['tx'],g['receiver'],g['day']) for g in diag['groups']}!=expected or
                any(g['count']!=300 for g in diag['groups'])):
            raise ValueError('Source identity/class factor grid differs from coefficient input')
        cells=diag.get('frontfilter_groups',[])
        if len(cells)!=90:raise ValueError('Complete source coefficient moments missing')
        for cell in cells:
            key=(v,s,cell['tx'],cell['receiver'],cell['day'])
            if key in raw:raise ValueError('Repeated original source coefficient cell')
            raw[key]=dict(variant=v,model_seed=s,**cell)
    rows=read_csv(evidence/'source_frontfilter_cells.csv'); indexed=validate_cells(rows)
    if indexed!=raw:raise ValueError('Derived coefficient CSV differs from complete source moments')
    return rows,selection


def markdown(result):
    text='# Source coefficient association\n\n'
    text+='固定 E200 的全部源 V：8 个模型，各 27000 包，6 TX × 5 RX × 3 day，每单元 300 包。系数排列为 real4、imaginary4。按每模型完整平衡析因分解，再计算四 seed 均值与样本 SD；四 seed 共用数据，不能当作四份独立数据集。\n\n'
    text+='七个主效应/交互项为单元均值的正交分解，within-cell 为单元内 population trace variance 的平均，不除以 300。总量是 8 个系数分量方差之和。逐 seed CSV 与 JSON 保留全均值、各项原始值及重构误差。\n\n'
    text+='|结构|因素|trace variance 均值 ± seed SD|占比均值 ± seed SD|有效占比 seed|\n|---|---|---:|---:|---:|\n'
    for row in result['four_seed_summary']:
        fraction='N/A' if row['fraction_mean'] is None else f"{row['fraction_mean']:.6f}"+(f" ± {row['fraction_seed_sd']:.6f}" if row['fraction_seed_sd'] is not None else '')
        text+=f"|{row['variant']}|{row['factor']}|{row['trace_variance_mean']:.8g} ± {row['trace_variance_seed_sd']:.8g}|{fraction}|{row['fraction_defined_seeds']}/4|\n"
    text+='\n总方差不超过 1e-10 时占比记 N/A，避免将静态系数的浮点消减微差放大成因素占比；实际方差原值保留，不静默归零。此容差只影响数值展示，不是性能或晋级门槛。不同 seed 的滤波基各自学习，系数坐标不保证跨模型对齐；这里汇总各模型内部方差，不平均不同模型的系数向量来推断物理因素。\n\n'
    text+='这些是源系数与 TX/RX/day 标签的观测关联，非因果归因，不能证明信道恢复、TX/RX 解耦或泛化。within-cell 同时包含包间变化和未建模因素，不等同于噪声。该诊断禁止用于选模、重选 epoch、调参或选择性重跑。未访问 target，不改变已有源冻结。\n'
    return text


def run(root):
    evidence=Path(root)/'automation_reports/CV-SincNet'/RUN/'evidence'
    rows,selection=completed_source_cells(evidence)
    result=analyze_cells(rows);result.update(source_run=RUN,source_selection=selection,
        input_ref='source_frontfilter_cells.csv',status='VERIFIED')
    flat=[]
    for model in result['models']:
        row={k:v for k,v in model.items() if k not in ('components','fractions','global_mean')}
        row['global_mean']=json.dumps(model['global_mean'],allow_nan=False)
        row.update({key+'_trace_variance':value for key,value in model['components'].items()})
        row.update({key+'_fraction':value for key,value in model['fractions'].items()});flat.append(row)
    prefix=evidence/'source_coefficient_association'
    # All validation and arithmetic precede output creation. Existing source artifacts are read-only.
    with Path(str(prefix)+'_by_seed.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    Path(str(prefix)+'.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    Path(str(prefix)+'.md').write_text(markdown(result),encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',analysis=result['analysis'],models=8,cells=720,source_only=True)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
