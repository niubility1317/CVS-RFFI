"""Source-only full-curve report; preserves fixed E200 selection."""
import argparse,csv,json,math,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cvs_frontfilter_identity.prepare import RUN
from experiments.cvs_frontfilter_identity.collect import validate_completed, validate_frontfilter_diagnostics
from experiments.cvs_frontfilter_identity.dispatch import CONTROL_RUN,CONTROL,CANDIDATES,ANCHOR_CONTROL,ANCHOR_CONTROL_RUN,SEEDS
from experiments.cvs_frontfilter_identity.model import VARIANTS, filter_contract

FILTER_BLOCKS = ('frontfilter',)
LEARNING_METRICS = ('train_CE', 'V_CE', 'CE_gap', 'V', 'worst_RX', 'score')
OUTPUT_SCALE_METRICS = ('relative_input_change_mean', 'relative_input_change_max',
                        'input_norm_ratio_min', 'input_norm_ratio_max')
BOUND_METRICS = ('coefficient_l1_mean', 'coefficient_l1_max', 'kernel_l1_mean', 'kernel_l1_max',
                 'basis_l1_max', 'bound_active_fraction', 'basis_bound_active_fraction',
                 'coefficient_packet_variance', 'fixed_coefficient_delta_operator_bound_max')
COMPONENT_GRADIENT_METRICS = ('basis_gradient_norm', 'context_gradient_norm', 'exit_gradient_norm')
OUTPUT_METRICS = (*OUTPUT_SCALE_METRICS, *BOUND_METRICS, *COMPONENT_GRADIENT_METRICS)
FULL_V_METRICS = (*OUTPUT_SCALE_METRICS, 'input_norm_ratio_eligible_count',
                  'coefficient_l1_mean', 'coefficient_l1_max', 'kernel_l1_mean', 'kernel_l1_max',
                  'coefficient_trace_variance', 'coefficient_within_cell_trace_variance',
                  'coefficient_between_cell_mean_scatter')


def frontfilter_output_values(diagnostic, variant):
    """Keep actual last-batch scalars; absent eligible inputs/gradients stay null."""
    record = validate_frontfilter_diagnostics(diagnostic, 28, filter_contract(variant))
    return {key: None if record[key] is None else float(record[key]) for key in OUTPUT_METRICS}


def finite(value, label, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f'{label} outside its measured range')
    return float(value)


def mean_sd(values):
    """Sample SD over the four paired model seeds, never over pooled epochs."""
    values = [v for v in values if v is not None]
    return {'mean': st.mean(values) if values else None,
            'SD': st.stdev(values) if len(values) > 1 else None,
            'measured_seeds': len(values)}


def summarize_source_filter_cells(rows):
    """Aggregate full V cells using their sample counts; no training or selection."""
    from experiments.cvs_frontfilter_identity.collect import validate_frontfilter_groups
    expected={(v,s) for v in VARIANTS for s in SEEDS}
    keys=[(r['resolved']['variant'],r['resolved']['model_seed']) for r in rows]
    if len(keys)!=8 or set(keys)!=expected:
        raise ValueError('Full V filter summary requires eight registered models')
    cells=[]; by_seed=[]
    for row in rows:
        variant=row['resolved']['variant']; seed=row['resolved']['model_seed']
        groups=validate_frontfilter_groups(row['source_diagnostics'],filter_contract(variant))
        count=sum(g['count'] for g in groups)
        base=dict(variant=variant,model_seed=seed)
        cells.extend(dict(base,**g) for g in groups)
        r=dict(base,count=count,cells=len(groups))
        for key in ('relative_input_change_mean','coefficient_l1_mean','kernel_l1_mean'):
            r[key]=sum(g['count']*g[key] for g in groups)/count
        for key in ('relative_input_change_max','coefficient_l1_max','kernel_l1_max'):
            r[key]=max(g[key] for g in groups)
        eligible=[g for g in groups if g['input_norm_ratio_eligible_count']]
        r['input_norm_ratio_eligible_count']=sum(g['input_norm_ratio_eligible_count'] for g in groups)
        r['input_norm_ratio_min']=min(g['input_norm_ratio_min'] for g in eligible) if eligible else None
        r['input_norm_ratio_max']=max(g['input_norm_ratio_max'] for g in eligible) if eligible else None
        mean=[sum(g['count']*g['coefficient_mean'][j] for g in groups)/count for j in range(8)]
        within=sum(g['count']*g['coefficient_trace_variance'] for g in groups)/count
        between=sum(g['count']*sum((g['coefficient_mean'][j]-mean[j])**2 for j in range(8)) for g in groups)/count
        r.update(coefficient_mean=mean,coefficient_within_cell_trace_variance=within,
            coefficient_between_cell_mean_scatter=between,coefficient_trace_variance=within+between)
        by_seed.append(r)
    summary=[]
    for variant in VARIANTS:
        records=[r for r in by_seed if r['variant']==variant]; aggregate=dict(variant=variant,seeds=4)
        for key in FULL_V_METRICS:
            aggregate.update({f'{key}_{name}':value for name,value in mean_sd(r[key] for r in records).items()})
        summary.append(aggregate)
    return dict(source_frontfilter_cells=cells,source_frontfilter_by_seed=by_seed,
                source_frontfilter_summary=summary)


def summarize_source_telemetry(rows):
    """Describe complete source telemetry without reading data or selecting an epoch.

    This is an arithmetic summary, not a replacement for validate_completed's
    independent completion/provenance/log checks at the analyze entry point.
    """
    expected = {(v, s) for v in CANDIDATES for s in SEEDS}
    keys = [(r['resolved']['variant'], r['resolved']['model_seed']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Expected exactly four registered seeds for every source candidate')
    indexed = dict(zip(keys, rows))
    learning = []
    series = {}
    gradient_epochs = []
    gradient_seeds = []
    output_epochs = []
    last_outputs = []
    for variant in CANDIDATES:
        for seed in sorted(SEEDS):
            row = indexed[variant, seed]
            epochs = row['epochs']
            if [e['epoch'] for e in epochs] != list(range(1, 201)):
                raise ValueError('Complete ordered E1 through E200 telemetry is required')
            base = dict(variant=variant, model_seed=seed)
            points = []
            for epoch in epochs:
                train = finite(epoch.get('clean_ce'), 'training CE', 0)
                val = finite(epoch.get('source_val_ce'), 'source V CE', 0)
                accuracy = finite(epoch.get('source_val_accuracy'), 'source V accuracy', 0, 1)
                worst = finite(epoch.get('source_val_worst_rx'), 'worst source RX', 0, 1)
                point = dict(train_CE=train, V_CE=val, CE_gap=val-train,
                             V=accuracy, worst_RX=worst, score=.5*(accuracy+worst))
                points.append(point)
                if variant not in VARIANTS:
                    continue
                norm = finite(epoch.get('frontfilter_gradient_norm'), 'frontfilter epoch-mean gradient', 0)
                gradient_epochs.append(dict(base, epoch=epoch['epoch'], frontfilter_gradient_norm=norm))
                diagnostic = epoch.get('frontfilter_diagnostics', {}).get('frontfilter', {})
                values = frontfilter_output_values(diagnostic, variant)
                output_epochs.append(dict(base, epoch=epoch['epoch'], packets=28,
                    input_norm_ratio_eligible_packets=diagnostic['records'][0]['input_norm_ratio_eligible_packets'], **values))
                if epoch['epoch'] == 200:
                    last_outputs.append(dict(output_epochs[-1], block=FILTER_BLOCKS[0]))
            series[variant, seed] = points
            final = dict(base, epoch=200, **points[-1])
            for metric in LEARNING_METRICS:
                early = st.mean(p[metric] for p in points[150:175])
                late = st.mean(p[metric] for p in points[175:200])
                final[f'{metric}_E151_175'] = early
                final[f'{metric}_E176_200'] = late
                final[f'{metric}_late_delta'] = late-early
            learning.append(final)
            if variant in VARIANTS:
                norms = [float(e['frontfilter_gradient_norm']) for e in epochs]
                gradient_seeds.append(dict(base, epochs=200, audited_steps=row.get('log_audit', {}).get('steps'),
                    E1_epoch_mean=norms[0], E200_epoch_mean=norms[-1], epoch_mean_min=min(norms),
                    epoch_mean_max=max(norms), full_run_epoch_mean=st.mean(norms),
                    E151_200_epoch_mean=st.mean(norms[150:]), zero_epoch_means=sum(x == 0 for x in norms)))

    def grouped(records, groups, fields):
        output = []
        for identity in groups:
            members = [r for r in records if all(r[k] == v for k, v in identity.items())]
            if len(members) != 4 or {r['model_seed'] for r in members} != SEEDS:
                raise ValueError('Four matched seeds required for each aggregate')
            aggregate = dict(identity, seeds=4)
            for field in fields:
                aggregate.update({f'{field}_{key}': value for key, value in mean_sd(r[field] for r in members).items()})
            output.append(aggregate)
        return output

    learning_fields = [k for k in learning[0] if k not in ('variant', 'model_seed', 'epoch')]
    learning_summary = grouped(learning, [dict(variant=v) for v in CANDIDATES], learning_fields)
    curve_summary = []
    for variant in CANDIDATES:
        for epoch in range(1, 201):
            record = dict(variant=variant, epoch=epoch, seeds=4)
            for metric in LEARNING_METRICS:
                record.update({f'{metric}_{key}': value for key, value in mean_sd(series[variant, s][epoch-1][metric] for s in sorted(SEEDS)).items()})
            curve_summary.append(record)
    paired = []
    for seed in sorted(SEEDS):
        static = series['frontfilter_static', seed][-1]
        dynamic = series['frontfilter_dynamic', seed][-1]
        paired.append(dict(model_seed=seed, epoch=200,
            **{f'{metric}_delta_pp': 100*(dynamic[metric]-static[metric]) for metric in ('score', 'V', 'worst_RX')}))
    paired_summary = []
    for metric in ('score', 'V', 'worst_RX'):
        deltas = [r[f'{metric}_delta_pp'] for r in paired]
        paired_summary.append(dict(comparison='frontfilter_dynamic minus frontfilter_static', metric=metric,
            epoch=200, seeds=4, mean_delta_pp=st.mean(deltas), paired_SD_pp=st.stdev(deltas),
            positive_seeds=sum(x > 0 for x in deltas), zero_seeds=sum(x == 0 for x in deltas)))
    gradient_fields = ('E1_epoch_mean', 'E200_epoch_mean', 'epoch_mean_min', 'epoch_mean_max',
                       'full_run_epoch_mean', 'E151_200_epoch_mean', 'zero_epoch_means')
    return dict(
        scope=dict(selection_epoch=200, descriptive_windows=[[151, 175], [176, 200]],
            train_CE='Unweighted mean of 50 training-batch mean CEs: 49 batches of 128 and one of 28; train mode.',
            V_CE='Sample mean over all 27000 source-validation samples; eval mode.',
            CE_gap='V_CE minus train_CE within each seed; different data, averaging and modes; not an error-rate gap.',
            aggregation='Four model seeds, arithmetic mean and sample SD (ddof=1); shared data, not four independent datasets.',
            gradient='All 200 epoch means of 50 measured step norms over the added front filter; component norms are the existing final-backward measurements, not epoch means or raw-step extrema.',
            outputs='Each epoch last training batch of 28 packets; complete E1-E200 curves do not make these full-source measurements or channel-causal evidence.',
            bounds='Complex-modulus L1 bounds apply with coefficients held fixed. Operator singular values 0.75 to 1.25 do not prove invertibility of the packet-conditioned nonlinear map, channel recovery or TX/RX separation.',
            norms='relative_input_change is ||G_x(x)-x||/||x||; input_norm_ratio is ||G_x(x)||/||x|| over nonzero-input packets. Coefficient/kernel/basis L1 uses complex modulus, not componentwise absolute values.',
            variance='Coefficient variance is the mean of population variances across the last-batch packets over real/imaginary and basis components. Static coefficients are shared across packets.',
            comparison='Static adds 48 parameters; dynamic adds 320. The two variants are not parameter-matched.',
            coordinate_diagnostics='CFO/coherence measurements refer to original input x, not G_x(x), and do not measure filter compensation effectiveness.',
            target_results_read=False, epoch_reselection=False),
        source_learning_by_seed=learning, source_learning_summary=learning_summary,
        source_curve_summary=curve_summary, frontfilter_pairwise_by_seed=paired,
        frontfilter_pairwise_summary=paired_summary, frontfilter_gradient_epochs=gradient_epochs,
        frontfilter_gradient_by_seed=gradient_seeds,
        frontfilter_output_epochs=output_epochs,
        frontfilter_output_curve_summary=grouped(output_epochs,
            [dict(variant=v, epoch=e) for v in VARIANTS for e in range(1, 201)], OUTPUT_METRICS),
        frontfilter_gradient_summary=grouped(gradient_seeds, [dict(variant=v) for v in VARIANTS], gradient_fields),
        frontfilter_last_epoch=last_outputs,
        frontfilter_last_epoch_summary=grouped(last_outputs,
            [dict(variant=v, block=b) for v in VARIANTS for b in FILTER_BLOCKS], OUTPUT_METRICS))


def telemetry_report(analysis):
    """Render descriptive source evidence, without making a selection decision."""
    def formatted(row, key):
        avg, sd = row[key+'_mean'], row[key+'_SD']
        return 'N/A' if avg is None else f'{avg:.6f}' + (f'±{sd:.6f}' if sd is not None else '')

    text = '\n## 训练CE与源V CE\n\n固定E200；表中为四个模型seed的均值±样本SD。CE差值先按同seed计算V CE−训练CE，再汇总；不是误差率差。训练CE是50个batch均值的等权平均（49×128+28样本），源V CE是全部27000个验证样本的均值；二者的数据、平均方式和train/eval模式不同。\n\n'
    text += '|结构|E200训练CE|E200源V CE|E200 CE差值|后段训练CE变化|后段源V CE变化|后段CE差值变化|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in analysis['source_learning_summary']:
        fields = ('train_CE', 'V_CE', 'CE_gap', 'train_CE_late_delta', 'V_CE_late_delta', 'CE_gap_late_delta')
        text += '|'+row['variant']+'|'+'|'.join(f"{row[k+'_mean']:.6f}±{row[k+'_SD']:.6f}" for k in fields)+'|\n'
    text += '\n后段变化固定定义为E176–200均值减E151–175均值，按seed配对后汇总；完整E1–200四seed均值/SD见[source_curve_summary.csv](evidence/source_curve_summary.csv)。这些窗口只描述曲线，不重新选择epoch，也不设鲁棒性门槛。\n\n## 静态与动态前置滤波的固定E200配对差异\n\n以下均为frontfilter_dynamic−frontfilter_static，同seed配对，单位为百分点。static新增48个参数，dynamic新增320个参数，二者不是参数完全匹配的比较。\n\n|指标|均值±配对SD|正差seed|零差seed|\n|---|---:|---:|---:|\n'
    for row in analysis['frontfilter_pairwise_summary']:
        text += f"|{row['metric']}|{row['mean_delta_pp']:+.4f}±{row['paired_SD_pp']:.4f}|{row['positive_seeds']}/4|{row['zero_seeds']}/4|\n"
    text += '\n四seed共用相同数据，只描述此次源训练差异；小幅均值变化不能单独支持稳定提升或独立clean提升。\n\n## 已有前置滤波遥测\n\n下表只用E200最后一个训练batch的28个样本，按四seed汇总。相对输入变化为||G_x(x)−x||/||x||，输入范数比为||G_x(x)||/||x||；后者只统计非零输入。表中最小/最大值先在各seed末批计算，再汇总四seed，不能解释为全源极值。\n\n'
    text += '|结构|相对变化均值|相对变化最大值|范数比最小值|范数比最大值|\n|---|---:|---:|---:|---:|\n'
    for row in analysis['frontfilter_last_epoch_summary']:
        text += '|'+row['variant']+'|'+'|'.join(formatted(row,k) for k in OUTPUT_SCALE_METRICS)+'|\n'
    text += '\nL1均指复模L1。系数边界活跃比例是未约束系数L1大于1的包比例；basis边界活跃比例是原始basis L1大于1的基底比例。系数跨包方差使用population variance，再对实虚部与基底分量平均。static跨包共享系数，未测量项记N/A，各字段实测seed数保留在CSV。\n\n'
    for fields in (BOUND_METRICS[:5], BOUND_METRICS[5:], COMPONENT_GRADIENT_METRICS):
        text += '|结构|'+'|'.join(fields)+'|\n|---|'+'|'.join('---:' for _ in fields)+'|\n'
        for row in analysis['frontfilter_last_epoch_summary']:
            text += '|'+row['variant']+'|'+'|'.join(formatted(row,k) for k in fields)+'|\n'
        text += '\n'
    text += '固定系数时线性算子的奇异值界为0.75至1.25；这不证明动态非线性映射可逆、物理信道恢复或TX/RX解耦。CFO与coherence诊断读取原输入x，不是G_x(x)，不能用来评价滤波后的补偿效果。这28个样本不能代表全源分布，也不能支持信道变化的因果结论。全部200轮末批遥测均保留；完整曲线不改变每个测量仅覆盖末批的范围。\n\n|结构|全200轮梯度均值|E200梯度均值|E151–200梯度均值|\n|---|---:|---:|---:|\n'
    for row in analysis['frontfilter_gradient_summary']:
        text += '|'+row['variant']+'|'+'|'.join(f"{row[k+'_mean']:.6f}±{row[k+'_SD']:.6f}" for k in ('full_run_epoch_mean', 'E200_epoch_mean', 'E151_200_epoch_mean'))+'|\n'
    text += '\n总体梯度使用每个新模型全部200条epoch记录，每条是50个已有step梯度范数的均值，分别覆盖static的48个或dynamic的320个新增参数。collector另行审计每模型10000步；当前离线产物不含原始step数组，未计算step极值。basis/context/exit组件梯度来自每轮最后一次CE backward，不能当作epoch均值；dynamic的exit属于context，不能相加当作独立梯度。static没有context，记N/A。梯度非零仅表示CE更新经过这些参数。\n\n[CE与后段逐seed](evidence/source_learning_by_seed.csv) · [static/dynamic配对逐seed](evidence/frontfilter_pairwise_by_seed.csv) · [全部1600条梯度epoch均值](evidence/frontfilter_gradient_epochs.csv) · [梯度逐seed摘要](evidence/frontfilter_gradient_by_seed.csv) · [全部1600条末批遥测](evidence/frontfilter_output_epochs.csv) · [四seed遥测曲线](evidence/frontfilter_output_curve_summary.csv) · [E200逐seed](evidence/frontfilter_last_epoch.csv) · [完整分析与口径](evidence/source_learning_analysis.json)。\n'
    return text

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

def analyze(root):
    folder=root/'automation_reports/CV-SincNet'/RUN;e=folder/'evidence';data=read(e/'source_research_complete.json')
    selection=validate_completed(data)
    old=read(root/'automation_reports/CV-SincNet'/CONTROL_RUN/'evidence/source_research_complete.json')
    controls=[r for r in old['rows'] if r['resolved']['variant']==CONTROL]
    assert len(controls)==4
    anchor=read(root/'automation_reports/CV-SincNet'/ANCHOR_CONTROL_RUN/'evidence/source_research_complete.json')
    controls += [r for r in anchor['rows'] if r['resolved']['variant']==ANCHOR_CONTROL]
    assert len(controls)==8
    learning_analysis=summarize_source_telemetry(controls+data['rows'])
    full_filter_analysis=summarize_source_filter_cells(data['rows'])
    curves=[];final=[];resources=[];neural=[];cells=[]
    for row in controls+data['rows']:
        variant=row['resolved']['variant'];seed=row['resolved']['model_seed'];base=dict(variant=variant,model_seed=seed)
        assert [x['epoch'] for x in row['epochs']]==list(range(1,201))
        for epoch in row['epochs']:
            curves.append(dict(base,epoch=epoch['epoch'],train_CE=epoch['clean_ce'],V_CE=epoch['source_val_ce'],V=epoch['source_val_accuracy'],worst_RX=epoch['source_val_worst_rx'],score=.5*(epoch['source_val_accuracy']+epoch['source_val_worst_rx']),gradient_norm=epoch['gradient_norm'],frontfilter_gradient_norm=epoch.get('frontfilter_gradient_norm')))
            for r in epoch.get('frontfilter_diagnostics',{}).get('frontfilter',{}).get('records',[]):neural.append(dict(base,epoch=epoch['epoch'],**r))
        last=curves[-1];final.append(last)
        p=row['profile'];resources.append(dict(base,parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],resident_bytes=p['resident_state_bytes'],conv_linear_MACs=p['conv_linear_macs_per_sample'],inference_batch1_ms=p['inference_batch1_ms'],training_batch128_ms=p['training_batch128_ms'],peak_training_bytes=max(x['peak_cuda_allocated_bytes'] for x in row['epochs']),hardware=p['hardware'],torch_version=p['torch_version']))
        for r in row['source_diagnostics']['groups']:cells.append(dict(base,**{k:v for k,v in r.items() if not isinstance(v,(list,dict))}))
    for name,rows in [('source_curves',curves),('source_final',final),('source_resources',resources),('frontfilter_outputs',neural),('source_cells',cells)]:table(e/(name+'.csv'),rows)
    summary=[]
    for variant in CANDIDATES:
        rows=[r for r in final if r['variant']==variant];profiles=[r for r in resources if r['variant']==variant]
        differences=[100*(r['score']-next(c['score'] for c in final if c['variant']==CONTROL and c['model_seed']==r['model_seed'])) for r in rows]
        summary.append(dict(variant=variant,V=st.mean(r['V'] for r in rows),worst_RX=st.mean(r['worst_RX'] for r in rows),score=st.mean(r['score'] for r in rows),paired_score_pp=st.mean(differences),paired_score_SD=st.stdev(differences),positive_seeds=sum(d>0 for d in differences),parameters=profiles[0]['parameters'],inference_ms=st.mean(r['inference_batch1_ms'] for r in profiles),training_ms=st.mean(r['training_batch128_ms'] for r in profiles)))
    table(e/'source_summary.csv',summary)
    for name,rows in learning_analysis.items():
        if name!='scope':table(e/(name+'.csv'),rows)
    write(e/'source_learning_analysis.json',learning_analysis)
    for name,rows in full_filter_analysis.items():table(e/(name+'.csv'),rows)
    write(e/'source_frontfilter_analysis.json',dict(
        scope='Entire frozen E200 source V: 27000 packets/model, all 90 balanced TX/RX/day cells. '
              'Observed filtered-input changes and coefficient variation, not channel recovery or causal TX/RX separation. '
              'Scalar means are sample-weighted within seed; four-seed mean and sample SD. '
              'Coefficient trace variance sums eight component population variances; full V uses within-cell plus between-cell mean scatter. '
              'It is eight times the last-batch component-mean variance only for an identical sample set.',
        **full_filter_analysis))
    figure,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for variant,color in zip(CANDIDATES,['#334155','#7046b1','#0284c7','#db7a13']):
        for ax,key,label in zip(axes.flat,['V','worst_RX','train_CE','V_CE'],['Source V accuracy (%)','Worst seen-source RX (%)','Training CE','Source V CE']):
            values=[st.mean(r[key] for r in curves if r['variant']==variant and r['epoch']==epoch) for epoch in range(1,201)]
            ax.plot(range(1,201),[x*100 if key in ('V','worst_RX') else x for x in values],label=variant,color=color,linewidth=1.5)
            ax.set(title=label,xlabel='Epoch');ax.grid(alpha=.15)
            if key.endswith('CE'):ax.set_yscale('log')
    axes[0,0].legend(fontsize=8)
    for ext in ('png','pdf'):figure.savefig(e/('source_curves.'+ext),dpi=180)
    plt.close(figure)
    text='# CVS全主干前置滤波：完整源训练结果\n\n8个scratch模型完成200轮×50步。源选择由完整16记录独立重算，包含同主干shallow和当前源赢家Anchor；两种新结构均从scratch Shallow初始化，既有控制仅复用元数据。三个原身份路径均接收G_x(x)，没有原输入旁路、额外辅助路径或滤波后单位RMS。训练策略和单一CE保持不变。以下是源验证指标，不是测试结果。\n\n'
    text+='|结构|源V/%|最差源RX/%|固定源分数/%|相对控制/百分点±配对SD|正提升seed|参数|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{100*r['V']:.4f}|{100*r['worst_RX']:.4f}|{100*r['score']:.4f}|{r['paired_score_pp']:+.4f}±{r['paired_score_SD']:.4f}|{r['positive_seeds']}/4|{r['parameters']}|\n"
    text+=f"\n固定规则选中`{selection['selected_variant']}`。"+('后续只对该候选4模型执行预登记clean测试。' if selection['new_candidate_selected'] else '保留既有控制，未选候选不访问query，复用已有控制clean完成证据。')+'\n\n![完整3200轮含控制曲线](evidence/source_curves.png)\n\n'
    text+='|结构|batch1推理/ms|batch128训练/ms|\n|---|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{r['inference_ms']:.4f}|{r['training_ms']:.4f}|\n"
    text+='\n资源在实际RTX3090/FP32环境测量，并行运行可能影响时间。MAC计数包含实际aten卷积（包括功能式复卷积）及矩阵乘法，未计FFT、归一化、门控、池化和逐元素运算；不是总FLOPs。峰值内存与常驻状态保留逐seed值。新模块同样只由CE更新；零出口的首步内部零梯度是初始化性质，不能与训练后失活混同。\n\n'
    text+=telemetry_report(learning_analysis)
    text+='\n## 全部源V的前置滤波测量\n\n每模型固定E200、完整27000包与90个TX/RX/day单元；输入变化和L1按样本数加权，先按seed汇总再计算四seed均值±样本SD。下表完整源V测量与前文训练末批遥测分开。\n\n|结构|全V相对输入变化|全V系数L1均值|全Vkernel L1均值|全V系数trace variance|\n|---|---:|---:|---:|---:|\n'
    for r in full_filter_analysis['source_frontfilter_summary']:
        text+='|'+r['variant']+'|'+'|'.join(f"{r[k+'_mean']:.6f}±{r[k+'_SD']:.6f}" for k in (
            'relative_input_change_mean','coefficient_l1_mean','kernel_l1_mean','coefficient_trace_variance'))+'|\n'
    text+='\n系数trace variance为8个实数分量的population variance之和，以单元内方差加单元均值间散度重构全V值；与末批coefficient_packet_variance的分量平均口径不同。完整源V极值、有效包数和每seed值见CSV；这些变化及TX/RX/day关联不能证明物理信道恢复或因果解耦。\n\n[全部720个前置滤波单元](evidence/source_frontfilter_cells.csv) · [全V逐seed](evidence/source_frontfilter_by_seed.csv) · [四seed摘要](evidence/source_frontfilter_summary.csv) · [口径与完整输出](evidence/source_frontfilter_analysis.json)。\n'
    text+='\n源V使用已见源RX；四seed不是独立数据集。新增模块是否提高独立clean识别必须由冻结测试证明。没有追加损失、增强、重加权、采样策略、teacher或目标适应。D92的适应三阶段/K×新增类为N/A。\n\n[逐seed](evidence/source_final.csv) · [完整曲线](evidence/source_curves.csv) · [实际前置滤波输出](evidence/frontfilter_outputs.csv) · [全部TX/RX/day单元](evidence/source_cells.csv) · [资源](evidence/source_resources.csv) · [80000步审计](evidence/source_completion_validation.json)。\n'
    (folder/'report.md').write_text(text,encoding='utf-8')
    write(e/'source_analysis_validation.json',dict(status='VERIFIED',models=8,source_epochs=1600,control_epochs=1600,selection=selection,summary=summary,frontfilter_output_records=len(neural),target_results_read=False,learning_analysis_scope=learning_analysis['scope'],frontfilter_gradient_epoch_records=len(learning_analysis['frontfilter_gradient_epochs']),frontfilter_last_epoch_records=len(learning_analysis['frontfilter_last_epoch']),source_frontfilter_cells=len(full_filter_analysis['source_frontfilter_cells'])))
    print(json.dumps(dict(status='VERIFIED',summary=summary,selection=selection),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
