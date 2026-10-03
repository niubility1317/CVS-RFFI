"""Source-only full-curve report; preserves fixed E200 selection."""
import argparse,csv,json,math,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cvs_neural_readout_identity.prepare import RUN
from experiments.cvs_neural_readout_identity.collect import validate_completed
from experiments.cvs_neural_readout_identity.dispatch import CONTROL_RUN,CONTROL,CANDIDATES,ANCHOR_CONTROL,ANCHOR_CONTROL_RUN,SEEDS
from experiments.cvs_neural_readout_identity.model import VARIANTS

READOUT_BLOCKS = ('time.readout', 'behavior.readout')
LEARNING_METRICS = ('train_CE', 'V_CE', 'CE_gap', 'V', 'worst_RX', 'score')
OUTPUT_METRICS = ('relative_output_change_mean', 'attention_effective_tokens_mean',
                  'attention_entropy_mean', 'projection_norm')


def finite(value, label, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f'{label} outside its measured range')
    return float(value)


def mean_sd(values):
    """Sample SD over the four paired model seeds, never over pooled epochs."""
    values = list(values)
    return {'mean': st.mean(values), 'SD': st.stdev(values)}


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
                norm = finite(epoch.get('readout_gradient_norm'), 'readout epoch-mean gradient', 0)
                gradient_epochs.append(dict(base, epoch=epoch['epoch'], readout_gradient_norm=norm))
                records = epoch.get('readout_diagnostics', {}).get('learned_readout', {}).get('records', [])
                if len(records) != 2 or {r.get('block') for r in records} != set(READOUT_BLOCKS):
                    raise ValueError('Both measured readout branches are required at every epoch')
                for record in records:
                    if record.get('packets') != 28:
                        raise ValueError('Readout diagnostic scope must be the last training batch of 28 packets')
                    values = {key: finite(record.get(key), key, -1e-6 if key == 'attention_entropy_mean' else 0) for key in OUTPUT_METRICS}
                    # Preserve the collector's FP32 diagnostic tolerances.
                    if not 1-1e-5 <= values['attention_effective_tokens_mean'] <= 64+1e-3:
                        raise ValueError('Effective tokens outside 64-position attention range')
                    if values['attention_entropy_mean'] > math.log(64)+1e-5:
                        raise ValueError('Attention entropy outside 64-position attention range')
                    if epoch['epoch'] == 200:
                        last_outputs.append(dict(base, epoch=200, block=record['block'], packets=28, **values))
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
                norms = [float(e['readout_gradient_norm']) for e in epochs]
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
        attention = series['readout_attention', seed][-1]
        complex_attention = series['readout_complex_attention', seed][-1]
        paired.append(dict(model_seed=seed, epoch=200,
            **{f'{metric}_delta_pp': 100*(complex_attention[metric]-attention[metric]) for metric in ('score', 'V', 'worst_RX')}))
    paired_summary = []
    for metric in ('score', 'V', 'worst_RX'):
        deltas = [r[f'{metric}_delta_pp'] for r in paired]
        paired_summary.append(dict(comparison='readout_complex_attention minus readout_attention', metric=metric,
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
            gradient='All 200 epoch means of 50 measured step norms; time and behavior readout parameters combined. No raw-step extrema or per-branch gradients available in this artifact.',
            outputs='E200 last training batch of 28 packets; not full-source or channel-causal evidence.',
            attention='Entropy/effective tokens describe within-head temporal distributions across 64 positions. Head similarity/diversity and channel disentanglement were not measured.',
            target_results_read=False, epoch_reselection=False),
        source_learning_by_seed=learning, source_learning_summary=learning_summary,
        source_curve_summary=curve_summary, readout_pairwise_by_seed=paired,
        readout_pairwise_summary=paired_summary, readout_gradient_epochs=gradient_epochs,
        readout_gradient_by_seed=gradient_seeds,
        readout_gradient_summary=grouped(gradient_seeds, [dict(variant=v) for v in VARIANTS], gradient_fields),
        readout_last_epoch=last_outputs,
        readout_last_epoch_summary=grouped(last_outputs,
            [dict(variant=v, block=b) for v in VARIANTS for b in READOUT_BLOCKS], OUTPUT_METRICS))


def telemetry_report(analysis):
    """Render descriptive source evidence, without making a selection decision."""
    text = '\n## 训练CE与源V CE\n\n固定E200；表中为四个模型seed的均值±样本SD。CE差值先按同seed计算V CE−训练CE，再汇总；不是误差率差。训练CE是50个batch均值的等权平均（49×128+28样本），源V CE是全部27000个验证样本的均值；二者的数据、平均方式和train/eval模式不同。\n\n'
    text += '|结构|E200训练CE|E200源V CE|E200 CE差值|后段训练CE变化|后段源V CE变化|后段CE差值变化|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in analysis['source_learning_summary']:
        fields = ('train_CE', 'V_CE', 'CE_gap', 'train_CE_late_delta', 'V_CE_late_delta', 'CE_gap_late_delta')
        text += '|'+row['variant']+'|'+'|'.join(f"{row[k+'_mean']:.6f}±{row[k+'_SD']:.6f}" for k in fields)+'|\n'
    text += '\n后段变化固定定义为E176–200均值减E151–175均值，按seed配对后汇总；完整E1–200四seed均值/SD见[source_curve_summary.csv](evidence/source_curve_summary.csv)。这些窗口只描述曲线，不重新选择epoch，也不设鲁棒性门槛。\n\n## 两种读出的固定E200配对差异\n\n以下均为complex_attention−attention，同seed配对，单位为百分点。\n\n|指标|均值±配对SD|正差seed|零差seed|\n|---|---:|---:|---:|\n'
    for row in analysis['readout_pairwise_summary']:
        text += f"|{row['metric']}|{row['mean_delta_pp']:+.4f}±{row['paired_SD_pp']:.4f}|{row['positive_seeds']}/4|{row['zero_seeds']}/4|\n"
    text += '\n四seed共用相同数据，只描述此次源训练差异；小幅均值变化不能单独支持稳定提升或独立clean提升。\n\n## 已有读出遥测\n\n下表只用E200最后一个训练batch的28个样本，分支指标按四seed汇总。输出变化是相对skip输出的逐样本L2范数比均值；投影范数是project权重的Frobenius范数。\n\n|结构/分支|相对输出变化|有效tokens|attention entropy|投影范数|\n|---|---:|---:|---:|---:|\n'
    for row in analysis['readout_last_epoch_summary']:
        text += '|'+row['variant']+'/'+row['block']+'|'+'|'.join(f"{row[k+'_mean']:.6f}±{row[k+'_SD']:.6f}" for k in OUTPUT_METRICS)+'|\n'
    text += '\n这28个样本不能代表全源分布，也不能支持信道变化的因果结论。有效tokens与entropy反映各head在64个时间位置上的分布，不证明head之间的多样性；当前未测量head相似度或信道解耦。\n\n|结构|全200轮梯度均值|E200梯度均值|E151–200梯度均值|\n|---|---:|---:|---:|\n'
    for row in analysis['readout_gradient_summary']:
        text += '|'+row['variant']+'|'+'|'.join(f"{row[k+'_mean']:.6f}±{row[k+'_SD']:.6f}" for k in ('full_run_epoch_mean', 'E200_epoch_mean', 'E151_200_epoch_mean'))+'|\n'
    text += '\n梯度使用每个新模型全部200条epoch记录，每条是50个已有step梯度范数的均值，合并time与behavior两分支的新增参数。原collector另行审计每模型10000步；当前离线产物不含原始step数组，未计算step极值或单分支梯度。epoch均值的范围、零值计数和已有步数审计信息保留在逐seed表中；梯度非零仅表示CE更新经过这些参数。\n\n[CE与后段逐seed](evidence/source_learning_by_seed.csv) · [读出配对逐seed](evidence/readout_pairwise_by_seed.csv) · [全部1600条读出梯度epoch均值](evidence/readout_gradient_epochs.csv) · [梯度逐seed摘要](evidence/readout_gradient_by_seed.csv) · [E200分支逐seed](evidence/readout_last_epoch.csv) · [完整分析与口径](evidence/source_learning_analysis.json)。\n'
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
    curves=[];final=[];resources=[];neural=[];cells=[]
    for row in controls+data['rows']:
        variant=row['resolved']['variant'];seed=row['resolved']['model_seed'];base=dict(variant=variant,model_seed=seed)
        assert [x['epoch'] for x in row['epochs']]==list(range(1,201))
        for epoch in row['epochs']:
            curves.append(dict(base,epoch=epoch['epoch'],train_CE=epoch['clean_ce'],V_CE=epoch['source_val_ce'],V=epoch['source_val_accuracy'],worst_RX=epoch['source_val_worst_rx'],score=.5*(epoch['source_val_accuracy']+epoch['source_val_worst_rx']),gradient_norm=epoch['gradient_norm'],readout_gradient_norm=epoch.get('readout_gradient_norm')))
            for r in epoch.get('readout_diagnostics',{}).get('learned_readout',{}).get('records',[]):neural.append(dict(base,epoch=epoch['epoch'],**r))
        last=curves[-1];final.append(last)
        p=row['profile'];resources.append(dict(base,parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],resident_bytes=p['resident_state_bytes'],conv_linear_MACs=p['conv_linear_macs_per_sample'],inference_batch1_ms=p['inference_batch1_ms'],training_batch128_ms=p['training_batch128_ms'],peak_training_bytes=max(x['peak_cuda_allocated_bytes'] for x in row['epochs']),hardware=p['hardware'],torch_version=p['torch_version']))
        for r in row['source_diagnostics']['groups']:cells.append(dict(base,**{k:v for k,v in r.items() if not isinstance(v,(list,dict))}))
    for name,rows in [('source_curves',curves),('source_final',final),('source_resources',resources),('readout_outputs',neural),('source_cells',cells)]:table(e/(name+'.csv'),rows)
    summary=[]
    for variant in CANDIDATES:
        rows=[r for r in final if r['variant']==variant];profiles=[r for r in resources if r['variant']==variant]
        differences=[100*(r['score']-next(c['score'] for c in final if c['variant']==CONTROL and c['model_seed']==r['model_seed'])) for r in rows]
        summary.append(dict(variant=variant,V=st.mean(r['V'] for r in rows),worst_RX=st.mean(r['worst_RX'] for r in rows),score=st.mean(r['score'] for r in rows),paired_score_pp=st.mean(differences),paired_score_SD=st.stdev(differences),positive_seeds=sum(d>0 for d in differences),parameters=profiles[0]['parameters'],inference_ms=st.mean(r['inference_batch1_ms'] for r in profiles),training_ms=st.mean(r['training_batch128_ms'] for r in profiles)))
    table(e/'source_summary.csv',summary)
    for name,rows in learning_analysis.items():
        if name!='scope':table(e/(name+'.csv'),rows)
    write(e/'source_learning_analysis.json',learning_analysis)
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
    text='# CVS可学习注意力读出：完整源训练结果\n\n8个scratch模型完成200轮×50步。源选择由完整16记录独立重算，包含同主干shallow和当前源赢家Anchor，训练策略和单一CE保持不变。以下是源验证指标，不是测试结果。\n\n'
    text+='|结构|源V/%|最差源RX/%|固定源分数/%|相对控制/百分点±配对SD|正提升seed|参数|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{100*r['V']:.4f}|{100*r['worst_RX']:.4f}|{100*r['score']:.4f}|{r['paired_score_pp']:+.4f}±{r['paired_score_SD']:.4f}|{r['positive_seeds']}/4|{r['parameters']}|\n"
    text+=f"\n固定规则选中`{selection['selected_variant']}`。"+('后续只对该候选4模型执行预登记clean测试。' if selection['new_candidate_selected'] else '保留既有控制，未选候选不访问query，复用已有控制clean完成证据。')+'\n\n![完整3200轮含控制曲线](evidence/source_curves.png)\n\n'
    text+='|结构|batch1推理/ms|batch128训练/ms|\n|---|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{r['inference_ms']:.4f}|{r['training_ms']:.4f}|\n"
    text+='\n资源在实际RTX3090/FP32环境测量，并行运行可能影响时间。MAC计数包含实际aten卷积（包括功能式复卷积）及矩阵乘法，未计FFT、归一化、门控、池化和逐元素运算；不是总FLOPs。峰值内存与常驻状态保留逐seed值。新模块同样只由CE更新；零出口的首步内部零梯度是初始化性质，不能与训练后失活混同。\n\n'
    text+=telemetry_report(learning_analysis)
    text+='\n源V使用已见源RX；四seed不是独立数据集。新增容量是否提高独立clean识别必须由冻结测试证明。没有追加损失、增强、重加权、采样策略、teacher或目标适应。D92的适应三阶段/K×新增类为N/A。\n\n[逐seed](evidence/source_final.csv) · [完整曲线](evidence/source_curves.csv) · [实际新增分支输出](evidence/readout_outputs.csv) · [全部TX/RX/day单元](evidence/source_cells.csv) · [资源](evidence/source_resources.csv) · [80000步审计](evidence/source_completion_validation.json) · [原设计](../../../docs/CVS_NEURAL_READOUT_20261003.md)。\n'
    (folder/'report.md').write_text(text,encoding='utf-8')
    write(e/'source_analysis_validation.json',dict(status='VERIFIED',models=8,source_epochs=1600,control_epochs=1600,selection=selection,summary=summary,readout_output_records=len(neural),target_results_read=False,learning_analysis_scope=learning_analysis['scope'],readout_gradient_epoch_records=len(learning_analysis['readout_gradient_epochs']),readout_last_epoch_records=len(learning_analysis['readout_last_epoch'])))
    print(json.dumps(dict(status='VERIFIED',summary=summary,selection=selection),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
