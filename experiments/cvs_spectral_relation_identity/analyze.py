"""Offline descriptive source analysis after complete, independently audited E200.

No checkpoint, IQ, target score or optimizer is read. The stored source selection
is independently reproduced by collect.validate_completed before any report.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

from .collect import RELATION_SCALARS, validate_completed
from .dispatch import CANDIDATES, CONTROL, SEEDS, select_source_candidate
from .model import VARIANTS
from .prepare import RELEASE, RUN

METRICS = ('V', 'worst_RX', 'score')
CURVE_METRICS = (*METRICS, 'train_CE', 'V_CE', 'CE_gap')


def finite(value, name, lower=0., upper=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < lower
            or (upper is not None and value > upper)):
        raise ValueError('Invalid measured '+name)
    return float(value)


def stats(values):
    values = list(values)
    if len(values) != 4:
        raise ValueError('Exactly four model seeds required')
    return dict(mean=st.mean(values), SD=st.stdev(values))


def aggregate(rows, variants, fields):
    output = []
    for variant in variants:
        members = [r for r in rows if r['variant'] == variant]
        if len(members) != 4 or {r['model_seed'] for r in members} != SEEDS:
            raise ValueError('Four unique paired model seeds required')
        record = dict(variant=variant, seeds=4)
        for field in fields:
            record.update({field+'_'+k:v for k,v in stats(r[field] for r in members).items()})
        output.append(record)
    return output


def summarize_records(records):
    selection = select_source_candidate(records)
    final = [dict(variant=r['variant'], model_seed=r['seed'],
                  V=r['accuracy'], worst_RX=r['worst_rx'],
                  score=.5*(r['accuracy']+r['worst_rx']),
                  parameters=r['parameters'], conv_linear_MACs=r['macs']) for r in records]
    indexed = {(r['variant'], r['model_seed']):r for r in final}
    summary = aggregate(final, CANDIDATES, METRICS)
    pairs = [(v, CONTROL) for v in CANDIDATES if v != CONTROL]
    pairs.append(('relation_frequency_energy', 'relation_packet_energy'))
    paired, pair_summary = [], []
    for left, right in pairs:
        for metric in METRICS:
            deltas = [100*(indexed[left,s][metric]-indexed[right,s][metric]) for s in sorted(SEEDS)]
            paired.extend(dict(left=left, right=right, metric=metric, model_seed=s,
                               delta_pp=d) for s,d in zip(sorted(SEEDS), deltas))
            pair_summary.append(dict(left=left, right=right, metric=metric,
                mean_delta_pp=st.mean(deltas), paired_SD_pp=st.stdev(deltas),
                positive_seeds=sum(d>0 for d in deltas), zero_seeds=sum(d==0 for d in deltas)))
    return dict(selection=selection, source_final=final, source_summary=summary,
                source_paired_by_seed=paired, source_paired_summary=pair_summary)


def summarize_curves(rows):
    expected = {(v,s) for v in VARIANTS for s in SEEDS}
    keys = [(r['resolved']['variant'],r['resolved']['model_seed']) for r in rows]
    if len(keys)!=8 or set(keys)!=expected:
        raise ValueError('Exactly eight new source models required')
    curves, late, gradients, outputs = [], [], [], []
    for key,row in zip(keys,rows):
        base = dict(variant=key[0],model_seed=key[1])
        epochs = row['epochs']
        if [e['epoch'] for e in epochs]!=list(range(1,201)):
            raise ValueError('Complete ordered E1 through E200 required')
        points, norms = [], []
        for e in epochs:
            train = finite(e['clean_ce'], 'train CE')
            val = finite(e['source_val_ce'], 'V CE')
            acc = finite(e['source_val_accuracy'], 'V accuracy', upper=1.)
            worst = finite(e['source_val_worst_rx'], 'worst RX', upper=1.)
            norm = finite(e['spectral_relation_gradient_norm'], 'relation gradient')
            points.append(dict(base,epoch=e['epoch'],V=acc,worst_RX=worst,
                               score=.5*(acc+worst),train_CE=train,V_CE=val,CE_gap=val-train))
            norms.append(norm)
            curves.append(dict(points[-1],spectral_relation_gradient_norm=norm))
            for record in e['spectral_relation_diagnostics']['spectral_relation']['records']:
                outputs.append(dict(base,epoch=e['epoch'],**{k:v for k,v in record.items()
                                    if not isinstance(v,(list,dict))}))
        last = dict(base)
        for metric in CURVE_METRICS:
            last[metric+'_E200'] = points[-1][metric]
            last[metric+'_E151_175'] = st.mean(p[metric] for p in points[150:175])
            last[metric+'_E176_200'] = st.mean(p[metric] for p in points[175:200])
            last[metric+'_late_delta'] = last[metric+'_E176_200']-last[metric+'_E151_175']
        late.append(last)
        gradients.append(dict(base,epoch_means=200,E1=norms[0],E200=norms[-1],
            mean=st.mean(norms),minimum=min(norms),maximum=max(norms),
            E151_200_mean=st.mean(norms[150:]),zero_epoch_means=sum(n==0 for n in norms)))
    curve_summary = []
    for epoch in range(1,201):
        for record in aggregate([r for r in curves if r['epoch']==epoch], VARIANTS, CURVE_METRICS):
            curve_summary.append(dict(epoch=epoch,**record))
    late_fields = [k for k in late[0] if k not in ('variant','model_seed')]
    return dict(source_curves=curves,source_curve_summary=curve_summary,
                source_learning_by_seed=late,source_learning_summary=aggregate(late,VARIANTS,late_fields),
                source_gradient_by_seed=gradients,spectral_relation_epoch_outputs=outputs)


def summarize_cells(rows):
    cells, by_seed = [], []
    expected = {(t,r,d) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}
    for row in rows:
        base = dict(variant=row['resolved']['variant'],model_seed=row['resolved']['model_seed'])
        groups = row['source_diagnostics']['spectral_relation_groups']
        if (len(groups)!=90 or {(g['tx'],g['receiver'],g['day']) for g in groups}!=expected
                or any(g['count']!=300 for g in groups)):
            raise ValueError('Complete 90 source V cells of 300 packets required')
        cells.extend(dict(base,**g) for g in groups)
        for receiver in ('ALL',1,3,4,6,8):
            selected = [g for g in groups if receiver=='ALL' or g['receiver']==receiver]
            n = sum(g['count'] for g in selected)
            record = dict(base,receiver=receiver,count=n)
            for metric in RELATION_SCALARS:
                record[metric+'_mean'] = sum(finite(g[metric+'_mean'],metric)*g['count'] for g in selected)/n
                record[metric+'_min'] = min(finite(g[metric+'_min'],metric) for g in selected)
                record[metric+'_max'] = max(finite(g[metric+'_max'],metric) for g in selected)
            by_seed.append(record)
    summary = []
    for receiver in ('ALL',1,3,4,6,8):
        summary.extend(dict(receiver=receiver,**r) for r in aggregate(
            [r for r in by_seed if r['receiver']==receiver],VARIANTS,[k+'_mean' for k in RELATION_SCALARS]))
    return dict(source_relation_cells=cells,source_relation_by_seed=by_seed,source_relation_summary=summary)


def resource_rows(rows):
    output = []
    for row in rows:
        p = row['profile']
        output.append(dict(variant=row['resolved']['variant'],model_seed=row['resolved']['model_seed'],
            parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],
            resident_bytes=p['resident_state_bytes'],conv_linear_MACs=p['conv_linear_macs_per_sample'],
            inference_batch1_ms=p['inference_batch1_ms'],training_batch128_ms=p['training_batch128_ms'],
            peak_training_bytes=max(e['peak_cuda_allocated_bytes'] for e in row['epochs']),
            hardware=p['hardware'],torch_version=p['torch_version']))
    return output


def physics_rows(rows):
    output = []
    for row in rows:
        p = row['physical_diagnostics']
        base = dict(variant=row['resolved']['variant'],model_seed=row['resolved']['model_seed'])
        output.append(dict(base,test='ideal_gain',**{k:v for k,v in p['ideal_gain'].items()
                                                    if not isinstance(v,(list,dict))}))
        output.append(dict(base,test='whole_model_phase',**p['whole_model_phase']))
        output.extend(dict(base,test='finite_fir_'+r['name'],**r) for r in p['finite_fir_sensitivity'])
        output.extend(dict(base,test=k,**r) for k,r in p['tx_rx_sensitivity'].items())
    return output


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def report_text(analysis):
    selection = analysis['selection']
    text = '# CVS频点内时间关系：完整源训练分析\n\n'
    text += '8个新模型完成固定E200训练。16条源记录独立复算，包含同主干Shallow和既有Anchor控制；控制仅复用源元数据。以下均为源验证结果。\n\n'
    text += '|结构|源V/%±SD|最差源RX/%±SD|源分数/%±SD|\n|---|---:|---:|---:|\n'
    for row in analysis['source_summary']:
        text += '|'+row['variant']+'|'+'|'.join(f"{100*row[k+'_mean']:.4f}±{100*row[k+'_SD']:.4f}" for k in METRICS)+'|\n'
    text += '\n同seed配对差异，单位为百分点；SD为4个模型seed的样本标准差。\n\n|比较|指标|均值±配对SD|正差seed|\n|---|---|---:|---:|\n'
    for row in analysis['source_paired_summary']:
        text += f"|{row['left']}−{row['right']}|{row['metric']}|{row['mean_delta_pp']:+.4f}±{row['paired_SD_pp']:.4f}|{row['positive_seeds']}/4|\n"
    text += f"\n固定源规则选中`{selection['selected_variant']}`。"
    text += ('后续仅该新候选的4个模型进入预登记clean测试，与44条冻结基准同行比较。' if selection['new_candidate_selected']
             else '保留既有控制，复用其已冻结的clean证据；未选中新候选不访问query。')
    text += '\n\n![完整新候选源曲线](evidence/source_curves.png)\n\n'
    text += '曲线覆盖全部1600条新模型epoch记录。E151–175与E176–200均值之差仅描述后段变化，不重选epoch。训练CE为50个batch均值的等权平均（49×128+28），源V CE为27000样本均值，train/eval模式不同；CE差不等于泛化误差率差。控制完整曲线在本次collector中为N/A，未补造。\n\n'
    text += '全源V关系遥测覆盖每模型27000个物理ID、90个TX/RX/day单元，每单元300样本。逐样本标量经独立NumPy复算后汇总；ID为不透明标识，坐标核对范围是存储元数据及跨模型同ID一致性。输出范数比、Q范数与floor比例用于描述分支使用情况，不代表识别增益或物理信道恢复。四seed共用同一数据划分。\n\n'
    text += '理想逐频复增益抵消性质只适用于归一化关系统计且变化前后均未触发绝对/相对floor的频点。公开合成诊断完整保留纳入和排除频点数、全域与合格域误差、有限FIR、TX非线性及RX IQ敏感性。有限窗多径和整网不保证不变。公开诊断不训练、不决定选模。\n\n'
    text += '资源逐seed记录硬件、推理/训练时间、峰值显存和常驻字节数；并行任务可能影响时间。MAC只覆盖统计到的卷积与矩阵乘，不含FFT、归一化及逐元素运算，不等于总FLOPs。控制时间/显存未在本次复测，记N/A。\n\n'
    text += '[逐seed结果](evidence/source_final.csv) · [完整曲线](evidence/source_curves.csv) · [后段变化](evidence/source_learning_by_seed.csv) · [全V单元](evidence/source_relation_cells.csv) · [RX汇总](evidence/source_relation_summary.csv) · [梯度](evidence/source_gradient_by_seed.csv) · [公开物理诊断](evidence/source_public_physics.csv) · [资源](evidence/source_resources.csv)。\n\n'
    text += '单一TX CE、scratch、原数据和训练策略保持不变。是否提升独立clean识别需冻结后测试证明；源V已见RX不能单独证明跨未知信道/接收机泛化。Phase2适应三阶段与K×新增类指标为N/A。\n'
    return text


def analyze(root, input_path=None):
    input_path = input_path or root/'local_artifacts'/RELEASE/'source_research_complete.json'
    data = json.loads(input_path.read_text(encoding='utf-8'))
    selection = validate_completed(data)
    result = summarize_records(data['source_controls']+[r['source_record'] for r in data['rows']])
    if selection != result['selection']:
        raise ValueError('Analysis and independent completion selection differ')
    result.update(summarize_curves(data['rows']))
    result.update(summarize_cells(data['rows']))
    result['source_resources'] = resource_rows(data['rows'])
    result['source_public_physics'] = physics_rows(data['rows'])
    folder = root/'automation_reports/CV-SincNet'/RUN
    evidence = folder/'evidence'
    evidence.mkdir(parents=True,exist_ok=True)
    for key,value in result.items():
        if isinstance(value,list): write_csv(evidence/(key+'.csv'),value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for variant in VARIANTS:
        points = [r for r in result['source_curve_summary'] if r['variant']==variant]
        for ax,metric in zip(axes.flat,('V','worst_RX','train_CE','V_CE')):
            scale = 100 if metric in METRICS else 1
            ax.plot([r['epoch'] for r in points],[scale*r[metric+'_mean'] for r in points],label=variant)
            ax.set(xlabel='Epoch',ylabel=metric+(' (%)' if scale==100 else ''))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    for extension in ('png','pdf'): figure.savefig(evidence/('source_curves.'+extension),dpi=180)
    plt.close(figure)
    (folder/'source_analysis.md').write_text(report_text(result),encoding='utf-8')
    validation = dict(status='VERIFIED',source_run=RUN,source_models=8,source_epochs=1600,
        controls=8,control_curves='N/A: not collected',selection=selection,
        summary=result['source_summary'],paired_summary=result['source_paired_summary'],
        all_source_scalars_recounted=True,source_relation_cells=len(result['source_relation_cells']),
        target_results_read=False,epoch_reselection=False,input=str(input_path))
    write_json(evidence/'source_analysis_validation.json',validation)
    print(json.dumps(validation,ensure_ascii=False,allow_nan=False))
    return validation


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--input',type=Path)
    args = parser.parse_args()
    analyze(args.root,args.input)
