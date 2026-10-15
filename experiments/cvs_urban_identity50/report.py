"""Full-budget summaries and paired comparisons; never consumed by training."""
import gzip,json
from pathlib import Path
from . import design as d


def overall(root):
    done=d.read(root/'scoring/complete.json')
    if done['status']!='SCORED_COMPLETE' or done['views']!=list(d.VIEWS):raise ValueError('Incomplete seven-view row')
    records=d.read(root/'scoring/scores.json')['results']
    byview={r['view']:r for r in records if r['dimension']=='overall'}
    receivers=sorted({r['stratum'] for r in records if r['dimension']=='receiver'})
    rx={(r['view'],r['stratum']):r['accuracy'] for r in records if r['dimension']=='receiver'}
    urx={receiver:sum(rx[v,receiver] for v in d.URBAN)/3 for receiver in receivers}
    return dict(arm=records[0]['arm'],model_seed=2026092701,
        urban_mean=sum(byview[v]['accuracy'] for v in d.URBAN)/3,
        urban_macro_f1=sum(byview[v]['macro_f1'] for v in d.URBAN)/3,
        low_urban=byview['practical_low_urban']['accuracy'],mid_urban=byview['practical_mid_urban']['accuracy'],
        clean=byview['clean']['accuracy'],low_suburban=byview['practical_low_suburban']['accuracy'],
        suburban_mean=sum(byview[v]['accuracy'] for v in d.SCENES[:3])/3,
        worst_urban_RX=min(urx.values()),urban_by_RX=urx,
        views={v:dict(accuracy=r['accuracy'],macro_f1=r['macro_f1']) for v,r in byview.items()})


def collect(rows):
    if set(rows)!={r['row_id'] for r in d.rows()}:raise ValueError('Incomplete registered two-row matrix')
    results=[];costs=[];curves={};clipping=[]
    for row in rows:
        root=d.BASE/row;results.append(overall(root));source=root/'source'
        counts=d.read(source/'training_mechanism_complete.json')['counts'];done=d.read(source/'completion.json')
        counts_main={k:counts.get(k,0) for k in ('clean_L_packets','satellite_L_packets','teacher_L_clean_packets','native_U_packets')}
        costs.append(dict(row_id=row,parent_epochs=0,incremental_epochs=50,total_epochs=50,updates=11100,
            main_forward_packets=sum(v*(2 if k=='native_U_packets' else 1) for k,v in counts_main.items()),
            counts=counts_main,source_panel_forward_packets=50*720*4,training_wall_seconds=done['elapsed_seconds'],
            source_calibration='Shared 2250 forwards/0 updates; prior A E50 diagnostic only, no weight inheritance',
            prediction=d.read(root/'prediction/complete.json')))
        with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(line) for line in f]
        if len(steps)!=11100 or done['optimizer_steps']!=11100:raise ValueError('Incomplete update/curve budget')
        curves[row]=[]
        for epoch in range(1,51):
            batch=steps[(epoch-1)*222:epoch*222];values={}
            for key in ('urban/clean_CE','urban/satellite_CE','urban/auxiliary_loss','urban/joint_direction_cosine','urban/satellite_joint_variance'):
                v=[s['metrics'][key] for s in batch];values[key]=sum(v)/len(v)
            curves[row].append(dict(epoch=epoch,**values))
        active=[r['metrics'] for r in steps if r['metrics'].get('train/grad_clip_active',0)]
        clipping.append(dict(row_id=row,steps=len(steps),clip_active_steps=len(active),
            clipped_steps=sum(m['train/grad_before_clip']>m['train/grad_clip_limit'] for m in active),
            fraction=sum(m['train/grad_before_clip']>m['train/grad_clip_limit'] for m in active)/len(active) if active else 0.))
    reference=d.reference.BASE
    reference_results=[overall(reference/r) for r in ('B0-s2026092701','C3-s2026092701')]
    all_results=reference_results+results;index={r['arm']:r for r in all_results};comparisons=[]
    keys=('urban_mean','low_urban','mid_urban','clean','low_suburban','suburban_mean','urban_macro_f1','worst_urban_RX')
    for treatment,control in (('OUTPUT_KL','C3'),('JOINT_DIR','C3'),('JOINT_DIR','OUTPUT_KL'),('JOINT_DIR','B0')):
        comparisons.append(dict(treatment=treatment,control=control,model_seed=2026092701,
            **{k+'_difference':index[treatment][k]-index[control][k] for k in keys}))
    status='MET_SINGLE_SEED' if index['JOINT_DIR']['low_urban']>=.6 else 'NOT_MET_SINGLE_SEED'
    result=dict(rows=all_results,pairs=comparisons,seed_count=1,cross_seed_confirmation='NOT_RUN',
        performance_status=status,target_feedback_forbidden=True,
        comparison_limit='B/C match objective-only; A uses student margin target/quality weights and has fewer forwards, disclosed')
    d.write(d.BASE/'summary.json',result);d.write(d.BASE/'resources.json',dict(rows=costs,clipping=clipping))
    d.write(d.BASE/'training_curves.json',curves)
    lines=['# SIG_TIME joint160：输出与方向迁移对照','',
        '固定原网络、默认头、训练物理信道和七种正式 IQ。B/C 从头 E50（33+17），11100 更新；只替换额外约束。',
        '源校准系数、共同课程已在训练前固定。B/C 各额外使用 clean EMA L 前向；不增加 U 特征目标，不训练诊断分类器。',
        'A 复用已有 E50 C3。A 的间隔参考/质量权重与 B/C 有差异，且前向量较少；B 对 C 是严格匹配的 KL/方向目标对照。','',
        '| 方法 | urban (%) | low urban (%) | mid urban (%) | clean (%) | low suburban (%) | urban Macro-F1 (%) | 最差 urban RX (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in all_results:lines.append('| '+r['arm']+' | '+' | '.join(f'{100*r[k]:.4f}' for k in ('urban_mean','low_urban','mid_urban','clean','low_suburban','urban_macro_f1','worst_urban_RX'))+' |')
    lines+=['','仅一个模型 seed，不构成跨 seed 重现证据。预算结束不等于收敛，负差保留。','',
        '本轮是第一阶段实际身份方向实验。容差目标、同信道中间视图和分组 SIG_TIME 局部教学尚未运行。源特征错误修复、错误 TX 距离和分支塌缩检查以冻结后源诊断为准。']
    (d.BASE/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out=d.BASE/'figures';out.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        for row,values in curves.items():
            for ax,key in zip(axes,('urban/clean_CE','urban/satellite_CE','urban/auxiliary_loss')):
                ax.plot(range(1,51),[v[key] for v in values],label=row.split('-')[0]);ax.set_title(key.split('/')[1]);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        axes[0].legend();fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    except Exception as exc:d.write(d.BASE/'plot_failure.json',dict(error=repr(exc),scores_complete=True))
    return result
