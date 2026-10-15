"""Complete eight-row metrics/costs; reporting never feeds the source decision."""
import gzip,json
from . import design as d
from experiments.cvs_urban_identity50.report import overall


def collect(rows):
    if set(rows)!={r['row_id'] for r in d.rows()}:raise ValueError('Incomplete eight-position pilot')
    records=[];costs=[];curves={}
    for row in rows:
        root=d.BASE/row;source=root/'source';records.append(overall(root))
        done=d.read(source/'completion.json');mechanism=d.read(source/'training_mechanism_complete.json')
        with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf-8') as stream:steps=[json.loads(line) for line in stream]
        if len(steps)!=11100 or done['optimizer_steps']!=11100:raise ValueError('Incomplete real E50 update/curve budget')
        counts=mechanism['counts'];native_U=counts['native_U_packets']
        main=sum(counts[k] for k in ('clean_L_packets','satellite_L_packets','teacher_L_clean_packets'))+2*native_U
        if main!=6190200:raise ValueError('Matched original forward-sample budget differs')
        clipped=[v['metrics'] for v in steps if v['metrics'].get('train/grad_clip_active',0)]
        costs.append(dict(row_id=row,parent_epochs=0,incremental_epochs=50,total_epochs=50,updates=11100,
            main_forward_packets=main,counts=counts,source_developer_forward_packets=mechanism['source_dev_forward_packets'],
            source_calibration_forward_packets=3600 if row.startswith('A0-') else 0,
            native_V_forward_accounting='full native per-epoch + final health validation, separate from main and L_dev; preserved execution/args',
            training_wall_seconds=done['elapsed_seconds'],
            clipping_active_steps=len(clipped),clipped_steps=sum(v['train/grad_before_clip']>v['train/grad_clip_limit'] for v in clipped),
            prediction=d.read(root/'prediction/complete.json')))
        curve=[]
        for epoch in range(1,51):
            block=steps[(epoch-1)*222:epoch*222]
            keys=('urban/clean_CE','urban/satellite_CE','urban/auxiliary_weighted_loss',
                *('urban/'+key+'_direction_loss' for key in ('T','F','P','R','S','J')))
            curve.append(dict(epoch=epoch,**{key:sum(v['metrics'][key] for v in block)/222 for key in keys}))
        curves[row]=curve
    index={r['arm']:r for r in records};baseline=index['A0']
    metrics=('urban_mean','low_urban','mid_urban','clean','low_suburban','suburban_mean','urban_macro_f1','worst_urban_RX')
    pairs=[dict(treatment=arm,control='A0',model_seed=2026092701,
        **{key+'_difference':index[arm][key]-baseline[key] for key in metrics}) for arm in d.ARMS[1:]]
    result=dict(rows=records,pairs=pairs,seed_count=1,cross_seed_confirmation='NOT_RUN',
        performance_status='SINGLE_SEED_REPORTED_NO_BEST_POSITION_CLAIM',target_feedback_forbidden=True,
        source_family_decision_ref=str(d.BASE/'source_family_decision.json'),
        low_urban_60_by_arm={arm:index[arm]['low_urban']>=.6 for arm in d.ARMS},
        comparison_limit='All eight use L_train5580/L_dev720; old full-L E50 results are historical context, not strict matched controls')
    d.write(d.BASE/'summary.json',result);d.write(d.BASE/'resources.json',dict(rows=costs));d.write(d.BASE/'training_curves.json',curves)
    lines=['# SIG_TIME 同分支位置对齐 E50 单 seed 小试','',
        '全部八组从零 E50（33+17），11100 更新，原普通拼接 CE、旧 U 门控、默认头和正式 IQ。',
        'L_s 内部固定分为训练5580/开发720；V只作健康诊断。前5轮共用无对齐规则，epoch6起唯一位置方向损失。',
        '全局梯度一次源校准和ALL组合总强度均在A0固定epoch5完成；来源不是已看过target的成熟模型。','',
        '| 组别 | urban (%) | low urban (%) | mid urban (%) | clean (%) | suburban (%) | urban Macro-F1 (%) | 最差 urban RX (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in d.ARMS:
        value=index[arm];lines.append('| '+arm+' | '+' | '.join(f'{100*value[k]:.4f}' for k in
            ('urban_mean','low_urban','mid_urban','clean','suburban_mean','urban_macro_f1','worst_urban_RX'))+' |')
    lines+=['','一个seed不能证明位置最优或跨seed可重复。保留全部负差；扩完整三seed家族只读取源开发判据，目标指标不回流。',
        '稀疏组合仅在三seed合法源单项证据后按预登记规则补充。梯度下降、分支gain下降或替换诊断改善均不单独证明鲁棒性。']
    (d.BASE/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out=d.BASE/'figures';out.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for row,curve in curves.items():
            for ax,key in zip(axes,('urban/clean_CE','urban/satellite_CE','urban/auxiliary_weighted_loss')):
                ax.plot(range(1,51),[r[key] for r in curve],label=row.rsplit('-s',1)[0]);ax.set_title(key.split('/')[1]);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        axes[0].legend(fontsize=7,ncol=2);fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    except Exception as exc:d.write(d.BASE/'plot_failure.json',dict(error=repr(exc),scores_complete=True))
    return result
