"""Complete logs and all actual rows; target scores cannot enter training config."""
import gzip,json,math
from pathlib import Path
from . import design as d

def collect(rows):
    all_scores=[];resources=[];training={}
    for row in rows:
        root=d.BASE/row;done=d.read(root/'scoring/complete.json')
        if done['status']!='SCORED_COMPLETE' or done['views']!=list(d.VIEWS):raise ValueError('Missing row/scenario scoring')
        all_scores.extend(d.read(root/'scoring/scores.json')['results'])
        source=d.read(root/'source/completion.json');counts=d.read(root/'source/training_mechanism_complete.json')['counts']
        training[row]=[]
        with gzip.open(root/'source/step_metrics.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:training[row].append(json.loads(line))
        if len(training[row])!=6660:raise ValueError('Training curves do not cover complete update budget')
        U=counts.get('native_U_packets',0);L=counts['clean_L_packets'];sat=counts.get('satellite_L_packets',0);satU=counts.get('satellite_U_packets',0)
        resources.append(dict(row_id=row,parent_epochs=0,incremental_epochs=30,total_epochs=30,updates=6660,
            observed_forward_packets=dict(L_clean=L,L_satellite=sat,U_teacher_clean=U,U_student_strong=U,U_satellite=satU),
            main_forward_packet_total=L+sat+2*U+satU,source_panel_forwards=30*720*4,
            source_validation_native_count_per_pass=27000,native_validation_calls='see complete epoch/execution logs',
            training_wall_seconds=source['elapsed_seconds'],prediction=d.read(root/'prediction/complete.json'),
            source_reference_E50='diagnostic only, not inherited; reported separately'))
    lookup={(s['arm'],s['model_seed'],s['view'],s['dimension'],s['stratum']):s for s in all_scores}
    per_seed=[]
    for arm in d.ARMS:
        for seed in d.SEEDS:
            if (arm,seed,'clean','overall','ALL') not in lookup:continue
            overall={v:lookup[arm,seed,v,'overall','ALL'] for v in d.VIEWS}
            receivers=sorted({s['stratum'] for s in all_scores if s['arm']==arm and s['model_seed']==seed and s['dimension']=='receiver'})
            urx=[sum(lookup[arm,seed,v,'receiver',rx]['accuracy'] for v in d.URBAN)/3 for rx in receivers]
            per_seed.append(dict(arm=arm,model_seed=seed,urban_mean=sum(overall[v]['accuracy'] for v in d.URBAN)/3,
                urban_macro_f1=sum(overall[v]['macro_f1'] for v in d.URBAN)/3,
                worst_urban_RX=min(urx),low_urban=overall['practical_low_urban']['accuracy'],
                mid_urban=overall['practical_mid_urban']['accuracy'],clean=overall['clean']['accuracy'],
                suburban_mean=sum(overall[v]['accuracy'] for v in d.SCENES[:3])/3,
                views={v:dict(accuracy=s['accuracy'],macro_f1=s['macro_f1']) for v,s in overall.items()}))
    summary=[];pairs=[]
    for arm in d.ARMS:
        values=[p for p in per_seed if p['arm']==arm]
        if not values:continue
        item=dict(arm=arm,seed_count=len(values),complete_three_seeds=len(values)==3)
        for key in ('urban_mean','urban_macro_f1','worst_urban_RX','low_urban','mid_urban','clean','suburban_mean'):
            a=[v[key] for v in values];mean=sum(a)/len(a)
            item[key]=mean;item[key+'_sd']=math.sqrt(sum((v-mean)**2 for v in a)/(len(a)-1)) if len(a)>1 else None
        summary.append(item)
    contrasts=[('B0','C0'),('C1','B0'),('C2','C1'),('C3','C2'),('C4','C3')]+[(a,'B0') for a in ('C2','C3','C4')]
    indexed={(p['arm'],p['model_seed']):p for p in per_seed}
    for treatment,control in contrasts:
        seeds=[s for s in d.SEEDS if (treatment,s) in indexed and (control,s) in indexed]
        if not seeds:continue
        item=dict(treatment=treatment,control=control,seeds=seeds,complete_three_seeds=len(seeds)==3)
        for key in ('urban_mean','low_urban','mid_urban','clean','suburban_mean','urban_macro_f1','worst_urban_RX'):
            ds=[indexed[treatment,s][key]-indexed[control,s][key] for s in seeds]
            item[key+'_difference']=sum(ds)/len(ds);item[key+'_per_seed_differences']=ds;item[key+'_positive_seeds']=sum(v>0 for v in ds)
        pairs.append(item)
    primary=next((s for s in summary if s['arm']==d.PRIMARY),None)
    assessment=dict(primary=d.PRIMARY,low_urban_threshold=.6,
        performance_status='PARTIAL' if not primary or not primary['complete_three_seeds'] else 'MET' if primary['low_urban']>=.6 else 'NOT_MET',
        comparisons=pairs,source_selection_frozen_before_target_analysis=True,target_feedback_forbidden=True)
    d.write(d.BASE/'scores.json',dict(results=all_scores));d.write(d.BASE/'summary.json',dict(summary=summary,per_seed=per_seed,pairs=pairs,**assessment))
    d.write(d.BASE/'resources.json',dict(rows=resources,diagnostic=d.read(d.BASE/'source_diagnostic/results.json').get('seconds')))
    lines=['# 固定 SIG_TIME 的 practical urban 增强实验','',
        '各臂从头 E30，20 轮有标签＋10 轮旧门控/EMA。所有模型使用原默认头联合训练。',
        'B0 是普通拼接基线；C1 的等权双视图 CE 在相同 logits 下与拼接 CE 数学等价。差异来自执行路径和随机 Dropout，不能把单独归一化本身宣称为新算法。',
        '训练只使用 low suburban、mid urban、low urban 三环境。正式预测复用完全相同的 clean 与六 practical 固定 IQ。',
        'E30 是固定预算，不是收敛证明。E50 源 checkpoint 只供冻结诊断，不用于初始化；历史 benchmark 已暴露。','',
        '| 方法 | 实际 seed 数 | urban 均值 (%) | low urban (%) | mid urban (%) | clean (%) | suburban (%) | urban Macro-F1 (%) | 最差 urban RX (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in summary:lines.append('| '+v['arm']+' | '+str(v['seed_count'])+' | '+' | '.join(f"{100*v[k]:.3f}" for k in ('urban_mean','low_urban','mid_urban','clean','suburban_mean','urban_macro_f1','worst_urban_RX'))+' |')
    lines+=['','| 对比 | urban 差 (百分点) | low urban 差 | clean 差 | 正收益 seed（urban） |','|---|---:|---:|---:|---:|']
    for v in pairs:lines.append(f"| {v['treatment']} − {v['control']} | {100*v['urban_mean_difference']:.3f} | {100*v['low_urban_difference']:.3f} | {100*v['clean_difference']:.3f} | {v['urban_mean_positive_seeds']}/{len(v['seeds'])} |")
    lines+=['','逐 row/逐 seed 与全部场景见 summary.json、scores.json；所有负差保留。缺少 seed 的汇总不会标为三种子结果。',
        '源域分层、固定参考 clean 正确集合保持率、白输入代理/实际波形功率、受控与总效应见 source_diagnostic/；训练覆盖与门控/梯度变化见各 row/source/。',
        '低 urban 准确率 60% 是验收目标。urban +3、low +5、clean 下降不超过 0.5 个百分点是额外探索目标，均不得用于训练选择、晋级或重跑。']
    (d.BASE/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try:plots(training,per_seed,pairs)
    except Exception as exc:d.write(d.BASE/'plot_failure.json',dict(error=repr(exc),scores_complete=True,repair_without_retraining=True))
    return assessment

def plots(training,per_seed,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=d.BASE/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for row,records in training.items():
        if not row.endswith(str(d.SEEDS[0])):continue
        label=row.split('-')[0]
        for ax,key in zip(axes,('urban/clean_CE','urban/satellite_CE','urban/margin_loss')):
            curve=[]
            for ep in range(30):
                vals=[v['metrics'].get(key) for v in records[ep*222:(ep+1)*222]];vals=[v for v in vals if v is not None]
                curve.append(sum(vals)/len(vals) if vals else float('nan'))
            ax.plot(range(1,31),curve,label=label);ax.set_xlabel('Epoch');ax.set_title(key.split('/')[1]);ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    arms=[a for a in d.ARMS if any(p['arm']==a for p in per_seed)];fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,key in zip(axes,('low_urban','clean')):
        for i,arm in enumerate(arms):
            values=[100*p[key] for p in per_seed if p['arm']==arm];ax.scatter([i]*len(values),values);ax.plot([i-.2,i+.2],[sum(values)/len(values)]*2,color='black')
        ax.set_xticks(range(len(arms)),arms);ax.set_ylabel('Accuracy (%)');ax.set_title(key);ax.grid(alpha=.2)
    axes[0].axhline(60,color='red',linestyle='--',label='60% goal');fig.savefig(out/'per_seed_low_urban_clean.png',dpi=180);plt.close(fig)
    diagnostic=d.read(d.BASE/'source_diagnostic/results.json')['summary'];fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    base={ (r['scene'],r['state'],r['lock']):r for r in diagnostic if r['condition']=='full'}
    types=('no_noise/fixed','no_noise/total','flat_H/fixed','no_carrier/fixed','no_RX_IQ/fixed')
    for j,condition in enumerate(types):
        points=[r for r in diagnostic if r['condition']==condition and r['scene']=='practical_low_urban']
        ax.bar([i+(j-2)*.13 for i in range(len(points))],[100*(r['accuracy']-base[r['scene'],r['state'],r['lock']]['accuracy']) for r in points],width=.13,label=condition)
    ax.set_xticks(range(len(points)),[r['state']+'/'+r['lock'] for r in points],rotation=25,ha='right');ax.set_ylabel('Source accuracy difference (pp)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'source_controlled_attribution.png',dpi=180);plt.close(fig)
