"""Offline arithmetic and figures from the completed fixed five-row matrix.

Consumes saved predictions' independent scores, source diagnostics and logs.
Does not import training code, load model weights, access IQ, or open truth.
"""
import argparse
from collections import defaultdict
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path
import re
from itertools import zip_longest

ARMS=('R_LEGACY','B00','B10','B01','B11')
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
VIEWS=('clean',*SCENES)
COLORS=dict(zip(ARMS,('#777777','#0072B2','#009E73','#D55E00','#CC79A7')))
LABELS=dict(zip(VIEWS,('Clean','High S','Mid S','Low S','High U','Mid U','Low U')))


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def table(path,records):
    fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)


def mean(values):return sum(values)/len(values) if values else None


def merge(parts):
    import numpy as np
    n=sum(x['count'] for x in parts);correct=sum(x['correct'] for x in parts)
    reference=sum(x['reference_clean_correct'] for x in parts)
    cm=sum((np.asarray(x['confusion'],dtype=np.int64) for x in parts),np.zeros((6,6),np.int64))
    denominator=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(6),where=denominator>0).mean()
    if int(cm.sum())!=n or int(np.trace(cm))!=correct:raise ValueError('Source confusion recount differs')
    return dict(count=n,correct=correct,accuracy=correct/n if n else None,macro_f1=float(f1),
        margin=sum(x['count']*x['margin'] for x in parts)/n if n else None,
        reference_clean_correct=reference,
        clean_correct_retention=(reference-sum(x['new_errors'] for x in parts))/reference if reference else None)


def source_groups(panel):
    output=[]
    for view in SCENES:
        cells=panel['views'][view]['state_quality']
        if sum(x['count'] for x in cells)!=720:raise ValueError('Complete fixed source panel required')
        for dimension,key in (('state','state'),('lock','lock'),('added_noise_SNR_bin','added_noise_SNR_bin')):
            grouped=defaultdict(list)
            for x in cells:grouped[x[key]].append(x)
            for value,parts in sorted(grouped.items()):output.append(dict(view=view,dimension=dimension,stratum=value,**merge(parts)))
    return output


def audit_channel_pairs(root):
    arms=('B01','B11');counts={a:Counter() for a in arms};sessions={a:defaultdict(set) for a in arms}
    stats={a:defaultdict(list) for a in arms};mismatches=0;steps=0
    fields=('quality_snr_db','signal_to_added_noise_db','output_frequency_hz','residual_kernel_length','residual_kernel_energy')
    paths=[root/(a+'-s2026092701')/'source/channel_training.jsonl.gz' for a in arms]
    with gzip.open(paths[0],'rt',encoding='utf-8') as first,gzip.open(paths[1],'rt',encoding='utf-8') as second:
        for line_a,line_b in zip_longest(first,second):
            steps+=1
            if line_a is None or line_b is None:raise ValueError('Incomplete actual channel trace')
            pair=[json.loads(line) for line in (line_a,line_b)]
            if any(x['step']!=steps or x['epoch']!=(steps-1)//222+1 or len(x['records'])!=128 for x in pair):raise ValueError('Actual channel step budget differs')
            for left,right in zip(pair[0]['records'],pair[1]['records']):
                if {k:v for k,v in left.items() if k!='correct'}!={k:v for k,v in right.items() if k!='correct'}:mismatches+=1
                for arm,r in zip(arms,(left,right)):
                    counts[arm][(r['TX'],r['RX'],r['scenario'],r['state_end'],r['receiver_lock_status'])]+=1
                    sessions[arm][r['session_id']].add(r['TX'])
                    for key in fields:
                        value=r.get(key)
                        if isinstance(value,(int,float)) and math.isfinite(value):stats[arm][key].append(value)
                    stats[arm]['agc_limited'].append(int(bool(r.get('agc_limited'))))
    if steps!=4440:raise ValueError('Incomplete E20 actual channel trace')
    result={}
    for arm in arms:
        mechanism=read(root/(arm+'-s2026092701')/'source/training_mechanism_complete.json')
        expected=Counter({(r['TX'],r['RX'],r['scene'],r['state'],r['lock']):r['count'] for r in mechanism['coverage']})
        if counts[arm]!=expected or sum(counts[arm].values())!=568320:raise ValueError('Observer/actual trace recount differs')
        result[arm]=dict(packets=sum(counts[arm].values()),TX_RX_scene_cells=len({k[:3] for k in counts[arm]}),sessions=len(sessions[arm]),
            TX_count_per_session={k:len(v) for k,v in sessions[arm].items()},
            numeric_ranges={k:dict(count=len(v),min=min(v),max=max(v),mean=mean(v)) for k,v in stats[arm].items()},
            AGC_limited=sum(stats[arm]['agc_limited']))
    for arm in ('R_LEGACY','B00','B10'):
        with gzip.open(root/(arm+'-s2026092701')/'source/channel_training.jsonl.gz','rt',encoding='utf-8') as f:
            if next(f,None) is not None:raise ValueError('Inactive satellite arm emitted channel packets')
    return dict(status='ACTUAL_TRACE_RECOUNT_VERIFIED',steps_per_satellite_row=steps,rows=result,
        compared_packet_pairs=steps*128,metadata_mismatched_pairs=mismatches,excluded_comparison_field='correct',
        interpretation='Training observations, not final-state test accuracy; no model forward or fitting')


def parse_source(root):
    source=root/'source';epochs=[json.loads(line) for line in (source/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    with (source/'metrics_epoch.csv').open(encoding='utf-8-sig',newline='') as f:csv_rows=list(csv.DictReader(f))
    with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(line) for line in f]
    if [x['epoch'] for x in epochs]!=list(range(1,21)) or len(csv_rows)!=20 or [x['step'] for x in steps]!=list(range(1,4441)):
        raise ValueError('Completed E20 full logs required')
    logs=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf-8',errors='replace')
    errors={p:len(re.findall(p,logs,re.I)) for p in ('Traceback','CUDA out of memory','FloatingPointError','Killed')}
    critical=('train/loss','train/grad_total','train/grad_backbone','train/grad_before_clip','mixedIQ/mixed_L_loss','mixedIQ/native_CE')
    nonfinite={k:sum(not isinstance(x['metrics'].get(k),(int,float)) or not math.isfinite(x['metrics'][k]) for x in steps) for k in critical}
    by_epoch=defaultdict(list)
    for x in steps:by_epoch[x['epoch']].append(x['metrics'])
    curves=[]
    for epoch,records in sorted(by_epoch.items()):
        keys={k for r in records for k in r if k.startswith('mixedIQ/') or k in critical or k in ('train/grad_clip_active','train/pseudo_total','train/pseudo_selected','train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')}
        values={k:mean([r[k] for r in records if isinstance(r.get(k),(int,float)) and math.isfinite(r[k])]) for k in sorted(keys)}
        curves.append(dict(epoch=epoch,steps=len(records),**values))
    max_loss_difference=max(abs(x['metrics']['train/loss']-x['metrics']['mixedIQ/mixed_L_loss']) for x in steps if x['epoch']<=13)
    proof=dict(epochs=len(epochs),steps=len(steps),CSV_rows=len(csv_rows),errors=errors,critical_nonfinite=nonfinite,
        label_phase_max_total_minus_mixed_L=max_loss_difference,
        U_truth_available_max=max(x['metrics']['train/pseudo_truth_available'] for x in steps),
        skipped_nonfinite_loss=sum(x['metrics']['train/skipped_nonfinite_loss'] for x in steps),
        skipped_nonfinite_grad=sum(x['metrics']['train/skipped_nonfinite_grad'] for x in steps))
    if any(errors.values()) or any(nonfinite.values()) or proof['U_truth_available_max'] or max_loss_difference>1e-5:
        raise ValueError(('Source log anomaly',proof))
    return dict(epochs=epochs,curves=curves,proof=proof)


def validate(root):
    done=read(root/'complete.json');result=read(root/'results.json');qualification=read(root/'source_qualification.json')
    if done.get('status')!='ANALYZED' or done.get('actual_model_count')!=5 or result['seed_count']!=1 or qualification['target_scores_read'] or result['target_feedback']:
        raise ValueError('Complete registered matrix and frozen source qualification required')
    matrix=[];panels={};logs={};costs=[];coverage=[]
    launches={x['row_id']:x for x in read(root/'launch.json')['rows']}
    for arm in ARMS:
        rid=arm+'-s2026092701';row=root/rid;source=row/'source'
        scored=read(row/'scoring/complete.json');prediction=read(row/'prediction/complete.json');mechanism=read(source/'training_mechanism_complete.json');completion=read(source/'completion.json')
        if scored['status']!='SCORED_COMPLETE' or tuple(scored['views'])!=VIEWS or prediction['truth_read'] or prediction['query_fit'] or not prediction['all_parameters_frozen']:
            raise ValueError('Complete frozen seven-view independent scoring required')
        if completion['epoch']!=20 or completion['optimizer_steps']!=4440 or completion['head_fit_updates'] or mechanism['head_fit_updates'] or mechanism['U_truth_read'] or not mechanism['default_head_joint_training']:
            raise ValueError('Fixed E20/default-head budget differs')
        records=read(row/'scoring/scores.json')['results'];overall={x['view']:x for x in records if x['dimension']=='overall'}
        for view in VIEWS:
            accuracy=overall[view]['accuracy'];f1=overall[view]['macro_f1']
            if abs(accuracy-result['rows'][arm]['metrics'][view])>1e-12:raise ValueError('Saved metric aggregate differs')
            receivers=[x['accuracy'] for x in records if x['view']==view and x['dimension']=='receiver']
            matrix.append(dict(seed=2026092701,arm=arm,view=view,accuracy_percent=100*accuracy,macro_F1_percent=100*f1,worst_RX_percent=100*min(receivers),clean_minus_view_pp=100*(overall['clean']['accuracy']-accuracy)))
        panels[arm]=read(source/'source_panel_results.json');logs[arm]=parse_source(row)
        for view in VIEWS:
            actual=panels[arm]['views'][view];recount=merge([actual])
            if actual['count']!=720 or abs(recount['accuracy']-actual['accuracy'])>1e-12 or abs(recount['macro_f1']-actual['macro_f1'])>3e-7:raise ValueError('Source default-head recount differs')
        costs.append(dict(arm=arm,epochs=20,updates=4440,parent_weights_loaded=False,parent_training_epochs=0,
            source_seconds=completion['elapsed_seconds'],formal_prediction_seconds=prediction['wall_seconds'],
            row_wall_seconds=(row/'complete.json').stat().st_mtime-launches[rid]['launched_at'],
            L_forward_packets=mechanism['L_forward_packets'],student_U_forward_packets=mechanism['native_student_U_forward_packets'],
            EMA_forward_packets=mechanism['EMA_eval_forward_packets'],student_source_eval_forward_packets=mechanism['student_eval_forward_packets'],
            frozen_source_panel_forward_packets=5040,formal_forward_packets=prediction['actual_forward_packets']))
        for x in mechanism.get('coverage',[]):coverage.append(dict(arm=arm,**x))
    for pair,values in result['paired_difference_pp'].items():
        a,b=pair.split('-')
        for metric,delta in values.items():
            if abs(delta-100*(result['rows'][a]['metrics'][metric]-result['rows'][b]['metrics'][metric]))>1e-10:raise ValueError('Paired difference arithmetic differs')
    for metric,delta in result['interaction_pp'].items():
        expected=100*sum(sign*result['rows'][arm]['metrics'][metric] for arm,sign in (('B11',1),('B10',-1),('B01',-1),('B00',1)))
        if abs(delta-expected)>1e-10:raise ValueError('Factorial interaction arithmetic differs')
    return result,qualification,matrix,panels,logs,costs,coverage


def report(output,result,qualification,panels,costs,summary):
    lines=['# SIG_TIME＋RDrop：固定 E20 混合 IQ 五组结果','',
        '执行代码：`1fe3f00bdddfdb0af494f74c6570db65490ddf72`。一个模型种子 2026092701，五个独立 scratch E20；每个模型 4,440 次更新，full L6300/U56700/V27000。原 SIG_TIME/joint160/NoExplicitCFO、默认 cosine 头联合训练，旧 U 门控与 ownEMA 不变。',
        '', '所有 row 在原 clean＋六种 practical 上固定预测后独立 truth-last 评分。以下是已暴露固定基准上的单种子探索，不是新的独立盲测或跨种子复现。',
        '', '## 七视图默认头准确率','',
        '|row|'+'|'.join(LABELS[v]+' %' for v in VIEWS)+'|','|---|'+'---:|'*7]
    for arm in ARMS:lines.append('|'+arm+'|'+'|'.join(f"{100*result['rows'][arm]['metrics'][v]:.4f}" for v in VIEWS)+'|')
    lines+=['','|row|urban 均值 %|urban 宏 F1 %|最差 urban RX %|','|---|---:|---:|---:|']
    for arm in ARMS:lines.append('|'+arm+'|'+'|'.join(f"{100*result['rows'][arm]['metrics'][k]:.4f}" for k in ('urban_mean','urban_macro_f1','worst_urban_RX'))+'|')
    lines+=['','## 配对差和交互项','','|比较|clean Δpp|low urban Δpp|mid urban Δpp|urban 均值 Δpp|','|---|---:|---:|---:|---:|']
    for pair,values in result['paired_difference_pp'].items():lines.append('|'+pair+'|'+'|'.join(f'{values[k]:+.4f}' for k in ('clean','practical_low_urban','practical_mid_urban','urban_mean'))+'|')
    lines.append('|B11−B10−B01＋B00|'+'|'.join(f"{result['interaction_pp'][k]:+.4f}" for k in ('clean','practical_low_urban','practical_mid_urban','urban_mean'))+'|')
    lines+=['','B10−B00 检验新增响应变化；B01−B00 检验 practical；B11−B01 检验响应变化对 practical 的增量。B00−R_LEGACY 同时改变了旧普通 L 增强和前向数量，不能单独归因为某个算子。R_LEGACY 不能被较弱的重组对照替代。',
        '', '## 源规则与正式探索条件','',
        '源资格：'+str(qualification['passed'])+'；后续扩种子资格只来自该固定源规则，目标成绩不能覆盖。',
        '', '|源条件|通过|','|---|---|']
    for key,value in qualification['checks'].items():lines.append(f'|{key}|{value}|')
    lines+=['','|正式探索目标|通过|','|---|---|']
    for key,value in result['exploration_checks'].items():lines.append(f'|{key}|{value}|')
    lines+=['','正式状态：`'+result['performance_status']+'`。一个种子不足以证明跨 seed 复现。固定 E20 结束不等于收敛。',
        '', '## 源域机制与观察边界','',
        '|row|full V clean %|V 最差 RX %|固定 V720 low urban %|low 保持率 %|low 默认头间隔/30|','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        source=qualification['source_V'][arm];low=panels[arm]['views']['practical_low_urban']
        lines.append(f"|{arm}|{100*source['source_val_accuracy']:.4f}|{100*source['source_val_worst_rx']:.4f}|{100*low['accuracy']:.4f}|{100*low['clean_correct_retention']:.4f}|{low['margin']:.6f}|")
    lines+=['','保持率始终基于冻结参照模型在同一 V720 上的 clean 正确集合；分母固定，不按每个新模型重选。分层图只描述合法源域相同受损观测，不把源状态差异解释成目标域因果结论。源新增噪声 SNR 分箱也不等于总 SNR。',
        '', '## 实际成本','',
        '|row|L 前向|student U 前向|EMA 前向|student 源评估前向|源训练秒|全 row 墙钟秒|','|---|---:|---:|---:|---:|---:|---:|']
    for cost in costs:lines.append('|'+cost['arm']+'|'+'|'.join(str(cost[k]) if isinstance(cost[k],int) else f'{cost[k]:.3f}' for k in ('L_forward_packets','student_U_forward_packets','EMA_forward_packets','student_source_eval_forward_packets','source_seconds','row_wall_seconds'))+'|')
    lines+=['','每 row 另有冻结源 panel 5,040 前向及正式预测 1,176,002 前向（含 2 条 smoke）；五个模型累计 100 model-epoch、22,200 更新。共同源准备 722 前向另计。历史冻结源安全实验 135,002 前向／74.915770 秒、0 更新；其 E20 原参照历史成本 4,440 更新／923.560061 秒，不是本轮任何模型的初始化或继承成本。',
        '', f"当前矩阵墙钟 {summary['matrix_wall_seconds']:.3f} 秒，按已保存文件时间戳计算；每 row 墙钟从已登记启动时间到 row 完成，包含各阶段与开销。GPU 与其他健康任务共享，墙钟不作为纯算法速度归因。",
        '', '## 图和可检查数据','']
    for path in summary['figures']:lines.append('- ['+Path(path).stem+']('+Path(path).as_posix()+')')
    lines+=['','同目录 CSV 包含七视图逐 row／seed 结果、配对差、交互项、全 20 轮曲线、源分层、实际覆盖与成本。分析程序只读取已保存日志、源诊断和独立评分，不访问 IQ、模型权重或 truth。',
        '', '整体目标保持未完成，不能把本轮完成或局部正差写成 clean 与 urban 双突破。新方案若继续研究，须依据正式分数读取前已冻结的源问题另行登记，不用本轮目标分数选择幅度、采样或训练预算。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def figures(output,result,panels,logs,costs,coverage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    paths=[]
    def save(fig,name):
        path=output/(name+'.png');fig.savefig(path);plt.close(fig);paths.append(str(path))
    metrics=(*VIEWS,'urban_mean','urban_macro_f1','worst_urban_RX')
    data=np.array([[100*result['rows'][a]['metrics'][k] for k in metrics] for a in ARMS])
    fig,ax=plt.subplots(figsize=(14,4.5),layout='constrained');im=ax.imshow(data,aspect='auto',cmap='viridis')
    ax.set_xticks(range(len(metrics)),[LABELS.get(k,{'urban_mean':'Urban mean','urban_macro_f1':'Urban F1','worst_urban_RX':'Worst U RX'}.get(k,k)) for k in metrics],rotation=25,ha='right');ax.set_yticks(range(5),ARMS)
    for i in range(5):
        for j in range(len(metrics)):ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',color='white' if data[i,j]<np.median(data) else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Percent');ax.set_title('Fixed default head / original seven test views / one seed')
    save(fig,'formal_metrics')
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),layout='constrained')
    effects=[('B10-B00','clean'),('B11-B00','clean'),('B11-R_LEGACY','clean'),('B11-B10','clean'),('B11-B01','practical_low_urban'),('B11-B01','urban_mean')]
    labels=[p+' '+LABELS.get(k,'urban mean') for p,k in effects];values=[result['paired_difference_pp'][p][k] for p,k in effects]
    axes[0].barh(labels,values,color=['#0072B2' if v>=0 else '#D55E00' for v in values]);axes[0].axvline(0,color='black',lw=.7);axes[0].invert_yaxis();axes[0].set(title='Matched factorial effects',xlabel='Difference (percentage points)')
    keys=('clean','practical_low_urban','practical_mid_urban','urban_mean');values=[result['interaction_pp'][k] for k in keys]
    axes[1].bar([LABELS.get(k,'Urban mean') for k in keys],values,color=['#0072B2' if v>=0 else '#D55E00' for v in values]);axes[1].axhline(0,color='black',lw=.7);axes[1].tick_params(axis='x',rotation=25);axes[1].set(title='B11 - B10 - B01 + B00',ylabel='Interaction (percentage points)')
    x=np.arange(3)
    for i,a in enumerate(ARMS):axes[2].bar(x+(i-2)*.15,[result['rows'][a]['suburban_gap_pp'][s] for s in SCENES[:3]],.14,label=a,color=COLORS[a])
    axes[2].axhline(2,color='#777777',ls='--',lw=.8);axes[2].axhline(-2,color='#777777',ls='--',lw=.8);axes[2].set_xticks(x,[LABELS[s] for s in SCENES[:3]]);axes[2].set(title='Suburban gap to own clean',ylabel='Clean - suburban (pp)');axes[2].legend(fontsize=8)
    fig.suptitle('Single seed exploratory effects; no confidence interval or replication claim');save(fig,'factorial_effects')
    fig,ax=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    for a in ARMS:
        epochs=logs[a]['epochs'];curves=logs[a]['curves']
        ax[0,0].plot([e['epoch'] for e in epochs],[e['val_tx_acc'] for e in epochs],label=a,color=COLORS[a])
        ax[1,0].plot(range(1,21),[e['train/grad_before_clip'] for e in curves],label=a,color=COLORS[a])
    ax[0,0].set(title='Full source V clean',xlabel='Epoch',ylabel='Accuracy (%)');ax[0,0].legend(fontsize=8,ncol=3)
    for key,label,color in [('clean_CE','Raw','#0072B2'),('generalization_CE','Response','#009E73'),('satellite_CE','Practical','#D55E00')]:ax[0,1].plot(range(1,21),[e['mixedIQ/'+key] for e in logs['B11']['curves']],label=label,color=color)
    ax[0,1].set(title='B11 per-view smoothed CE',xlabel='Epoch',ylabel='CE');ax[0,1].legend(fontsize=8)
    ax[1,0].set(title='Mean gradient norm before clipping',xlabel='Epoch',ylabel='Gradient norm')
    for a in ('B01','B11'):
        for key,style,label in [('clean_margin','-','raw'),('s_margin','--','practical')]:ax[1,1].plot(range(1,21),[e['mixedIQ/'+key] for e in logs[a]['curves']],style,label=a+' '+label,color=COLORS[a])
    ax[1,1].set(title='Training default-head margin / 30',xlabel='Epoch',ylabel='Normalized margin');ax[1,1].legend(fontsize=8)
    for item in ax.flat:item.axvline(13.5,color='#aaaaaa',ls=':',lw=.8);item.grid(alpha=.15)
    fig.suptitle('All 20 completed epochs; dotted line marks original U-stage start');save(fig,'training_curves')
    grouped={a:source_groups(panels[a]) for a in ARMS}
    fig,ax=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    specs=[('state',['LOS','PARTIAL','BLOCKED'],'accuracy','Accuracy (%)'),('state',['LOS','PARTIAL','BLOCKED'],'clean_correct_retention','Fixed parent clean-correct retention (%)'),('added_noise_SNR_bin',['<0','0-3','3-10','10-20','>=20'],'accuracy','Accuracy (%)'),('state',['LOS','PARTIAL','BLOCKED'],'margin','Mean default-head margin / 30')]
    for axis,(dimension,labels,key,title) in zip(ax.flat,specs):
        counts=[]
        for a,shift in (('B01',-.18),('B11',.18)):
            cells={r['stratum']:r for r in grouped[a] if r['view']=='practical_low_urban' and r['dimension']==dimension}
            values=[cells.get(label,{}).get(key) for label in labels];values=[float('nan') if val is None else val*(1 if key=='margin' else 100) for val in values]
            axis.bar(np.arange(len(labels))+shift,values,.34,label=a,color=COLORS[a]);counts=[cells.get(label,{}).get('count',0) for label in labels]
        axis.set_xticks(range(len(labels)),[f'{label}\nn={count}' for label,count in zip(labels,counts)]);axis.set_title(title);axis.legend(fontsize=8);axis.grid(axis='y',alpha=.15)
    fig.suptitle('Frozen source V720, low urban; same observations / not target causal attribution');save(fig,'source_state_quality')
    fig,axes=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
    for column,arm in enumerate(('B01','B11')):
        rows=[r for r in coverage if r['arm']==arm]
        for line,(field,classes,colors) in enumerate((('state',('LOS','PARTIAL','BLOCKED'),('#0072B2','#E69F00','#D55E00')),('lock',('locked','degraded','unlocked'),('#009E73','#E69F00','#CC79A7')))):
            axis=axes[line,column];bottom=np.zeros(6)
            for group,color in zip(classes,colors):
                y=np.array([sum(r['count'] for r in rows if r['scene']==s and r[field]==group) for s in SCENES]);axis.bar(range(6),y,bottom=bottom,label=group,color=color);bottom+=y
            axis.set_xticks(range(6),[LABELS[s] for s in SCENES]);axis.set(title=arm+' '+field,ylabel='L practical exposures');axis.legend(fontsize=8)
    fig.suptitle('Actual training coverage; scene mixture with native state priors');save(fig,'training_coverage')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');bottom=np.zeros(5)
    for key,label in [('L_forward_packets','L'),('student_U_forward_packets','Student U'),('EMA_forward_packets','EMA'),('student_source_eval_forward_packets','Source eval'),('frozen_source_panel_forward_packets','Source panel'),('formal_forward_packets','Formal prediction')]:
        y=np.array([c[key]/1e6 for c in costs]);axes[0].bar(ARMS,y,bottom=bottom,label=label);bottom+=y
    axes[0].set(title='Measured packet forwards',ylabel='Million packets');axes[0].legend(fontsize=8,ncol=2)
    axes[1].bar(ARMS,[c['row_wall_seconds']/60 for c in costs],label='All stages');axes[1].plot(ARMS,[c['source_seconds']/60 for c in costs],'ko--',ms=3,label='Native source training');axes[1].set(title='Observed wall time with shared GPUs',ylabel='Minutes');axes[1].legend(fontsize=8)
    fig.suptitle('Every model scratch E20; extra forwards and wall costs remain explicit');save(fig,'measured_cost')
    return paths,grouped


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--trace-only',action='store_true');args=parser.parse_args()
    if args.trace_only:
        value=audit_channel_pairs(args.input);args.output.mkdir(parents=True,exist_ok=True);write(args.output/'channel_trace_audit.json',value)
        print(json.dumps(value));return
    result,qualification,matrix,panels,logs,costs,coverage=validate(args.input)
    for a,b in (('B01','B00'),('B10','R_LEGACY'),('B01','R_LEGACY')):
        result['paired_difference_pp'][a+'-'+b]={k:100*(result['rows'][a]['metrics'][k]-result['rows'][b]['metrics'][k]) for k in result['rows'][a]['metrics']}
    args.output.mkdir(parents=True,exist_ok=False)
    paths,grouped=figures(args.output,result,panels,logs,costs,coverage)
    table(args.output/'seven_view_results.csv',matrix)
    table(args.output/'paired_effects.csv',[dict(pair=pair,metric=key,difference_pp=value) for pair,values in result['paired_difference_pp'].items() for key,value in values.items()])
    table(args.output/'factorial_interaction.csv',[dict(metric=k,interaction_pp=v) for k,v in result['interaction_pp'].items()])
    table(args.output/'source_state_quality.csv',[dict(arm=a,**r) for a in ARMS for r in grouped[a]])
    table(args.output/'costs.csv',costs);table(args.output/'coverage.csv',coverage)
    table(args.output/'training_curves.csv',[dict(arm=a,**r) for a in ARMS for r in logs[a]['curves']])
    summary=dict(status='OFFLINE_ANALYZED',seed_count=1,figures=paths,log_checks={a:logs[a]['proof'] for a in ARMS},
        source_qualification=qualification['passed'],source_checks=qualification['checks'],formal_exploration_checks=result['exploration_checks'],
        performance_status=result['performance_status'],matrix_wall_seconds=(args.input/'complete.json').stat().st_mtime-(args.input/'dispatcher.json').stat().st_mtime,
        shared_source_preparation_forward_packets=722,prior_source_safety_forward_packets=135002,
        prior_source_safety_seconds=74.91577004594728,reference_parent_epochs=20,reference_parent_updates=4440,reference_parent_seconds=923.5600609779358,
        reference_parent_loaded_into_new_models=False,goal_completed=False,claim='Fixed exposed benchmark; one seed; no replication claim')
    write(args.output/'analysis_summary.json',summary)
    report(args.output,result,qualification,panels,costs,summary)
    print(json.dumps(dict(status=summary['status'],output=str(args.output),figures=len(paths),seed_count=1,source_pass=qualification['passed'],formal_checks=result['exploration_checks'])))


if __name__=='__main__':main()
