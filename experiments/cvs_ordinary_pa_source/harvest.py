"""Independent default-head recount from source decisions; no IQ/CP imports."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from . import design as d
from experiments.cvs_ordinary_fold20.harvest import metrics,read,write,SSH


def collect(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for row in d.rows():
        rid=row['row_id'];dest=output/rid;dest.mkdir(exist_ok=True)
        names=['completion.json','resolved_config.json','RAW.json','paired.jsonl.gz','progress.json']
        subprocess.run(['scp',*SSH,*['N607:'+(d.BASE/rid/v).as_posix() for v in names],str(dest)],check=True,timeout=60)
    subprocess.run(['scp',*SSH,'N607:'+(d.BASE/'worker.log').as_posix(),str(output)],check=True,timeout=60)


def distribution(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty waveform statistics required')
    return dict(count=len(x),mean=float(x.mean()),min=float(x.min()),median=float(np.median(x)),
        p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),max=float(x.max()))


def analyze(output,state):
    output=Path(output)
    if state['failure'] or state['failure_dispatch'] or state['worker'] or state['dispatcher'] or not state['completion']:
        raise ValueError('Source diagnostic must be terminal without technical failure')
    if state['completion']['packet_forwards']!=207910 or state['completion']['optimizer_updates'] or state['completion']['target_access']:
        raise ValueError('Fixed frozen source budget differs')
    folds=[];strata=[];wave=[];models={};proof={};paired_events={}
    for row in d.rows():
        rid=row['row_id'];rx=row['held_RX'];root=output/rid;done=read(root/'completion.json');resolved=read(root/'resolved_config.json')
        if (done['config']!=d.config(row) or done['packet_forwards']!=41582 or done['optimizer_updates'] or done['head_fit_updates']
            or done['V_used'] or done['U_used'] or done['target_access'] or not done['upstream_identical']
            or not done['non_PA_output_identical'] or not done['parameters_buffers_unchanged']):raise ValueError('Frozen row contract differs')
        raw=read(root/'RAW.json')['records'];by_index={v['index']:v for v in raw}
        if len(raw)!=1260 or len(by_index)!=1260 or len({v['id'] for v in raw})!=1260 or any(v['RX']!=rx for v in raw):
            raise ValueError('Unique held L1260 required')
        if any(sum(v['label']==t and v['day']==day for v in raw)!=70 for t in range(6) for day in (1,2,3)):
            raise ValueError('All source TX/day cells required')
        records=[];keys=set();active=0
        with gzip.open(root/'paired.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                v=json.loads(line);key=(v['index'],v['draw'])
                if key in keys or v['index'] not in by_index or v['draw'] not in range(16):raise ValueError('Repeated/foreign source decision')
                keys.add(key);b=by_index[v['index']];v.update(label=b['label'],day=b['day'],RX=rx,raw_correct=b['label']==b['prediction'])
                if not math.isfinite(v['PA_strength']) or not math.isfinite(v['PA_MSE']) or v['PA_MSE']<0:raise ValueError('Invalid PA effect')
                if v['PA_strength']==0 and (v['FULL']!=v['NO_PA'] or v['PA_MSE']!=0):raise ValueError('Unacted PA pair changed')
                active+=v['PA_strength']>0;records.append(v)
        if len(records)!=20160 or keys!={(i,j) for i in by_index for j in range(16)} or active!=done['PA_acted']:
            raise ValueError('Complete fixed16draw source decisions required')
        paired_events[rx]=records
        result=dict(RAW=metrics(raw),PA_acted=active)
        correct_count=sum(v['prediction']==v['label'] for v in raw)
        if correct_count==0:raise ValueError('Fixed RAW-correct set is empty')
        for arm in ('FULL','NO_PA'):
            result[arm]=metrics([dict(label=v['label'],prediction=v[arm]['prediction']) for v in records])
            kept=sum(v['raw_correct'] and v[arm]['prediction']==v['label'] for v in records)
            result[arm]['RAW_correct_retention']=kept/(correct_count*16)
            result[arm]['RAW_correct_exposures']=correct_count*16
            for dimension,values in (('TX',range(6)),('day',(1,2,3)),('PA_acted',(False,True))):
                for value in values:
                    selected=[v for v in records if (v['label'] if dimension=='TX' else v['day'] if dimension=='day' else v['PA_strength']>0)==value]
                    if not selected:raise ValueError('Missing source diagnostic stratum')
                    strata.append(dict(RX=rx,arm=arm,dimension=dimension,value=value,
                        **metrics([dict(label=v['label'],prediction=v[arm]['prediction']) for v in selected])))
            for key in ('power','max_abs','PAPR'):
                for subset in ('all','PA_acted'):
                    values=[v[arm]['waveform'][key] for v in records if subset=='all' or v['PA_strength']>0]
                    wave.append(dict(RX=rx,arm=arm,subset=subset,metric=key,**distribution(values)))
        result['paired_repaired']=sum(v['FULL']['prediction']!=v['label'] and v['NO_PA']['prediction']==v['label'] for v in records)
        result['paired_broken']=sum(v['FULL']['prediction']==v['label'] and v['NO_PA']['prediction']!=v['label'] for v in records)
        models[str(rx)]=result
        folds.append(dict(held_RX=rx,retention_gain_pp=100*(result['NO_PA']['RAW_correct_retention']-result['FULL']['RAW_correct_retention']),
            accuracy_gain_pp=100*(result['NO_PA']['accuracy']-result['FULL']['accuracy']),macroF1_gain_pp=100*(result['NO_PA']['macro_f1']-result['FULL']['macro_f1']),
            PA_acted=active,paired_repaired=result['paired_repaired'],paired_broken=result['paired_broken']))
        proof[str(rx)]=dict(unique_held_L=1260,paired_exposures=20160,all16draws_complete=True,PA_acted=active,upstream_identical=True,
            non_PA_output_identical=True,frozen_parameters_buffers=True,checkpoint_smoke_packets=2,actual_code=resolved['commit'],
            actual_modules=resolved['actual_modules'],actual_CWD=resolved['actual_CWD'],actual_GPU=resolved['GPU'],recipe=resolved['recipe'])
    q=d.qualify(folds);summary=dict(status='ANALYZED',source_decision=q,models=models,proof=proof,
        packet_forwards=207910,new_models=0,optimizer_updates=0,head_fit_updates=0,target_access=False,U_used=False,V_used=False,
        model_seed_count=1,source_fold_count=5,parent_weight_model_epochs=100,parent_all_fold_study_model_epochs=200,
        wall_seconds=state['completion']['wall_seconds'],goal_completed=False,automatic_training=False,
        postlaunch_live_snapshot='MISSED: 23s job was terminal at first independent read; no rerun to fill observation',
        local_observer_issue='Local inspect.py shadowed stdlib inspect; renamed before successful readback. No remote/data/model mutation or retry.')
    for name,rows in (('fold_results.csv',folds),('source_strata.csv',strata),('waveform_summary.csv',wave)):
        fields=list(dict.fromkeys(k for v in rows for k in v))
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write(output/'source_decision.json',q);write(output/'analysis_summary.json',summary)
    plot(output,models,folds,strata,wave,paired_events)
    lines=['# 额外 PA 对冻结默认头的源身份影响','',f"主判据：**{q['status']}**。零训练更新；五个N0 final20，原held L6300。",'',
        '|held RX|RAW准确率|FULL准确率|NO_PA准确率|保持率变化/pp|准确率变化/pp|实际PA作用数|','|---|---:|---:|---:|---:|---:|---:|']
    for v in folds:
        m=models[str(v['held_RX'])];lines.append(f"|{v['held_RX']}|{100*m['RAW']['accuracy']:.4f}%|{100*m['FULL']['accuracy']:.4f}%|{100*m['NO_PA']['accuracy']:.4f}%|{v['retention_gain_pp']:+.4f}|{v['accuracy_gain_pp']:+.4f}|{v['PA_acted']}|")
    lines+=['',f"NO_PA−FULL平均保持率{q['mean_retention_gain_pp']:+.6f}pp，事前条件逐项{q['checks']}。",'',
        'RAW只前向一次；FULL/NO_PA各16draw。每折1260个唯一源L样本，所有20160配对记录独立复算，RAW正确集合固定；没有按学生增强置信度筛样本。逐TX/day/PA作用子集和macroF1见CSV。', '',
        f"真实样本前向207910（含5个CPsmoke共10），0 optimizer/额外head更新，墙钟{summary['wall_seconds']:.6f}s。继承五个E20权重的100model-epoch另列，完整五foldN0/N1研发成本200model-epoch不隐藏。本地合成网络前向96/增强输入192包另计。",'',
        '每折前级逐值相同、非PA作用包输出相同、模型参数/缓冲区冻结已实际断言，source配对预测由离线工具独立复算。没有V/U/target、星地输入或新训练。', '',
        '这只证明该冻结头在当前条件下的输入效应，不能把它当成PA导致旧N1训练后RX6/8下降的因果证据，不能证明NO_PA是有效Ag，也没有正式clean/urban或跨seed结果。主判据FAIL不改强度/seed/RX或切换次要终点救援；PASS也不自动训练。', '',
        '独立首个读回时23s任务已终止，运行中PID/CWD/日志增长快照缺失；发布记录、实际resolved CWD/GPU/commit、5smoke、全部draw日志、终态进程退出及结果已保存。没有补跑。最初本地观察器文件名遮蔽stdlib inspect，已改名；没有远端失败或重启。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def plot(output,models,folds,strata,wave,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);x=np.arange(5)
    for arm,delta,color in (('RAW',-.24,'#999999'),('FULL',0,'#e66101'),('NO_PA',.24,'#386cb0')):
        axes[0,0].bar(x+delta,[100*models[str(r)][arm]['accuracy'] for r in d.RX],width=.24,label=arm,color=color)
    axes[0,0].set(xticks=x,xticklabels=d.RX,xlabel='Held source RX',ylabel='Accuracy (%)',ylim=(0,100));axes[0,0].legend()
    axes[0,1].bar(x,[v['retention_gain_pp'] for v in folds]);axes[0,1].axhline(0,color='black',lw=.8)
    axes[0,1].set(xticks=x,xticklabels=d.RX,xlabel='Held source RX',ylabel='NO_PA minus FULL retention (pp)')
    for ax,dimension,values in ((axes[1,0],'TX',range(6)),(axes[1,1],'day',(1,2,3))):
        matrix=np.array([[100*(next(v['accuracy'] for v in strata if v['RX']==r and v['arm']=='NO_PA' and v['dimension']==dimension and v['value']==c)-next(v['accuracy'] for v in strata if v['RX']==r and v['arm']=='FULL' and v['dimension']==dimension and v['value']==c)) for c in values] for r in d.RX])
        bound=max(float(np.abs(matrix).max()),.1);im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
        for i in range(5):
            for j in range(len(values)):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=8)
        ax.set(xticks=range(len(values)),xticklabels=list(values),yticks=range(5),yticklabels=d.RX,xlabel=dimension,ylabel='Held source RX');fig.colorbar(im,ax=ax,label='Accuracy gain (pp)')
    fig.suptitle('Frozen N0 source heads; original E13; paired identical upstream');fig.savefig(output/'source_PA_effects.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for arm,color in (('FULL','#e66101'),('NO_PA','#386cb0')):
        values=[v[arm]['waveform']['max_abs'] for rows in events.values() for v in rows if v['PA_strength']>0]
        values=np.sort(values);axes[0].plot(values,np.arange(1,len(values)+1)/len(values),label=arm,color=color)
    axes[0].set(xscale='log',xlabel='Max absolute IQ component, PA-acted packets',ylabel='Empirical CDF');axes[0].legend()
    for rx in d.RX:
        rows=events[rx];active=[v for v in rows if v['PA_strength']>0]
        delta=[10*np.log10(max(v['FULL']['waveform']['power'],1e-20)/max(v['NO_PA']['waveform']['power'],1e-20)) for v in active]
        axes[1].scatter([rx]*len(delta),delta,s=2,alpha=.15)
    axes[1].axhline(0,color='black',lw=.8);axes[1].set(xticks=d.RX,xlabel='Held source RX',ylabel='PA added power change (dB)')
    fig.savefig(output/'PA_waveform_effects.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    if not a.cached:collect(a.output)
    result=analyze(a.output,read(a.state));print({k:v for k,v in result.items() if k not in ('models','proof')})
