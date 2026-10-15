"""Offline five-fold source evidence; no native, IQ or checkpoint imports."""
import argparse
import csv
import gzip
from itertools import zip_longest
from pathlib import Path
import numpy as np
from experiments.cvs_ordinary_fold20.harvest import read,write,remote,INSPECT,metrics,parse_logs,transfer

ROOT=Path(__file__).resolve().parents[2]
RX=(1,3,4,6,8);ARMS=('N0','N1')


def inspect(output,name='progress.json'):
    spec=read(ROOT/'experiments/cvs_ordinary_confirm20/registration.json')
    script=INSPECT.replace("matched_sequence=read(root/'matched_sequence.json')",
        "matched_sequences=read(root/'matched_sequences.json'),reused_RX1=read(root/'reused_RX1.json')")
    value=remote(script.replace('CONFIG',repr(dict(root=spec['execution']['remote_run_root'],release=spec['code']['cwd']))))
    write(Path(output)/name,value)
    return value


def decision(records):
    if [v['held_RX'] for v in records]!=list(RX):raise ValueError('All five fixed folds required')
    mean=float(np.mean([v['accuracy_gain_pp'] for v in records]));f1=float(np.mean([v['macroF1_gain'] for v in records]))
    checks=dict(mean_accuracy=mean>=1.-1e-9,positive_folds=sum(v['accuracy_gain_pp']>0 for v in records)>=4,
        worst_fold=min(v['accuracy_gain_pp'] for v in records)>=-1.-1e-9,mean_macroF1=f1>=-1e-12,
        seen_health=all(v['seen_V_mean_drop_pp']<=.5+1e-9 for v in records),complete_health=all(v['healthy'] for v in records))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,mean_accuracy_gain_pp=mean,mean_macroF1_gain=f1)


def predictions(source,rx,saved):
    rows=read(source/'held_V_predictions.json')['records']
    keys=[tuple(v[k] for k in ('eq','RX','day','sig_i','base_index')) for v in rows]
    if len(rows)!=5400 or len(set(keys))!=5400 or any(v['RX']!=rx or v['day'] not in (1,2,3) for v in rows):
        raise ValueError('Exact unique held source V5400 required')
    recount=metrics(rows)
    if recount['confusion']!=saved['confusion'] or any(abs(recount[k]-saved[k])>1e-12 for k in ('accuracy','macro_f1')):
        raise ValueError('Independent source recount differs')
    return rows,{k:v for k,v in zip(keys,rows)}


def pair(output,rx,registered):
    models={};held={};logs={};costs=[];strata=[];curves=[];arguments={}
    for row in registered:
        rid=row['row_id'];arm=row['config']['arm'];source=output/rid/'source'
        result=read(source/'result.json');done=read(source/'completion.json');init=read(source/'initialization.json');freeze=read(source/'source_frozen.json')
        fit=[r for r in RX if r!=rx];contract=read(source/'source_contract.json')
        if (done['config']!=row['config'] or done['optimizer_steps']!=4440 or done['epoch']!=20 or done['initialization']!='scratch'
            or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
            or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact']
            or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
            or result['target_access'] or result['U_truth_read'] or result['head_fit_updates'] or not result['default_head_joint_training']
            or result['held_RX']!=rx or result['fit_RX']!=fit or contract['held_RX']!=rx or contract['source_receivers']!=fit
            or result['L_forward_packets']!=1136640 or result['training_loader_packets']!={'L_s':568320,'U_s':396368}
            or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):
            raise ValueError('Fixed source RX/default-head/scratch E20 contract differs')
        if ({(v['TX'],v['RX'],v['day']) for v in result['coverage']}!={(t,r,d) for t in range(6) for r in fit for d in (1,2,3)}
            or sum(v['count'] for v in result['coverage'])!=568320):raise ValueError('Training coverage incomplete')
        rows,held[arm]=predictions(source,rx,result['held_V'])
        for dimension,values in (('TX',range(6)),('day',(1,2,3))):
            for value in values:
                selected=[v for v in rows if v['label' if dimension=='TX' else 'day']==value]
                strata.append(dict(held_RX=rx,arm=arm,dimension=dimension,value=value,**metrics(selected)))
        for tx in range(6):
            for day in (1,2,3):
                selected=[v for v in rows if v['label']==tx and v['day']==day]
                if len(selected)!=300:raise ValueError('TX/day cell count differs')
                strata.append(dict(held_RX=rx,arm=arm,dimension='TX_day',value=f'{tx}/{day}',**metrics(selected)))
        logs[arm]=parse_logs(source,output/'logs'/(rid+'.log'))
        curves.extend(dict(held_RX=rx,arm=arm,**v) for v in logs[arm]['curves'])
        a=read(source/'actual_native_args.json');arguments[arm]=a.get('args',a)
        mechanism=read(source/'mechanism_progress.json')['records']
        if len(mechanism)!=20 or any(v<=0 for v in result['actual_feature_parameters_moved'].values()):raise ValueError('Joint updates incomplete')
        if arm=='N1' and (result['ordinary']['calls']!=4440 or result['ordinary']['packets']!=568320):raise ValueError('Ordinary generation count differs')
        models[arm]=result
        costs.append(dict(held_RX=rx,arm=arm,reused=False,epochs=20,updates=4440,parent_weight_epochs=0,
            L_forward_packets=result['L_forward_packets'],student_U_forward_packets=result['student_U_forward_packets'],
            EMA_forward_packets=result['EMA_eval_forward_packets'],student_source_eval_forward_packets=result['student_source_eval_forward_packets'],
            held_V_forward_packets=5400,source_training_seconds=result['elapsed_seconds'],formal_forward_packets=0))
    changed=sorted(k for k in set(arguments['N0'])|set(arguments['N1']) if arguments['N0'].get(k)!=arguments['N1'].get(k))
    if set(changed)-{'candidate_id','output_dir','use_aug'}:raise ValueError(('Unexpected actual args contrast',rx,changed))
    if held['N0'].keys()!=held['N1'].keys() or any(held['N0'][k]['label']!=held['N1'][k]['label'] for k in held['N0']):raise ValueError('Held pair sample mismatch')
    repaired=sum(v['prediction']!=v['label'] and held['N1'][k]['prediction']==v['label'] for k,v in held['N0'].items())
    broken=sum(v['prediction']==v['label'] and held['N1'][k]['prediction']!=v['label'] for k,v in held['N0'].items())
    count=0
    paths=[output/r['row_id']/'source/training_sequence.jsonl.gz' for r in registered]
    with gzip.open(paths[0],'rt') as a,gzip.open(paths[1],'rt') as b:
        for x,y in zip_longest(a,b):
            if x!=y:raise ValueError(('Actual full L/U sequence mismatch',rx))
            count+=1
    return dict(models=models,logs=logs,costs=costs,strata=strata,curves=curves,paired_repaired=repaired,paired_broken=broken,
        actual_sequence_batch_records=count,actual_native_arg_differences=changed)


def analyze(output,state,first_output):
    output=Path(output);first_output=Path(first_output);spec=read(ROOT/'experiments/cvs_ordinary_confirm20/registration.json')
    prior=read(first_output/'analysis_summary.json')
    if prior['status']!='ANALYZED' or prior['source_decision']['status']!='PASS' or prior['target_access']:raise ValueError('Completed RX1 source result required')
    folds={};records=[];costs=[];strata=[];curves=[]
    for rx in RX:
        if rx==1:
            models=state['reused_RX1']['results']
            if {a:prior['models'][a]['held_V'] for a in ARMS}!={a:models[a]['held_V'] for a in ARMS}:raise ValueError('RX1 source reuse differs')
            folds[rx]=dict(models=models,paired_repaired=prior['paired_repaired'],paired_broken=prior['paired_broken'],
                actual_sequence_batch_records=prior['actual_sequence_batch_records'],full_log_proof=prior['full_log_proof'],reused=True)
            costs.extend(dict(v,held_RX=1,reused=True) for v in prior['costs'])
            with (first_output/'source_strata.csv').open(encoding='utf-8',newline='') as f:
                for v in csv.DictReader(f):
                    strata.append(dict(held_RX=1,arm=v['arm'],dimension=v['dimension'],value=int(v['value']) if v['dimension']!='TX_day' else v['value'],
                        accuracy=float(v['accuracy']),macro_f1=float(v['macro_f1']),count=int(v['count'])))
        else:
            result=pair(output,rx,[r for r in spec['rows'] if r['config']['held_RX']==rx]);models=result['models'];folds[rx]=result
            costs+=result['costs'];strata+=result['strata'];curves+=result['curves']
        a,b=models['N1'],models['N0']
        records.append(dict(held_RX=rx,accuracy_gain_pp=100*(a['held_V']['accuracy']-b['held_V']['accuracy']),
            macroF1_gain=a['held_V']['macro_f1']-b['held_V']['macro_f1'],seen_V_mean_drop_pp=100*(b['seen_V']['source_val_accuracy']-a['seen_V']['source_val_accuracy']),
            healthy=bool(a['complete_health'] and b['complete_health'])))
    q=decision(records);saved=state['source_decision']
    if q['checks']!=saved['checks'] or q['status']!=saved['status'] or abs(q['mean_accuracy_gain_pp']-saved['mean_accuracy_gain_pp'])>1e-10:
        raise ValueError('Frozen five-fold source decision arithmetic differs')
    keys=('L_forward_packets','student_U_forward_packets','EMA_forward_packets','student_source_eval_forward_packets','held_V_forward_packets')
    total=lambda rows:sum(sum(v[k] for k in keys) for v in rows)
    summary=dict(status='ANALYZED',source_decision=saved,folds={str(r):dict(models=folds[r]['models'],paired_repaired=folds[r]['paired_repaired'],
        paired_broken=folds[r]['paired_broken'],actual_sequence_batch_records=folds[r]['actual_sequence_batch_records'],
        full_log_proof=folds[r].get('full_log_proof') or {a:v['proof'] for a,v in folds[r]['logs'].items()}) for r in RX},
        new_model_epochs=160,new_updates=35520,reused_model_epochs=40,reused_updates=8880,new_real_forward_packets=total([v for v in costs if not v['reused']]),
        reused_real_forward_packets=total([v for v in costs if v['reused']]),total_real_forward_packets=total(costs),
        matrix_wall_seconds=state['complete']['wall_seconds'],reused_RX1_matrix_wall_seconds=prior['matrix_wall_seconds'],
        model_seed_count=1,source_fold_count=5,source_folds_are_independent_model_seeds=False,target_access=False,goal_completed=False,
        formal_test_status='NOT_REGISTERED_SOURCE_ONLY',costs=costs)
    for name,rows in (('fold_results.csv',records),('training_curves.csv',curves),('source_strata.csv',strata),('measured_costs.csv',costs)):
        fields=list(dict.fromkeys(k for v in rows for k in v))
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write(output/'analysis_summary.json',summary);plot(output,folds,records,strata,costs)
    lines=['# 普通 IQ 增强的五个源 RX 留出结果','',f"源资格：**{q['status']}**。一个model seed，五个源RX折；RX1复用已有结果，其余8个独立E20完成。",'',
        '|held RX|N0准确率|N1准确率|增益/pp|macroF1增益/pp|seenV下降/pp|','|---|---:|---:|---:|---:|---:|']
    for v in records:
        m=folds[v['held_RX']]['models'];lines.append(f"|{v['held_RX']}|{100*m['N0']['held_V']['accuracy']:.6f}%|{100*m['N1']['held_V']['accuracy']:.6f}%|{v['accuracy_gain_pp']:+.6f}|{100*v['macroF1_gain']:+.6f}|{v['seen_V_mean_drop_pp']:+.6f}|")
    lines+=['',f"平均准确率差{q['mean_accuracy_gain_pp']:+.6f}pp，macroF1差{100*q['mean_macroF1_gain']:+.6f}pp。源门槛逐项：{q['checks']}。",'',
        f"新8模型共160model-epoch/35520updates/真实前向{summary['new_real_forward_packets']}；复用RX1 pair40model-epoch/8880updates/真实前向{summary['reused_real_forward_packets']}。新矩阵墙钟{summary['matrix_wall_seconds']:.3f}s。本地合成前向另计。",'',
        '所有新行完整20epoch/4440step/nativeCSV/stdout已解析；所有四对实际L/U物理样本序列逐条相同。默认头联合更新、源TX/RX/day覆盖和U真值隐藏按实际日志核对；保存的heldV默认头决策独立复算。', '',
        '五个fold训练集相互重叠，不是五个model seed；只有FINAL heldV读出，不存在heldV训练曲线。该结果不等于正式跨RX clean/urban收益，不认证每个旧PA等算子的身份保持。E20用尽不是收敛。', '',
        '无target、无As、无旧权重。FAIL结束普通增强作为Ag的本候选，不换RX/强度/seed救援；PASS也不自动启动target或扩seed。原正式practical IQ与信道保持。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def plot(output,folds,records,strata,costs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'N0':'#386cb0','N1':'#e66101'}
    for rx in RX[1:]:
        fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
        for ax,key in zip(axes.flat,('legacy_rdrop/paired_CE','legacy_rdrop/symmetric_KL','train/grad_before_clip','train/pseudo_selected','val_tx_acc','lr')):
            for arm in ARMS:
                log=folds[rx]['logs'][arm];rows=log['epochs'] if key in ('val_tx_acc','lr') else log['curves']
                ax.plot([v['epoch'] for v in rows],[v.get(key,0) for v in rows],label=arm,color=colors[arm])
            ax.axvline(13.5,color='grey',ls='--',lw=.8);ax.set(xlabel='Epoch',ylabel=key);ax.legend()
        fig.suptitle(f'Complete E20 training; held source RX{rx} (FINAL only)');fig.savefig(output/f'training_RX{rx}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);x=np.arange(5)
    for arm,offset in (('N0',-.18),('N1',.18)):
        axes[0,0].bar(x+offset,[100*folds[r]['models'][arm]['held_V']['accuracy'] for r in RX],width=.36,label=arm,color=colors[arm])
    axes[0,0].set(xticks=x,xticklabels=[str(r) for r in RX],xlabel='Held source RX',ylabel='Accuracy (%)',ylim=(0,100));axes[0,0].legend()
    axes[0,1].bar(x,[v['accuracy_gain_pp'] for v in records],color='#4c956c');axes[0,1].axhline(0,color='black',lw=.8)
    axes[0,1].set(xticks=x,xticklabels=[str(r) for r in RX],xlabel='Held source RX',ylabel='N1 minus N0 (pp)')
    for ax,dimension,values in ((axes[1,0],'TX',range(6)),(axes[1,1],'day',(1,2,3))):
        matrix=np.array([[100*(next(v['accuracy'] for v in strata if v['held_RX']==r and v['arm']=='N1' and v['dimension']==dimension and v['value']==c)-next(v['accuracy'] for v in strata if v['held_RX']==r and v['arm']=='N0' and v['dimension']==dimension and v['value']==c)) for c in values] for r in RX])
        bound=max(float(np.abs(matrix).max()),1.);im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
        for i in range(5):
            for j in range(len(values)):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=8)
        ax.set(xticks=range(len(values)),xticklabels=list(values),yticks=range(5),yticklabels=list(RX),xlabel=dimension,ylabel='Held source RX');fig.colorbar(im,ax=ax,label='Accuracy gain (pp)')
    fig.suptitle('Five source RX folds; one model seed; RX1 reused');fig.savefig(output/'fivefold_source_results.png',dpi=170);plt.close(fig)


def finish(output,first_output):
    output=Path(output);state=inspect(output,'terminal_state.json')
    if state['failure'] or not state['complete'] or state['dispatcher_live'] or len(state['rows'])!=8 or any(v['live'] for v in state['rows']):
        raise ValueError('Fixed source confirmation must be terminal before full analysis')
    if not state['matched_sequences'] or state['matched_sequences']['status']!='VERIFIED':raise ValueError('Real sample sequences incomplete')
    spec=read(ROOT/'experiments/cvs_ordinary_confirm20/registration.json');transfer(output,spec)
    return analyze(output,state,first_output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--name',default='progress.json')
    parser.add_argument('--finish',action='store_true');parser.add_argument('--first-output')
    args=parser.parse_args()
    if args.finish:
        if not args.first_output:parser.error('--first-output required for completed five-fold analysis')
        value=finish(args.output,args.first_output);print({k:v for k,v in value.items() if k not in ('folds','costs')})
    else:
        value=inspect(args.output,args.name);print(dict(epochs={v['launch']['row_id']:v['epochs'] for v in value['rows']},complete=value['complete'],failure=value['failure']))
