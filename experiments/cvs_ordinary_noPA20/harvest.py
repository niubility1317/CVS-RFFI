"""Complete source training analysis without native/IQ/checkpoint imports."""
import argparse
import csv
import gzip
from itertools import zip_longest
from pathlib import Path
import subprocess
import numpy as np
from experiments.cvs_ordinary_fold20.harvest import read,write,metrics,parse_logs,SSH
ROOT=Path(__file__).resolve().parents[2]

def collect(output,spec):
    output=Path(output);row=spec['rows'][0];source=output/row['row_id']/'source';source.mkdir(parents=True,exist_ok=True)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json',
        'module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz',
        'training_sequence.jsonl.gz','held_V_predictions.json','ordinary_base_config.json','PA_removal_proof.json']
    subprocess.run(['scp',*SSH,*['N607:'+row['config']['output_root']+'/'+v for v in names],str(source)],check=True,timeout=60)
    subprocess.run(['scp',*SSH,'N607:'+spec['execution']['remote_run_root']+'/worker.log',str(output)],check=True,timeout=60)

def analyze(output,state,prior_output):
    output=Path(output);prior_output=Path(prior_output);spec=read(ROOT/'experiments/cvs_ordinary_noPA20/registration.json');row=spec['rows'][0]
    if state['failure'] or state['worker'] or state['dispatcher'] or not state['completion']:raise ValueError('Terminal N2 required')
    source=output/row['row_id']/'source';result=read(source/'result.json');done=read(source/'completion.json');init=read(source/'initialization.json')
    freeze=read(source/'source_frozen.json');contract=read(source/'source_contract.json');pa=read(source/'PA_removal_proof.json')
    if (done['config']!=row['config'] or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch'
        or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
        or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only'
        or result['target_access'] or result['U_truth_read'] or result['head_fit_updates'] or not result['default_head_joint_training']
        or result['L_forward_packets']!=1136640 or result['training_loader_packets']!={'L_s':568320,'U_s':396368}
        or contract['held_RX']!=1 or contract['source_receivers']!=[3,4,6,8]
        or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}
        or pa['actual_PA_calls']!=0 or pa['actual_aug_enable_pa_normal'] or not pa['guard_enabled'] or pa['ordinary_calls']!=4440):
        raise ValueError('Fixed defaulthead/scratch20/source/NoPA contract differs')
    if {(v['TX'],v['RX'],v['day']) for v in result['coverage']}!={(t,r,d) for t in range(6) for r in (3,4,6,8) for d in (1,2,3)} or sum(v['count'] for v in result['coverage'])!=568320:
        raise ValueError('Complete training coverage required')
    predictions=read(source/'held_V_predictions.json')['records'];keys=[tuple(v[k] for k in ('eq','RX','day','sig_i','base_index')) for v in predictions]
    if len(predictions)!=5400 or len(set(keys))!=5400 or any(v['RX']!=1 for v in predictions):raise ValueError('Exact heldRX1 V5400 required')
    recount=metrics(predictions)
    if recount['confusion']!=result['held_V']['confusion'] or abs(recount['macro_f1']-result['held_V']['macro_f1'])>1e-12:raise ValueError('Default-head source recount differs')
    log=parse_logs(source,output/'worker.log');prior=read(prior_output/'analysis_summary.json');models=dict(prior['models'],N2=result);strata=[];curves=[]
    with (prior_output/'source_strata.csv').open(encoding='utf-8',newline='') as f:
        for v in csv.DictReader(f):strata.append(dict(arm=v['arm'],dimension=v['dimension'],value=int(v['value']) if v['dimension']!='TX_day' else v['value'],accuracy=float(v['accuracy']),macro_f1=float(v['macro_f1']),count=int(v['count'])))
    with (prior_output/'training_curves.csv').open(encoding='utf-8',newline='') as f:
        for v in csv.DictReader(f):curves.append({k:(v[k] if k=='arm' else float(v[k]) if v[k] else None) for k in v})
    curves += [dict(arm='N2',**v) for v in log['curves']]
    for dimension,values in (('TX',range(6)),('day',(1,2,3))):
        for value in values:strata.append(dict(arm='N2',dimension=dimension,value=value,**metrics([v for v in predictions if v['label' if dimension=='TX' else 'day']==value])))
    for tx in range(6):
        for day in (1,2,3):
            selected=[v for v in predictions if v['label']==tx and v['day']==day]
            if len(selected)!=300:raise ValueError('Held TX/day count differs')
            strata.append(dict(arm='N2',dimension='TX_day',value=f'{tx}/{day}',**metrics(selected)))
    a=read(source/'actual_native_args.json');a=a.get('args',a);b=read(prior_output/'N1-RX1-s2026092701/source/actual_native_args.json');b=b.get('args',b)
    changed=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
    if set(changed)-{'run_id','candidate_id','output_dir','aug_enable_pa_normal'} or a['aug_enable_pa_normal'] or not a['use_aug']:
        raise ValueError(('Unexpected actual training differences',changed))
    files=[source/'training_sequence.jsonl.gz']+[prior_output/(arm+'-RX1-s2026092701')/'source/training_sequence.jsonl.gz' for arm in ('N0','N1')];count=0
    with gzip.open(files[0],'rt') as x,gzip.open(files[1],'rt') as y,gzip.open(files[2],'rt') as z:
        for a1,b1,c1 in zip_longest(x,y,z):
            if a1!=b1 or a1!=c1:raise ValueError('All actual N2/N0/N1 physical sequences must match')
            count+=1
    paired={}
    n2={k:v for k,v in zip(keys,predictions)}
    for arm in ('N0','N1'):
        rows=read(prior_output/(arm+'-RX1-s2026092701')/'source/held_V_predictions.json')['records'];other={tuple(v[k] for k in ('eq','RX','day','sig_i','base_index')):v for v in rows}
        if other.keys()!=n2.keys() or any(other[k]['label']!=v['label'] for k,v in n2.items()):raise ValueError('Same held physical samples/labels required')
        paired[arm]=dict(repaired=sum(v['prediction']==v['label'] and other[k]['prediction']!=v['label'] for k,v in n2.items()),
            broken=sum(v['prediction']!=v['label'] and other[k]['prediction']==v['label'] for k,v in n2.items()))
    n0,n1=models['N0'],models['N1'];v=result['held_V']
    checks=dict(vs_N0_accuracy=100*(v['accuracy']-n0['held_V']['accuracy'])>=1.-1e-9,vs_N0_macroF1=v['macro_f1']>=n0['held_V']['macro_f1']-1e-12,
        seen_health=100*(n0['seen_V']['source_val_accuracy']-result['seen_V']['source_val_accuracy'])<=.5+1e-9,
        N1_accuracy_retained=100*(n1['held_V']['accuracy']-v['accuracy'])<=.5+1e-9,N1_macroF1_retained=100*(n1['held_V']['macro_f1']-v['macro_f1'])<=.5+1e-9,
        complete_health=bool(result['complete_health']))
    q=state['source_decision']
    if q['checks']!=checks or q['status']!=('PASS' if all(checks.values()) else 'FAIL'):raise ValueError('Frozen source qualification independently differs')
    packets=sum(result[k] for k in ('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets'))
    summary=dict(status='ANALYZED',source_decision=q,models=models,actual_native_arg_differences=changed,full_log_proof=log['proof'],
        paired=paired,actual_sequence_batch_records=count,new_model_epochs=20,new_updates=4440,new_real_forward_packets=packets,
        reused_control_model_epochs=40,reused_control_updates=8880,reused_control_real_forward_packets=prior['real_forward_packets'],
        matrix_wall_seconds=state['completion']['wall_seconds'],model_seed_count=1,source_fold_count=1,target_access=False,goal_completed=False,
        PA_removal_proof=pa,automatic_extension=False)
    for name,rows in (('source_strata.csv',strata),('training_curves.csv',curves)):
        fields=list(dict.fromkeys(k for v in rows for k in v))
        with (output/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write(output/'analysis_summary.json',summary);plot(output,models,curves,strata,log,prior_output)
    lines=['# 普通增强关闭 PA：RX1 独立 E20 训练结果','',f"源资格：**{q['status']}**。一个新scratchE20，复用N0/N1结果不继承权重。",'',
        '|模型|heldRX1准确率|macroF1|seenV均值|seenV最差RX|','|---|---:|---:|---:|---:|']
    for arm,m in models.items():lines.append(f"|{arm}|{100*m['held_V']['accuracy']:.6f}%|{100*m['held_V']['macro_f1']:.6f}%|{100*m['seen_V']['source_val_accuracy']:.6f}%|{100*m['seen_V']['source_val_worst_rx']:.6f}%|")
    lines+=['',f"固定门槛：{q['checks']}。N2−N0 accuracy {q['vs_N0_accuracy_gain_pp']:+.6f}pp；N2−N1 {q['vs_N1_accuracy_gain_pp']:+.6f}pp。",'',
        f"新20epoch/4440step/nativeCSV与完整stdout已解析；全部{count}个真实L/U批次与两旧控制逐条匹配。PA guard实际零调用，其余ordinary保留。真实网络前向{packets}，矩阵墙钟{summary['matrix_wall_seconds']:.3f}s；原控制40modelepoch/8880update/4819956前向另列，本地512合成网络前向另计。",'',
        '源only，0target/新头/新U机制，heldL不入训练；只在FINAL评价源heldV。默认头决策与macroF1独立复算，逐TX/day正负结果完整保存。E20预算用尽不是收敛。', '',
        'PA输入诊断PASS不能替代本训练结果；本单fold也不能证明完整Ag或clean/urban双收益，跨seed未验证。FAIL按原规则结束NoPA候选，PASS仅允许另登记其余固定四RX；不自动进入target或混合训练。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary

def plot(output,models,curves,strata,new_log,prior_output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'N0':'#386cb0','N1':'#e66101','N2':'#4c956c'}
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for ax,key in zip(axes.flat,('legacy_rdrop/paired_CE','legacy_rdrop/symmetric_KL','train/grad_before_clip','train/pseudo_selected','val_tx_acc','lr')):
        for arm in ('N0','N1','N2'):
            if key in ('val_tx_acc','lr'):
                if arm=='N2':rows=new_log['epochs']
                else:rows=[__import__('json').loads(v) for v in (prior_output/(arm+'-RX1-s2026092701')/'source/epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            else:rows=[v for v in curves if v['arm']==arm]
            ax.plot([v['epoch'] for v in rows],[v.get(key,0) for v in rows],label=arm,color=colors[arm])
        ax.axvline(13.5,color='grey',ls='--',lw=.8);ax.set(xlabel='Epoch',ylabel=key);ax.legend()
    fig.savefig(output/'complete_training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,dimension in zip(axes,('TX','day')):
        for arm in ('N0','N1','N2'):
            rows=[v for v in strata if v['arm']==arm and v['dimension']==dimension];ax.plot([v['value'] for v in rows],[100*v['accuracy'] for v in rows],marker='o',label=arm,color=colors[arm])
        ax.set(xlabel=dimension,ylabel='Held RX1 accuracy (%)');ax.legend()
    fig.savefig(output/'held_source_strata.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--prior-output',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    spec=read(ROOT/'experiments/cvs_ordinary_noPA20/registration.json')
    if not a.cached:collect(a.output,spec)
    value=analyze(a.output,read(a.state),a.prior_output);print({k:v for k,v in value.items() if k not in ('models','full_log_proof')})
