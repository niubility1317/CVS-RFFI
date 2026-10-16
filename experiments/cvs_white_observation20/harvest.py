"""Offline source-only results; no training/IQ/checkpoint imports."""
import argparse,csv,gzip,json,math,subprocess
from itertools import zip_longest
from pathlib import Path
from experiments.cvs_ordinary_fold20.harvest import read,write,metrics,parse_logs,SSH
ROOT=Path(__file__).resolve().parents[2]

def collect(output,spec):
    output=Path(output);row=spec['rows'][0];source=output/row['row_id']/'source';source.mkdir(parents=True,exist_ok=True)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json',
        'module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz',
        'training_sequence.jsonl.gz','held_V_predictions.json','observation_noise_config.json','white_observation_proof.json',
        'F_implementation_proof.json','F_optimizer_steps.jsonl.gz']
    subprocess.run(['scp',*SSH,*['N607:'+row['output_root']+'/'+v for v in names],str(source)],check=True,timeout=60)
    subprocess.run(['scp',*SSH,'N607:'+row['log_path'],str(output)],check=True,timeout=60)

def analyze(output,spec,state,control_root):
    output=Path(output);control_root=Path(control_root);row=spec['rows'][0];source=output/row['row_id']/'source'
    if state['failure'] or state['dispatcher_live'] or any(v['live'] for v in state['rows']) or not state['complete']:raise ValueError('Terminal white experiment required')
    result=read(source/'result.json');control=read(control_root/'result.json');done=read(source/'completion.json');init=read(source/'initialization.json')
    freeze=read(source/'source_frozen.json');contract=read(source/'source_contract.json');proof=read(source/'white_observation_proof.json')
    fproof=read(source/'F_implementation_proof.json');actual=proof['actual']
    if (done['config']!=row['config'] or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch'
        or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
        or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact']
        or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
        or result['L_forward_packets']!=2273280 or result['training_loader_packets']!={'L_s':568320,'U_s':396368}
        or result['U_truth_read'] or result['target_access'] or result['head_fit_updates'] or not result['default_head_joint_training']
        or contract['held_RX']!=1 or contract['source_receivers']!=[3,4,6,8]
        or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}
        or actual['kind']!='WHITE_FIXED_ENERGY' or actual['calls']!=4440 or actual['packets']!=568320
        or not 29.999<=actual['signal_to_added_noise_db_min']<=30.001 or not 29.999<=actual['signal_to_added_noise_db_max']<=30.001
        or actual['hardware_changes'] or actual['labels_used'] or actual['total_SNR_claim']
        or proof['whole_ordinary_calls'] or proof['As_calls'] or proof['fixed_raw_weight']!=.5 or proof['white_weight']!=.5
        or fproof['steps']!=4440 or fproof['physical_clip_calls']!=4440 or fproof['optimizer']!='CENTER_UNITS'):
        raise ValueError('Actual fixed white/F/source/default-head contract differs')
    control_done=read(control_root/'completion.json');control_contract=read(control_root/'source_contract.json')
    if control_contract['role_ids']!=contract['role_ids'] or control_done['config']['arm']!='A0' or control['L_forward_packets']!=2273280 or control['target_access']:
        raise ValueError('Exact matching historical source control required')
    if {(v['TX'],v['RX'],v['day']) for v in result['coverage']}!={(t,r,d) for t in range(6) for r in (3,4,6,8) for d in (1,2,3)} or sum(v['count'] for v in result['coverage'])!=568320:
        raise ValueError('Full source training coverage required')
    with gzip.open(source/'F_optimizer_steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(v) for v in f]
    if [v['step'] for v in steps]!=list(range(1,4441)) or any(v['physical_clip_calls']!=v['step'] or not math.isclose(v['sinc_lr'],v['base_lr']*1000,rel_tol=1e-12) or v['sinc_weight_decay']!=0 for v in steps):raise ValueError('Complete F physical optimizer mapping required')
    logs=parse_logs(source,output/(row['row_id']+'.log'));curves=[dict(arm='W30',**v) for v in logs['curves']]
    control_logs=parse_logs(control_root,control_root.parents[1]/'logs'/(control_root.parent.name+'.log'))
    curves += [dict(arm='A0',**v) for v in control_logs['curves']]
    predictions={};strata=[]
    for arm,path,v in (('A0',control_root,control),('W30',source,result)):
        pred=read(path/'held_V_predictions.json')['records'];keys=[tuple(z[k] for k in ('eq','RX','day','sig_i','base_index')) for z in pred]
        if len(pred)!=5400 or len(set(keys))!=5400 or any(z['RX']!=1 for z in pred):raise ValueError('Unique heldV5400 required')
        recount=metrics(pred)
        if recount['confusion']!=v['held_V']['confusion'] or abs(recount['macro_f1']-v['held_V']['macro_f1'])>1e-12:raise ValueError('Independent default-head recount differs')
        predictions[arm]=dict(zip(keys,pred))
        for dim,values in (('TX',range(6)),('day',(1,2,3)),('TX_day',[f'{t}/{d}' for t in range(6) for d in (1,2,3)])):
            for value in values:
                selected=[z for z in pred if (z['label'] if dim=='TX' else z['day'] if dim=='day' else f"{z['label']}/{z['day']}")==value]
                strata.append(dict(arm=arm,dimension=dim,value=value,**metrics(selected)))
    if predictions['A0'].keys()!=predictions['W30'].keys() or any(z['label']!=predictions['A0'][k]['label'] for k,z in predictions['W30'].items()):raise ValueError('Matched held physical IDs/truth required')
    repaired=sum(z['prediction']==z['label'] and predictions['A0'][k]['prediction']!=z['label'] for k,z in predictions['W30'].items())
    broken=sum(z['prediction']!=z['label'] and predictions['A0'][k]['prediction']==z['label'] for k,z in predictions['W30'].items())
    count=0
    with gzip.open(source/'training_sequence.jsonl.gz','rt') as x,gzip.open(control_root/'training_sequence.jsonl.gz','rt') as y:
        for a,b in zip_longest(x,y):
            if a!=b:raise ValueError('All actual L/U physical sequences differ')
            count+=1
    h,b=result['held_V'],control['held_V'];acc=100*(h['accuracy']-b['accuracy']);f1=100*(h['macro_f1']-b['macro_f1'])
    drop=100*(control['seen_V']['source_val_accuracy']-result['seen_V']['source_val_accuracy'])
    checks=dict(held_accuracy=acc>=1.-1e-9,held_macroF1=f1>=-1e-9,seen_health=drop<=.5+1e-9,complete_health=bool(result['complete_health'] and control['complete_health']))
    q=state['source_decision']
    if q['checks']!=checks or q['status']!=('PASS' if all(checks.values()) else 'FAIL'):raise ValueError('Independent qualification differs')
    costs={arm:sum(v[k] for k in ('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets')) for arm,v in (('A0',control),('W30',result))}
    summary=dict(status='ANALYZED',source_decision=q,models=dict(A0=control,W30=result),paired=dict(repaired=repaired,broken=broken),
        full_log_proof=dict(A0=control_logs['proof'],W30=logs['proof']),actual_sequence_batch_records=count,new_model_epochs=20,new_updates=4440,
        new_real_forward_packets=costs['W30'],reused_control_model_epochs=20,reused_control_updates=4440,reused_control_real_forward_packets=costs['A0'],
        white_observation_proof=proof,matrix_wall_seconds=state['complete']['wall_seconds'],model_seed_count=1,source_fold_count=1,target_access=False,goal_completed=False,automatic_extension=False)
    for name,rows in (('training_curves.csv',curves),('source_strata.csv',strata)):
        fields=list(dict.fromkeys(k for v in rows for k in v))
        with (output/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    # Use this two-arm figure directly; no synthetic third result.
    plot(output,curves,strata,dict(A0=control_logs['epochs'],W30=logs['epochs']))
    write(output/'analysis_summary.json',summary)
    lines=['# F/E20：有界接收观测噪声源RX1结果','',f"源资格 **{q['status']}**，0目标访问；一个新scratch20，A0结果复用不载权重。",'',
        '|模型|heldRX1准确率|MacroF1|seenV均值|seenV最差RX|','|---|---:|---:|---:|---:|']
    for arm,v in (('A0',control),('W30',result)):lines.append(f"|{arm}|{100*v['held_V']['accuracy']:.6f}%|{100*v['held_V']['macro_f1']:.6f}%|{100*v['seen_V']['source_val_accuracy']:.6f}%|{100*v['seen_V']['source_val_worst_rx']:.6f}%|")
    lines+=['',f'W30−A0 accuracy {acc:+.6f}pp，macroF1 {f1:+.6f}pp；修复{repaired}/破坏{broken}；seenV损失{drop:+.6f}pp。', '',
        f"新20epoch/4440step/nativeCSV/fullstdout与复用A0日志完整解析，全部{count}个L/U批次逐条相同。实际新增噪声比{actual['signal_to_added_noise_db_min']:.6f}至{actual['signal_to_added_noise_db_max']:.6f}dB，wholeordinary/As调用0。",'',
        f"新20模型轮/4440updates/{costs['W30']}真实前向，墙钟{summary['matrix_wall_seconds']:.3f}s；复用控制20模型轮/4440updates/{costs['A0']}前向另列。本地合成检查另计。",'',
        '一个已使用过的源RX折、一个modelseed，不能替代完整DG/正式clean/practical/跨seed。FAIL关闭WHITE30且不扫SNR/概率或换RX/seed救援；PASS只支持另登记固定其他4RX，不直接进target或As。', '',
        '固定能量Gaussian方向不等于精确AWGN；30dB只是新增噪声比，不是原录音总SNR。所有有标签视图参与CE，零功率包保留；默认头和F按原联合方式训练。完整曲线/逐TX/day正负结果保存，预算耗尽不等于收敛。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary

def plot(output,curves,strata,epochs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'A0':'#386cb0','W30':'#4c956c'}
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for ax,key in zip(axes.flat,('legacy_rdrop/paired_CE','legacy_rdrop/symmetric_KL','train/grad_before_clip','train/pseudo_selected','val_tx_acc','lr')):
        for arm in colors:
            rows=epochs[arm] if key in ('val_tx_acc','lr') else [v for v in curves if v['arm']==arm]
            ax.plot([v['epoch'] for v in rows],[v.get(key,0) for v in rows],label=arm,color=colors[arm])
        ax.axvline(13.5,color='grey',ls='--',lw=.8);ax.set(xlabel='Epoch',ylabel=key);ax.legend()
    fig.savefig(output/'complete_training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,dim in zip(axes,('TX','day')):
        for arm in colors:
            rows=[v for v in strata if v['arm']==arm and v['dimension']==dim]
            ax.plot([v['value'] for v in rows],[100*v['accuracy'] for v in rows],marker='o',label=arm,color=colors[arm])
        ax.set(xlabel=dim,ylabel='Held source RX1 accuracy (%)');ax.legend()
    fig.savefig(output/'held_source_strata.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--control-root',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    spec=read(ROOT/'experiments/cvs_white_observation20/registration.json')
    if not a.cached:collect(a.output,spec)
    v=analyze(a.output,spec,read(a.state),a.control_root);print(json.dumps({k:z for k,z in v.items() if k not in ('models','full_log_proof')},ensure_ascii=False))
