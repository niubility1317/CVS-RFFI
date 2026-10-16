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
        'training_sequence.jsonl.gz','held_V_predictions.json','origin_jitter_config.json','origin_jitter_proof.json',
        'F_implementation_proof.json','F_optimizer_steps.jsonl.gz']
    subprocess.run(['scp',*SSH,*['N607:'+row['output_root']+'/'+v for v in names],str(source)],check=True,timeout=60)
    subprocess.run(['scp',*SSH,'N607:'+row['log_path'],str(output)],check=True,timeout=60)

def analyze(output,spec,state,control_root):
    output=Path(output);control_root=Path(control_root);row=spec['rows'][0];source=output/row['row_id']/'source'
    if state['failure'] or state['dispatcher_live'] or any(v['live'] for v in state['rows']) or not state['complete']:raise ValueError('Terminal origin jitter experiment required')
    result=read(source/'result.json');control=read(control_root/'result.json');done=read(source/'completion.json');init=read(source/'initialization.json')
    freeze=read(source/'source_frozen.json');contract=read(source/'source_contract.json');proof=read(source/'origin_jitter_proof.json')
    fproof=read(source/'F_implementation_proof.json');actual=proof['actual']
    if (done['config']!=row['config'] or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch'
        or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
        or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact']
        or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
        or result['L_forward_packets']!=2273280 or result['training_loader_packets']!={'L_s':568320,'U_s':396368}
        or result['U_truth_read'] or result['target_access'] or result['head_fit_updates'] or not result['default_head_joint_training']
        or contract['held_RX']!=1 or contract['source_receivers']!=[3,4,6,8]
        or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}
        or actual['kind']!='CAUSAL_REFLECT_ORIGIN_JITTER' or actual['calls']!=4440 or actual['packets']!=568320
        or (actual['delay_min'],actual['delay_max'])!=(0,24) or len(actual['delay_counts'])!=25 or min(actual['delay_counts'])<=0
        or sum(actual['delay_counts'])!=568320 or actual['synthetic_prefix_max']!=24 or actual['real_observation_min']!=232
        or actual['hardware_changes'] or actual['labels_used'] or actual['circular_roll'] or actual['post_normalization'] or actual['all_other_operators']
        or proof['original_ordinary_calls'] or proof['As_calls'] or proof['raw_weight']!=.5 or proof['jitter_weight']!=.5 or not proof['same_IQ_within_RDpair']
        or fproof['steps']!=4440 or fproof['physical_clip_calls']!=4440 or fproof['optimizer']!='CENTER_UNITS'):
        raise ValueError('Actual fixed origin/F/source/default-head contract differs')
    control_done=read(control_root/'completion.json');control_contract=read(control_root/'source_contract.json')
    if control_contract['role_ids']!=contract['role_ids'] or control_done['config']['arm']!='A0' or control['L_forward_packets']!=2273280 or control['target_access']:
        raise ValueError('Exact matching historical source control required')
    native_args=read(source/'actual_native_args.json');native_args=native_args.get('args',native_args)
    control_args=read(control_root/'actual_native_args.json');control_args=control_args.get('args',control_args)
    differences=sorted(k for k in set(native_args)|set(control_args) if native_args.get(k)!=control_args.get(k))
    if any(k not in ('run_id','candidate_id','output_dir','use_aug') and not k.startswith('aug_') for k in differences):
        raise ValueError(('Unexpected native/U recipe difference',differences))
    mechanism=read(source/'mechanism_progress.json')['records']
    if len(mechanism)!=20 or any(v<=0 for v in result['actual_feature_parameters_moved'].values()):raise ValueError('F/default head must actually update')
    if {(v['TX'],v['RX'],v['day']) for v in result['coverage']}!={(t,r,d) for t in range(6) for r in (3,4,6,8) for d in (1,2,3)} or sum(v['count'] for v in result['coverage'])!=568320:
        raise ValueError('Full source training coverage required')
    with gzip.open(source/'F_optimizer_steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(v) for v in f]
    if [v['step'] for v in steps]!=list(range(1,4441)) or any(v['physical_clip_calls']!=v['step'] or not math.isclose(v['sinc_lr'],v['base_lr']*1000,rel_tol=1e-12) or v['sinc_weight_decay']!=0 for v in steps):raise ValueError('Complete F physical optimizer mapping required')
    logs=parse_logs(source,output/(row['row_id']+'.log'));curves=[dict(arm='OJ24',**v) for v in logs['curves']]
    control_logs=parse_logs(control_root,control_root.parents[1]/'logs'/(control_root.parent.name+'.log'))
    curves += [dict(arm='A0',**v) for v in control_logs['curves']]
    predictions={};strata=[];margins={}
    for arm,path,v in (('A0',control_root,control),('OJ24',source,result)):
        pred=read(path/'held_V_predictions.json')['records'];keys=[tuple(z[k] for k in ('eq','RX','day','sig_i','base_index')) for z in pred]
        if len(pred)!=5400 or len(set(keys))!=5400 or any(z['RX']!=1 for z in pred):raise ValueError('Unique heldV5400 required')
        recount=metrics(pred)
        if recount['confusion']!=v['held_V']['confusion'] or abs(recount['accuracy']-v['held_V']['accuracy'])>1e-12 or abs(recount['macro_f1']-v['held_V']['macro_f1'])>1e-12:raise ValueError('Independent default-head recount differs')
        predictions[arm]=dict(zip(keys,pred))
        if any(not math.isfinite(z['margin']) for z in pred):raise ValueError('Finite scale-normalized default-head margins required')
        margins[arm]=dict(count=len(pred),mean=sum(z['margin'] for z in pred)/len(pred),definition='(true logit - strongest wrong logit)/30; original default head')
        for dim,values in (('TX',range(6)),('day',(1,2,3)),('TX_day',[f'{t}/{d}' for t in range(6) for d in (1,2,3)])):
            for value in values:
                selected=[z for z in pred if (z['label'] if dim=='TX' else z['day'] if dim=='day' else f"{z['label']}/{z['day']}")==value]
                strata.append(dict(arm=arm,dimension=dim,value=value,**metrics(selected),mean_margin_over30=sum(z['margin'] for z in selected)/len(selected)))
    if predictions['A0'].keys()!=predictions['OJ24'].keys() or any(z['label']!=predictions['A0'][k]['label'] for k,z in predictions['OJ24'].items()):raise ValueError('Matched held physical IDs/truth required')
    repaired=sum(z['prediction']==z['label'] and predictions['A0'][k]['prediction']!=z['label'] for k,z in predictions['OJ24'].items())
    broken=sum(z['prediction']!=z['label'] and predictions['A0'][k]['prediction']==z['label'] for k,z in predictions['OJ24'].items())
    reference_correct=sum(z['prediction']==z['label'] for z in predictions['A0'].values())
    kept=reference_correct-broken
    count=0
    with gzip.open(source/'training_sequence.jsonl.gz','rt') as x,gzip.open(control_root/'training_sequence.jsonl.gz','rt') as y:
        for a,b in zip_longest(x,y):
            if a!=b:raise ValueError('All actual L/U physical sequences differ')
            count+=1
    if count!=5994:raise ValueError('Complete 4440L+1554U batch trace required')
    h,b=result['held_V'],control['held_V'];acc=100*(h['accuracy']-b['accuracy']);f1=100*(h['macro_f1']-b['macro_f1'])
    drop=100*(control['seen_V']['source_val_accuracy']-result['seen_V']['source_val_accuracy'])
    checks=dict(held_accuracy=acc>=1.-1e-9,held_macroF1=f1>=-1e-9,seen_health=drop<=.5+1e-9,complete_health=bool(result['complete_health'] and control['complete_health']))
    q=state['source_decision']
    if q['checks']!=checks or q['status']!=('PASS' if all(checks.values()) else 'FAIL'):raise ValueError('Independent qualification differs')
    costs={arm:sum(v[k] for k in ('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets')) for arm,v in (('A0',control),('OJ24',result))}
    summary=dict(status='ANALYZED',source_decision=q,models=dict(A0=control,OJ24=result),paired=dict(repaired=repaired,broken=broken,
        fixed_A0_clean_correct_count=reference_correct,retained=kept,retention=kept/reference_correct),default_head_margins=margins,
        full_log_proof=dict(A0=control_logs['proof'],OJ24=logs['proof']),actual_sequence_batch_records=count,new_model_epochs=20,new_updates=4440,
        new_real_forward_packets=costs['OJ24'],reused_control_model_epochs=20,reused_control_updates=4440,reused_control_real_forward_packets=costs['A0'],
        origin_jitter_proof=proof,actual_native_arg_differences=differences,matrix_wall_seconds=state['complete']['wall_seconds'],model_seed_count=1,source_fold_count=1,target_access=False,goal_completed=False,automatic_extension=False)
    for name,rows in (('training_curves.csv',curves),('source_strata.csv',strata)):
        fields=list(dict.fromkeys(k for v in rows for k in v))
        with (output/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    # Use this two-arm figure directly; no synthetic third result.
    plot(output,curves,strata,dict(A0=control_logs['epochs'],OJ24=logs['epochs']))
    plot_origin(output,actual,mechanism,result,control)
    write(output/'analysis_summary.json',summary)
    lines=['# F/E20：有界观察原点抖动的源 RX1 结果','',f"源规则 **{q['status']}**，0 目标访问；一个新 scratch E20，A0 结果复用，未加载其权重。",'',
        '|模型|held RX1 准确率|宏平均 F1|seen V 均值|seen V 最差 RX|','|---|---:|---:|---:|---:|']
    for arm,v in (('A0',control),('OJ24',result)):lines.append(f"|{arm}|{100*v['held_V']['accuracy']:.6f}%|{100*v['held_V']['macro_f1']:.6f}%|{100*v['seen_V']['source_val_accuracy']:.6f}%|{100*v['seen_V']['source_val_worst_rx']:.6f}%|")
    lines+=['',f'OJ24−A0 准确率 {acc:+.6f} 个百分点，宏平均 F1 {f1:+.6f} 个百分点；修复 {repaired}／新增错误 {broken}；seen V 损失 {drop:+.6f} 个百分点。', '',
        f"新 20 epoch／4440 step／native CSV／完整 stdout 与复用 A0 日志完整解析，全部 {count} 个 L/U 批次逐条相同。25 种偏移全部覆盖，568320 个实际新视图。原 ordinary／As 调用均为 0。",'',
        f"新成本 20 模型轮／4440 更新／{costs['OJ24']} 真实网络前向，墙钟 {summary['matrix_wall_seconds']:.3f} s；复用控制 20 模型轮／4440 更新／{costs['A0']} 前向另列。本地合成检查另计。",'',
        '一个已使用过的源 RX 折、一个模型 seed，不能替代完整 DG、正式 clean/practical 或跨 seed 证据。失败关闭 OJ24，不扫描偏移范围、概率或更换 RX/seed 救援；通过只支持另登记固定其他 4 个源 RX，不直接进入 target 或 As。', '',
        'OJ24 是 causal-reflect 观察原点变化；前 d 点是人工反射过去，末 d 点被截去，不称为真实长记录、未来补齐或循环平移。无后归一化，窗口功率可以改变。所有合法有标签视图均参与原默认头监督，F 与头联合训练。完整曲线和逐 TX/day 正负结果保存，预算耗尽不等于收敛。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary

def plot(output,curves,strata,epochs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'A0':'#386cb0','OJ24':'#4c956c'}
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

def plot_origin(output,actual,mechanism,result,control):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    axes[0].bar(range(25),actual['delay_counts'],color='#4c956c')
    axes[0].set(xlabel='Causal reflect delay (samples)',ylabel='Generated L views',title='Observed delay coverage')
    x=[v['epoch'] for v in mechanism]
    for key,label in [('window_energy_ratio_min','Minimum'),('window_energy_ratio_max','Maximum')]:
        axes[1].plot(x,[v['ordinary'][key] for v in mechanism],label=label)
    axes[1].axhline(1.,ls='--',lw=.8,color='grey');axes[1].legend()
    axes[1].set(xlabel='Epoch',ylabel='Cumulative output/input window energy ratio',title='No post normalization')
    costs=('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets')
    bottom=[0.,0.]
    for key in costs:
        values=[control[key]/1e6,result[key]/1e6]
        axes[2].bar(['Reused A0','New OJ24'],values,bottom=bottom,label=key)
        bottom=[a+b for a,b in zip(bottom,values)]
    axes[2].set(ylabel='Real packet forwards (millions)',title='Training / teacher / source evaluation')
    axes[2].legend(fontsize=6)
    fig.savefig(output/'origin_and_cost.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--control-root',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    spec=read(ROOT/'experiments/cvs_origin_jitter20/registration.json')
    if not a.cached:collect(a.output,spec)
    v=analyze(a.output,spec,read(a.state),a.control_root);print(json.dumps({k:z for k,z in v.items() if k not in ('models','full_log_proof')},ensure_ascii=False))
