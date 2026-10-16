"""Read saved source predictions/logs only; never opens IQ or checkpoints."""
import argparse,csv,gzip,json,math,re,subprocess
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
from experiments.cvs_ordinary_fold20.harvest import read,write,metrics,SSH

ROOT=Path(__file__).resolve().parents[2]
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
KEYS=('eq','RX','day','sig_i','base_index')

def collect(output,spec):
    output=Path(output)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json',
        'resolved_config.json','module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz',
        'training_sequence.jsonl.gz','held_V_predictions.json','ordinary_base_config.json','F_implementation_proof.json','F_optimizer_steps.jsonl.gz',
        'FcrossRX_pair_proof.json','supcon_steps.jsonl.gz','practical_channel_audit.json','source_practical_evaluation.json']
    names += [s+'_source_predictions.jsonl.gz' for s in SCENES[1:]]
    for row in spec['rows']:
        source=output/row['row_id']/'source';source.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*SSH,*['N607:'+row['output_root']+'/'+n for n in names],str(source)],check=True,timeout=60)
        (output/'logs').mkdir(exist_ok=True)
        subprocess.run(['scp',*SSH,'N607:'+row['log_path'],str(output/'logs')],check=True,timeout=60)

def full_logs(source,stdout):
    epochs=[json.loads(v) for v in (source/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    with (source/'metrics_epoch.csv').open(encoding='utf-8-sig',newline='') as f:native=list(csv.DictReader(f))
    with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(v) for v in f]
    with gzip.open(source/'supcon_steps.jsonl.gz','rt',encoding='utf-8') as f:supcon=[json.loads(v) for v in f]
    if [v['epoch'] for v in epochs]!=list(range(1,21)) or len(native)!=20 or [v['step'] for v in steps]!=list(range(1,4441)) or [v['step'] for v in supcon]!=list(range(1,4441)):
        raise ValueError('Whole E20/4440/native CSV/SupCon logs required')
    critical=('train/loss','train/grad_total','train/grad_backbone','train/grad_before_clip','legacy_rdrop/paired_CE','legacy_rdrop/symmetric_KL','F_supcon/loss','F_supcon/weight','pair/primary_CE','pair/secondary_CE')
    bad={k:sum(not isinstance(v['metrics'].get(k),(int,float)) or not math.isfinite(v['metrics'][k]) for v in steps) for k in critical}
    text=stdout.read_text(encoding='utf-8',errors='replace');errors={k:len(re.findall(k,text,re.I)) for k in ('Traceback','CUDA out of memory','FloatingPointError','Killed')}
    truth=max(v['metrics'].get('train/pseudo_truth_available',0) for v in steps)
    skipped={k:sum(v['metrics'].get(k,0) for v in steps) for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')}
    diffs=[]
    for s,aux in zip(steps,supcon):
        e=s['epoch'];m=s['metrics'];w=.1*min(max((e-5)/10.,0.),1.)
        if aux['epoch']!=e or not math.isclose(aux['weight'],w,abs_tol=1e-12) or not math.isclose(m['F_supcon/loss'],aux['loss'],abs_tol=1e-6) or aux['extra_model_forwards'] or aux['U_truth_read'] or aux['head_fit_updates']:
            raise ValueError('Original L-only contrastive schedule/readout differs')
        if bool(aux['details'])!=(e>5) or (aux['details'] and (len(aux['details'])!=4 or any(v['valid_anchors']<=0 for v in aux['details']))):
            raise ValueError('Actual two primary forwards/two levels and valid anchors required')
        if e<=13:diffs.append(abs(m['train/loss']-m['legacy_rdrop/paired_CE']-m['legacy_rdrop/symmetric_KL']-w*m['F_supcon/loss']))
    if any(bad.values()) or any(errors.values()) or truth or any(skipped.values()) or max(diffs)>1e-5:raise ValueError(('Full log anomaly',bad,errors,truth,skipped,max(diffs)))
    curves=[]
    for e in range(1,21):
        selected=[v['metrics'] for v in steps if v['epoch']==e];keys={k for v in selected for k,x in v.items() if isinstance(x,(int,float)) and math.isfinite(x)}
        means={k:float(np.mean([v[k] for v in selected if isinstance(v.get(k),(int,float)) and math.isfinite(v[k])])) for k in keys}
        curves.append(dict(epoch=e,steps=len(selected),**means))
    return dict(curves=curves,epochs=epochs,proof=dict(epochs=20,steps=4440,native_CSV_rows=20,SupCon_steps=4440,SupCon_applied=3330,
        critical_nonfinite=bad,stdout_errors=errors,U_truth_available_max=truth,skipped=skipped,max_label_objective_error=max(diffs),stdout_bytes=stdout.stat().st_size,
        warnings=[v for v in text.splitlines() if 'Warning' in v or 'WARNING' in v]))

def packet_key(v):return tuple(v[k] for k in KEYS)
def bucket(x,edges):
    if x is None:return 'undefined'
    if not math.isfinite(x):raise ValueError('Nonfinite serialized source metadata')
    for edge in edges:
        if x<edge:return '<'+str(edge)
    return '>='+str(edges[-1])
def groups(v):
    p=v.get('physics',{})
    yield 'TX',str(v['label']);yield 'day',str(v['day'])
    if not p:return
    for key in ('state_end','receiver_lock_status','agc_limited','boundary','residual_kernel_length','numerical_filter_latency_samples','phase_tracking_applied','iq_compensation_applied'):
        yield key,str(p[key])
    for key,edges in [('quality_snr_db',(-10,0,10,20,30)),('signal_to_added_noise_db',(-10,0,10,20,30)),('output_frequency_hz',(-10000,-1000,-100,0,100,1000,10000)),('residual_kernel_energy',(.01,.1,1,10))]:
        yield key,bucket(p[key],edges)

def check_predictions(records,saved,scene):
    keys=[packet_key(v) for v in records]
    if len(keys)!=5400 or len(set(keys))!=5400 or any(v['RX']!=1 or v['day'] not in (1,2,3) or not math.isfinite(v['margin']) for v in records):raise ValueError('Unique held RX1 V5400 required')
    v=metrics(records)
    if v['confusion']!=saved['confusion'] or not math.isclose(v['accuracy'],saved['accuracy'],abs_tol=1e-12) or not math.isclose(v['macro_f1'],saved['macro_f1'],abs_tol=1e-12):raise ValueError('Independent saved default-head recount differs')
    if scene!='clean' and any(z['physics']['scene']!=scene or z['physics']['channel_equalization_applied'] or z['physics']['true_total_snr_known'] or z['physics']['ordinary_before_satellite'] or z['physics']['boundary']!='synthetic_reflect' for z in records):raise ValueError('Full original practical condition differs')
    return dict(zip(keys,records))

def analyze(output,spec,state):
    output=Path(output)
    if state['failure'] or state['row_failures'] or state['dispatcher_live'] or any(v['live'] for v in state['rows']) or not state['complete'] or state['complete']['status']!='ARTIFACTS_COMPLETE':raise ValueError('Both source rows must finish before completed analysis')
    models={};predictions={};logs={};costs=[];curves=[];strata=[];fusion=[];arguments={}
    for row in spec['rows']:
        arm=row['config']['arm'];s=output/row['row_id']/'source';r=read(s/'result.json');done=read(s/'completion.json');init=read(s/'initialization.json');freeze=read(s/'source_frozen.json');contract=read(s/'source_contract.json')
        if (done['config']!=row['config'] or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
            or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
            or r['optimizer_updates']!=4440 or r['L_forward_packets']!=2273280 or r['training_loader_packets']!={'L_s':568320,'U_s':396368} or r['U_truth_read'] or r['target_access'] or r['head_fit_updates'] or not r['default_head_joint_training']
            or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):raise ValueError('Actual F sourcefold/defaulthead/scratchE20 contract differs')
        fp=read(s/'F_implementation_proof.json');pair=read(s/'FcrossRX_pair_proof.json');ev=read(s/'source_practical_evaluation.json');sat=arm=='C1'
        if fp['steps']!=4440 or fp['physical_clip_calls']!=4440 or fp['optimizer']!='CENTER_UNITS' or pair['pair_weights']!=[.5,.5] or pair['student_confidence_gate'] or not pair['all_L_views_supervised'] or pair['ordinary_before_satellite'] or ev['optimizer_updates'] or ev['EMA_updates'] or ev['packet_forwards']!=32400:raise ValueError('F/paired/sourceevaluation execution differs')
        if r['training_practical']['calls']!=(4440 if sat else 0) or r['training_practical']['packets']!=(568320 if sat else 0) or r['supcon']['applied_calls']!=3330 or r['supcon']['calls']!=4440 or not r['supcon']['only_primary_pair']:raise ValueError('Actual As/SupCon counts differ')
        with gzip.open(s/'F_optimizer_steps.jsonl.gz','rt',encoding='utf-8') as f:opts=[json.loads(v) for v in f]
        if [v['step'] for v in opts]!=list(range(1,4441)) or any(v['physical_clip_calls']!=v['step'] or not math.isclose(v['sinc_lr'],v['base_lr']*1000,rel_tol=1e-12) or v['sinc_weight_decay']!=0 for v in opts):raise ValueError('Complete F physical optimizer trajectory differs')
        if {(v['TX'],v['RX'],v['day']) for v in r['coverage']}!={(t,rx,day) for t in range(6) for rx in (3,4,6,8) for day in (1,2,3)} or sum(v['count'] for v in r['coverage'])!=568320:raise ValueError('Source exposure incomplete')
        coverage=r['TX_RX_scene_state_coverage']
        if sat and ({(v['TX'],v['RX'],v['scene']) for v in coverage}!={(t,rx,sc) for t in range(6) for rx in (3,4,6,8) for sc in SCENES[3:4]+SCENES[5:]} or {(v['TX'],v['RX'],v['state']) for v in coverage}!={(t,rx,st) for t in range(6) for rx in (3,4,6,8) for st in ('LOS','PARTIAL','BLOCKED')} or sum(v['count'] for v in coverage)!=568320):raise ValueError('TX x RX x scene/state coverage incomplete')
        logs[arm]=full_logs(s,output/'logs'/(row['row_id']+'.log'));curves.extend(dict(arm=arm,**v) for v in logs[arm]['curves'])
        arguments[arm]=read(s/'actual_native_args.json');arguments[arm]=arguments[arm].get('args',arguments[arm]);predictions[arm]={}
        for scene in SCENES:
            if scene=='clean':records=read(s/'held_V_predictions.json')['records'];saved=r['held_V']
            else:
                with gzip.open(s/(scene+'_source_predictions.jsonl.gz'),'rt',encoding='utf-8') as f:records=[json.loads(v) for v in f]
                saved=r['practical'][scene]
            predictions[arm][scene]=check_predictions(records,saved,scene);grouped=defaultdict(list)
            for v in records:
                for key,value in groups(v):grouped[(key,value)].append(v)
            ref=predictions['C0']['clean']
            for (dim,value),rr in sorted(grouped.items()):
                eligible=[z for z in rr if ref[packet_key(z)]['label']==ref[packet_key(z)]['prediction']]
                kept=sum(z['label']==z['prediction'] for z in eligible)
                strata.append(dict(arm=arm,scene=scene,dimension=dim,value=value,**metrics(rr),mean_margin_over30=float(np.mean([z['margin'] for z in rr])),
                    fixed_C0_clean_correct_count=len(eligible),fixed_retained=kept,fixed_retention=kept/len(eligible) if eligible else None))
        mechanism=read(s/'mechanism_progress.json')['records']
        if len(mechanism)!=20 or any(v<=0 for v in r['actual_feature_parameters_moved'].values()):raise ValueError('Actual F and defaulthead joint training required')
        for v in mechanism:
            g=v['default_fusion_gain'];fusion.append(dict(arm=arm,epoch=v['epoch'],mean=float(np.mean(g)),minimum=min(g),maximum=max(g),std=float(np.std(g))))
        costs.append(dict(arm=arm,parent_weight_epochs=0,epochs=20,updates=4440,**{k:r[k] for k in ('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets','source_practical_forward_packets','elapsed_seconds','source_practical_seconds')}))
        models[arm]=r
    args_diff=sorted(k for k in set(arguments['C0'])|set(arguments['C1']) if arguments['C0'].get(k)!=arguments['C1'].get(k))
    if set(args_diff)-{'candidate_id','output_dir'}:raise ValueError(('Unexpected native recipe difference',args_diff))
    paths=[output/r['row_id']/'source/training_sequence.jsonl.gz' for r in spec['rows']];n=0
    with gzip.open(paths[0],'rt') as a,gzip.open(paths[1],'rt') as b:
        for x,y in zip_longest(a,b):
            if x!=y:raise ValueError('Actual whole L/U physical sequence differs')
            n+=1
    if n!=5994:raise ValueError('Complete L/U sequence required')
    table=[];paired_strata=[];reference=predictions['C0']['clean'];fixed={k for k,v in reference.items() if v['label']==v['prediction']}
    for scene in SCENES:
        x,y=(predictions[a][scene] for a in ('C0','C1'))
        if x.keys()!=y.keys() or x.keys()!=reference.keys() or any(x[k]['label']!=y[k]['label'] or x[k]['label']!=reference[k]['label'] or x[k].get('physics')!=y[k].get('physics') for k in x):raise ValueError('Matched samples/truth/physical streams required')
        scores={a:metrics(list(predictions[a][scene].values())) for a in ('C0','C1')}
        repaired=sum(x[k]['prediction']!=x[k]['label'] and y[k]['prediction']==y[k]['label'] for k in x);broken=sum(x[k]['prediction']==x[k]['label'] and y[k]['prediction']!=y[k]['label'] for k in x)
        grouped=defaultdict(list)
        for k,v in x.items():
            for dim,value in groups(v):grouped[(dim,value)].append(k)
        for (dim,value),kk in sorted(grouped.items()):
            repair=sum(x[k]['prediction']!=x[k]['label'] and y[k]['prediction']==y[k]['label'] for k in kk)
            broke=sum(x[k]['prediction']==x[k]['label'] and y[k]['prediction']!=y[k]['label'] for k in kk)
            paired_strata.append(dict(scene=scene,dimension=dim,value=value,count=len(kk),repaired=repair,broken=broke,net=repair-broke,accuracy_gain_pp=100*(repair-broke)/len(kk)))
        for arm in ('C0','C1'):
            rr=predictions[arm][scene];kept=sum(rr[k]['label']==rr[k]['prediction'] for k in fixed)
            table.append(dict(arm=arm,scene=scene,**scores[arm],mean_margin_over30=float(np.mean([v['margin'] for v in rr.values()])),fixed_C0_clean_correct_count=len(fixed),fixed_retained=kept,fixed_retention=kept/len(fixed),
                C1_minus_C0_accuracy_pp=100*(scores['C1']['accuracy']-scores['C0']['accuracy']),C1_minus_C0_macroF1_pp=100*(scores['C1']['macro_f1']-scores['C0']['macro_f1']),repaired=repaired,broken=broken))
    # Recompute the preregistered source decision without importing training/design code.
    urban=lambda a,key:np.mean([models[a]['practical'][s][key] for s in SCENES[4:]])
    ug=100*(urban('C1','accuracy')-urban('C0','accuracy'));lg=100*(models['C1']['practical'][SCENES[-1]]['accuracy']-models['C0']['practical'][SCENES[-1]]['accuracy'])
    fg=100*(urban('C1','macro_f1')-urban('C0','macro_f1'));cd=100*(models['C0']['held_V']['accuracy']-models['C1']['held_V']['accuracy']);sd=100*(models['C0']['seen_V']['source_val_accuracy']-models['C1']['seen_V']['source_val_accuracy'])
    checks=dict(urban=ug>=3.-1e-9,low=lg>=5.-1e-9,urban_F1=fg>=-1e-9,clean=cd<=.5+1e-9,seen=sd<=.5+1e-9,complete_health=all(v['complete_health'] for v in models.values()))
    q=state['source_decision']
    if q['checks']!=checks or q['status']!=('PASS' if all(checks.values()) else 'FAIL'):raise ValueError('Independent source qualifier differs')
    cost_keys=('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets','source_practical_forward_packets')
    summary=dict(status='ANALYZED',source_decision=q,models=models,scene_results=table,actual_native_arg_differences=args_diff,full_log_proof={a:v['proof'] for a,v in logs.items()},costs=costs,
        real_forward_packets=sum(sum(v[k] for k in cost_keys) for v in costs),new_model_epochs=40,new_updates=8880,actual_sequence_batch_records=n,matrix_wall_seconds=state['complete']['wall_seconds'],model_seed_count=1,source_fold_count=1,target_access=False,formal_test_status='NOT_REGISTERED_SOURCE_ONLY',goal_completed=False,automatic_extension=False)
    for name,rows in [('scene_results.csv',table),('source_strata.csv',strata),('paired_strata.csv',paired_strata),('training_curves.csv',curves),('fusion_gain_curves.csv',fusion),('measured_costs.csv',costs)]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rows for k in v)));w.writeheader();w.writerows(rows)
    plot(output,table,strata,curves,logs,fusion);write(output/'analysis_summary.json',summary)
    lines=['# F＋跨 RX 同 TX：配对 practical 的源 RX1 结果','',f"源规则 **{q['status']}**；两组 scratch E20 完成，0 目标访问。",'',
        '|场景|C0 准确率|C1 准确率|差值 pp|C0 F1|C1 F1|','|---|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        a,b=[v for v in table if v['scene']==scene];lines.append(f"|{scene}|{100*a['accuracy']:.4f}%|{100*b['accuracy']:.4f}%|{b['C1_minus_C0_accuracy_pp']:+.4f}|{100*a['macro_f1']:.4f}%|{100*b['macro_f1']:.4f}%|")
    lines+=['',f'urban 均值 {ug:+.6f} pp，low {lg:+.6f} pp，urban F1 {fg:+.6f} pp；clean 损失 {cd:+.6f} pp，seen V 损失 {sd:+.6f} pp。', '',
        f"完整 40 epoch／8880 更新／全部 stdout、native CSV、SupCon 和物理梯度轨迹已解析。全 L/U 批次 {n} 条逐条一致；默认头逐包保存预测独立复算，75,600 个 held source 预测。同名实际信道元数据完全一致。",'',
        f"总真实网络前向 {summary['real_forward_packets']}，矩阵墙钟 {summary['matrix_wall_seconds']:.3f} s；实际源六 practical 评分耗时、U/EMA/训练/验证成本见 measured_costs.csv，父模型权重成本0。本地合成检查另计。",'',
        'C0=.5 RD(共同旧 ordinary)+.5 RD(同一 IQ)；C1=.5 RD(共同旧 ordinary)+.5 RD(practical(raw))。两组四次 L 前向，原浅/中层跨 RX SupCon 仅复用前两次；无新 Ag、无额外跨视图一致性、所有有标签 satellite 直接监督。旧 ordinary 不是已验证的新泛化增强。', '',
        '一个已使用过的源 RX1 折和一个模型 seed；不是正式目标测试或跨 seed 证据。低 urban 超过60%、clean 净收益、suburban 接近 clean 及联合 Ag/As 目标均须独立后续证据。预算用尽不等于收敛。固定配方 FAIL 不扫描强度、权重、场景或更换 RX/seed 救援；PASS 只支持另登记源确认，不自动进入目标测试。', '',
        '分层和固定 C0 clean 正确集合的保持率见 CSV。quality_snr_db 是白输入期望功率代理，signal_to_added_noise_db 只表示新增噪声，均不冒充原录音总 SNR。EQ 关闭、G=1，噪声未被 EQ 着色；固定 FIR 延迟与反射边界保持。状态分层是机制关联，不等于单因素因果归因。', '',
        '门控图记录实际默认融合 gain 参数随训练的变化；本矩阵没有额外逐样本分支/门控干预，不能将参数变化写成门控因果结论。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary

def plot(output,table,strata,curves,logs,fusion):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    keys=('pair/primary_CE','pair/secondary_CE','F_supcon/loss','train/grad_before_clip','legacy_rdrop/symmetric_KL','train/pseudo_selected','val_tx_acc','lr')
    for ax,key in zip(axes.flat,keys):
        for arm in ('C0','C1'):
            rr=logs[arm]['epochs'] if key in ('val_tx_acc','lr') else [v for v in curves if v['arm']==arm]
            ax.plot([v['epoch'] for v in rr],[v.get(key,0) for v in rr],label=arm)
        ax.axvline(5.5,ls=':',color='grey',lw=.8);ax.axvline(13.5,ls='--',color='grey',lw=.8);ax.set(xlabel='Epoch',ylabel=key);ax.legend()
    fig.savefig(output/'complete_training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for arm,offset in [('C0',-.18),('C1',.18)]:
        rr=[v for v in table if v['arm']==arm];x=np.arange(7)+offset
        axes[0].bar(x,[100*v['accuracy'] for v in rr],width=.35,label=arm);axes[1].bar(x,[100*v['fixed_retention'] for v in rr],width=.35,label=arm)
    for ax in axes:ax.set_xticks(range(7),['clean','high','mid','low sub','high urb','mid urb','low urb'],rotation=30);ax.legend()
    axes[0].set(ylabel='Held source RX1 accuracy (%)');axes[1].set(ylabel='Retention on fixed C0 clean-correct set (%)')
    fig.savefig(output/'source_scene_results.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for ax,dim in zip(axes.flat,('state_end','receiver_lock_status','quality_snr_db','signal_to_added_noise_db','TX','day')):
        rr=[v for v in strata if v['scene']=='practical_low_urban' and v['dimension']==dim];values=list(dict.fromkeys(v['value'] for v in rr))
        for arm in ('C0','C1'):
            lookup={v['value']:v for v in rr if v['arm']==arm};ax.plot(values,[100*lookup[v]['accuracy'] if v in lookup else np.nan for v in values],marker='o',label=arm)
        ax.set(xlabel=dim,ylabel='Low urban source accuracy (%)');ax.tick_params(axis='x',rotation=25);ax.legend()
    fig.savefig(output/'low_urban_strata.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm in ('C0','C1'):
        rr=[v for v in fusion if v['arm']==arm];axes[0].plot([v['epoch'] for v in rr],[v['mean'] for v in rr],label=arm)
        ss=[v for v in table if v['arm']==arm];axes[1].plot(range(7),[v['mean_margin_over30'] for v in ss],marker='o',label=arm)
    axes[0].set(xlabel='Epoch',ylabel='Mean actual default fusion gain');axes[1].set_xticks(range(7),['clean','high','mid','low sub','high urb','mid urb','low urb'],rotation=30);axes[1].set(ylabel='Default-head margin / 30')
    for ax in axes:ax.legend()
    fig.savefig(output/'fusion_and_margins.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    spec=read(ROOT/'experiments/cvs_f_crossrx_practical20/registration.json')
    if not a.cached:collect(a.output,spec)
    result=analyze(a.output,spec,read(a.state));print(json.dumps({k:v for k,v in result.items() if k not in ('models','full_log_proof','scene_results','costs')},ensure_ascii=False))
