"""Independent saved-source analysis. No training/model/IQ/checkpoint imports."""
import argparse,csv,gzip,json,math,subprocess
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
from experiments.cvs_f_crossrx_practical20.harvest import full_logs,check_predictions,packet_key,groups,SCENES,read,write,SSH
ROOT=Path(__file__).resolve().parents[2]
FIT=(3,4,6,8);ARMS=('SAME_RX','CROSS_RX');COST_KEYS=('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets','source_practical_forward_packets')

def collect(output,spec):
    output=Path(output);(output/'logs').mkdir(parents=True,exist_ok=True)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json','resolved_config.json','module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz','training_sequence.jsonl.gz','held_V_predictions.json','ordinary_base_config.json','F_implementation_proof.json','F_optimizer_steps.jsonl.gz','supcon_steps.jsonl.gz','source_practical_evaluation.json','real_pair_bank.json','real_pair_sequence.jsonl.gz','real_pair_proof.json']+[s+'_source_predictions.jsonl.gz' for s in SCENES[1:]]
    for row in spec['rows']:
        dest=output/row['row_id']/'source';dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*SSH,*['N607:'+row['output_root']+'/'+name for name in names],str(dest)],check=True,timeout=60)
        subprocess.run(['scp',*SSH,'N607:'+row['log_path'],str(output/'logs')],check=True,timeout=60)

def validate_pair_record(r,allowed,labels,mode):
    a,b=tuple(r['primary']),tuple(r['partner']);tx=r['TX']
    if (mode not in ARMS or r['mode']!=mode or a not in allowed or b not in allowed or a==b or tx not in range(6) or labels[a]!=tx or labels[b]!=tx
        or a[0]!=1 or b[0]!=1 or a[1] not in FIT or b[1] not in FIT or a[2]!=b[2] or a[2] not in (1,2,3)
        or ((a[1]==b[1])!=(mode=='SAME_RX'))):raise ValueError('Illegal/mismatched real partner observation')

def check_pair_trace(source,mode,saved):
    with gzip.open(source/'training_sequence.jsonl.gz','rt',encoding='utf-8') as f:primary=[json.loads(line)['ids'] for line in f if '"L_s"' in line]
    if len(primary)!=4440 or any(len(v)!=128 for v in primary):raise ValueError('Complete original L128 primary trace required')
    allowed={tuple(k) for batch in primary for k in batch};labels={};counts=Counter();used=set();calls=0
    if len(allowed)!=5040:raise ValueError('Only exact source training L5040 may be paired')
    with gzip.open(source/'real_pair_sequence.jsonl.gz','rt',encoding='utf-8') as f:
        for anchor,line in zip_longest(primary,f):
            if anchor is None or line is None:raise ValueError('One real-partner record per primary update required')
            r=json.loads(line);calls+=1
            if r['step']!=calls or len(r['records'])!=128 or [v['primary'] for v in r['records']]!=anchor:raise ValueError('Partner trace detached from actual primary physical order')
            for item in r['records']:
                a,b=tuple(item['primary']),tuple(item['partner']);tx=item['TX']
                if tx not in range(6) or labels.setdefault(a,tx)!=tx:raise ValueError('Only consistent true L labels permitted')
                counts[(tx,a[1],b[1],a[2])]+=1;used.add(b)
    if calls!=4440 or set(labels)!=allowed or used!=allowed:raise ValueError('Every legal L physical packet must be covered')
    with gzip.open(source/'real_pair_sequence.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            for r in json.loads(line)['records']:validate_pair_record(r,allowed,labels,mode)
    expected={(t,r,other,day) for t in range(6) for r in FIT for other in FIT for day in (1,2,3) if (r==other)==(mode=='SAME_RX')}
    recorded={(r['TX'],r['primary_RX'],r['partner_RX'],r['day']):r['count'] for r in saved['strata']}
    if set(counts)!=expected or dict(counts)!=recorded or saved['calls']!=4440 or saved['packets']!=568320 or saved['unique_partner_packets']!=5040 or saved['U_used'] or saved['V_used'] or saved['held_RX_used'] or saved['target_access']:raise ValueError('Complete legal partner coverage differs from runtime proof')
    return dict(status='VERIFIED',steps=calls,primary_physical_ids=5040,partner_physical_ids=5040,partner_exposures=568320,cells=len(counts),same_true_TX_day=True,distinct_physical_packet=True)

def qualify(models,recorded):
    c=models['CROSS_RX'];pairs={}
    for name in ('C0','SAME_RX'):
        b=models[name];pairs[name]=dict(accuracy_gain_pp=100*(c['held_V']['accuracy']-b['held_V']['accuracy']),macroF1_gain_pp=100*(c['held_V']['macro_f1']-b['held_V']['macro_f1']),seen_drop_pp=100*(b['seen_V']['source_val_accuracy']-c['seen_V']['source_val_accuracy']))
    checks=dict(vs_C0=pairs['C0']['accuracy_gain_pp']>=1.-1e-9,vs_same=pairs['SAME_RX']['accuracy_gain_pp']>=.5-1e-9,F1=all(r['macroF1_gain_pp']>=-1e-9 for r in pairs.values()),seen=all(r['seen_drop_pp']<=.5+1e-9 for r in pairs.values()),complete_health=all(r['complete_health'] for r in models.values()))
    if recorded['checks']!=checks or recorded['status']!=('PASS' if all(checks.values()) else 'FAIL') or not recorded['source_only'] or recorded['target_access'] or recorded['model_seeds']!=1:raise ValueError('Independent real-pair scientific decision differs')
    if any(not math.isclose(recorded['paired_comparisons'][a][k],v,abs_tol=1e-9) for a,z in pairs.items() for k,v in z.items()):raise ValueError('Independent real-pair gains differ')
    return recorded

def analyze(output,spec,state,reference_root,cached=False):
    output=Path(output);refroot=Path(reference_root)
    if state['dispatcher_live'] or any(r['live'] for r in state['rows']) or state['failure'] or state['row_failures'] or not state['completion'] or state['completion']['status']!='ARTIFACTS_COMPLETE':raise ValueError('Both actual source workers must finish before analysis')
    if not cached:collect(output,spec)
    rows={r['config']['arm']:r for r in spec['rows']};paths={a:output/r['row_id']/'source' for a,r in rows.items()};paths['C0']=refroot/'C0-RX1-s2026092701/source'
    configs={a:r['config'] for a,r in rows.items()};configs['C0']=read(ROOT/'experiments/cvs_f_crossrx_practical20/registration.json')['rows'][0]['config']
    models={};predictions={};logs={};curves=[];costs=[];strata=[];fusion=[];trace_proofs={};arguments={}
    for arm in ('C0',*ARMS):
        s=paths[arm];r=read(s/'result.json');done=read(s/'completion.json');init=read(s/'initialization.json');freeze=read(s/'source_frozen.json');contract=read(s/'source_contract.json')
        if (done['config']!=configs[arm] or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
            or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
            or r['optimizer_updates']!=4440 or r['L_forward_packets']!=2273280 or r['training_loader_packets']!={'L_s':568320,'U_s':396368} or r['U_truth_read'] or r['target_access'] or r['head_fit_updates'] or not r['default_head_joint_training']
            or contract['held_RX']!=1 or contract['source_receivers']!=list(FIT) or r['held_RX']!=1 or r['fit_RX']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):raise ValueError('Actual source roles/final20/defaulthead provenance differs')
        proof=read(s/'F_implementation_proof.json');ev=read(s/'source_practical_evaluation.json')
        if proof['steps']!=4440 or proof['physical_clip_calls']!=4440 or proof['optimizer']!='CENTER_UNITS' or ev['optimizer_updates'] or ev['EMA_updates'] or ev['packet_forwards']!=32400 or r['supcon']['calls']!=4440 or r['supcon']['applied_calls']!=3330 or not r['supcon']['only_primary_pair']:raise ValueError('Original F/primary SupCon/sourceevaluation differs')
        with gzip.open(s/'F_optimizer_steps.jsonl.gz','rt',encoding='utf-8') as f:opts=[json.loads(line) for line in f]
        if [v['step'] for v in opts]!=list(range(1,4441)) or any(v['physical_clip_calls']!=v['step'] or not math.isclose(v['sinc_lr'],v['base_lr']*1000,rel_tol=1e-12) or v['sinc_weight_decay']!=0 for v in opts):raise ValueError('Complete physical F updates required')
        if arm!='C0':
            pair=read(s/'real_pair_proof.json');bank=read(s/'real_pair_bank.json')
            if pair['status']!='VERIFIED' or pair['pair_weights']!=[.5,.5] or not pair['all_L_views_supervised'] or not pair['partner_has_own_true_RX_day_domains'] or bank['status']!='VERIFIED' or not bank['read_after_own_scratch_checkpoint_smoke'] or 'domain_label_map' not in bank['raw_domain_remap']:raise ValueError('Real partner/defaulthead execution differs')
            trace_proofs[arm]=check_pair_trace(s,arm,r['real_pair'])
        stdout=(refroot if arm=='C0' else output)/'logs'/((('C0-RX1-s2026092701') if arm=='C0' else rows[arm]['row_id'])+'.log')
        logs[arm]=full_logs(s,stdout);curves.extend(dict(arm=arm,**v) for v in logs[arm]['curves']);arguments[arm]=read(s/'actual_native_args.json');predictions[arm]={}
        for scene in SCENES:
            if scene=='clean':records=read(s/'held_V_predictions.json')['records'];saved=r['held_V']
            else:
                with gzip.open(s/(scene+'_source_predictions.jsonl.gz'),'rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
                saved=r['practical'][scene]
            predictions[arm][scene]=check_predictions(records,saved,scene,held_RX=1)
            grouped=defaultdict(list)
            for v in records:
                for dim,value in groups(v):grouped[(dim,value)].append(v)
            ref=predictions['C0']['clean']
            for (dim,value),rr in sorted(grouped.items()):
                eligible=[v for v in rr if ref[packet_key(v)]['prediction']==ref[packet_key(v)]['label']];kept=sum(v['label']==v['prediction'] for v in eligible)
                strata.append(dict(arm=arm,scene=scene,dimension=dim,value=value,**metrics(rr),mean_margin_over30=float(np.mean([v['margin'] for v in rr])),fixed_C0_clean_correct_count=len(eligible),fixed_retained=kept,fixed_retention=kept/len(eligible) if eligible else None))
        mechanism=read(s/'mechanism_progress.json')['records']
        if len(mechanism)!=20 or any(v<=0 for v in r['actual_feature_parameters_moved'].values()):raise ValueError('Complete normal joint training required')
        for v in mechanism:
            gain=v['default_fusion_gain'];fusion.append(dict(arm=arm,epoch=v['epoch'],mean=float(np.mean(gain)),minimum=min(gain),maximum=max(gain),std=float(np.std(gain))))
        cost=dict(arm=arm,reused_reference=arm=='C0',epochs=20,updates=4440,parent_weight_epochs=0,**{k:r[k] for k in (*COST_KEYS,'elapsed_seconds','source_practical_seconds')})
        cost.update(bank_IQ_reads=5040 if arm!='C0' else 0,partner_physical_exposures=568320 if arm!='C0' else 0,bank_read_seconds=r.get('partner_ordinary',{}).get('bank_read_seconds',0.),partner_view_construction_seconds=r.get('partner_ordinary',{}).get('partner_view_construction_seconds',0.))
        costs.append(cost);models[arm]=r
    for arm in ARMS:
        diff={k for k in set(arguments[arm])|set(arguments['C0']) if arguments[arm].get(k)!=arguments['C0'].get(k)}
        if diff-{'candidate_id','output_dir','run_id'}:raise ValueError(('Unchanged native recipe required',arm,diff))
    with gzip.open(paths['C0']/'training_sequence.jsonl.gz','rt') as a,gzip.open(paths['SAME_RX']/'training_sequence.jsonl.gz','rt') as b,gzip.open(paths['CROSS_RX']/'training_sequence.jsonl.gz','rt') as c:
        count=0
        for x,y,z in zip_longest(a,b,c):
            if x!=y or x!=z:raise ValueError('Actual primary L/U physical sequences differ from matching C0')
            count+=1
    if count!=5994:raise ValueError('All original L/U batches required')
    table=[];paired=[];reference=predictions['C0']['clean'];fixed={k for k,v in reference.items() if v['label']==v['prediction']}
    for scene in SCENES:
        base=predictions['C0'][scene]
        for arm in ('C0',*ARMS):
            rr=predictions[arm][scene]
            if rr.keys()!=reference.keys() or any(rr[k]['label']!=base[k]['label'] or rr[k].get('physics')!=base[k].get('physics') for k in rr):raise ValueError('Same source physical labels and practical stream required')
            kept=sum(rr[k]['label']==rr[k]['prediction'] for k in fixed);table.append(dict(arm=arm,scene=scene,**metrics(list(rr.values())),mean_margin_over30=float(np.mean([v['margin'] for v in rr.values()])),fixed_C0_clean_correct_count=len(fixed),fixed_retained=kept,fixed_retention=kept/len(fixed)))
        for name in ('C0','SAME_RX'):
            a,b=predictions[name][scene],predictions['CROSS_RX'][scene];grouped=defaultdict(list);grouped[('all','all')]=list(a)
            for k,v in a.items():
                for dim,value in groups(v):grouped[(dim,value)].append(k)
            for (dim,value),keys in sorted(grouped.items()):
                repair=sum(a[k]['prediction']!=a[k]['label'] and b[k]['prediction']==b[k]['label'] for k in keys);broken=sum(a[k]['prediction']==a[k]['label'] and b[k]['prediction']!=b[k]['label'] for k in keys)
                paired.append(dict(comparison='CROSS_RX-'+name,scene=scene,dimension=dim,value=value,count=len(keys),repaired=repair,broken=broken,accuracy_gain_pp=100*(repair-broken)/len(keys)))
    q=qualify(models,state['source_decision']);newcost=sum(sum(r[k] for k in COST_KEYS) for r in costs if not r['reused_reference'])
    result=dict(status='ANALYZED',source_decision=q,models=models,scene_results=table,full_log_proof={a:r['proof'] for a,r in logs.items()},real_pair_trace_proof=trace_proofs,costs=costs,
        new_real_forward_packets=newcost,new_model_epochs=40,new_updates=8880,reused_C0_epochs=20,reused_C0_updates=4440,actual_primary_sequence_records=count,matrix_wall_seconds=state['completion']['wall_seconds'],
        actual_training_commit=state['submit']['commit'],model_seed_count=1,source_fold_count=1,target_access=False,formal_test_status='NOT_REGISTERED_SOURCE_ONLY',goal_completed=False,automatic_extension=False)
    for name,rows in [('scene_results.csv',table),('source_strata.csv',strata),('paired_strata.csv',paired),('training_curves.csv',curves),('fusion_gain_curves.csv',fusion),('measured_costs.csv',costs)]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
    plot(output,table,curves,fusion,strata);write(output/'analysis_summary.json',result)
    lines=['# 真实同 TX 观测配对：源域完整结果','',f"固定源规则 **{q['status']}**，一个model seed／一个held源RX1，0target。C0仅复用原结果，两个新模型从零E20。",'',
        '|scene|C0 accuracy|SAME_RX|CROSS_RX|CROSS−C0 pp|CROSS−SAME pp|','|---|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        rr={a:next(r for r in table if r['arm']==a and r['scene']==scene) for a in ('C0',*ARMS)}
        lines.append(f"|{scene}|{100*rr['C0']['accuracy']:.4f}%|{100*rr['SAME_RX']['accuracy']:.4f}%|{100*rr['CROSS_RX']['accuracy']:.4f}%|{100*(rr['CROSS_RX']['accuracy']-rr['C0']['accuracy']):+.4f}|{100*(rr['CROSS_RX']['accuracy']-rr['SAME_RX']['accuracy']):+.4f}|")
    lines+=['',f"预登记两个主比较／F1／seenV／健康检查逐项：{q['checks']}；数值 {q['paired_comparisons']}。不以SAME_RX事后替代主候选；失败不调比例/换RX/seed救援。",'',
        f"完整40新模型轮／8880更新／全部日志／F clip与SupCon，原L/U共{count}批与C0逐条一致；每行568320真实partner均同TX/day、不同物理包，全部5040合法L ID和所有允许RX配对cell覆盖，原U truth始终隐藏。",'',
        f"新增网络包{newcost}；两行各5040 bank IQ读取、568320 secondary物理曝光和ordinary生成；bank读取及partner视图构造墙钟见measured_costs.csv，后者不含骨干前向。矩阵墙钟{result['matrix_wall_seconds']:.3f}s。C0既有E20／4440更新和网络费用独立列出，无父权重继承。预算用尽不等于收敛。",'',
        '两新行都保留.5原primary监督、.5partner监督／四L前向、原F+CROSS_RX／默认头联合训练／原U与ownEMA。新算子只选真实观测，旧ordinary原处理是共有条件。CROSS−C0包含独立样本与独立ordinary多样性的共同变化，CROSS−SAME才检验跨RX的增量。不能把它写成数字IQ变换或模拟替换RX硬件。', '',
        '首折源资格即使PASS，也不是正式target、跨模型seed、独立Ag+As组合或lowurban>60%的证据。只支持另登记源确认。若FAIL保持结束，不由任何目标分数推翻。没有训练As、额外对齐或新部署模块/分类头，正式IQ与practical配置保持。全部TX/day/state/quality负结果、repair/break、固定C0 clean正确集合保持率见CSV。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result

def metrics(records):
    from experiments.cvs_ordinary_fold20.harvest import metrics as original
    return original(records)
def plot(output,table,curves,fusion,strata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=('C0',*ARMS);fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,key in zip(axes.flat,('pair/primary_CE','pair/secondary_CE','F_supcon/loss','train/grad_before_clip','legacy_rdrop/symmetric_KL','train/pseudo_selected')):
        for arm in arms:
            rr=[r for r in curves if r['arm']==arm];ax.plot([r['epoch'] for r in rr],[r.get(key,0.) for r in rr],label=arm)
        ax.axvline(13.5,ls='--',color='grey',lw=.7);ax.set(xlabel='Epoch',ylabel=key);ax.legend(fontsize=8)
    fig.savefig(output/'complete_training_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,arm in enumerate(arms):
        rr=[r for r in table if r['arm']==arm];x=np.arange(7)+(i-1)*.25
        axes[0].bar(x,[100*r['accuracy'] for r in rr],width=.24,label=arm);axes[1].plot(range(7),[r['mean_margin_over30'] for r in rr],marker='o',label=arm)
    for ax in axes:ax.set_xticks(range(7),['clean','high','mid','low sub','high urb','mid urb','low urb'],rotation=30);ax.legend()
    axes[0].set_ylabel('Held source RX1 accuracy (%)');axes[1].set_ylabel('Default-head margin / 30');fig.savefig(output/'source_scene_results.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,dim in zip(axes,('state_end','receiver_lock_status','TX')):
        rr=[r for r in strata if r['scene']=='practical_low_urban' and r['dimension']==dim];values=list(dict.fromkeys(r['value'] for r in rr))
        for arm in arms:
            lookup={r['value']:r for r in rr if r['arm']==arm};ax.plot(values,[100*lookup[v]['accuracy'] for v in values],marker='o',label=arm)
        ax.set(xlabel=dim,ylabel='Source low urban accuracy (%)');ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=8)
    fig.savefig(output/'low_urban_strata.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--reference-root',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    spec=read(ROOT/'experiments/cvs_f_realpair20/registration.json');result=analyze(a.output,spec,read(a.state),a.reference_root,a.cached)
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','scene_results','full_log_proof','costs')},ensure_ascii=False))
