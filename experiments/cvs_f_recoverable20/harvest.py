"""Independent saved-source recount; no model, IQ or checkpoint imports."""
import argparse,csv,gzip,json,math,subprocess
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
from experiments.cvs_f_crossrx_practical20.harvest import full_logs,check_predictions,packet_key,groups,SCENES,read,write,metrics,SSH
from .trace import check_trace
from .transport import transport
ROOT=Path(__file__).resolve().parents[2]
FIT=(1,3,4,8);ARMS=('C0','C1','C2')
COST_KEYS=('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets','source_practical_forward_packets')

def collect(output,spec):
    output=Path(output);(output/'logs').mkdir(parents=True,exist_ok=True)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json','resolved_config.json','module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz','training_sequence.jsonl.gz','held_V_predictions.json','ordinary_base_config.json','F_implementation_proof.json','F_optimizer_steps.jsonl.gz','supcon_steps.jsonl.gz','source_practical_evaluation.json','FcrossRX_pair_proof.json','practical_channel_audit.json','curriculum_history.json','curriculum_proof.json','curriculum_draws.jsonl.gz']+[s+'_source_predictions.jsonl.gz' for s in SCENES[1:]]
    for row in spec['rows']:
        dest=output/row['row_id']/'source';dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*SSH,*['N607:'+row['output_root']+'/'+name for name in names],str(dest)],check=True,timeout=120)
        subprocess.run(['scp',*SSH,'N607:'+row['log_path'],str(output/'logs')],check=True,timeout=60)

def qualify(models,recorded):
    b,a,c=(models[k] for k in ARMS);mean=lambda r,k:float(np.mean([r['practical'][s][k] for s in SCENES[4:]]))
    values=dict(clean_drop_vs_C0_pp=100*(b['held_V']['accuracy']-c['held_V']['accuracy']),seen_drop_vs_C0_pp=100*(b['seen_V']['source_val_accuracy']-c['seen_V']['source_val_accuracy']),
        urban_gain_vs_C1_pp=100*(mean(c,'accuracy')-mean(a,'accuracy')),low_gain_vs_C1_pp=100*(c['practical'][SCENES[-1]]['accuracy']-a['practical'][SCENES[-1]]['accuracy']),urban_F1_gain_vs_C1_pp=100*(mean(c,'macro_f1')-mean(a,'macro_f1')))
    checks=dict(clean=values['clean_drop_vs_C0_pp']<=.5+1e-9,seen=values['seen_drop_vs_C0_pp']<=.5+1e-9,urban=values['urban_gain_vs_C1_pp']>=-1e-9,low=values['low_gain_vs_C1_pp']>=-1e-9,urban_F1=values['urban_F1_gain_vs_C1_pp']>=-1e-9,
        activation=c['curriculum']['max_eta']>0 and c['training_practical']['hard_requested']>0,complete_health=all(r['complete_health'] for r in models.values()))
    if recorded['checks']!=checks or recorded['status']!=('PASS' if all(checks.values()) else 'FAIL') or not recorded['source_only'] or recorded['target_access'] or recorded['model_seed_count']!=1 or any(not math.isclose(recorded[k],v,abs_tol=1e-9) for k,v in values.items()):raise ValueError('Independent fixed curriculum qualification differs')
    return recorded

def analyze(output,spec,state,reference_root,cached=False):
    if state['dispatcher_live'] or any(r['live'] for r in state['rows']) or state['failure'] or state['row_failures'] or not state['completion'] or state['completion']['status']!='ARTIFACTS_COMPLETE':raise ValueError('Actual row and seven source views must finish')
    if len(spec['rows'])!=1 or spec['rows'][0]['config']['arm']!='C2' or spec['rows'][0]['config']['held_RX']!=6:raise ValueError('One fixed RX6 course row')
    output=Path(output);refroot=Path(reference_root)
    if not cached:collect(output,spec)
    row=spec['rows'][0];paths={a:refroot/f'{a}-RX6-s2026092701/source' for a in ARMS[:2]};paths['C2']=output/row['row_id']/'source'
    refs=read(ROOT/'experiments/cvs_f_crossrx_folds20/registration.json')['rows'];configs={a:next(r['config'] for r in refs if r['config']['arm']==a and r['config']['held_RX']==6) for a in ARMS[:2]};configs['C2']=row['config']
    models={};predictions={};logs={};curves=[];costs=[];strata=[];fusion=[];arguments={};trace=None
    for arm in ARMS:
        source=paths[arm];r=read(source/'result.json');done=read(source/'completion.json');init=read(source/'initialization.json');freeze=read(source/'source_frozen.json');contract=read(source/'source_contract.json')
        if (done['config']!=configs[arm] or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
            or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
            or r['optimizer_updates']!=4440 or r['L_forward_packets']!=2273280 or r['training_loader_packets']!={'L_s':568320,'U_s':396368} or r['U_truth_read'] or r['target_access'] or r['head_fit_updates'] or not r['default_head_joint_training']
            or contract['held_RX']!=6 or contract['source_receivers']!=list(FIT) or r['held_RX']!=6 or r['fit_RX']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):raise ValueError('Actual source roles/final20/defaulthead provenance differs')
        proof=read(source/'F_implementation_proof.json');pair=read(source/'FcrossRX_pair_proof.json');ev=read(source/'source_practical_evaluation.json')
        if proof['steps']!=4440 or proof['physical_clip_calls']!=4440 or proof['optimizer']!='CENTER_UNITS' or ev['optimizer_updates'] or ev['EMA_updates'] or ev['packet_forwards']!=32400 or r['supcon']['calls']!=4440 or r['supcon']['applied_calls']!=3330 or not r['supcon']['only_primary_pair'] or pair['pair_weights']!=[.5,.5] or pair['L_forwards_per_sample']!=4 or pair['student_confidence_gate'] or not pair['all_L_views_supervised'] or pair['ordinary_before_satellite'] or not pair['U_recipe_unchanged']:raise ValueError('Original F/SupCon/paired/sourceevaluation differs')
        with gzip.open(source/'F_optimizer_steps.jsonl.gz','rt',encoding='utf-8') as f:opts=[json.loads(line) for line in f]
        if [v['step'] for v in opts]!=list(range(1,4441)) or any(v['physical_clip_calls']!=v['step'] or not math.isclose(v['sinc_lr'],v['base_lr']*1000,rel_tol=1e-12) or v['sinc_weight_decay']!=0 for v in opts):raise ValueError('Complete original F trajectory required')
        if {(v['TX'],v['RX'],v['day']) for v in r['coverage']}!={(t,rx,d) for t in range(6) for rx in FIT for d in (1,2,3)} or sum(v['count'] for v in r['coverage'])!=568320:raise ValueError('Complete source L coverage required')
        if arm!='C0':
            coverage=r['TX_RX_scene_state_coverage'];s=r['training_practical']
            if s['calls']!=4440 or s['packets']!=568320 or {(v['TX'],v['RX'],v['scene']) for v in coverage}!={(t,rx,sc) for t in range(6) for rx in FIT for sc in ('practical_low_suburban','practical_mid_urban','practical_low_urban')} or {(v['TX'],v['RX'],v['state']) for v in coverage}!={(t,rx,st) for t in range(6) for rx in FIT for st in ('LOS','PARTIAL','BLOCKED')} or sum(v['count'] for v in coverage)!=568320:raise ValueError('Actual full TX/RX/scene/state support required')
        if arm=='C2':
            p=read(source/'curriculum_proof.json')
            if p['status']!='VERIFIED' or p['epochs']!=20 or p['steps']!=4440 or p['nominal_support_floor']!=.5 or not p['all_labeled_CE'] or p['new_head'] or p['extra_model_forwards'] or p['held_V_used_for_progress'] or p['U_truth_read'] or p['target_access']:raise ValueError('FitL-only fixed curriculum proof required')
            trace=check_trace(source,r)
        stdout=(refroot if arm!='C2' else output)/'logs'/(f'{arm}-RX6-s2026092701.log')
        logs[arm]=full_logs(source,stdout);curves.extend(dict(arm=arm,**v) for v in logs[arm]['curves']);a=read(source/'actual_native_args.json');arguments[arm]=a.get('args',a);predictions[arm]={}
        for scene in SCENES:
            if scene=='clean':records=read(source/'held_V_predictions.json')['records'];saved=r['held_V']
            else:
                with gzip.open(source/(scene+'_source_predictions.jsonl.gz'),'rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
                saved=r['practical'][scene]
            predictions[arm][scene]=check_predictions(records,saved,scene,held_RX=6);grouped=defaultdict(list)
            for v in records:
                for dim,value in groups(v):grouped[(dim,value)].append(v)
            ref=predictions['C0']['clean']
            for (dim,value),rr in sorted(grouped.items()):
                eligible=[v for v in rr if ref[packet_key(v)]['prediction']==ref[packet_key(v)]['label']];kept=sum(v['label']==v['prediction'] for v in eligible)
                strata.append(dict(arm=arm,scene=scene,dimension=dim,value=value,**metrics(rr),mean_margin_over30=float(np.mean([v['margin'] for v in rr])),fixed_C0_clean_correct_count=len(eligible),fixed_retained=kept,fixed_retention=kept/len(eligible) if eligible else None))
        mechanism=read(source/'mechanism_progress.json')['records']
        if len(mechanism)!=20 or any(v<=0 for v in r['actual_feature_parameters_moved'].values()):raise ValueError('Complete normal joint training required')
        for v in mechanism:
            g=v['default_fusion_gain'];fusion.append(dict(arm=arm,epoch=v['epoch'],mean=float(np.mean(g)),minimum=min(g),maximum=max(g),std=float(np.std(g))))
        costs.append(dict(arm=arm,reused_reference=arm!='C2',epochs=20,updates=4440,parent_weight_epochs=0,**{k:r[k] for k in (*COST_KEYS,'elapsed_seconds','source_practical_seconds')},physics_proposals=r['training_practical'].get('proposal_packets',r['training_practical']['packets']),physics_seconds=r['training_practical'].get('generation_seconds',0.)))
        models[arm]=r
    for arm in ARMS[1:]:
        differences={k for k in set(arguments[arm])|set(arguments['C0']) if arguments[arm].get(k)!=arguments['C0'].get(k)}
        if differences-{'candidate_id','output_dir','run_id'}:raise ValueError(('Unchanged native recipe required',differences))
        count=0
        with gzip.open(paths['C0']/'training_sequence.jsonl.gz','rt') as a,gzip.open(paths[arm]/'training_sequence.jsonl.gz','rt') as b:
            for x,y in zip_longest(a,b):
                if x!=y:raise ValueError('Complete original primary L/U differs')
                count+=1
        if count!=5994:raise ValueError('All original L/U batches required')
    table=[];paired=[];reference=predictions['C0']['clean'];fixed={k for k,v in reference.items() if v['label']==v['prediction']}
    for scene in SCENES:
        base=predictions['C0'][scene]
        for arm in ARMS:
            rr=predictions[arm][scene]
            if rr.keys()!=reference.keys() or any(rr[k]['label']!=base[k]['label'] or rr[k].get('physics')!=base[k].get('physics') for k in rr):raise ValueError('Same source labels and original physical draws required')
            kept=sum(rr[k]['label']==rr[k]['prediction'] for k in fixed);table.append(dict(arm=arm,scene=scene,**metrics(list(rr.values())),mean_margin_over30=float(np.mean([v['margin'] for v in rr.values()])),fixed_C0_clean_correct_count=len(fixed),fixed_retained=kept,fixed_retention=kept/len(fixed)))
        for a in ARMS[:2]:
            x,y=predictions[a][scene],predictions['C2'][scene];grouped=defaultdict(list);grouped[('all','all')]=list(x)
            for k,v in x.items():
                for dim,value in groups(v):grouped[(dim,value)].append(k)
            for (dim,value),keys in sorted(grouped.items()):
                repair=sum(x[k]['prediction']!=x[k]['label'] and y[k]['prediction']==y[k]['label'] for k in keys);broken=sum(x[k]['prediction']==x[k]['label'] and y[k]['prediction']!=y[k]['label'] for k in keys)
                paired.append(dict(comparison='C2-'+a,scene=scene,dimension=dim,value=value,count=len(keys),repaired=repair,broken=broken,accuracy_gain_pp=100*(repair-broken)/len(keys)))
    q=qualify(models,state['source_decision']);newcost=sum(costs[2][k] for k in COST_KEYS)
    summary=dict(status='ANALYZED',source_decision=q,models=models,scene_results=table,full_log_proof={a:r['proof'] for a,r in logs.items()},curriculum_trace_proof=trace,costs=costs,new_real_forward_packets=newcost,new_model_epochs=20,new_updates=4440,reused_reference_model_epochs=40,actual_primary_sequence_records=count,matrix_wall_seconds=state['completion']['wall_seconds'],actual_training_commit=state['submit']['commit'],model_seed_count=1,source_fold_count=1,target_access=False,formal_test_status='NOT_REGISTERED_SOURCE_ONLY',goal_completed=False,automatic_extension=False)
    for name,records in [('scene_results.csv',table),('source_strata.csv',strata),('paired_strata.csv',paired),('training_curves.csv',curves),('fusion_gain_curves.csv',fusion),('measured_costs.csv',costs),('error_transport.csv',transport(predictions))]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)));writer.writeheader();writer.writerows(records)
    plot(output,table,curves,strata,read(paths['C2']/'curriculum_history.json')['records'],fusion);write(output/'analysis_summary.json',summary)
    lines=['# 可恢复状态课程：完整源域结果','',f"固定源规则 **{q['status']}**。一个新scratchE20，一个modelseed，held sourceRX6，0target。",'', '|scene|C0 accuracy|C1 accuracy|C2 accuracy|C2-C0 pp|C2-C1 pp|','|---|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        rr={a:next(r for r in table if r['arm']==a and r['scene']==scene) for a in ARMS}
        lines.append(f"|{scene}|{100*rr['C0']['accuracy']:.6f}%|{100*rr['C1']['accuracy']:.6f}%|{100*rr['C2']['accuracy']:.6f}%|{100*(rr['C2']['accuracy']-rr['C0']['accuracy']):+.6f}|{100*(rr['C2']['accuracy']-rr['C1']['accuracy']):+.6f}|")
    lines+=['',f'固定clean/seen保留且urban/low/F1不低于C1、实际激活、全健康判据：{q}。FAIL关闭固定课程，不扫描eta/范围/warm/proposals/比例/RX/seed救援。', '', f"新增20模型轮/4440更新/{newcost}网络包，矩阵墙钟{summary['matrix_wall_seconds']:.6f}s；原C0/C1共40模型轮只复用结果，无父权重。",'', '全部5994条primary L/U序列逐条相同，568320次实测课程视图与有限提案独立复放。固定F+CROSS_RX/defaulthead、4L前向、原U/EMA/旧门控/cos不变；课程只有fitL进展、无新增损失或部署模块。', '', '一个已知失败源RX6/一个modelseed，不能证明其他源RX或跨seed；无正式target评分，不宣称low>60%或联合Ag+As目标已完成。新增噪声比不是总SNR，质量代理与原状态计算不改。分层关联不是物理因果。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary

def plot(output,table,curves,strata,history,fusion):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,key in zip(axes.flat,('pair/primary_CE','pair/secondary_CE','F_supcon/loss','train/grad_before_clip','legacy_rdrop/symmetric_KL','train/pseudo_selected')):
        for arm in ARMS:
            rr=[r for r in curves if r['arm']==arm];ax.plot([r['epoch'] for r in rr],[r.get(key,0.) for r in rr],label=arm)
        ax.axvline(13.5,ls='--',color='grey',lw=.7);ax.set(xlabel='Epoch',ylabel=key);ax.legend(fontsize=8)
    fig.savefig(output/'complete_training_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for i,arm in enumerate(ARMS):
        rr=[r for r in table if r['arm']==arm];axes[0].bar(np.arange(7)+(i-1)*.24,[100*r['accuracy'] for r in rr],width=.23,label=arm);axes[1].plot(range(7),[r['mean_margin_over30'] for r in rr],marker='o',label=arm);ff=[r for r in fusion if r['arm']==arm];axes[2].plot([r['epoch'] for r in ff],[r['mean'] for r in ff],label=arm)
    for ax in axes[:2]:ax.set_xticks(range(7),['clean','high','mid','low sub','high urb','mid urb','low urb'],rotation=30)
    for ax in axes:ax.legend(fontsize=8)
    axes[0].set_ylabel('Held source RX6 accuracy (%)');axes[1].set_ylabel('Default-head margin / 30');axes[2].set(xlabel='Epoch',ylabel='Mean default fusion gain');fig.savefig(output/'source_scene_results.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for ax,dim in zip(axes.flat,('state_end','receiver_lock_status','TX','day','quality_snr_db','signal_to_added_noise_db')):
        rr=[r for r in strata if r['scene']=='practical_low_urban' and r['dimension']==dim];values=list(dict.fromkeys(r['value'] for r in rr))
        for arm in ARMS:
            lookup={r['value']:r for r in rr if r['arm']==arm};ax.plot(values,[100*lookup[v]['accuracy'] for v in values],marker='o',label=arm)
        ax.set(xlabel=dim,ylabel='Source low urban accuracy (%)');ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=8)
    fig.savefig(output/'low_urban_strata.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True);epochs=list(range(1,21))
    axes[0].step(epochs,[r['eta_used'] for r in history],where='mid',label='eta used');axes[0].set(ylim=(-.02,.52),ylabel='Hard mixture fraction')
    for key in ('accuracy','p10_cell_accuracy','p10_cell_margin','mean_CE'):axes[1].plot(epochs,[r['metrics'][key] for r in history],label=key)
    axes[2].plot(epochs,[r['metrics']['gradient'] for r in history],label='native mean gradient')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend(fontsize=8)
    fig.savefig(output/'course_progress.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--reference-root',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    result=analyze(a.output,read(ROOT/'experiments/cvs_f_recoverable20/registration.json'),read(a.state),a.reference_root,a.cached)
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','scene_results','full_log_proof','costs')},ensure_ascii=False))
