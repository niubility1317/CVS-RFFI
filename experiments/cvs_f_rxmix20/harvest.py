"""Independent saved-source reader. No model, IQ or checkpoint imports."""
import argparse,csv,gzip,json,math,subprocess
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
from experiments.cvs_f_crossrx_practical20.harvest import full_logs,check_predictions,packet_key,groups,SCENES,read,write,metrics,SSH
ROOT=Path(__file__).resolve().parents[2]
FIT=(3,4,6,8);ARMS=('C0','RXMIX');COST_KEYS=('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets','source_practical_forward_packets')

def collect(output,spec):
    output=Path(output);(output/'logs').mkdir(parents=True,exist_ok=True)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json','resolved_config.json','module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz','training_sequence.jsonl.gz','held_V_predictions.json','ordinary_base_config.json','F_implementation_proof.json','F_optimizer_steps.jsonl.gz','supcon_steps.jsonl.gz','source_practical_evaluation.json','rxmix_sequence.jsonl.gz','rxmix_proof.json','rxmix_sessions.json']+[s+'_source_predictions.jsonl.gz' for s in SCENES[1:]]
    statistic=spec['rows'][0]['config']['rxmix']['statistics_root']
    payload='''import json,numpy as np
from pathlib import Path
p=Path(STATISTIC);s=json.loads((p/'summary.json').read_text())
with np.load(p/'source_ellipticity_statistics.npz',allow_pickle=False) as f:profile=f['RX_mean_profile'].tolist()
print(json.dumps(dict(statistics_root=str(p),source_summary_status=s['status'],fit_RX=s['config']['source_RX'],L_used=s['roles']['selected_L_packets'],target_access=s['roles']['target_access'],held_RX1_used=s['roles']['held_RX1_IQ_values_used'],profile=profile)))
'''.replace('STATISTIC',repr(statistic))
    r=subprocess.run(['ssh',*SSH,'-T','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'],input=payload,capture_output=True,text=True,encoding='utf-8',timeout=60)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    write(output/'private_source_ellipticity_profile.json',json.loads(r.stdout))
    for row in spec['rows']:
        dest=output/row['row_id']/'source';dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*SSH,*['N607:'+row['output_root']+'/'+name for name in names],str(dest)],check=True,timeout=60)
        subprocess.run(['scp',*SSH,'N607:'+row['log_path'],str(output/'logs')],check=True,timeout=60)

from .trace import check_trace

def qualify(models,recorded):
    b,c=models['C0'],models['RXMIX'];values=dict(clean_gain_pp=100*(c['held_V']['accuracy']-b['held_V']['accuracy']),macroF1_gain_pp=100*(c['held_V']['macro_f1']-b['held_V']['macro_f1']),seen_drop_pp=100*(b['seen_V']['source_val_accuracy']-c['seen_V']['source_val_accuracy']))
    checks=dict(clean=values['clean_gain_pp']>=1.-1e-9,F1=values['macroF1_gain_pp']>=-1e-9,seen=values['seen_drop_pp']<=.5+1e-9,complete_health=all(r['complete_health'] for r in models.values()))
    if recorded['checks']!=checks or recorded['status']!=('PASS' if all(checks.values()) else 'FAIL') or not recorded['source_only'] or recorded['target_access'] or recorded['model_seeds']!=1 or any(not math.isclose(recorded[k],v,abs_tol=1e-9) for k,v in values.items()):raise ValueError('Independent fixed RXMIX qualification differs')
    return recorded

def analyze(output,spec,state,reference_root,cached=False):
    if state['dispatcher_live'] or any(r['live'] for r in state['rows']) or state['failure'] or state['row_failures'] or not state['completion'] or state['completion']['status']!='ARTIFACTS_COMPLETE':raise ValueError('Actual row and seven source views must finish before analysis')
    if len(spec['rows'])!=1 or spec['rows'][0]['config']['arm']!='RXMIX':raise ValueError('One RXMIX row and frozen C0 reference only')
    output=Path(output);refroot=Path(reference_root)
    if not cached:collect(output,spec)
    row=spec['rows'][0];paths={'C0':refroot/'C0-RX1-s2026092701/source','RXMIX':output/row['row_id']/'source'}
    configs={'C0':read(ROOT/'experiments/cvs_f_crossrx_practical20/registration.json')['rows'][0]['config'],'RXMIX':row['config']}
    models={};predictions={};logs={};curves=[];costs=[];strata=[];fusion=[];arguments={};trace=None
    for arm in ARMS:
        source=paths[arm];r=read(source/'result.json');done=read(source/'completion.json');init=read(source/'initialization.json');freeze=read(source/'source_frozen.json');contract=read(source/'source_contract.json')
        if (done['config']!=configs[arm] or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
            or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
            or r['optimizer_updates']!=4440 or r['L_forward_packets']!=2273280 or r['training_loader_packets']!={'L_s':568320,'U_s':396368} or r['U_truth_read'] or r['target_access'] or r['head_fit_updates'] or not r['default_head_joint_training']
            or contract['held_RX']!=1 or contract['source_receivers']!=list(FIT) or r['held_RX']!=1 or r['fit_RX']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):raise ValueError('Actual source roles/final20/defaulthead provenance differs')
        proof=read(source/'F_implementation_proof.json');ev=read(source/'source_practical_evaluation.json')
        if proof['steps']!=4440 or proof['physical_clip_calls']!=4440 or proof['optimizer']!='CENTER_UNITS' or ev['optimizer_updates'] or ev['EMA_updates'] or ev['packet_forwards']!=32400 or r['supcon']['calls']!=4440 or r['supcon']['applied_calls']!=3330 or not r['supcon']['only_primary_pair']:raise ValueError('Original F/SupCon/sourceevaluation differs')
        with gzip.open(source/'F_optimizer_steps.jsonl.gz','rt',encoding='utf-8') as f:opts=[json.loads(line) for line in f]
        if [v['step'] for v in opts]!=list(range(1,4441)) or any(v['physical_clip_calls']!=v['step'] or not math.isclose(v['sinc_lr'],v['base_lr']*1000,rel_tol=1e-12) or v['sinc_weight_decay']!=0 for v in opts):raise ValueError('Complete original F trajectory required')
        if arm=='RXMIX':
            p=read(source/'rxmix_proof.json')
            if p['status']!='VERIFIED' or p['pair_weights']!=[.5,.5] or p['L_forwards_per_sample']!=4 or not all(p[k] for k in ('all_L_views_supervised','same_generated_ordinary_IQ','SupCon_only_first_pair','U_recipe_unchanged')) or p['new_head'] or p['target_access']:raise ValueError('Fixed RXMIX/defaulthead execution differs')
            trace=check_trace(source,r,configs[arm]['rxmix']['seed'])
        stdout=(refroot if arm=='C0' else output)/'logs'/((('C0-RX1-s2026092701') if arm=='C0' else row['row_id'])+'.log')
        logs[arm]=full_logs(source,stdout);curves.extend(dict(arm=arm,**v) for v in logs[arm]['curves']);a=read(source/'actual_native_args.json');arguments[arm]=a.get('args',a);predictions[arm]={}
        for scene in SCENES:
            if scene=='clean':records=read(source/'held_V_predictions.json')['records'];saved=r['held_V']
            else:
                with gzip.open(source/(scene+'_source_predictions.jsonl.gz'),'rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
                saved=r['practical'][scene]
            predictions[arm][scene]=check_predictions(records,saved,scene,held_RX=1);grouped=defaultdict(list)
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
        costs.append(dict(arm=arm,reused_reference=arm=='C0',epochs=20,updates=4440,parent_weight_epochs=0,**{k:r[k] for k in (*COST_KEYS,'elapsed_seconds','source_practical_seconds')},rxmix_exposures=568320 if arm=='RXMIX' else 0,rxmix_seconds=r.get('rxmix_seconds',0.)))
        models[arm]=r
    differences={k for k in set(arguments['RXMIX'])|set(arguments['C0']) if arguments['RXMIX'].get(k)!=arguments['C0'].get(k)}
    if differences-{'candidate_id','output_dir','run_id'}:raise ValueError(('Unchanged native recipe required',differences))
    count=0
    with gzip.open(paths['C0']/'training_sequence.jsonl.gz','rt') as a,gzip.open(paths['RXMIX']/'training_sequence.jsonl.gz','rt') as b:
        for x,y in zip_longest(a,b):
            if x!=y:raise ValueError('Actual primary L/U differs from C0')
            count+=1
    if count!=5994:raise ValueError('All original L/U batches required')
    table=[];paired=[];reference=predictions['C0']['clean'];fixed={k for k,v in reference.items() if v['label']==v['prediction']}
    for scene in SCENES:
        base,candidate=(predictions[a][scene] for a in ARMS)
        for arm in ARMS:
            rr=predictions[arm][scene]
            if rr.keys()!=reference.keys() or any(rr[k]['label']!=base[k]['label'] or rr[k].get('physics')!=base[k].get('physics') for k in rr):raise ValueError('Same original source labels and physics required')
            kept=sum(rr[k]['label']==rr[k]['prediction'] for k in fixed);table.append(dict(arm=arm,scene=scene,**metrics(list(rr.values())),mean_margin_over30=float(np.mean([v['margin'] for v in rr.values()])),fixed_C0_clean_correct_count=len(fixed),fixed_retained=kept,fixed_retention=kept/len(fixed)))
        grouped=defaultdict(list);grouped[('all','all')]=list(base)
        for k,v in base.items():
            for dim,value in groups(v):grouped[(dim,value)].append(k)
        for (dim,value),keys in sorted(grouped.items()):
            repair=sum(base[k]['prediction']!=base[k]['label'] and candidate[k]['prediction']==candidate[k]['label'] for k in keys);broken=sum(base[k]['prediction']==base[k]['label'] and candidate[k]['prediction']!=candidate[k]['label'] for k in keys)
            paired.append(dict(scene=scene,dimension=dim,value=value,count=len(keys),repaired=repair,broken=broken,accuracy_gain_pp=100*(repair-broken)/len(keys)))
    q=qualify(models,state['source_decision']);newcost=sum(costs[1][k] for k in COST_KEYS)
    summary=dict(status='ANALYZED',source_decision=q,models=models,scene_results=table,full_log_proof={a:r['proof'] for a,r in logs.items()},rxmix_trace_proof=trace,costs=costs,
        new_real_forward_packets=newcost,new_model_epochs=20,new_updates=4440,reused_C0_epochs=20,reused_C0_updates=4440,actual_primary_sequence_records=count,matrix_wall_seconds=state['completion']['wall_seconds'],
        actual_training_commit=state['submit']['commit'],model_seed_count=1,source_fold_count=1,target_access=False,formal_test_status='NOT_REGISTERED_SOURCE_ONLY',goal_completed=False,automatic_extension=False)
    for name,records in [('scene_results.csv',table),('source_strata.csv',strata),('paired_strata.csv',paired),('training_curves.csv',curves),('fusion_gain_curves.csv',fusion),('measured_costs.csv',costs)]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)));writer.writeheader();writer.writerows(records)
    plot(output,table,curves,strata);write(output/'analysis_summary.json',summary)
    lines=['# 有界接收RXMIX：完整源域结果','',f"固定源规则 **{q['status']}**。一个新scratchE20，一个modelseed，held sourceRX1，0target。",'', '|scene|C0 accuracy|RXMIX accuracy|Delta pp|C0 macroF1|RXMIX macroF1|','|---|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        rr={a:next(r for r in table if r['arm']==a and r['scene']==scene) for a in ARMS}
        lines.append(f"|{scene}|{100*rr['C0']['accuracy']:.6f}%|{100*rr['RXMIX']['accuracy']:.6f}%|{100*(rr['RXMIX']['accuracy']-rr['C0']['accuracy']):+.6f}|{rr['C0']['macro_f1']:.6f}|{rr['RXMIX']['macro_f1']:.6f}|")
    lines+=['',f"固定clean>=+1pp/F1非负/seenV损失<=.5pp及完整健康判据：{q}。不调bound/ratio/coordinate/RX/seed救援。",'',f"新增20模型轮/4440更新/{newcost}网络包，墙钟{summary['matrix_wall_seconds']:.6f}s。C0原20轮独立列，不继承权重。全部5994primary L/U记录逐条相同。",'', '独立复放568320次有界RXMIX，5040物理包、240个epoch/RX/day session；私有源profile只用于幅度验证、不提交Git。已知响应逆与逐点能量界为数值证据，不证明身份保持或硬件因果。', '', '固定F+CROSS_RX/默认头联合训练、第一对SupCon、原U/ownEMA/旧门控/cos不改。0.5RD ordinary+0.5RD RXMIX(sameordinary)，4L前向，一次ordinary，无新增损失/部署头/As。', '', '完整七源视图、20轮/4440步日志、全部TX/day/state/quality、修复/新错、固定C0clean正确集合保持率、同scale30间隔及曲线保留；未改正式IQ/channel/SNR。单seed源证据不是正式target结果或跨seed复现；真正Ag/As四格、low>60%及suburban接近自身clean仍未证明。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary

def plot(output,table,curves,strata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,key in zip(axes.flat,('pair/primary_CE','pair/secondary_CE','F_supcon/loss','train/grad_before_clip','legacy_rdrop/symmetric_KL','train/pseudo_selected')):
        for arm in ARMS:
            rr=[r for r in curves if r['arm']==arm];ax.plot([r['epoch'] for r in rr],[r.get(key,0.) for r in rr],label=arm)
        ax.axvline(13.5,ls='--',color='grey',lw=.7);ax.set(xlabel='Epoch',ylabel=key);ax.legend(fontsize=8)
    fig.savefig(output/'complete_training_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,arm in enumerate(ARMS):
        rr=[r for r in table if r['arm']==arm];axes[0].bar(np.arange(7)+(i-.5)*.32,[100*r['accuracy'] for r in rr],width=.3,label=arm);axes[1].plot(range(7),[r['mean_margin_over30'] for r in rr],marker='o',label=arm)
    for ax in axes:ax.set_xticks(range(7),['clean','high','mid','low sub','high urb','mid urb','low urb'],rotation=30);ax.legend()
    axes[0].set_ylabel('Held source RX1 accuracy (%)');axes[1].set_ylabel('Default-head margin / 30');fig.savefig(output/'source_scene_results.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,dim in zip(axes,('state_end','receiver_lock_status','TX')):
        rr=[r for r in strata if r['scene']=='practical_low_urban' and r['dimension']==dim];values=list(dict.fromkeys(r['value'] for r in rr))
        for arm in ARMS:
            lookup={r['value']:r for r in rr if r['arm']==arm};ax.plot(values,[100*lookup[v]['accuracy'] for v in values],marker='o',label=arm)
        ax.set(xlabel=dim,ylabel='Source low urban accuracy (%)');ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=8)
    fig.savefig(output/'low_urban_strata.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--reference-root',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    result=analyze(a.output,read(ROOT/'experiments/cvs_f_rxmix20/registration.json'),read(a.state),a.reference_root,a.cached)
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','scene_results','full_log_proof','costs')},ensure_ascii=False))
