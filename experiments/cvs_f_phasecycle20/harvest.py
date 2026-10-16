"""Independent saved-source reader. No model, IQ or checkpoint imports."""
import argparse,csv,gzip,json,math,subprocess
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
from experiments.cvs_f_crossrx_practical20.harvest import full_logs,check_predictions,packet_key,groups,SCENES,read,write,metrics,SSH
ROOT=Path(__file__).resolve().parents[2]
FIT=(3,4,6,8);ARMS=('C0','C4');COST_KEYS=('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets','source_practical_forward_packets')

def collect(output,spec):
    output=Path(output);(output/'logs').mkdir(parents=True,exist_ok=True)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json','resolved_config.json','module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz','training_sequence.jsonl.gz','held_V_predictions.json','ordinary_base_config.json','F_implementation_proof.json','F_optimizer_steps.jsonl.gz','supcon_steps.jsonl.gz','source_practical_evaluation.json','phase_cycle_sequence.jsonl.gz','phase_cycle_proof.json']+[s+'_source_predictions.jsonl.gz' for s in SCENES[1:]]
    for row in spec['rows']:
        dest=output/row['row_id']/'source';dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*SSH,*['N607:'+row['output_root']+'/'+name for name in names],str(dest)],check=True,timeout=60)
        subprocess.run(['scp',*SSH,'N607:'+row['log_path'],str(output/'logs')],check=True,timeout=60)

def phase_record(r,primary,step,visits,offsets,rng,counts):
    if r['step']!=step or r['ids']!=primary or len(primary)!=128 or len(r['k'])!=128:raise ValueError('C4 trace must follow actual L128 order')
    for row,k in zip(primary,r['k']):
        p=tuple(row)
        if len(p)!=5 or p[0]!=1 or p[1] not in FIT or p[2] not in (1,2,3) or k not in (1,2,3):raise ValueError('Illegal source/phase in actual trace')
        if p not in offsets:offsets[p]=int(rng.integers(3))
        if k!=1+(offsets[p]+visits[p])%3:raise ValueError('Label-blind original C4 cycle differs')
        visits[p]+=1;counts[(p,k)]+=1

def check_trace(source,result,seed):
    with gzip.open(source/'training_sequence.jsonl.gz','rt',encoding='utf-8') as f:primary=[json.loads(line)['ids'] for line in f if '"L_s"' in line]
    if len(primary)!=4440:raise ValueError('Whole actual L sequence required')
    visits=Counter();offsets={};counts=Counter();rng=np.random.default_rng(seed);steps=0
    with gzip.open(source/'phase_cycle_sequence.jsonl.gz','rt',encoding='utf-8') as f:
        for ids,line in zip_longest(primary,f):
            if ids is None or line is None:raise ValueError('One phase record per actual L batch required')
            steps+=1;phase_record(json.loads(line),ids,steps,visits,offsets,rng,counts)
    saved=result['phase_cycle'];imbalance=max(max(counts[(p,k)] for k in (1,2,3))-min(counts[(p,k)] for k in (1,2,3)) for p in visits)
    if len(visits)!=5040 or len(counts)!=15120 or sum(counts.values())!=568320 or imbalance>1:raise ValueError('Complete per-physical C4 coverage required')
    expected={(t,r,day,k) for t in range(6) for r in FIT for day in (1,2,3) for k in (1,2,3)}
    cov=result['phase_TX_RX_day_coverage'];by_rx=Counter()
    if {(v['TX'],v['RX'],v['day'],v['k']) for v in cov}!=expected or sum(v['count'] for v in cov)!=568320:raise ValueError('Actual TX/RX/day/phase coverage incomplete')
    for v in cov:by_rx[(v['RX'],v['day'],v['k'])]+=v['count']
    actual=Counter()
    for (p,k),n in counts.items():actual[(p[1],p[2],k)]+=n
    checks=dict(calls=saved['calls']==4440,packets=saved['packets']==568320,physical=saved['physical_IDs']==5040,
        phase_cells=saved['physical_ID_rotations']==15120,balance=saved['max_per_ID_rotation_count_imbalance']==imbalance,
        power=saved['max_pointwise_power_error']==0.,inverse=saved['max_exact_inverse_error']==0.,
        no_label=not saved['TX_used_for_selection'],RNG=not saved['global_RNG_advance'],normalization=not saved['new_normalization'],boundary=not saved['boundary_change'],noise=not saved['added_noise'],coverage=actual==by_rx)
    if not all(checks.values()):raise ValueError(('Runtime C4 proof differs',checks))
    return dict(status='VERIFIED',steps=steps,primary_physical_IDs=5040,physical_ID_rotations=15120,phase_exposures=568320,max_per_ID_imbalance=imbalance,checks=checks)

def qualify(models,recorded):
    b,c=models['C0'],models['C4'];values=dict(clean_gain_pp=100*(c['held_V']['accuracy']-b['held_V']['accuracy']),macroF1_gain_pp=100*(c['held_V']['macro_f1']-b['held_V']['macro_f1']),seen_drop_pp=100*(b['seen_V']['source_val_accuracy']-c['seen_V']['source_val_accuracy']))
    checks=dict(clean=values['clean_gain_pp']>=1.-1e-9,F1=values['macroF1_gain_pp']>=-1e-9,seen=values['seen_drop_pp']<=.5+1e-9,complete_health=all(r['complete_health'] for r in models.values()))
    if recorded['checks']!=checks or recorded['status']!=('PASS' if all(checks.values()) else 'FAIL') or not recorded['source_only'] or recorded['target_access'] or recorded['model_seeds']!=1 or any(not math.isclose(recorded[k],v,abs_tol=1e-9) for k,v in values.items()):raise ValueError('Independent fixed C4 qualification differs')
    return recorded

def analyze(output,spec,state,reference_root,cached=False):
    if state['dispatcher_live'] or any(r['live'] for r in state['rows']) or state['failure'] or state['row_failures'] or not state['completion'] or state['completion']['status']!='ARTIFACTS_COMPLETE':raise ValueError('Actual row and seven source views must finish before analysis')
    if len(spec['rows'])!=1 or spec['rows'][0]['config']['arm']!='C4':raise ValueError('One C4 row and frozen C0 reference only')
    output=Path(output);refroot=Path(reference_root)
    if not cached:collect(output,spec)
    row=spec['rows'][0];paths={'C0':refroot/'C0-RX1-s2026092701/source','C4':output/row['row_id']/'source'}
    configs={'C0':read(ROOT/'experiments/cvs_f_crossrx_practical20/registration.json')['rows'][0]['config'],'C4':row['config']}
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
        if arm=='C4':
            p=read(source/'phase_cycle_proof.json')
            if p['status']!='VERIFIED' or p['pair_weights']!=[.5,.5] or p['L_forwards_per_sample']!=4 or not all(p[k] for k in ('all_L_views_supervised','same_generated_ordinary_IQ','SupCon_only_first_pair','U_recipe_unchanged')) or p['new_head'] or p['target_access']:raise ValueError('Fixed C4/defaulthead execution differs')
            trace=check_trace(source,r,configs[arm]['phase_cycle']['phase_seed'])
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
        costs.append(dict(arm=arm,reused_reference=arm=='C0',epochs=20,updates=4440,parent_weight_epochs=0,**{k:r[k] for k in (*COST_KEYS,'elapsed_seconds','source_practical_seconds')},phase_cycle_exposures=568320 if arm=='C4' else 0,phase_cycle_seconds=r.get('phase_cycle_seconds',0.)))
        models[arm]=r
    differences={k for k in set(arguments['C4'])|set(arguments['C0']) if arguments['C4'].get(k)!=arguments['C0'].get(k)}
    if differences-{'candidate_id','output_dir','run_id'}:raise ValueError(('Unchanged native recipe required',differences))
    count=0
    with gzip.open(paths['C0']/'training_sequence.jsonl.gz','rt') as a,gzip.open(paths['C4']/'training_sequence.jsonl.gz','rt') as b:
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
    summary=dict(status='ANALYZED',source_decision=q,models=models,scene_results=table,full_log_proof={a:r['proof'] for a,r in logs.items()},phase_cycle_trace_proof=trace,costs=costs,
        new_real_forward_packets=newcost,new_model_epochs=20,new_updates=4440,reused_C0_epochs=20,reused_C0_updates=4440,actual_primary_sequence_records=count,matrix_wall_seconds=state['completion']['wall_seconds'],
        actual_training_commit=state['submit']['commit'],model_seed_count=1,source_fold_count=1,target_access=False,formal_test_status='NOT_REGISTERED_SOURCE_ONLY',goal_completed=False,automatic_extension=False)
    for name,records in [('scene_results.csv',table),('source_strata.csv',strata),('paired_strata.csv',paired),('training_curves.csv',curves),('fusion_gain_curves.csv',fusion),('measured_costs.csv',costs)]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)));writer.writeheader();writer.writerows(records)
    plot(output,table,curves,strata);write(output/'analysis_summary.json',summary)
    lines=['# 精确 C4 相位轮转：源域完整结果','',f"固定源规则 **{q['status']}**。一个新scratch E20、一个modelseed、held sourceRX1；0target。",'', '|scene|C0 accuracy|C4 accuracy|Δ pp|C0 macroF1|C4 macroF1|','|---|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        rr={a:next(r for r in table if r['arm']==a and r['scene']==scene) for a in ARMS}
        lines.append(f"|{scene}|{100*rr['C0']['accuracy']:.4f}%|{100*rr['C4']['accuracy']:.4f}%|{100*(rr['C4']['accuracy']-rr['C0']['accuracy']):+.4f}|{rr['C0']['macro_f1']:.6f}|{rr['C4']['macro_f1']:.6f}|")
    lines+=['',f"原clean≥+1pp/F1不降/seenV损失≤.5pp及完整健康规则：{q}。不调角度、概率、比例、RX/seed或预算救援。",'',f"完整20新模型轮／4440更新／{newcost}网络包；C0既有20轮成本单列，父权重成本0。完整5994条primary L/U与C0相同；568320次C4轮转、5040包各三元素、15120物理phase cell独立重放核实，逐点功率／逆恢复执行值为0误差。矩阵墙钟{summary['matrix_wall_seconds']:.6f}s；构造墙钟含旋转、核对及trace，见measured_costs.csv。",'', '原F＋跨RX SupCon/默认头联合训练/原U/ownEMA不改，无新部署模块或分类头。0.5RD同ordinary＋0.5RD(C4同IQ)，四L前向、一次ordinary；不做跨视图一致性。新增输入变化可逆不等于训练有效，也不等于替换RX硬件或拟合相位bank。', '', '全部七源视图默认头预测独立复算、同物理标签和原practical流；全部20轮/4440步/nativeCSV/stdout/F clip/SupCon解析。所有TX/day/state/quality、修复/新错、固定C0 clean正确集合保持率、间隔与曲线保留；不修改正式IQ/信道/SNR。没有As训练、正式target或跨modelseed复现；真正四格混合、lowurban>60%及suburban接近clean仍未证明。预算耗尽不等于收敛。']
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
    result=analyze(a.output,read(ROOT/'experiments/cvs_f_phasecycle20/registration.json'),read(a.state),a.reference_root,a.cached)
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','scene_results','full_log_proof','costs')},ensure_ascii=False))
