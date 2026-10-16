"""Independent saved-source reader. No model, IQ or checkpoint imports."""
import argparse,csv,gzip,json,math,os,subprocess
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
from experiments.cvs_f_crossrx_practical20.harvest import full_logs,check_predictions as original_check_predictions,packet_key,groups,SCENES,read,write,metrics,SSH
ROOT=Path(__file__).resolve().parents[2]
SCP=['C:/Program Files/Git/usr/bin/scp.exe','-S','C:/Program Files/Git/usr/bin/ssh.exe'] if os.name=='nt' else ['scp']
FIT=(3,4,6,8);ARMS=('C0','LA16');COST_KEYS=('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets','source_practical_forward_packets')

def check_predictions(records,saved,scene,held_RX=1):
    mapped=original_check_predictions(records,saved,scene,held_RX)
    actual=metrics(records)
    if actual['count']!=saved['count'] or actual['correct']!=saved['correct']:
        raise ValueError('Saved source packet/correct counters disagree with independent decisions')
    return mapped

def collect(output,spec):
    output=Path(output);(output/'logs').mkdir(parents=True,exist_ok=True)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json','resolved_config.json','module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz','training_sequence.jsonl.gz','held_V_predictions.json','ordinary_base_config.json','F_implementation_proof.json','F_optimizer_steps.jsonl.gz','supcon_steps.jsonl.gz','source_practical_evaluation.json','localatten_sequence.jsonl.gz','localatten_proof.json']+[s+'_source_predictions.jsonl.gz' for s in SCENES[1:]]
    for row in spec['rows']:
        dest=output/row['row_id']/'source';dest.mkdir(parents=True,exist_ok=True)
        subprocess.run([*SCP,*SSH,*['N607:'+row['output_root']+'/'+name for name in names],'.'],cwd=dest,check=True,timeout=60)
        subprocess.run([*SCP,*SSH,'N607:'+row['log_path'],'.'],cwd=output/'logs',check=True,timeout=60)

def local_record(r,primary,step,seed,counts,physical,values):
    if r['step']!=step or r['ids']!=primary or len(primary)!=128:raise ValueError('Local trace must follow original L128 order')
    starts=np.random.default_rng(np.random.SeedSequence([seed,step])).integers(0,241,size=128).tolist()
    if r['starts']!=starts or r['outside_max_error']!=0.:raise ValueError('Label/RX-blind original position stream or outside identity differs')
    keys=('depth','relative_L2','removed_power_fraction','zero_energy')
    if any(len(r[k])!=128 for k in keys):raise ValueError('Every labeled packet needs full local statistics')
    for i,(p,s) in enumerate(zip(primary,starts)):
        if len(p)!=5 or p[0]!=1 or p[1] not in FIT or p[2] not in (1,2,3) or any(int(v)!=v or v<0 for v in p):raise ValueError('Illegal source physical ID')
        dep,rel,powr=(r[k][i] for k in keys[:3]);zero=r['zero_energy'][i]
        if type(zero) is not bool or not all(np.isfinite(v) for v in (dep,rel,powr)) or not (0<=dep<=.5 and 0<=rel<=.1+1e-7 and -1e-12<=powr<=2*rel-rel*rel+1e-6):raise ValueError('Actual local attenuation trace violates frozen bound')
        if zero and (rel!=0. or powr!=0.):raise ValueError('Zero-power packet must remain with zero change,not be dropped')
        counts[s]+=1;physical.add(tuple(p));values.append((dep,rel,powr,zero))

def exposure_curves(counts,depth_sums,packets):
    sample_exposure=np.zeros(256,dtype=np.int64);mean_loss=np.zeros(256)
    w=np.sin(np.pi*(np.arange(16)+.5)/16)**2
    for start,n in counts.items():
        sample_exposure[start:start+16]+=n;mean_loss[start:start+16]+=depth_sums[start]*w/packets
    return sample_exposure,mean_loss

def check_trace(source,result,seed):
    with gzip.open(source/'training_sequence.jsonl.gz','rt',encoding='utf-8') as f:primary=[json.loads(line)['ids'] for line in f if '\"L_s\"' in line]
    if len(primary)!=4440:raise ValueError('Complete actual L sequence required')
    counts=Counter();physical=set();values=[];steps=0;rxregions=Counter();depth_sums=Counter()
    with gzip.open(source/'localatten_sequence.jsonl.gz','rt',encoding='utf-8') as f:
        for ids,line in zip_longest(primary,f):
            if ids is None or line is None:raise ValueError('Exactly one local trace per actual L update required')
            steps+=1;r=json.loads(line);local_record(r,ids,steps,seed,counts,physical,values)
            for p,start,dep in zip(ids,r['starts'],r['depth']):
                rxregions[(p[1],p[2],start//16)]+=1;depth_sums[start]+=dep
    if len(physical)!=5040 or set(counts)!=set(range(241)) or sum(counts.values())!=568320:raise ValueError('Full5040 source packets and all241 starts required')
    a=np.array(values,dtype=float);saved=result['local_attenuation'];cov=result['local_TX_RX_day_region_coverage']
    expect={(tx,rx,day,j) for tx in range(6) for rx in FIT for day in (1,2,3) for j in range(16)}
    if {(v['TX'],v['RX'],v['day'],v['region']) for v in cov}!=expect or sum(v['count'] for v in cov)!=568320:raise ValueError('Complete TX/RX/day/position-region coverage required')
    reported=Counter()
    for v in cov:reported[(v['RX'],v['day'],v['region'])]+=v['count']
    if reported!=rxregions:raise ValueError('Saved region coverage does not match original physical trace')
    required=dict(calls=4440,packets=568320,physical_IDs=5040,width=16,start_positions=241,max_depth=float(a[:,0].max()),max_relative_L2=float(a[:,1].max()),max_removed_power_fraction=float(a[:,2].max()),mean_relative_L2=float(a[:,1].mean()),mean_removed_power_fraction=float(a[:,2].mean()),zero_energy_packets_kept=int(a[:,3].sum()),outside_max_error=0.)
    for k,v in required.items():
        if not math.isclose(saved[k],v,rel_tol=1e-10,abs_tol=1e-10):raise ValueError(('Complete local-statistic reduction differs',k,saved[k],v))
    if saved['start_counts']!={str(k):v for k,v in sorted(counts.items())} or any(saved[k] for k in ('TX_used_for_selection','RX_used_for_selection','global_RNG_advance','new_normalization','time_indices_changed','boundary_extension','added_noise','synthetic_RX_hardware')):raise ValueError('Fixed observation scope differs')
    sample_exposure,mean_loss=exposure_curves(counts,depth_sums,568320)
    return dict(status='VERIFIED',sample_window_exposure_count=sample_exposure.tolist(),mean_reconstructed_coefficient_loss=mean_loss.tolist(),start_counts={str(k):v for k,v in sorted(counts.items())},steps=steps,packets=568320,physical_IDs=5040,start_positions=241,region_cells=len(cov),**{k:v for k,v in required.items() if k not in ('calls','packets','physical_IDs','start_positions')},
        scope='Positions replayed independently;stats reduced and bounds checked from saved trace,not recomputed from original IQ',new_IQ_reads=0,checkpoint_loads=0,model_forwards=0)

def qualify(models,recorded):
    b,c=models['C0'],models['LA16'];values=dict(clean_gain_pp=100*(c['held_V']['accuracy']-b['held_V']['accuracy']),macroF1_gain_pp=100*(c['held_V']['macro_f1']-b['held_V']['macro_f1']),seen_drop_pp=100*(b['seen_V']['source_val_accuracy']-c['seen_V']['source_val_accuracy']))
    practical={s:100*(c['practical'][s]['accuracy']-b['practical'][s]['accuracy']) for s in SCENES[1:]}
    checks=dict(clean=values['clean_gain_pp']>=1.-1e-9,F1=values['macroF1_gain_pp']>=-1e-9,seen=values['seen_drop_pp']<=.5+1e-9,practical=all(v>=-.5-1e-9 for v in practical.values()),complete_health=all(r['complete_health'] for r in models.values()))
    if recorded['checks']!=checks or recorded['status']!=('PASS' if all(checks.values()) else 'FAIL') or not recorded['source_only'] or recorded['target_access'] or recorded['model_seeds']!=1 or any(not math.isclose(recorded[k],v,abs_tol=1e-9) for k,v in values.items()) or recorded['practical_gain_pp']!=practical:raise ValueError('Independent fixed localatten qualification differs')
    return recorded

def analyze(output,spec,state,reference_root,cached=False,make_plots=True):
    if state['dispatcher_live'] or any(r['live'] for r in state['rows']) or state['failure'] or state['row_failures'] or not state['completion'] or state['completion']['status']!='ARTIFACTS_COMPLETE':raise ValueError('Actual row and seven source views must finish before analysis')
    if len(spec['rows'])!=1 or spec['rows'][0]['config']['arm']!='LA16':raise ValueError('One LA16 row and frozen C0 reference only')
    output=Path(output);refroot=Path(reference_root)
    if not cached:collect(output,spec)
    row=spec['rows'][0];paths={'C0':refroot/'C0-RX1-s2026092701/source','LA16':output/row['row_id']/'source'}
    configs={'C0':read(ROOT/'experiments/cvs_f_crossrx_practical20/registration.json')['rows'][0]['config'],'LA16':row['config']}
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
        if arm=='LA16':
            p=read(source/'localatten_proof.json')
            if p['status']!='VERIFIED' or p['pair_weights']!=[.5,.5] or p['L_forwards_per_sample']!=4 or not all(p[k] for k in ('all_L_views_supervised','same_generated_ordinary_IQ','SupCon_only_first_pair','U_recipe_unchanged')) or p['new_head'] or p['target_access']:raise ValueError('Fixed LA16/defaulthead execution differs')
            trace=check_trace(source,r,configs[arm]['local_attenuation']['seed'])
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
        costs.append(dict(arm=arm,reused_reference=arm=='C0',epochs=20,updates=4440,parent_weight_epochs=0,**{k:r[k] for k in (*COST_KEYS,'elapsed_seconds','source_practical_seconds')},local_attenuation_exposures=568320 if arm=='LA16' else 0,local_attenuation_seconds=r.get('local_attenuation_seconds',0.)))
        models[arm]=r
    differences={k for k in set(arguments['LA16'])|set(arguments['C0']) if arguments['LA16'].get(k)!=arguments['C0'].get(k)}
    if differences-{'candidate_id','output_dir','run_id'}:raise ValueError(('Unchanged native recipe required',differences))
    count=0
    with gzip.open(paths['C0']/'training_sequence.jsonl.gz','rt') as a,gzip.open(paths['LA16']/'training_sequence.jsonl.gz','rt') as b:
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
    summary=dict(status='ANALYZED',source_decision=q,models=models,scene_results=table,full_log_proof={a:r['proof'] for a,r in logs.items()},local_attenuation_trace_proof=trace,costs=costs,
        new_real_forward_packets=newcost,new_model_epochs=20,new_updates=4440,reused_C0_epochs=20,reused_C0_updates=4440,actual_primary_sequence_records=count,matrix_wall_seconds=state['completion']['wall_seconds'],
        actual_training_commit=state['submit']['commit'],model_seed_count=1,source_fold_count=1,target_access=False,formal_test_status='NOT_REGISTERED_SOURCE_ONLY',goal_completed=False,automatic_extension=False)
    for name,records in [('scene_results.csv',table),('source_strata.csv',strata),('paired_strata.csv',paired),('training_curves.csv',curves),('fusion_gain_curves.csv',fusion),('measured_costs.csv',costs)]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)));writer.writeheader();writer.writerows(records)
    with (output/'local_mask_exposure.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sample_index','window_exposure_count','mean_reconstructed_coefficient_loss'])
        writer.writerows(zip(range(256),trace['sample_window_exposure_count'],trace['mean_reconstructed_coefficient_loss']))
    if make_plots:plot(output,table,curves,strata)
    write(output/'analysis_summary.json',summary)
    lines=['# 有界局部观察衰减：完整源域结果','',f"固定源规则 **{q['status']}**。一个scratch E20,一个modelseed,held sourceRX1;0As训练/0target。",'', '|scene|C0 accuracy|LA16 accuracy|Δ pp|C0 macroF1|LA16 macroF1|','|---|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        rr={a:next(r for r in table if r['arm']==a and r['scene']==scene) for a in ARMS}
        lines.append(f"|{scene}|{100*rr['C0']['accuracy']:.6f}%|{100*rr['LA16']['accuracy']:.6f}%|{100*(rr['LA16']['accuracy']-rr['C0']['accuracy']):+.6f}|{100*rr['C0']['macro_f1']:.6f}%|{100*rr['LA16']['macro_f1']:.6f}%|")
    lines+=['',f"原clean>=+1pp/F1不降/seenV及逐项practical下降<=.5pp和完整健康条件：{q}。失败不调width/depth/L2bound/probability/ratio/RX/seed/预算。",'',f"新增20模型轮/4440更新/{newcost}网络包；C0既有E20成本单列，0父权重继承。全5994主L/U序列与C0一致；568320窗口/5040原L/241起点独立复放，保存的边界/深度/L2/功率变化及全部聚合核对。未重新访问IQ、模型或CP，不把保存统计复算写成原始IQ独立重算。矩阵墙钟{summary['matrix_wall_seconds']:.6f}s；局部构造包含bound计算、CPU同步与trace、不含NN。",'', 'F+CROSS_RX/defaultcos/NoExplicitCFO/原U/ownEMA不变。0.5RD原ordinary+0.5RD局部衰减同ordinary，四L前向、一次ordinary、无新一致性/头/参数。所有合法有标签包CE，0置信度过滤和物理样本丢弃。', '', '完整七源视图预测独立复算，同一物理标签和原practical流；全部20轮/4440步/CSV/stdout/Fclip/SupCon解析。保存全部TX/day/state/lock/quality、间隔、repair-break、固定C0 clean正确集合保持率和成本。单seed源结果不等于正式target、多seed或联合突破。lowurban>60%、suburban接近clean、真正Ag/As四格交互仍未证明。']
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
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--reference-root',required=True);p.add_argument('--cached',action='store_true');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    result=analyze(a.output,read(ROOT/'experiments/cvs_f_localatten20/registration.json'),read(a.state),a.reference_root,a.cached,not a.no_plots)
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','scene_results','full_log_proof','costs','local_attenuation_trace_proof')},ensure_ascii=False))
