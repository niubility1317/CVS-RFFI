"""P74 source-only CPU readout of saved default-head contribution interventions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

MODELS=('NoAs','As')
SCENES=('practical_mid_urban','practical_low_urban')
BRANCHES=('TF','PA','REF','SIG')
KINDS=('zero','swap_nonoise','swap_clean')
STATES=('LOS','PARTIAL','BLOCKED')
LOCKS=('locked','degraded','unlocked')
RX=(1,3,4,6,8)
DAYS=(1,2,3)
SEED=2026092701


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def csv_write(path,rows):
    rows=list(rows);columns=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def noise_bin(value):
    if value is None:return 'MISSING'
    value=float(value)
    if not np.isfinite(value):return 'MISSING'
    return ('<3','3-8','8-15','>=15')[int(np.searchsorted([3.,8.,15.],value,side='right'))]


def margin30(logits,labels):
    logits=np.asarray(logits);labels=np.asarray(labels)
    require(logits.shape==(len(labels),6) and np.isfinite(logits).all(),'Finite original six-class logits required')
    other=logits.copy();other[np.arange(len(labels)),labels]=-np.inf
    return (logits[np.arange(len(labels)),labels]-other.max(1))/30.


def metrics(logits,labels,receivers,cohort):
    logits=np.asarray(logits);labels=np.asarray(labels);receivers=np.asarray(receivers);cohort=np.asarray(cohort,dtype=bool)
    require(labels.ndim==1 and np.issubdtype(labels.dtype,np.integer) and ((labels>=0)&(labels<6)).all() and
        receivers.shape==cohort.shape==labels.shape,'Aligned visible source labels, RX and fixed cohort required')
    margins=margin30(logits,labels);n=len(labels);pred=logits.argmax(1);correct=pred==labels
    cm=np.bincount(labels*6+pred,minlength=36).reshape(6,6);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    by_rx={str(rx):float(correct[receivers==rx].mean()) for rx in sorted(set(receivers.tolist()))}
    worst=min(by_rx.values()) if by_rx else None;cohort_n=int(cohort.sum())
    return dict(count=n,correct=int(correct.sum()),accuracy=float(correct.mean()) if n else None,
        macro_f1=float(np.divide(2*tp,den,out=np.zeros(6,float),where=den>0).mean()) if n else None,
        worst_RX=worst,worst_RX_IDs=[int(k) for k,v in by_rx.items() if v==worst],
        mean_margin_over30=float(margins.mean()) if n else None,median_margin_over30=float(np.median(margins)) if n else None,
        fixed_NoAs30_clean_correct_count=cohort_n,fixed_NoAs30_clean_correct_retained=int((correct&cohort).sum()),
        satellite_retention=float(correct[cohort].mean()) if cohort_n else None,confusion=cm.tolist(),
        empty_cell=n==0,small_cell=n<30)


def paired(baseline,changed,labels,cohort):
    baseline=np.asarray(baseline);changed=np.asarray(changed);labels=np.asarray(labels);cohort=np.asarray(cohort,bool)
    require(baseline.shape==changed.shape==(len(labels),6) and cohort.shape==labels.shape,'Same physical observations required for paired effects')
    old=baseline.argmax(1)==labels;new=changed.argmax(1)==labels;n=len(labels)
    repaired=int((~old&new).sum());errors=int((old&~new).sum())
    delta=margin30(changed,labels)-margin30(baseline,labels)
    return dict(count=n,control_correct=int(old.sum()),new_correct=int(new.sum()),repairs=repaired,new_errors=errors,
        net_repairs=repaired-errors,accuracy_delta_pp=100*(repaired-errors)/n if n else None,
        mean_margin_over30_delta=float(delta.mean()) if n else None,
        cohort_repairs=int((~old&new&cohort).sum()),cohort_new_errors=int((old&~new&cohort).sum()),
        fixed_NoAs30_clean_correct_count=int(cohort.sum()),
        control_retention=float(old[cohort].mean()) if cohort.any() else None,
        new_retention=float(new[cohort].mean()) if cohort.any() else None)


def load_inputs(root,expected_n=1800):
    root=Path(root);completion=read(root/'complete.json');metadata=read(root/'panel_metadata.json')
    require(completion.get('source_only') is True and completion.get('target_access') is False,
        'Only a completed source-only diagnostic may be analyzed')
    require(all(completion.get(k,0)==0 for k in ('training_updates','optimizer_updates','head_fit_updates')),
        'Contribution readout cannot include training or fitted heads')
    arrays={};shapes={'ids':(expected_n,),'TX':(expected_n,),'RX':(expected_n,),'day':(expected_n,),
        'clean':(expected_n,6),'clean_zero':(4,expected_n,6),'full':(2,2,expected_n,6),
        'no_noise_fixed':(2,2,expected_n,6),'interventions':(2,2,3,4,expected_n,6)}
    for model in MODELS:
        with np.load(root/model/'predictions.npz',allow_pickle=False) as f:values={key:f[key].copy() for key in shapes}
        require(all(values[k].shape==shape for k,shape in shapes.items()),'Exact scene/draw/kind/branch saved axes required: '+model)
        require(values['ids'].dtype.kind in 'US' and len(set(values['ids'].tolist()))==expected_n,'Unique opaque source V IDs required')
        require(all(np.issubdtype(values[k].dtype,np.integer) for k in ('TX','RX','day')) and
            ((values['TX']>=0)&(values['TX']<6)).all() and set(values['RX'].tolist())<=set(RX) and
            set(values['day'].tolist())<=set(DAYS),'Original visible source TX/RX/day required')
        require(all(np.isfinite(values[k]).all() for k in ('clean','clean_zero','full','no_noise_fixed','interventions')),
            'Nonfinite saved default-head predictions')
        arrays[model]=values
    for key in ('ids','TX','RX','day'):require(np.array_equal(arrays['NoAs'][key],arrays['As'][key]),'Both frozen models must use identical physical panel order')
    records=metadata.get('full_records');require(isinstance(records,list) and len(records)==4*expected_n,'All four scene/draw full receiver records required')
    physics={};base=arrays['NoAs']
    for record in records:
        scene,draw,index=record.get('scene'),record.get('draw'),record.get('index');key=(scene,draw,index)
        require(scene in SCENES and type(draw) is int and draw in (0,1) and type(index) is int and 0<=index<expected_n and key not in physics,
            'Duplicate or invalid full-observation metadata key')
        require(str(record.get('physical_id'))==str(base['ids'][index]) and all(record.get(k)==int(base[k][index]) for k in ('TX','RX','day')),
            'Source metadata must align to opaque IDs and true source roles')
        full=record['conditions']['full'];state=full.get('state_end',full.get('state'));lock=full.get('receiver_lock_status')
        require(state in STATES and lock in LOCKS,'Actual original full state and lock required')
        physics[key]=dict(state=state,lock=lock,noise_bin=noise_bin(full.get('signal_to_added_noise_db')))
    require(len(physics)==4*expected_n,'Incomplete original receiver-state coverage')
    return arrays,physics,completion,metadata


def stratification(physics,scene,draws,n,base):
    records=[physics[(scene,draw,i)] for draw in draws for i in range(n)]
    values={key:np.asarray([r[key] for r in records]) for key in ('state','lock','noise_bin')}
    result=[('overall','ALL',np.ones(len(records),bool))]
    for key,labels in (('state',STATES),('lock',LOCKS),('noise_bin',('<3','3-8','8-15','>=15','MISSING'))):
        result.extend((key,label,values[key]==label) for label in labels)
    result.extend(('state_x_lock',state+'/'+lock,(values['state']==state)&(values['lock']==lock)) for state in STATES for lock in LOCKS)
    for key,labels in (('RX',RX),('day',DAYS)):
        raw=np.tile(base[key],len(draws));result.extend((key,str(label),raw==label) for label in labels)
    return result


def analyze_arrays(arrays,physics):
    base=arrays['NoAs'];n=len(base['ids']);cohort=base['clean'].argmax(1)==base['TX']
    metric_rows=[];pair_rows=[];clean_rows=[];overall={}
    for model,values in arrays.items():
        full_clean=metrics(values['clean'],base['TX'],base['RX'],cohort)
        clean_rows.append(dict(model=model,model_seed=SEED,condition='clean',branch='ALL',**full_clean))
        for bi,branch in enumerate(BRANCHES):
            value=metrics(values['clean_zero'][bi],base['TX'],base['RX'],cohort)
            effect=paired(values['clean'],values['clean_zero'][bi],base['TX'],cohort)
            clean_rows.append(dict(model=model,model_seed=SEED,condition='clean_zero',branch=branch,**value,
                clean_zero_cost_pp=100*(full_clean['accuracy']-value['accuracy']),
                **{k:v for k,v in effect.items() if k not in value}))
        for si,scene in enumerate(SCENES):
            for draw in (0,1,-1):
                draws=(0,1) if draw==-1 else (draw,);y=np.tile(base['TX'],len(draws));rx=np.tile(base['RX'],len(draws));fixed=np.tile(cohort,len(draws))
                full=np.concatenate([values['full'][d,si] for d in draws])
                conditions=[('full','ALL',full),('no_noise_fixed','ALL',np.concatenate([values['no_noise_fixed'][d,si] for d in draws]))]
                conditions.extend((kind,branch,np.concatenate([values['interventions'][d,si,ki,bi] for d in draws]))
                    for ki,kind in enumerate(KINDS) for bi,branch in enumerate(BRANCHES))
                strata=stratification(physics,scene,draws,n,base)
                for kind,branch,z in conditions:
                    for dimension,stratum,mask in strata:
                        tags=dict(model=model,model_seed=SEED,scene=scene,draw=draw,condition=kind,branch=branch,dimension=dimension,stratum=stratum)
                        value=metrics(z[mask],y[mask],rx[mask],fixed[mask]);effect=paired(full[mask],z[mask],y[mask],fixed[mask])
                        metric_rows.append(dict(tags,**value));pair_rows.append(dict(tags,control='full',**effect))
                        if dimension=='overall':overall[(model,scene,draw,kind,branch)]=dict(value,effect=effect)
    return dict(metrics=metric_rows,paired=pair_rows,clean=clean_rows,overall=overall,
        fixed_NoAs30_clean_correct_count=int(cohort.sum()),unique_source_V=n)


def attribution_rules(readout):
    overall=readout['overall'];clean={r['branch']:r for r in readout['clean'] if r['model']=='As' and r['condition']=='clean_zero'}
    rows=[]
    for branch in BRANCHES:
        conditions=[]
        for scene in SCENES:
            for draw in (0,1):
                whole=overall[('As',scene,draw,'no_noise_fixed','ALL')]['effect']['accuracy_delta_pp']
                swap=overall[('As',scene,draw,'swap_nonoise',branch)]['effect']['accuracy_delta_pp']
                zero=overall[('As',scene,draw,'zero',branch)]['effect']['accuracy_delta_pp']
                conditions.append(dict(scene=scene,draw=draw,whole_no_noise_gain_pp=whole,swap_nonoise_gain_pp=swap,zero_gain_pp=zero,
                    noise_rule_met=bool(whole>0 and swap>=2.-1e-10 and swap>=.5*whole-1e-10),zero_rule_met=bool(zero>=1.-1e-10)))
        rows.append(dict(branch=branch,conditions=conditions,
            noise_specific_local_evidence=all(r['noise_rule_met'] for r in conditions),
            clean_zero_cost_pp=clean[branch]['clean_zero_cost_pp'],
            lightweight_masking_worth_research=all(r['zero_rule_met'] for r in conditions) and clean[branch]['clean_zero_cost_pp']<=.1+1e-10))
    gains={branch:float(np.mean([overall[('As',scene,-1,'swap_nonoise',branch)]['effect']['accuracy_delta_pp'] for scene in SCENES])) for branch in BRANCHES}
    selected=max(BRANCHES,key=lambda branch:gains[branch])
    return dict(model='As',branches=rows,all_four_scene_draw_conditions_required=True,
        model_seed_count=1,channel_draw_count=2,channel_draws_are_not_model_seeds=True,
        interpretation='Frozen counterfactual diagnostics only; not a deployment module, trainable upper bound, training benefit or automatic promotion rule',
        automatic_training=False,automatic_E50=False,target_feedback=False,
        descriptive_fixed_swap=dict(branch=selected,criterion='Highest As swap_nonoise mean net accuracy change across BOTH scenes and BOTH draws; branch-order tie break',
            source_analysis_selected=True,per_draw_selection=False,gains_pp=gains,not_a_training_candidate=True))


def plots(out,readout,rules):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[];overall=readout['overall'];colors=('#64748b','#2563eb','#d97706')
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=Path(out)/(name+'.'+ext);fig.savefig(path,dpi=180);files.append(str(path))
        plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(15,14))
    for mi,model in enumerate(MODELS):
        for si,scene in enumerate(SCENES):
            for draw in (0,1):
                ax=axes[2*mi+si,draw];x=np.arange(4)
                for ki,kind in enumerate(KINDS):
                    ax.bar(x+(ki-1)*.24,[overall[(model,scene,draw,kind,b)]['effect']['accuracy_delta_pp'] for b in BRANCHES],.24,label=kind,color=colors[ki])
                ax.axhline(0,color='black',lw=.6);ax.set(xticks=x,xticklabels=BRANCHES,title=f'{model} / {scene} / draw {draw}',ylabel='Counterfactual - full (pp)');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);save(fig,'branch_effects_all_draws')
    branch=rules['descriptive_fixed_swap']['branch'];labels=[s+'/'+k for s in STATES for k in LOCKS]
    fig,axes=plt.subplots(2,2,figsize=(17,9))
    for mi,model in enumerate(MODELS):
        for draw in (0,1):
            ax=axes[mi,draw]
            for ci,(condition,b,label) in enumerate((('full','ALL','Full'),('no_noise_fixed','ALL','No added noise / fixed state'),('swap_nonoise',branch,'Fixed swap '+branch+' (source-selected)'))):
                rr=[r for r in readout['metrics'] if r['model']==model and r['scene']==SCENES[1] and r['draw']==draw and r['dimension']=='state_x_lock' and r['condition']==condition and r['branch']==b]
                values={r['stratum']:r for r in rr};ax.plot(range(9),[100*values[k]['accuracy'] if values[k]['accuracy'] is not None else np.nan for k in labels],marker='.',label=label,color=colors[ci])
            ax.set(xticks=range(9),xticklabels=labels,title=f'{model} / low urban / draw {draw}',ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelrotation=35);ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'low_state_lock_fixed_swap')
    fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(4)
    for mi,model in enumerate(MODELS):
        values={r['branch']:r['clean_zero_cost_pp'] for r in readout['clean'] if r['model']==model and r['condition']=='clean_zero'}
        ax.bar(x+(mi-.5)*.32,[values[b] for b in BRANCHES],.32,label=model,color=colors[mi])
    ax.axhline(.1,color='#dc2626',ls='--',label='0.1 pp diagnostic cost bound');ax.axhline(0,color='black',lw=.6)
    ax.set(xticks=x,xticklabels=BRANCHES,ylabel='Clean accuracy cost: full - zero (pp)',title='Source clean contribution-removal cost');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    save(fig,'clean_zero_cost');return files


def markdown(out,readout,rules,figures):
    overall=readout['overall'];lines=['# P74：冻结模型的噪声特异贡献干预','',
        '仅源域反事实读出，原P43 NoAs30／As30、默认cos30头均冻结。缓存源诊断载频为2.45 GHz，不冒称训练载频2.462 GHz。full的接收／补偿状态固定；去掉的是新增噪声，不是原录音总噪声。TF、PA、REF、whole SIG贡献经过真实postLN／gain，替换后重做原最终归一化和默认头。','',
        '| 模型 | 场景 | draw | full % | no_noise_fixed % | 去新增噪声 Δpp |','|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        for scene in SCENES:
            for draw in (0,1,-1):
                full=overall[(model,scene,draw,'full','ALL')];noise=overall[(model,scene,draw,'no_noise_fixed','ALL')]
                lines.append(f'| {model} | {scene} | {draw} | {100*full["accuracy"]:.6f} | {100*noise["accuracy"]:.6f} | {noise["effect"]["accuracy_delta_pp"]:+.6f} |')
    lines+=['','draw −1为两次信道观测合并；两模型是处理条件，模型seed仅1个，两个draw不是跨seed复现。',
        f'固定NoAs30 clean正确集合：{readout["fixed_NoAs30_clean_correct_count"]}/{readout["unique_source_V"]}。它不是As20父集合；所有卫星干预保持率复用同一集合。', '',
        '| As 分支 | 四条件噪声局部证据 | clean置零代价 pp | 轻量屏蔽值得研究 |','|---|---|---:|---|']
    for r in rules['branches']:lines.append(f'| {r["branch"]} | {r["noise_specific_local_evidence"]} | {r["clean_zero_cost_pp"]:+.6f} | {r["lightweight_masking_worth_research"]} |')
    lines+=['','噪声局部规则要求每个mid／low×draw条件的整体去噪增量为正，单分支swap_nonoise≥2 pp且≥整体去噪增量一半。屏蔽解释要求四条件zero增量均≥1 pp且同分支As clean代价≤0.1 pp。它们只约束解释，不是自动训练、模型晋级或E50条件。','',
        '完整accuracy、宏F1、最差RX、分类间隔÷30、配对修复／新增错误及状态、lock、状态×lock、新增噪声比、RX/day表见metrics.csv与paired_effects.csv。空格保留，不挑最好draw。贡献混合未必是可获得的真实输入，不能称训练可达上界或已缓解实际部署退化。','',
        '图中固定swap分支为'+rules['descriptive_fixed_swap']['branch']+'，按As两个场景、两个draw合并净变化在四分支中选定，仅供源分析展示；没有逐draw挑选，也不是部署候选。', '',
        '本分析新增NN／IQ读取／checkpoint读取／训练更新／头拟合均0。没有正式目标数据、训练收益或跨seed复现结论。','', '实际图表：']
    lines.extend('- '+f for f in figures);Path(out/'readout.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def run(root,out=None):
    started=time.perf_counter();root=Path(root);out=Path(out) if out else root/'analysis_r01'
    arrays,physics,completion,metadata=load_inputs(root)
    require(not out.exists(),'Preserve existing analysis; select a new output directory');out.mkdir(parents=False)
    data=analyze_arrays(arrays,physics);rules=attribution_rules(data)
    csv_write(out/'metrics.csv',data['metrics']);csv_write(out/'paired_effects.csv',data['paired']);csv_write(out/'clean_zero.csv',data['clean'])
    rule_rows=[dict(branch=r['branch'],clean_zero_cost_pp=r['clean_zero_cost_pp'],**condition) for r in rules['branches'] for condition in r['conditions']]
    csv_write(out/'attribution_rules.csv',rule_rows);write(out/'attribution_rules.json',rules)
    figures=plots(out,data,rules);markdown(out,data,rules,figures)
    overall=[dict(model=k[0],scene=k[1],draw=k[2],condition=k[3],branch=k[4],**v) for k,v in data['overall'].items()]
    result=dict(status='ANALYZED_SOURCE_ONLY',source_only=True,target_access=False,model_seed_count=1,model_seed=SEED,
        frozen_models=list(MODELS),models_are_treatments_not_seeds=True,channel_draw_count=2,source_unique_V=data['unique_source_V'],
        cached_source_carrier_Hz=2450000000,training_carrier_Hz=2462000000,
        fixed_NoAs30_clean_correct_count=data['fixed_NoAs30_clean_correct_count'],cohort_is_As20_parent=False,
        metrics_rows=len(data['metrics']),paired_effect_rows=len(data['paired']),overall=overall,clean=data['clean'],rules=rules,figures=figures,
        NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,head_fit_updates=0,
        automatic_training=False,automatic_E50=False,counterfactual_is_deployment_improvement=False,counterfactual_is_trainable_upper_bound=False,
        input_completion=completion,wall_seconds=time.perf_counter()-started)
    write(out/'summary.json',result);write(out/'complete.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args()
    result=run(a.root,a.out);print(json.dumps(dict(status=result['status'],out=str(a.out or a.root/'analysis_r01'),NN=0),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
