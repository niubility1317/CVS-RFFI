"""Aggregate the two fixed source seeds only after both independent recounts."""
import csv,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FIRST=ROOT/'local_artifacts/F_crossrx_satmix20_20261016/final'
SECOND=ROOT/'local_artifacts/F_crossrx_satmix_confirm20_20261016/final'
SEEDS=(2026092701,2026092702)
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def read(path):return json.loads(path.read_text(encoding='utf8'))
def stats(values):return dict(mean=statistics.mean(values),sample_SD=statistics.stdev(values),per_seed=dict(zip(map(str,SEEDS),values)))


def analyze():
    reports=[read(folder/'analysis_summary.json') for folder in (FIRST,SECOND)]
    for seed,r in zip(SEEDS,reports):
        assert r['model_seed']==seed and r['status']=='ANALYZED' and r['actual_execution']=='VERIFIED'
        assert r['source_only'] and not r['target_access'] and not r['joint_goal_complete'] and r['all_L_U_sequences_identical']
        assert r['new_optimizer_updates']==8880 and r['new_model_epochs']==40 and r['head_fit_updates']==0
    contracts=[];args=[];strata=[];configs=[]
    for folder,seed in zip((FIRST,SECOND),SEEDS):
        for arm in ('B00','B01'):
            src=folder/(arm+'-RX1-s'+str(seed))/'source'
            c=read(src/'source_contract.json');contracts.append(c)
            assert c['held_RX']==1 and set(c['source_receivers'])=={3,4,6,8}
            args.append(read(src/'actual_native_args.json'))
            configs.append(read(src/'resolved_config.json')['config'])
        with (folder/'source_strata.csv').open(encoding='utf8',newline='') as f:strata.append(list(csv.DictReader(f)))
    assert all(c['role_ids']==contracts[0]['role_ids'] for c in contracts)
    assert all(c['channel']==configs[0]['channel'] and c['supcon']==configs[0]['supcon'] for c in configs)
    assert all(c['training_mechanism']==dict(configs[0]['training_mechanism'],As=c['arm']=='B01') for c in configs)
    differences=[sorted(k for k,v in a.items() if v!=args[0].get(k)) for a in args]
    assert all(set(keys)<={'seed','run_id','candidate_id','output_dir'} for keys in differences)
    metrics={}
    for view in VIEWS:
        item={arm:{metric:stats([100*r['scores'][arm][view][metric] for r in reports]) for metric in ('accuracy','macro_F1')} for arm in ('B00','B01')}
        item['paired_accuracy_gain_pp']=stats([r['paired'][view]['accuracy_gain_pp'] for r in reports])
        item['paired_macro_F1_gain_pp']=stats([r['paired'][view]['macro_F1_gain_pp'] for r in reports])
        assert math.isclose(item['paired_accuracy_gain_pp']['mean'],item['B01']['accuracy']['mean']-item['B00']['accuracy']['mean'],abs_tol=1e-10)
        item['strictly_positive_accuracy_gain_seed_count']=sum(r['paired'][view]['accuracy_gain_pp']>0 for r in reports)
        item['repairs_breaks_per_seed']={str(seed):{k:r['paired'][view][k] for k in ('repaired','broken')} for seed,r in zip(SEEDS,reports)}
        metrics[view]=item
    gaps={arm:{view:stats([r['suburban_gaps'][arm][view]['absolute_gap_pp'] for r in reports]) for view in VIEWS[1:4]} for arm in ('B00','B01')}
    by_stratum=[]
    for row in strata[0]:
        match=next(v for v in strata[1] if all(v[k]==row[k] for k in ('view','dimension','value','count')))
        values=[float(v['gain_pp']) for v in (row,match)]
        by_stratum.append(dict(view=row['view'],dimension=row['dimension'],value=int(row['value']),count_per_seed=int(row['count']),paired_accuracy_gain_pp=stats(values),both_seeds_negative=all(v<0 for v in values)))
    qualified=all(r['qualification']['qualified'] for r in reports)
    value=dict(status='TWO_FIXED_SOURCE_SEEDS_ANALYZED',source_only=True,model_seeds=list(SEEDS),model_seed_count=2,physical_held_RX=1,physical_packets_per_scene=1260,unique_physical_packets_across_seeds=1260,not_2520_independent_packets=True,metrics=metrics,suburban_absolute_gaps_pp=gaps,strata=by_stratum,qualification=dict(qualified=qualified,rule='Both independent pairs must pass every originally registered source condition;mean gains never rescue a failed confirmation',per_seed={str(seed):r['qualification'] for seed,r in zip(SEEDS,reports)},urban_mean_gain_pp=stats([r['qualification']['urban_mean_gain_pp'] for r in reports])),cost=dict(new_models=4,new_model_epochs=80,new_optimizer_updates=17760,actual_NN_forward_packets=sum(r['total_actual_NN_forward_packets'] for r in reports),practical_training_IQ=sum(r['execution']['B01']['As_training_observations'] for r in reports),training_wall_seconds_by_seed={str(seed):{arm:r['execution'][arm]['elapsed_training_seconds'] for arm in ('B00','B01')} for seed,r in zip(SEEDS,reports)}),statistic_scope='Arithmetic model-seed means and unbiased sample SD,n=2;not a confidence interval or independent physical-sample replication',same_role_ids_across_seeds=True,native_argument_differences=differences,target_access=False,head_fit_updates=0,joint_goal_complete=False,missing=['Supported IQ Ag','B10/B11 and factorial interaction','Formal full-source clean+six practical truth-last results','Final clean net gain,lowurban>60%,eachsuburban close to clean'])
    value['clean_interpretation']=dict(every_seed_strictly_positive=metrics['clean']['strictly_positive_accuracy_gain_seed_count']==2,mean_strictly_positive=metrics['clean']['paired_accuracy_gain_pp']['mean']>0,source_guardrail='Original confirmation permits up to0.5pp clean loss;passing it alone does not prove positive clean gain in every seed')
    value['fixed_channel_seeds']={k:configs[0]['channel'][k] for k in ('seed','receiver_seed','eval_seed')}
    (SECOND/'two_seed_source_summary.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(status=value['status'],qualified=qualified,model_seeds=list(SEEDS),clean_gain_pp=metrics['clean']['paired_accuracy_gain_pp'],low_urban_gain_pp=metrics['practical_low_urban']['paired_accuracy_gain_pp'],joint_goal_complete=False),ensure_ascii=False,indent=2))
    return value


def figures(value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=['clean','high sub','mid sub','low sub','high urban','mid urban','low urban']
    x=np.arange(7);fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for arm,offset,color in (('B00',-.18,'#306b9c'),('B01',.18,'#c36935')):
        means=[value['metrics'][view][arm]['accuracy']['mean'] for view in VIEWS]
        sd=[value['metrics'][view][arm]['accuracy']['sample_SD'] for view in VIEWS]
        axes[0].bar(x+offset,means,.35,yerr=sd,capsize=3,color=color,label=arm)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Source accuracy (%)',ylim=(0,100),title='Mean and sample SD across two model seeds');axes[0].legend(frameon=False)
    for seed,marker in zip(SEEDS,('o','s')):
        gains=[value['metrics'][view]['paired_accuracy_gain_pp']['per_seed'][str(seed)] for view in VIEWS]
        axes[1].plot(x,gains,marker=marker,label='seed'+str(seed))
    axes[1].axhline(0,color='#666666',lw=1)
    axes[1].set(xticks=x,xticklabels=labels,ylabel='B01 minus B00 (pp)',title='Every paired seed retained');axes[1].legend(frameon=False)
    for ax in axes:ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Fixed RX1 source holdout / same1260 physical packets / no formal target or joint claim\nSource confirmation: '+('PASS' if value['qualification']['qualified'] else 'FAIL'),fontsize=12)
    fig.savefig(SECOND/'two_seed_source_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':figures(analyze())
