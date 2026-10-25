"""CPU-only complete paired summaries; no result can modify a source worker."""
import csv
from pathlib import Path
from statistics import mean,stdev
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.plan import TASKS,VIEWS
from .training_plan import rows,TRAIN_RUN
from .plan import PROTOCOLS,SEEDS


def moments(values):
    return dict(mean=mean(values),sd=stdev(values) if len(values)>1 else None,n=len(values))


def combine_populations(values):
    parts=[values[t] for t in ('cross_rx','cross_both')];n=sum(v['count'] for v in parts)
    return dict(count=n,accuracy=sum(v['count']*v['accuracy'] for v in parts)/n)


def analyze(rt):
    root=Path(rt['run_root']);refs=read(root/'reference_scores.json');records=[];cells={}
    for row in rows():
        rid=row['row_id'];paths=dict(refs[rid],ChannelPL=str(root/'readouts'/rid/'scores/scores.json'))
        for method,path in paths.items():
            result=read(path)
            if result['status']!='SCORED' or not result['truth_last'] or result['model_seed']!=row['model_seed']:
                raise ValueError('Complete same-seed independent score required')
            if method=='ChannelPL' and (result['run_id']!=TRAIN_RUN or result['row_id']!=rid):
                raise ValueError('Wrong ChannelPL scored source')
            overall={(r['test'],r['readout'],r['view']):r for r in result['results'] if r['dimension']=='overall' and r['stratum']=='ALL'}
            for route in ('raw','DURA-LU'):
                for view in VIEWS:
                    values={t:overall[t,route,view] for t in TASKS}
                    values['all_target_dates']=combine_populations(values)
                    for task,value in values.items():
                        key=(row['protocol'],row['model_seed'],method,task,route,view)
                        cells[key]=100.*value['accuracy']
                        records.append(dict(protocol=row['protocol'],model_seed=row['model_seed'],method=method,
                            test=task,readout=route,view=view,count=value['count'],accuracy_percent=cells[key]))
                for task in (*TASKS,'all_target_dates'):
                    key=(row['protocol'],row['model_seed'],method,task,route,'six_practical_mean')
                    cells[key]=mean(cells[row['protocol'],row['model_seed'],method,task,route,v] for v in VIEWS[1:])
    summary=[]
    for p in PROTOCOLS:
        for task in (*TASKS,'all_target_dates'):
            for route in ('raw','DURA-LU'):
                for view in (*VIEWS,'six_practical_mean'):
                    value=dict(protocol=p,test=task,readout=route,view=view,seeds=3)
                    for method in ('supervised','ordinary_PL','REPL','ChannelPL'):
                        stat=moments([cells[p,s,method,task,route,view] for s in SEEDS])
                        value.update({method+'_'+k:v for k,v in stat.items() if k!='n'})
                    for control in ('supervised','ordinary_PL','REPL'):
                        stat=moments([cells[p,s,'ChannelPL',task,route,view]-cells[p,s,control,task,route,view] for s in SEEDS])
                        value.update({'ChannelPL_minus_'+control+'_'+k+'_pp':v for k,v in stat.items() if k!='n'})
                    summary.append(value)
    out=root/'analysis';out.mkdir(exist_ok=False)
    for name,values in [('per_seed_overall.csv',records),('paired_summary.csv',summary)]:
        with (out/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    primary=[]
    for route in ('raw','DURA-LU'):
        items={r['view']:r for r in summary if r['protocol']=='F0' and r['test']=='all_target_dates' and r['readout']==route}
        item=dict(readout=route)
        for endpoint in ('clean','six_practical_mean'):
            r=items[endpoint];gap=r['ChannelPL_mean']-max(r['ordinary_PL_mean'],r['supervised_mean'])
            item[endpoint+'_gain_over_best_control_pp']=gap
        item['goal_pass']=item['clean_gain_over_best_control_pp']>=1. and item['six_practical_mean_gain_over_best_control_pp']>1.5
        primary.append(item)
    macro=[]
    for task in TASKS:
        for route in ('raw','DURA-LU'):
            for endpoint in ('clean','six_practical_mean'):
                values=[r['ChannelPL_minus_ordinary_PL_mean_pp'] for r in summary if r['test']==task and r['readout']==route and r['view']==endpoint]
                macro.append(dict(test=task,readout=route,view=endpoint,settings=11,
                    equal_setting_macro_delta_pp=mean(values),positive_settings=sum(v>0 for v in values)))
    write(out/'complete.json',dict(status='ANALYZED',run_id=TRAIN_RUN,models=33,model_seeds=list(SEEDS),
        primary_original_F0_168000=primary,original_target_met_in_both_routes=all(v['goal_pass'] for v in primary),
        equal_setting_macro=macro,claim='Adaptive development on previously exposed benchmark; no new blind or significance claim.',
        no_target_feedback_to_training=True,new_G=0,new_IQ_cache=0))
