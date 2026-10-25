"""All three fixed variants are reported, with paired seeds and original raw-F goal."""
import csv
from pathlib import Path
from statistics import mean
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.plan import TASKS,VIEWS
from experiments.pamf_channel_pl.results import moments,combine_populations
from .plan import RUN,matrix_rows,SEEDS


def analyze(rt):
    root=Path(rt['run_root']);refs=read(root/'reference_scores.json');records=[];cells={}
    rows=matrix_rows(rt['run_id']);candidates=tuple(dict.fromkeys(r['method'] for r in rows));protocol=rows[0]['protocol']
    scientific_run=RUN if protocol=='F0' else rt['run_id']
    controls=('supervised','ordinary_PL','REPL')
    for row in rows:
        rid=row['row_id'];paths=dict(refs[rid],**{row['method']:str(root/'readouts'/rid/'scores/scores.json')})
        for method,path in paths.items():
            result=read(path)
            expected_row=rid if method==row['method'] else protocol+'-'+method+'-s'+str(row['model_seed'])
            if (result['status']!='SCORED' or not result['truth_last'] or result['model_seed']!=row['model_seed']
                or result['row_id']!=expected_row or method==row['method'] and result['run_id']!=scientific_run):
                raise ValueError('Complete matching fixed-row truth-last score required')
            overall={(r['test'],r['readout'],r['view']):r for r in result['results'] if r['dimension']=='overall' and r['stratum']=='ALL'}
            for route in ('raw','DURA-LU'):
                for view in VIEWS:
                    values={t:overall[t,route,view] for t in TASKS};values['all_target_dates']=combine_populations(values)
                    for task,value in values.items():
                        key=(row['model_seed'],method,task,route,view);accuracy=100.*value['accuracy']
                        if key in cells:
                            if cells[key]!=accuracy:raise ValueError('Repeated frozen control differs')
                            continue
                        cells[key]=accuracy
                        records.append(dict(model_seed=row['model_seed'],method=method,test=task,readout=route,
                            view=view,count=value['count'],accuracy_percent=accuracy))
                for task in (*TASKS,'all_target_dates'):
                    cells[row['model_seed'],method,task,route,'six_practical_mean']=mean(cells[row['model_seed'],method,task,route,v] for v in VIEWS[1:])
    summary=[]
    for task in (*TASKS,'all_target_dates'):
        for route in ('raw','DURA-LU'):
            for view in (*VIEWS,'six_practical_mean'):
                value=dict(protocol=protocol,test=task,readout=route,view=view,seeds=3)
                for method in (*controls,*candidates):
                    stat=moments([cells[s,method,task,route,view] for s in SEEDS])
                    value.update({method+'_'+k:v for k,v in stat.items() if k!='n'})
                for candidate in candidates:
                    for control in controls:
                        stat=moments([cells[s,candidate,task,route,view]-cells[s,control,task,route,view] for s in SEEDS])
                        value.update({candidate+'_minus_'+control+'_'+k+'_pp':v for k,v in stat.items() if k!='n'})
                summary.append(value)
    out=root/'analysis';out.mkdir(exist_ok=False)
    for name,values in [('per_seed_overall.csv',records),('paired_summary.csv',summary)]:
        with (out/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    endpoints=[]
    for candidate in candidates:
        for route in ('raw','DURA-LU'):
            items={r['view']:r for r in summary if r['test']=='all_target_dates' and r['readout']==route}
            item=dict(method=candidate,readout=route)
            for endpoint in ('clean','six_practical_mean'):
                r=items[endpoint];item[endpoint+'_gain_over_best_control_pp']=r[candidate+'_mean']-max(r[m+'_mean'] for m in controls[:2])
            item['goal_pass']=item['clean_gain_over_best_control_pp']>=1. and item['six_practical_mean_gain_over_best_control_pp']>1.5
            endpoints.append(item)
    write(out/'complete.json',dict(status='ANALYZED',run_id=scientific_run,new_students=len(rows),student_seeds=list(SEEDS),
        protocol=protocol,all_target_dates_endpoints=endpoints,original_F0_goal_applicable=protocol=='F0',
        original_raw_goal_met=protocol=='F0' and any(v['goal_pass'] for v in endpoints if v['readout']=='raw'),
        DURA_scope='same-contract frozen G-LU reference,reported separately',
        claim='Adaptive research on previously exposed benchmark; fixed E50/three variants, no target-driven training or checkpoint selection.',
        new_G=0,new_IQ_cache=0))
