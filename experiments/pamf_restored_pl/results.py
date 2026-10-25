"""Read completed independent scores only; exact joint goal and paired seeds."""
import csv
from fractions import Fraction
from pathlib import Path
from statistics import mean,stdev
from experiments.cric_dura_comparison.io import read
from experiments.practical_full_2x2.cache import atomic
from experiments.practical_full_2x2.plan import TASKS,VIEWS
from .plan import RUN,rows,SEEDS,METHODS


def partial(rt,require_all=False):
    root=Path(rt['run_root']);refs=read(root/'reference_scores.json');out=root/'analysis';out.mkdir(exist_ok=True)
    summaries=[];per_seed=[];groups=[];complete_rows=[]
    for candidate in METHODS:
        finished=[r for r in rows() if r['method']==candidate and (root/'readouts'/r['row_id']/'scores/complete.json').exists()
            and read(root/'readouts'/r['row_id']/'scores/complete.json')['status']=='SCORED_COMPLETE']
        complete_rows.extend(r['row_id'] for r in finished);cells={};pop={}
        for row in finished:
            seed=row['model_seed'];rid=row['row_id']
            paths=dict(refs[rid],**{candidate:str(root/'readouts'/rid/'scores/scores.json')})
            for method,path in paths.items():
                score=read(path);expected=rid if method==candidate else 'F0-'+method+'-s'+str(seed)
                if (score['status']!='SCORED' or not score['truth_last'] or score['model_seed']!=seed or score['row_id']!=expected
                        or method==candidate and score['run_id']!=RUN):raise ValueError('Matching independent truth-last score required')
                values={(r['test'],r['readout'],r['view']):r for r in score['results'] if r['dimension']=='overall' and r['stratum']=='ALL'}
                for route in ('raw','DURA-LU'):
                    for view in VIEWS:
                        correct={}
                        for task in TASKS:
                            r=values[task,route,view];cm=r['confusion'];n=r['count'];k=sum(cm[i][i] for i in range(len(cm)))
                            if sum(map(sum,cm))!=n or abs(k/n-r['accuracy'])>1e-12 or pop.setdefault(task,n)!=n:
                                raise ValueError('Fixed matched population/confusion/count required')
                            correct[task]=k;cells[seed,method,task,route,view]=100*Fraction(k,n)
                        n=pop['cross_rx']+pop['cross_both']
                        if n!=168000:raise ValueError('Original all-targetRX endpoint required')
                        cells[seed,method,'all_target_dates',route,view]=100*Fraction(correct['cross_rx']+correct['cross_both'],n)
                    for task in (*TASKS,'all_target_dates'):
                        cells[seed,method,task,route,'six_practical_mean']=sum(cells[seed,method,task,route,v] for v in VIEWS[1:])/6
            for task in (*TASKS,'all_target_dates'):
                for route in ('raw','DURA-LU'):
                    for view in (*VIEWS,'six_practical_mean'):
                        record=dict(candidate=candidate,model_seed=seed,test=task,readout=route,view=view)
                        for method in ('supervised','ordinary_PL','REPL',candidate):record[method+'_accuracy_percent']=float(cells[seed,method,task,route,view])
                        per_seed.append(record)
        group=dict(method=candidate,completed_seeds=sorted(r['model_seed'] for r in finished),required_seeds=list(SEEDS),
            status='WAITING_FOR_ALL_THREE_SEEDS',goal_pass=False)
        if len(finished)==3:
            group['status']='COMPLETE_THREE_SEED_SCORING';means={}
            for task in (*TASKS,'all_target_dates'):
                for route in ('raw','DURA-LU'):
                    for view in (*VIEWS,'six_practical_mean'):
                        record=dict(candidate=candidate,test=task,readout=route,view=view,seeds=3)
                        for method in ('supervised','ordinary_PL','REPL',candidate):
                            values=[cells[s,method,task,route,view] for s in SEEDS];m=sum(values)/3
                            means[method,task,route,view]=m
                            record[method+'_mean']=float(m);record[method+'_SD']=stdev(map(float,values))
                        for control in ('supervised','ordinary_PL','REPL'):
                            diffs=[cells[s,candidate,task,route,view]-cells[s,control,task,route,view] for s in SEEDS]
                            record['delta_vs_'+control+'_mean_pp']=float(sum(diffs)/3)
                            record['delta_vs_'+control+'_paired_SD_pp']=stdev(map(float,diffs))
                        summaries.append(record)
            gains=[]
            for route,endpoint,threshold,strict in [('raw','clean',Fraction(1),False),
                    ('raw','six_practical_mean',Fraction(3,2),True),('DURA-LU','six_practical_mean',Fraction(0),True)]:
                gain=means[candidate,'all_target_dates',route,endpoint]-max(means[m,'all_target_dates',route,endpoint] for m in ('supervised','ordinary_PL'))
                gains.append(dict(route=route,endpoint=endpoint,exact_gain_pp=str(gain),gain_pp=float(gain),
                    pass_=gain>threshold if strict else gain>=threshold))
            group.update(goals=gains,goal_pass=all(v['pass_'] for v in gains))
        groups.append(group)
    def save(name,records):
        if not records:return
        fields=list(dict.fromkeys(k for r in records for k in r))
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(records)
    save('per_seed_overall.csv',per_seed);save('paired_summary.csv',summaries)
    status='ANALYZED' if len(complete_rows)==9 else 'PARTIAL'
    if require_all and status!='ANALYZED':raise ValueError('All nine fixed rows required')
    atomic(out/'complete.json',dict(status=status,run_id=RUN,complete_rows=complete_rows,groups=groups,
        original_goal_met=any(g['goal_pass'] for g in groups),
        goal='rawclean>=bestcontrol+1pp AND rawsix>bestcontrol+1.5pp AND sameG-LU six>bothcontrols',
        target_feedback_to_training=False,claim='adaptive research on exposed benchmark;SD not significance;all fixed rows reported'))
