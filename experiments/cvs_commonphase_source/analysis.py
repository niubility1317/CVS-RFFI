"""Independent all-decision/logit/stratum recount; no Torch, IQ or model load."""
import json,math
from pathlib import Path
from experiments.cvs_iq_timing20.harvest import score
from . import design as d
OUT=d.ROOT/d.LOCAL/'final'

def analyze():
    complete=d.read(OUT/'completion.json');result=d.read(OUT/'result.json');smoke=d.read(OUT/'real_checkpoint_smoke.json')
    assert complete['status']=='ANALYZED_SOURCE_ONLY' and complete['source_decisions']==15120 and complete['actual_NN_packets']==15122 and complete['optimizer_updates']==0 and not complete['target_access'] and not complete['automatic_training']
    assert smoke['status']=='PASS' and smoke['synthetic_NN_packets']==2 and smoke['source_IQ_reads']==0 and not smoke['target_access']
    assert result['model_state_unchanged'] and result['default_head_only'] and result['NoExplicitCFO'] and result['head_fit_updates']==0 and not result['actual_optimizer_updates'] and not result['As_generations'] and not result['teacher_forwards']
    assert not any(result[k] for k in ('U_used','V_used','held_RX1_used','target_access'))
    expected=d.read(OUT/'fit_L_ids.json');parent=d.read(d.ROOT/'local_artifacts/F_crossrx_satmix20_20261016/final/B00-RX1-s2026092701/source/source_contract.json')
    assert len(expected)==len(set(expected))==5040 and set(expected)==set(parent['role_ids']['L_s'])
    predictions={};scores={};strata={};maxCE=0.
    for view,angle in d.VIEWS.items():
        p=d.read(OUT/'predictions'/(view+'.json'));predictions[view]=p
        assert p['view']==view and p['angle_rad']==angle and p['physical_ids']==expected and len(p['labels'])==len(p['predictions'])==len(p['logits'])==len(p['RX'])==len(p['day'])==5040 and p['role']=='fitL' and not p['head_fit_updates'] and not p['target_access']
        assert set(p['RX'])==set(d.FIT) and set(p['day'])=={1,2,3}
        if view!='identity':assert all(p[k]==predictions['identity'][k] for k in ('physical_ids','labels','RX','day'))
        assert all(len(v)==6 and all(math.isfinite(x) for x in v) for v in p['logits'])
        assert p['predictions']==[max(range(6),key=v.__getitem__) for v in p['logits']]
        s=score(p['labels'],p['predictions']);saved=result['scores'][view]
        assert s['count']==5040 and s['confusion']==saved['confusion']
        assert math.isclose(s['accuracy'],saved['accuracy'],abs_tol=1e-12) and math.isclose(s['macro_F1'],saved['macro_F1'],abs_tol=1e-12)
        ce=sum(max(z)+math.log(sum(math.exp(v-max(z)) for v in z))-z[y] for z,y in zip(p['logits'],p['labels']))/5040
        maxCE=max(maxCE,abs(ce-saved['mean_CE']));assert abs(ce-saved['mean_CE'])<1e-10
        scores[view]=dict(s,mean_CE=ce);strata[view]={}
        for name,values,allowed,count in (('RX',p['RX'],d.FIT,1260),('TX',p['labels'],range(6),840),('day',p['day'],(1,2,3),1680)):
            strata[view][name]={}
            for value in allowed:
                take=[i for i,v in enumerate(values) if v==value];a=score([p['labels'][i] for i in take],[p['predictions'][i] for i in take]);b=saved['by_'+name][str(value)]
                assert a['count']==count and a['confusion']==b['confusion'] and math.isclose(a['accuracy'],b['accuracy'],abs_tol=1e-12) and math.isclose(a['macro_F1'],b['macro_F1'],abs_tol=1e-12)
                strata[view][name][str(value)]=a
        cells=saved['cells'];assert len(cells)==72
        for cell in cells:
            take=[i for i,(r,day,tx) in enumerate(zip(p['RX'],p['day'],p['labels'])) if (r,day,tx)==(cell['RX'],cell['day'],cell['TX'])]
            s=score([p['labels'][i] for i in take],[p['predictions'][i] for i in take]);assert s['count']==70 and s['confusion']==cell['confusion'] and math.isclose(s['accuracy'],cell['accuracy'],abs_tol=1e-12) and math.isclose(s['macro_F1'],cell['macro_F1'],abs_tol=1e-12)
    paired={};conditions={}
    raw=predictions['identity']
    for view in ('phase_plus','phase_minus'):
        p=predictions[view];repair=sum(a!=y and b==y for a,b,y in zip(raw['predictions'],p['predictions'],raw['labels']));broken=sum(a==y and b!=y for a,b,y in zip(raw['predictions'],p['predictions'],raw['labels']))
        drop=100*(scores['identity']['accuracy']-scores[view]['accuracy']);byRX={str(r):100*(strata['identity']['RX'][str(r)]['accuracy']-strata[view]['RX'][str(r)]['accuracy']) for r in d.FIT}
        assert math.isclose(drop,100*(broken-repair)/5040,abs_tol=1e-10)
        paired[view]=dict(accuracy_drop_pp=drop,macro_F1_drop_pp=100*(scores['identity']['macro_F1']-scores[view]['macro_F1']),repaired=repair,broken=broken,RX_drop_pp=byRX,RX_at_least1pp=sum(v>=1 for v in byRX.values()))
        conditions[view+'_overall']=drop>=2;conditions[view+'_RX']=paired[view]['RX_at_least1pp']>=3
    decision='PASS' if all(conditions.values()) else 'FAIL'
    assert conditions==result['qualification']['conditions'] and decision==result['qualification']['decision']==complete['decision'] and max(result['power_relative_error_max'].values())<=1e-6
    log=(OUT/'phase_source.stdout.log').read_text(encoding='utf8',errors='replace');assert 'SMOKE PASS real matching checkpoint' in log and 'COMPLETE ' in log and not any(t in log for t in ('Traceback (most recent call last)','FloatingPointError'))
    summary=dict(status='ANALYZED_SOURCE_ONLY_VERIFIED',run_id=d.RUN,model_seed=2026092701,decision=decision,conditions=conditions,scores=scores,strata=strata,paired=paired,all_15120_default_head_decisions_and_logits_recounted=True,source_cells_recounted=216,max_CE_recount_error=maxCE,source_physical_packets=5040,source_NN_packets=15120,synthetic_smoke_NN_packets=2,total_NN_packets=15122,optimizer_updates=0,head_fit_updates=0,model_state_unchanged=True,source_fit_only=True,held_RX1_used=False,U_used=False,V_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,power_relative_error_max=result['power_relative_error_max'],diagnostic_wall_seconds=result['elapsed_seconds'],full_stdout_lines=len(log.splitlines()))
    d.write(OUT/'analysis_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ('status','decision','conditions','paired','total_NN_packets','diagnostic_wall_seconds')},ensure_ascii=False,indent=2));return summary

def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(4)
    for view,offset,color in (('identity',-.25,'#306b9c'),('phase_plus',0.,'#c36935'),('phase_minus',.25,'#718a49')):
        values=[100*a['strata'][view]['RX'][str(r)]['accuracy'] for r in d.FIT]
        axes[0].bar(x+offset,values,.24,color=color,label=view)
    axes[0].set(xticks=x,xticklabels=d.FIT,ylabel='Frozen default-head fit-L accuracy (%)',ylim=(0,100),xlabel='Fit RX');axes[0].legend(frameon=False)
    for view,offset,color in (('phase_plus',-.15,'#c36935'),('phase_minus',.15,'#718a49')):
        axes[1].bar(x+offset,[a['paired'][view]['RX_drop_pp'][str(r)] for r in d.FIT],.28,color=color,label=view)
    axes[1].axhline(1,color='#666666',ls='--',label='Preregistered per-RX1pp');axes[1].set(xticks=x,xticklabels=d.FIT,ylabel='Identity accuracy minus phase view (pp)',xlabel='Fit RX');axes[1].legend(frameon=False)
    fig.suptitle('F+CROSS_RX / bounded common phase / source sensitivity '+a['decision']+'\nNo update, no held RX or target; not Ag efficacy')
    fig.savefig(OUT/'phase_source_results.png',dpi=180);plt.close(fig)

if __name__=='__main__':figures(analyze())
