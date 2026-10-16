"""Synthetic metadata falsifications, no new model/IQ/checkpoint work."""
from collections import Counter
from copy import deepcopy
import csv,json,math,tempfile
from pathlib import Path
import numpy as np
from . import trace as t
from .harvest import qualify

def checks():
    negatives=0
    def reject(fn):
        nonlocal negatives
        try:fn()
        except (ValueError,KeyError,TypeError):negatives+=1;return
        raise AssertionError('Invalid metadata accepted')
    primary=[[1,(1,3,4,8)[i%4],i%3+1,i,10000+i] for i in range(128)]
    def fixture(step,eta,mode):
        e=(step-1)//222+1;ns=f'FcrossRX/train/e{e}/step{step}/L_S';rr=[]
        for p in primary:
            pid='source:'+str(p[4]);nominal=str(np.random.default_rng(t.seed(20261016421,ns,'scene',pid)).choice(t.SCENES));u=float(np.random.default_rng(t.seed(20261016651,ns,'mixture',pid)).random());hard=u<eta;pp=[]
            for i in range(5 if hard else 1):
                scene=nominal if i==0 else str(np.random.default_rng(t.seed(20261016651,ns,'urban_scene',i,pid)).choice(t.SCENES[1:]));good=hard and ((mode=='first' and i==0 and scene in t.SCENES[1:]) or (mode!='fallback' and i==2))
                prop=dict(index=i,scene=scene,seed=t.seed(20261016421,ns,scene,pid) if i==0 else t.seed(20261016651,ns,'proposal',i,scene,pid),state='LOS' if good else 'BLOCKED',lock='locked' if good else 'unlocked',added_noise_db=8. if good else -2.,quality_snr_db=8.,eligible=good)
                pp.append(prop)
                if good:break
            hits=[i for i,v in enumerate(pp) if v['eligible']] if hard else [];chosen=hits[0] if hits else 0
            rr.append(dict(sample_id=pid,session_id=f'rx{p[1]}:day{p[2]}:eq1',eta=eta,u=u,hard_requested=hard,hard_accepted=bool(hits),selected_index=chosen,nominal_scene=nominal,proposals=pp))
        return dict(step=step,epoch=e,eta=eta,records=rr)
    def replay(r,step,eta):
        c=Counter();v=Counter();cov=Counter();etas=[0.]*20;etas[(step-1)//222]=eta;t.batch(r,primary,step,etas,c,v,cov);return c
    fixtures=[(1,0.,'fallback'),(1333,.5,'late'),(1333,.5,'first'),(1333,.5,'fallback')]
    for step,eta,mode in fixtures:
        r=fixture(step,eta,mode);c=replay(r,step,eta);assert c['packets']==128
    r=fixture(1333,.5,'late')
    for key,value in [('sample_id','source:target'),('session_id','rx6:day1:eq1'),('eta',.6),('u',-.1),('hard_requested','wrong'),('selected_index',5),('hard_accepted','wrong'),('nominal_scene','practical_high')]:
        x=deepcopy(r);x['records'][0][key]=value;reject(lambda:replay(x,1333,.5))
    for key,value in [('seed',0),('index',3),('eligible','wrong'),('lock','bad'),('state','bad')]:
        x=deepcopy(r);x['records'][0]['proposals'][0][key]=value;reject(lambda:replay(x,1333,.5))
    fallback=fixture(1333,.5,'fallback');x=deepcopy(fallback);v=next(v for v in x['records'] if v['hard_requested']);v['selected_index']=4;reject(lambda:replay(x,1333,.5))
    cells=[]
    for i,(tx,rx,day) in enumerate((a,b,c) for a in range(6) for b in t.FIT for c in (1,2,3)):
        n=395 if i<48 else 394;cells.append(dict(TX=tx,RX=rx,day=day,count=n,correct=int(.9*n),margin_sum=.12*n,CE_sum=.3*n))
    count=sum(c['count'] for c in cells);m=dict(L_packets=count,source_cells=72,all_72_fit_cells=True,accuracy=sum(c['correct'] for c in cells)/count,mean_margin=sum(c['margin_sum'] for c in cells)/count,mean_CE=sum(c['CE_sum'] for c in cells)/count,p10_cell_accuracy=float(np.quantile([c['correct']/c['count'] for c in cells],.1)),p10_cell_margin=float(np.quantile([c['margin_sum']/c['count'] for c in cells],.1)),gradient=1.)
    rows=[];mech=[];eta=0.
    for e in range(1,21):
        nex=min(.5,round(eta+.1,10)) if e>=5 else eta;r=dict(epoch=e,eta_used=eta,eta_next=nex,absolute_ready=True,stable=True,metrics=m,cells=cells);rows.append(r);mech.append(dict(curriculum=dict(r,extra_model_forwards=0,held_V_used=False,U_truth_read=False,target_access=False)));eta=nex
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        def save(h=rows,mm=mech):
            (p/'curriculum_history.json').write_text(json.dumps(dict(records=h)),encoding='utf-8');(p/'mechanism_progress.json').write_text(json.dumps(dict(records=mm)),encoding='utf-8')
        with (p/'metrics_epoch.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['train_grad_before_clip']);w.writeheader();w.writerows([dict(train_grad_before_clip=1.) for _ in rows])
        save();assert len(t.history(p))==20
        for field,value in [('eta_next',.9),('absolute_ready',False),('stable',False),('epoch',9)]:
            x=deepcopy(rows);x[5][field]=value;save(x);reject(lambda:t.history(p))
        x=deepcopy(rows);x[5]['cells'][0]['RX']=6;save(x);reject(lambda:t.history(p))
        x=deepcopy(mech);x[5]['curriculum']['held_V_used']=True;save(rows,x);reject(lambda:t.history(p))
    models={a:dict(held_V=dict(accuracy=.8),seen_V=dict(source_val_accuracy=.9),practical={s:dict(accuracy=.5,macro_f1=.5) for s in ('practical_high_urban','practical_mid_urban','practical_low_urban')},complete_health=True) for a in ('C0','C1','C2')}
    models['C2'].update(curriculum=dict(max_eta=.5),training_practical=dict(hard_requested=1))
    q=dict(status='PASS',checks=dict(clean=True,seen=True,urban=True,low=True,urban_F1=True,activation=True,complete_health=True),source_only=True,target_access=False,model_seed_count=1,clean_drop_vs_C0_pp=0.,seen_drop_vs_C0_pp=0.,urban_gain_vs_C1_pp=0.,low_gain_vs_C1_pp=0.,urban_F1_gain_vs_C1_pp=0.)
    assert qualify(models,q)==q
    for field,value in [('target_access',True),('model_seed_count',3),('low_gain_vs_C1_pp',1.)]:
        x=deepcopy(q);x[field]=value;reject(lambda:qualify(models,x))
    x=deepcopy(models);x['C2']['curriculum']['max_eta']=0;reject(lambda:qualify(x,q))
    return dict(status='PASS',metadata_negative_cases=negatives,synthetic_draw_records=512,synthetic_history_epochs=20,IQ_checkpoint_network_access=0)
if __name__=='__main__':print(json.dumps(checks()))
