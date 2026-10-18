"""Current P43 frozen-head source controls; saved logits/metadata only.

No IQ, checkpoint, model, truth or remote operations. Every control uses strata
from the same full observation. Channel draws are repeated observations of one
model seed. This script never decides a training promotion.
"""
import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

RUN='20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01'
PARENT='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
PANEL_RUN='20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01'
LABELS=('NOAS30','AS30')
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CONDITIONS=('full','no_noise/fixed','no_noise/total','flat_H/fixed','flat_H/total','no_carrier/fixed','no_carrier/total','no_RX_IQ/fixed','no_RX_IQ/total','actual_power_quality','edge_boundary')
STATES=('LOS','PARTIAL','BLOCKED');LOCKS=('locked','degraded','unlocked')
BRANCHES=('time','frequency','PA','reference','signature')
BRANCH_METRICS=('clean_sat_cosine','clean_norm','sat_norm')
GROUPS=('Sinc_Hz','signature','reference','default_head','PA','time','frequency','remaining','whole')
PHYSICS=('state','lock','quality_snr_db','white_proxy_quality_db','actual_waveform_quality_db','waveform_minus_white_power_db','output_frequency_hz','estimated_frequency_hz','signal_to_added_noise_db','noise_coloration_dynamic_range_db','agc_gain_db','agc_limited','residual_kernel_energy','kernel_length','numerical_delay','interior_count','residual_scale')
FIXED_PARAMETERS=('receiver_lock_status','residual_scale','estimated_frequency_hz','phase_tracking_applied','channel_equalization_applied','agc_gain_db')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row if k!='confusion'))
    with Path(path).open('x',encoding='utf-8',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def validate_predictions(x,ids,reference=None,n=1800):
    if set(x)!=set(('ids','TX','RX','day','clean','satellite')):raise ValueError('Only source panel metadata and current default-head logits')
    if x['clean'].shape!=(n,6) or x['satellite'].shape!=(2,3,11,n,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Complete finite 2x3x11 source controls required')
    if x['ids'].shape!=(n,) or x['ids'].dtype.kind not in 'US' or not np.array_equal(x['ids'],ids) or len(set(x['ids'].tolist()))!=n:raise ValueError('Unique ordered original source IDs required')
    for k,allowed in [('TX',set(range(6))),('RX',{1,3,4,6,8}),('day',{1,2,3})]:
        if x[k].shape!=(n,) or x[k].dtype.kind not in 'iu' or not set(x[k].tolist())<=allowed:raise ValueError('Visible source integer labels/RX/day only')
        if reference is not None and not np.array_equal(x[k],reference[k]):raise ValueError('Current two models must use identical source panel order')
    if n==1800:
        cells=Counter(zip(x['TX'].tolist(),x['RX'].tolist(),x['day'].tolist()))
        if len(cells)!=90 or set(cells.values())!={20}:raise ValueError('Exact source 6TX x5RX x3day x20 panel required')
    return x


def validate_physics(physics,n=1800):
    if not set(PHYSICS)<=set(physics):raise ValueError('Original complete physics fields required')
    for k,v in physics.items():
        if v.shape!=(2,3,11,n) or v.dtype.kind not in 'iufb':raise ValueError('Original physics matrix shape/type differs')
        if np.isinf(v).any():raise ValueError('Producer uses finite numeric physics or NaN for missing ratios')
    for k in ('state','lock'):
        v=physics[k]
        if not np.isfinite(v).all() or not np.isin(v,(0,1,2)).all():raise ValueError('Known original full state/lock indices required')
    return physics


def validate_branch(x,n=1800):
    if tuple(x['names'].tolist())!=BRANCHES or tuple(x['metrics'].tolist())!=BRANCH_METRICS or x['values'].shape!=(2,3,n,5,3):raise ValueError('Existing five branch full-versus-clean readouts required')
    v=x['values']
    if not np.isfinite(v).all() or (v[...,1:]<0).any() or (np.abs(v[...,0])>1.00001).any():raise ValueError('Finite branch norms/cosines required')
    return v


def validate_result(r,label):
    if r.get('status')!='SOURCE_DIAGNOSED' or r.get('NN')!=121232 or r.get('prediction_NN')!=120600 or r.get('gradient_forward_NN')!=630 or r.get('synthetic_smoke_NN')!=2 or r.get('training_updates')!=0 or r.get('head_fit_updates')!=0 or r.get('target_access')is not False or r.get('state_unchanged')is not True:raise ValueError('Current frozen student/default-head source-only cost/state required')
    arm='B00'if label=='NOAS30'else'B01'
    if r.get('label')!=label or r.get('arm')!=arm:raise ValueError('Logical current label and original P43 arm differ')
    provenance=r.get('provenance',{})
    if not isinstance(provenance,dict)or provenance.get('run_id')!=PARENT or provenance.get('row_id')!=arm+'-s2026092701' or provenance.get('state_key')!='model' or PARENT not in str(provenance.get('checkpoint','')) or arm+'-s2026092701/source/final_ssdg.pth'not in str(provenance.get('checkpoint','')):raise ValueError('Exact P43 current FINAL30 student provenance required; no old P24 checkpoint fallback')
    gates=r.get('gates',{})
    if gates.get('kind')!='learned static coordinate gains; not invented per-packet gate weights':raise ValueError('Static existing coordinate gains must not become per-packet gate claims')
    for k in ('PA','reference','signature'):
        value=np.asarray(gates.get(k),float)
        if not value.size or not np.isfinite(value).all():raise ValueError('Finite actual learned static coordinate gains required')
    gradients=r.get('gradients',[])
    if len(gradients)!=6 or {(g['draw'],g['scene'])for g in gradients}!={(d,s)for d in (0,1)for s in SCENES}:raise ValueError('One shared clean90 plus six full90 gradient comparisons required')
    rows=[]
    for g in gradients:
        if g['count']!=90 or not all(math.isfinite(g[k])for k in ('clean_CE','satellite_CE')) or set(g['groups'])!=set(GROUPS) or 'not stochastic training gradients'not in g['scope']:raise ValueError('Deterministic frozen eval CE gradient scope required')
        for name,v in g['groups'].items():
            a,b,dot=v['clean_norm'],v['satellite_norm'],v['dot'];cos=v['cosine']
            if v['finite']is not True or not all(math.isfinite(float(q))for q in (a,b,dot)) or min(a,b)<0:raise ValueError('Finite readonly gradient summary required')
            if a*b:
                if cos is None or not math.isfinite(cos) or abs(cos-dot/(a*b))>1e-8 or abs(cos)>1.000001:raise ValueError('Gradient cosine/norm/dot mismatch')
            elif cos is not None:raise ValueError('Zero gradient norms require undefined cosine')
            rows.append(dict(label=label,draw=g['draw'],scene=g['scene'],count=90,group=name,clean_CE=g['clean_CE'],satellite_CE=g['satellite_CE'],**v,scope=g['scope']))
    for g in gradients:
        parts=[v for name,v in g['groups'].items()if name!='whole'];whole=g['groups']['whole']
        if not np.isclose(whole['clean_norm']**2,sum(v['clean_norm']**2 for v in parts),rtol=1e-8,atol=1e-12)or not np.isclose(whole['satellite_norm']**2,sum(v['satellite_norm']**2 for v in parts),rtol=1e-8,atol=1e-12)or not np.isclose(whole['dot'],sum(v['dot']for v in parts),rtol=1e-8,atol=1e-12):raise ValueError('Whole eval gradient must equal disjoint parameter groups')
    if len({g['clean_CE']for g in gradients})!=1:raise ValueError('The same clean90 gradient reference must be reused across six comparisons')
    return rows


def check_record(record,index,physics,x,n):
    draw=index//(3*n);si=index//n%3;i=index%n
    if draw>=2 or record.get('draw')!=draw or record.get('scene')!=SCENES[si] or record.get('index')!=i or record.get('physical_id')!=str(x['ids'][i]):raise ValueError('Physical record order differs')
    if any(record.get(k)!=int(x[k][i])for k in ('TX','RX','day')) or record.get('session')!='rx'+str(int(x['RX'][i]))+':day'+str(int(x['day'][i]))+':eq1' or type(record.get('seed'))is not int:raise ValueError('Physical source label/session/named seed differs')
    conditions=record.get('conditions',{})
    if tuple(conditions)!=CONDITIONS:raise ValueError('Fixed original eleven condition order required')
    full=conditions['full']
    for ci,condition in enumerate(CONDITIONS):
        v=conditions[condition]
        if v.get('state')!=STATES[int(physics['state'][draw,si,ci,i])] or v.get('receiver_lock_status')!=LOCKS[int(physics['lock'][draw,si,ci,i])]:raise ValueError('All-condition state/lock metadata differs')
        if v.get('true_total_SNR_known')is not False or v.get('channel_equalization_applied')is not False:raise ValueError('Added-noise ratio is not total SNR; existing EQoff means G1')
        if v.get('boundary')!=('edge'if condition=='edge_boundary'else'reflect'):raise ValueError('Original boundary control differs')
        for k in PHYSICS[2:]:
            a=physics[k][draw,si,ci,i];b=v.get(k)
            if b is None:
                if not np.isnan(a):raise ValueError('Missing/numeric physics mismatch: '+k)
            elif not math.isfinite(float(b)) or not np.isclose(a,b,rtol=1e-10,atol=1e-10):raise ValueError('Saved physics mismatch: '+k)
        if condition.endswith('/fixed') or condition=='edge_boundary':
            if any(v[k]!=full[k]for k in FIXED_PARAMETERS):raise ValueError('Fixed receiver state/compensation/AGC changed')
        if condition.endswith('/fixed') and v.get('effect')!='fixed_receiver_state_and_compensation':raise ValueError('Controlled effect label differs')
        if condition.endswith('/total') and v.get('effect')!='recomputed_receiver_state':raise ValueError('Total effect label differs')
    return draw,si


def load_inputs(project,run_root):
    project=Path(project);expected=project/'runs'/RUN;root=Path(run_root)if run_root else expected
    if root.resolve()!=expected.resolve():raise ValueError('Only exact current source-only run root allowed')
    panel=project/'runs'/PANEL_RUN/'diagnostic';role=read(panel/'data_roles.json');obs=read(panel/'observations/completion.json');engine=read(panel/'observations/actual_engine.json')
    if role.get('role')!='V' or role.get('counts')!={'L_s':6300,'U_s':56700,'V':27000} or role.get('V_panel')!=1800 or role.get('held_RX')is not None or role.get('extra_Ldev',False)or role.get('U_truth_used')is not False or role.get('target_access')is not False or role.get('updates')!=0:raise ValueError('Original all-source V1800 only')
    ids=np.asarray(role['physical_ids'])
    if obs.get('status')!='SOURCE_OBSERVATIONS_COMPLETE' or obs.get('unique_V')!=1800 or obs.get('draws')!=2 or tuple(obs.get('scenes',()))!=SCENES or tuple(obs.get('conditions',()))!=CONDITIONS or obs.get('physical_draws')!=10800 or obs.get('target_access')is not False or obs.get('training_updates')!=0:raise ValueError('Original unchanged observations matrix required')
    if engine.get('EQ_off_G_equals_one')is not True or engine.get('formal_test_changed')is not False:raise ValueError('Original residual EQoff/G1 engine audit required')
    for scene in SCENES:
        cfg=engine['actual_configs'][scene]
        expected_cfg=dict(processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,fs_hz=25000000.,fc_hz=2450000000.,boundary='reflect',quality_adaptation_enabled=True,phase_tracking_enabled=True,iq_compensation_enabled=True)
        if any(cfg.get(k)!=v for k,v in expected_cfg.items()):raise ValueError('Fixed existing source channel differs')
    with np.load(panel/'observations/physics.npz',allow_pickle=False)as f:physics=validate_physics({k:f[k].copy()for k in f.files})
    data={};branches={};results={};gradient_rows=[]
    for label in LABELS:
        dest=root/label
        with np.load(dest/'predictions.npz',allow_pickle=False)as f:x={k:f[k].copy()for k in f.files}
        data[label]=validate_predictions(x,ids,data.get('NOAS30'))
        with np.load(dest/'branch_metrics.npz',allow_pickle=False)as f:branches[label]=validate_branch({k:f[k].copy()for k in f.files})
        results[label]=read(dest/'result.json');gradient_rows.extend(validate_result(results[label],label))
    total=0
    with gzip.open(panel/'observations/physical_records.jsonl.gz','rt',encoding='utf-8')as stream:
        for total,line in enumerate(stream,1):check_record(json.loads(line),total-1,physics,data['NOAS30'],1800)
    if total!=10800:raise ValueError('Complete physical metadata matrix required')
    completion=read(root/'completion.json')
    if completion.get('NN')!=242464 or completion.get('target_access')is not False or completion.get('training_updates')!=0 or completion.get('head_fit_updates')!=0:raise ValueError('Complete two-current-model frozen source readout required')
    return data,physics,branches,results,gradient_rows,dict(run_root=str(root),panel=str(panel),physical_records=total,engine=engine,completion=completion,fullV27000_new_readout=False,IQ_read=False,checkpoint_read=False)


def metric(z,y,fixed,own):
    n=len(y)
    if z.shape!=(n,6) or y.shape!=(n,)or fixed.shape!=(n,)or own.shape!=(n,)or not np.isfinite(z).all():raise ValueError('Aligned finite source logits/cohorts required')
    p=z.argmax(1);cm=np.bincount(y*6+p,minlength=36).reshape(6,6);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(6,float),where=den>0)
    if n:
        other=z.copy();other[np.arange(n),y]=-np.inf;margin=(z[np.arange(n),y]-other.max(1))/30.
    else:margin=np.empty(0)
    out=dict(count=n,correct=int(tp.sum()),accuracy=float(tp.sum()/n)if n else None,macro_f1=float(f1.mean())if n else None,confusion=cm.tolist(),mean_margin_over30=float(margin.mean())if n else None,small_cell=n<30,empty_cell=n==0)
    for key,co in [('fixed_NOAS30_clean',fixed),('own_model_clean',own)]:
        nc=int(co.sum());retained=int(((p==y)&co).sum());out.update({key+'_correct_count':nc,key+'_correct_retained':retained,key+'_retention':retained/nc if nc else None})
    return out


def paired(z0,z1,y,fixed,own=None):
    n=len(y);a=z0.argmax(1)==y;b=z1.argmax(1)==y;repair=int((~a&b).sum());damage=int((a&~b).sum());net=repair-damage
    out=dict(count=n,control_correct=int(a.sum()),new_correct=int(b.sum()),repairs=repair,new_errors=damage,net_repairs=net,accuracy_delta_pp=100*net/n if n else None,small_cell=n<30,empty_cell=n==0)
    for key,co in [('fixed_NOAS30_clean',fixed),('own_model_clean',own)]:
        if co is None:continue
        nc=int(co.sum());out.update({key+'_correct_count':nc,key+'_repairs':int((~a&b&co).sum()),key+'_new_errors':int((a&~b&co).sum()),key+'_control_retention':float(a[co].mean())if nc else None,key+'_new_retention':float(b[co].mean())if nc else None})
    return out


def strata(y,rx,day,full):
    n=len(y);out=[('overall','ALL',np.ones(n,bool))]
    for key,values,labels in [('RX',rx,(1,3,4,6,8)),('day',day,(1,2,3)),('TX',y,range(6)),('state',full['state'],STATES),('lock',full['lock'],LOCKS)]:
        for i,label in enumerate(labels):out.append((key,str(label),values==(i if key in('state','lock')else label)))
    out.extend(('state_lock',state+'/'+lock,(full['state']==i)&(full['lock']==j))for i,state in enumerate(STATES)for j,lock in enumerate(LOCKS))
    snr=full['signal_to_added_noise_db']
    for lo,hi,name in [(-np.inf,3.,'<3'),(3.,8.,'3-8'),(8.,15.,'8-15'),(15.,np.inf,'>=15')]:out.append(('signal_to_added_noise_db',name,(snr>=lo)&(snr<hi)))
    out.append(('signal_to_added_noise_db','MISSING',np.isnan(snr)))
    return out


def analyze(data,physics,branches,results):
    n=len(data['NOAS30']['TX']);fixed0=data['NOAS30']['clean'].argmax(1)==data['NOAS30']['TX']
    metrics=[];effects=[];training=[];interactions=[];receiver=[];branch_rows=[]
    for label in LABELS:
        x=data[label];own0=x['clean'].argmax(1)==x['TX']
        for dim,values in [('overall',['ALL']),('RX',(1,3,4,6,8)),('day',(1,2,3)),('TX',range(6))]:
            for value in values:
                mask=np.ones(n,bool)if dim=='overall'else x[dim if dim!='TX'else'TX']==value
                metrics.append(dict(label=label,draw=-1,scene='clean',condition='clean',dimension=dim,stratum=str(value),**metric(x['clean'][mask],x['TX'][mask],fixed0[mask],own0[mask])))
    x=data['NOAS30'];other=data['AS30'];n=len(x['TX'])
    for dim,vals in [('overall',['ALL']),('RX',(1,3,4,6,8)),('day',(1,2,3)),('TX',range(6))]:
        for value in vals:
            mask=np.ones(n,bool)if dim=='overall'else x[dim]==value
            q=paired(x['clean'][mask],other['clean'][mask],x['TX'][mask],fixed0[mask])
            a=metric(x['clean'][mask],x['TX'][mask],fixed0[mask],fixed0[mask]);own=other['clean'].argmax(1)==other['TX'];b=metric(other['clean'][mask],other['TX'][mask],fixed0[mask],own[mask])
            q.update(macro_f1_delta_pp=100*(b['macro_f1']-a['macro_f1'])if mask.any()else None,mean_margin_over30_delta=b['mean_margin_over30']-a['mean_margin_over30']if mask.any()else None,own_cohort_crossmodel_comparison=False)
            training.append(dict(new='AS30',control='NOAS30',draw=-1,scene='clean',condition='clean',dimension=dim,stratum=str(value),full_state_stratification=False,**q))
    contexts=[(scene,(si,))for si,scene in enumerate(SCENES)]+[('midlow_pooled',(1,2))]
    for draw in (-1,0,1):
        ds=(0,1)if draw==-1 else(draw,)
        for scene,sis in contexts:
            positions=[(dd,si)for dd in ds for si in sis];copies=len(positions);y=np.tile(data['NOAS30']['TX'],copies);rx=np.tile(data['NOAS30']['RX'],copies);day=np.tile(data['NOAS30']['day'],copies);fixed=np.tile(fixed0,copies)
            full={k:np.concatenate([physics[k][dd,si,0]for dd,si in positions])for k in ('state','lock','signal_to_added_noise_db')}
            logits={label:np.concatenate([data[label]['satellite'][dd,si]for dd,si in positions],axis=1)for label in LABELS}
            owns={label:np.tile(data[label]['clean'].argmax(1)==data[label]['TX'],copies)for label in LABELS}
            for dimension,stratum,mask in strata(y,rx,day,full):
                context=dict(draw=draw,scene=scene,dimension=dimension,stratum=stratum,full_state_stratification=True)
                values={}
                for label in LABELS:
                    for ci,condition in enumerate(CONDITIONS):
                        z=logits[label][ci,mask];q=metric(z,y[mask],fixed[mask],owns[label][mask]);values[(label,condition)]=q
                        metrics.append(dict(label=label,condition=condition,**context,**q))
                        if ci:
                            p=paired(logits[label][0,mask],z,y[mask],fixed[mask],owns[label][mask])
                            p.update(macro_f1_delta_pp=100*(q['macro_f1']-values[(label,'full')]['macro_f1'])if mask.any()else None,mean_margin_over30_delta=q['mean_margin_over30']-values[(label,'full')]['mean_margin_over30']if mask.any()else None)
                            effects.append(dict(label=label,condition=condition,**context,**p))
                for ci,condition in enumerate(CONDITIONS):
                    a=values[('NOAS30',condition)];b=values[('AS30',condition)]
                    p=paired(logits['NOAS30'][ci,mask],logits['AS30'][ci,mask],y[mask],fixed[mask])
                    p.update(macro_f1_delta_pp=100*(b['macro_f1']-a['macro_f1'])if mask.any()else None,mean_margin_over30_delta=b['mean_margin_over30']-a['mean_margin_over30']if mask.any()else None,own_cohort_crossmodel_comparison=False)
                    training.append(dict(new='AS30',control='NOAS30',condition=condition,**context,**p))
                    if ci:
                        no=a['correct']-values[('NOAS30','full')]['correct'];sat=b['correct']-values[('AS30','full')]['correct'];net=sat-no
                        interactions.append(dict(condition=condition,**context,count=int(mask.sum()),NOAS30_removal_net=no,AS30_removal_net=sat,training_x_control_net=net,training_x_control_pp=100*net/int(mask.sum())if mask.any()else None,sensitivity_reduction_NOAS_minus_AS_pp=-100*net/int(mask.sum())if mask.any()else None,removal_effects_not_additive=True))
                for label in LABELS:
                    bb=np.concatenate([branches[label][dd,si]for dd,si in positions])
                    for bi,branch in enumerate(BRANCHES):
                        for mi,name in enumerate(BRANCH_METRICS):
                            vv=bb[mask,bi,mi]
                            branch_rows.append(dict(label=label,branch=branch,metric=name,**context,count=int(mask.sum()),mean=float(vv.mean())if len(vv)else None,median=float(np.median(vv))if len(vv)else None,small_cell=len(vv)<30,empty_cell=len(vv)==0,scope='Existing branch full-versus-clean readout; not a fitted diagnostic head'))
            for label in LABELS:
                for control in ('no_noise','flat_H','no_carrier','no_RX_IQ'):
                    ai=CONDITIONS.index(control+'/fixed');bi=CONDITIONS.index(control+'/total')
                    receiver.append(dict(label=label,draw=draw,scene=scene,control=control,dimension='overall',stratum='ALL',**paired(logits[label][ai],logits[label][bi],y,fixed,owns[label]),scope='Total minus fixed effect for the same damage; not a sum of damage effects'))
    gates=[];gradients=[]
    for label,result in results.items():
        gradients.extend(validate_result(result,label))
        for name in ('PA','reference','signature'):
            for i,value in enumerate(np.asarray(result['gates'][name]).reshape(-1)):gates.append(dict(label=label,branch=name,coordinate=i,value=float(value),kind=result['gates']['kind']))
    return dict(metrics=metrics,effects=effects,training_effects=training,interactions=interactions,receiver_effects=receiver,branch_metrics=branch_rows,gates=gates,gradients=gradients,fixed_NOAS30_clean_correct_count=int(fixed0.sum()),own_clean_correct_counts={label:int((x['clean'].argmax(1)==x['TX']).sum())for label,x in data.items()})


def plot(out,tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=out/(name+'.'+ext);fig.savefig(path,dpi=160);files.append(path.name)
        plt.close(fig)
    def selected(table,**kw):return [r for r in tables[table]if all(r.get(k)==v for k,v in kw.items())]
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,scene in zip(axes,SCENES):
        for label in LABELS:
            rr=selected('effects',label=label,scene=scene,draw=-1,dimension='overall')
            ax.plot(range(10),[next(r['accuracy_delta_pp']for r in rr if r['condition']==c)for c in CONDITIONS[1:]],marker='.',label=label)
        ax.axhline(0,color='grey',lw=.8);ax.set(xticks=range(10),xticklabels=CONDITIONS[1:],title=scene,ylabel='Control minus full accuracy (pp)');ax.tick_params(axis='x',labelrotation=65);ax.legend();ax.grid(alpha=.2)
    save(fig,'control_gain')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for row,scene in enumerate(SCENES[1:]):
        for col,dim in enumerate(('state_lock','signal_to_added_noise_db')):
            ax=axes[row,col];rr=selected('metrics',scene=scene,draw=-1,dimension=dim,condition='full');names=list(dict.fromkeys(r['stratum']for r in rr));xx=np.arange(len(names))
            for j,label in enumerate(LABELS):
                vv=[next(r for r in rr if r['label']==label and r['stratum']==name)for name in names]
                ax.bar(xx+(j-.5)*.35,[100*r['accuracy']if r['accuracy']is not None else np.nan for r in vv],.35,label=label)
            nn=[next(r['count']for r in rr if r['stratum']==name)for name in names]
            ax.set(xticks=xx,xticklabels=[name+'\nn='+str(n)for name,n in zip(names,nn)],title=scene+' / full '+dim,ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelrotation=40);ax.legend();ax.grid(axis='y',alpha=.2)
    save(fig,'source_strata')
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for si,scene in enumerate(SCENES):
        rr=selected('training_effects',scene=scene,draw=-1,dimension='overall');ax=axes[0,si]
        ax.plot(range(11),[next(r['accuracy_delta_pp']for r in rr if r['condition']==c)for c in CONDITIONS],marker='o');ax.axhline(0,color='grey');ax.set(xticks=range(11),xticklabels=CONDITIONS,title=scene,ylabel='AS30 minus NOAS30 (pp)');ax.tick_params(axis='x',labelrotation=65)
        rr=selected('interactions',scene=scene,dimension='overall');ax=axes[1,si]
        for draw,offset in ((0,-.18),(1,.18)):
            ax.bar(np.arange(10)+offset,[next(r['training_x_control_pp']for r in rr if r['draw']==draw and r['condition']==c)for c in CONDITIONS[1:]],.36,label='channel draw '+str(draw))
        ax.axhline(0,color='grey');ax.set(xticks=range(10),xticklabels=CONDITIONS[1:],ylabel='Training x control interaction (pp)');ax.tick_params(axis='x',labelrotation=65);ax.legend()
    save(fig,'training_effect')
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for label in LABELS:
        for mi,name in enumerate(('clean_sat_cosine','clean_norm','sat_norm')):
            rr=selected('branch_metrics',label=label,scene='midlow_pooled',draw=-1,dimension='overall',metric=name)
            if mi==0:axes[0].plot(range(5),[next(r['mean']for r in rr if r['branch']==b)for b in BRANCHES],marker='o',label=label)
        gg=selected('gradients',label=label);groups=[g for g in GROUPS if g!='remaining']
        axes[1].plot(range(len(groups)),[np.mean([r['cosine']for r in gg if r['group']==g and r['cosine']is not None])if any(r['group']==g and r['cosine']is not None for r in gg)else np.nan for g in groups],marker='.',label=label)
        axes[2].plot(range(len(groups)),[np.mean([r['satellite_norm']for r in gg if r['group']==g])for g in groups],marker='.',label=label)
    axes[0].set(xticks=range(5),xticklabels=BRANCHES,title='Full versus clean branch cosine, mid/low');axes[1].set(xticks=range(len(groups)),xticklabels=groups,title='Eval CE gradient cosine, six comparisons');axes[2].set(xticks=range(len(groups)),xticklabels=groups,title='Eval CE satellite gradient norm')
    for ax in axes:ax.tick_params(axis='x',labelrotation=45);ax.grid(alpha=.2);ax.legend()
    save(fig,'branch_gradients')
    return files


def run(project,run_root,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Preserve existing analysis; use a new output directory')
    data,physics,branches,results,_,audit=load_inputs(project,run_root)
    out.mkdir(parents=True,exist_ok=False);tables=analyze(data,physics,branches,results)
    write(out/'condition_confusions.json',tables['metrics'])
    for name,table in [('condition_metrics','metrics'),('control_effects','effects'),('As_training_effects','training_effects'),('training_control_interactions','interactions'),('total_minus_fixed','receiver_effects'),('branch_metrics','branch_metrics'),('static_gates','gates'),('eval_CE_gradients','gradients')]:csv_write(out/(name+'.csv'),tables[table])
    figures=plot(out,tables)
    overall=lambda table:[r for r in tables[table]if r.get('dimension')=='overall']
    summary=dict(status='SOURCE_CONTROLS_ANALYZED',run_id=RUN,checkpoints={label:results[label]['provenance']for label in LABELS},labels=list(LABELS),conditions=list(CONDITIONS),scenes=list(SCENES),source_unique_V=1800,channel_draws=2,model_seed_count=1,model_seed=2026092701,logit_scale=30.,stratification='Same baseline full state/lock/added-noise ratio for every control',fixed_NOAS30_clean_correct_count=tables['fixed_NOAS30_clean_correct_count'],own_clean_correct_counts=tables['own_clean_correct_counts'],cohorts_are_not_E20_parents=True,fullV27000_new_readout=False,overall_metrics=overall('metrics'),overall_control_effects=overall('effects'),overall_training_effects=overall('training_effects'),overall_interactions=overall('interactions'),total_minus_fixed=tables['receiver_effects'],static_gates={label:results[label]['gates']for label in LABELS},gradient_scope='Frozen eval-mode smoothed CE; not stochastic training gradients',table_counts={k:len(v)for k,v in tables.items()if isinstance(v,list)},audit=audit,actual_readout_NN=sum(r['NN']for r in results.values()),analysis_NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,head_fit_updates=0,formal_scores_read=False,target_access=False,promotion_gate=False,removal_effects_not_additive=True,physical_causality_complete=False,repeatable_crossseed_evidence=False,figures=figures,goal_complete=False)
    write(out/'summary.json',summary);write(out/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',run_id=RUN,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,target_access=False,figures=figures,missing=[],model_seed_count=1))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--run-root',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();summary=run(args.project,args.run_root,args.out)
    print(json.dumps({k:summary[k]for k in('status','run_id','actual_readout_NN','analysis_NN','figures')},ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
