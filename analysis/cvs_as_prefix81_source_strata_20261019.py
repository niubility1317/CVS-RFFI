"""Optional P81 posthoc source strata from saved logits and numeric physics.

No IQ, checkpoint, model, formal prediction, score or truth is opened. Groups
describe the original full observation; they do not select candidates or runs.
"""
import argparse,csv,json,math,time,traceback
from pathlib import Path
import numpy as np

RUN='20261019-phase1-FcrossRX-asPrefix-e30-s2026092701-m2-r01'
REFERENCE='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
PANEL_RUN='20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CONDITIONS=('full','no_noise/fixed','no_noise/total','flat_H/fixed','flat_H/total','no_carrier/fixed','no_carrier/total','no_RX_IQ/fixed','no_RX_IQ/total','actual_power_quality','edge_boundary')
ARMS=('NOAS30','AS30','RANDOM30','PREFIX30')
PAIRS=(('PREFIX30','AS30'),('PREFIX30','RANDOM30'))
STATES=('LOS','PARTIAL','BLOCKED');LOCKS=('locked','degraded','unlocked')
QUALITY_FIELDS=('quality_snr_db','signal_to_added_noise_db','actual_waveform_quality_db','white_proxy_quality_db')
QUALITY_LABELS=('<3','3-8','8-15','>=15')
LOGIT_SCALE=30.


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row if k!='confusion'))
    with Path(path).open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def rows(project,root):
    reference=Path(project)/'runs'/REFERENCE;root=Path(root)
    return {'NOAS30':reference/'B00-s2026092701','AS30':reference/'B01-s2026092701',
            'RANDOM30':root/'RANDOM-s2026092701','PREFIX30':root/'PREFIX-s2026092701'}


def validate_physics(physics,n):
    expected=(2,3,len(CONDITIONS),n)
    for name,value in physics.items():
        if value.shape!=expected or value.dtype.kind not in 'iufb' or np.isinf(value).any():raise ValueError('Saved numeric physics shape/type differs: '+name)
    for name in ('state','lock'):
        if name not in physics or not np.isfinite(physics[name]).all() or not np.isin(physics[name],(0,1,2)).all():raise ValueError('Saved full state/lock indices absent or malformed')
    return [name for name in QUALITY_FIELDS if name not in physics]


def validate_predictions(x,ids,reference=None):
    n=len(ids)
    if x['clean'].shape!=(n,6) or x['satellite'].shape!=(2,3,n,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Finite default-head source logits required')
    if not np.array_equal(x['ids'],ids):raise ValueError('Source predictions and physics-panel physical ID order differ')
    for name,allowed in [('TX',set(range(6))),('RX',{1,3,4,6,8}),('day',{1,2,3})]:
        if x[name].shape!=(n,) or x[name].dtype.kind not in 'iu' or not set(x[name].tolist())<=allowed:raise ValueError('Original source metadata differs: '+name)
        if reference is not None and not np.array_equal(x[name],reference[name]):raise ValueError('Four source prediction rows do not align')


def load_inputs(project,root,physics_path=None):
    project,root=Path(project),Path(root);panel=project/'runs'/PANEL_RUN/'diagnostic'
    role=read(panel/'data_roles.json');completion=read(panel/'observations/completion.json')
    if role.get('role')!='V' or role.get('counts')!={'L_s':6300,'U_s':56700,'V':27000} or role.get('V_panel')!=1800 or role.get('target_access') is not False or role.get('U_truth_used') is not False:raise ValueError('Existing legal all-source V1800 panel required')
    ids=np.asarray(role['physical_ids'])
    if ids.shape!=(1800,) or ids.dtype.kind not in 'US' or len(set(ids.tolist()))!=1800:raise ValueError('Unique saved source physical IDs required')
    if completion.get('status')!='SOURCE_OBSERVATIONS_COMPLETE' or completion.get('unique_V')!=1800 or completion.get('draws')!=2 or tuple(completion.get('scenes',()))!=SCENES or tuple(completion.get('conditions',()))!=CONDITIONS or completion.get('target_access') is not False:raise ValueError('Original draw/scene/condition order required')
    physics_path=Path(physics_path) if physics_path else panel/'observations/physics.npz'
    with np.load(physics_path,allow_pickle=False) as packed:
        physics={name:packed[name].copy() for name in ('state','lock',*QUALITY_FIELDS) if name in packed.files}
    missing=validate_physics(physics,1800);data={};readouts={}
    for arm,row in rows(project,root).items():
        q=read(row/'source_qualification/complete.json')
        if q.get('status')!='SOURCE_READOUT_COMPLETE' or q.get('target_access') is not False or q.get('formal_scores_read') is not False or q.get('training_updates')!=0 or q.get('head_fit_updates')!=0:raise ValueError('Completed source-only frozen readout required: '+arm)
        with np.load(row/'source_qualification/predictions.npz',allow_pickle=False) as packed:
            x={name:packed[name].copy() for name in ('ids','TX','RX','day','clean','satellite')}
        validate_predictions(x,ids,data.get('NOAS30'));data[arm]=x
        acc=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
        if not np.allclose(acc,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Source saved readout accuracy differs: '+arm)
        readouts[arm]=dict(source_readout_NN=q.get('NN'),source_accuracies=acc.tolist(),default_head_scale=LOGIT_SCALE)
    return data,physics,dict(panel_role='V',panel_physical_samples=1800,model_seed=2026092701,model_seed_count=1,channel_draws=2,
        condition='full',condition_index=0,physics_shape=list(physics['state'].shape),physics_file=str(physics_path),
        source_readouts=readouts,missing_quality_fields=missing,join='Saved role physical-ID order, then draw/scene/full-condition/sample indices',
        observation_IQ_opened=False,formal_scores_read=False,formal_predictions_read=False,checkpoint_reads=0,new_NN=0)


def groups(full):
    n=len(full['state']);out=[('overall','ALL',np.ones(n,bool))]
    for name,labels in [('state',STATES),('lock',LOCKS)]:out.extend((name,label,full[name]==i) for i,label in enumerate(labels))
    out.extend(('state_lock',s+'/'+l,(full['state']==i)&(full['lock']==j)) for i,s in enumerate(STATES) for j,l in enumerate(LOCKS))
    for name in QUALITY_FIELDS:
        values=full.get(name,np.full(n,np.nan));codes=np.full(n,-1,int);finite=np.isfinite(values)
        codes[finite]=np.searchsorted(np.array([3.,8.,15.]),values[finite],side='right')
        out.extend((name,label,codes==i) for i,label in enumerate(QUALITY_LABELS));out.append((name,'MISSING',codes==-1))
    return out


def metric(y,z,rx,cohort):
    n=len(y)
    if z.shape!=(n,6) or rx.shape!=(n,) or cohort.shape!=(n,) or not np.isfinite(z).all():raise ValueError('Aligned finite source logits/cohort required')
    p=z.argmax(1);correct=p==y;cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
    if n:
        other=z.copy();other[np.arange(n),y]=-np.inf;margin=(z[np.arange(n),y]-other.max(1))/LOGIT_SCALE
    else:margin=np.empty(0)
    nc=int(cohort.sum());retained=int((correct&cohort).sum())
    return dict(count=n,correct=int(correct.sum()),accuracy=float(correct.mean()) if n else None,macro_F1=float(f.mean()) if n else None,
        worst_RX=float(min(correct[rx==r].mean() for r in np.unique(rx))) if n else None,confusion=cm.tolist(),
        mean_true_class_margin_over30=float(margin.mean()) if n else None,
        mean_correct_prediction_margin_over30=float(margin[correct].mean()) if correct.any() else None,
        fixed_NOAS30_clean_correct_count=nc,fixed_NOAS30_clean_retained=retained,fixed_NOAS30_clean_retention=retained/nc if nc else None,
        empty_cell=n==0,small_cell=n<30)


def paired(y,old,new,cohort):
    a,b=old.argmax(1)==y,new.argmax(1)==y;repair=int((~a&b).sum());damage=int((a&~b).sum());n=len(y);nc=int(cohort.sum())
    return dict(count=n,control_correct=int(a.sum()),new_correct=int(b.sum()),repaired=repair,new_errors=damage,net=repair-damage,
        accuracy_pp=100*(repair-damage)/n if n else None,fixed_NOAS30_clean_correct_count=nc,
        cohort_repaired=int((~a&b&cohort).sum()),cohort_new_errors=int((a&~b&cohort).sum()),
        control_cohort_retention=float(a[cohort].mean()) if nc else None,new_cohort_retention=float(b[cohort].mean()) if nc else None)


def analyze(data,physics):
    base=data['NOAS30'];n=len(base['TX']);fixed=base['clean'].argmax(1)==base['TX'];metrics=[];effects=[]
    for scene in ('clean',*SCENES):
        si=SCENES.index(scene) if scene!='clean' else None
        for draw in ((-1,) if scene=='clean' else (0,1,-1)):
            ds=(0,1) if draw==-1 else (draw,);copies=1 if scene=='clean' else len(ds)
            y=np.tile(base['TX'],copies);rx=np.tile(base['RX'],copies);co=np.tile(fixed,copies)
            logits={arm:x['clean'] if scene=='clean' else np.concatenate([x['satellite'][d,si] for d in ds]) for arm,x in data.items()}
            current_groups=[('overall','ALL',np.ones(n,bool))] if scene=='clean' else groups({name:np.concatenate([value[d,si,0] for d in ds]) for name,value in physics.items()})
            for dimension,stratum,mask in current_groups:
                context=dict(scene=scene,draw=draw,dimension=dimension,stratum=stratum);current={}
                for arm in ARMS:
                    current[arm]=metric(y[mask],logits[arm][mask],rx[mask],co[mask]);metrics.append(dict(arm=arm,model_seed=2026092701,**context,**current[arm]))
                for new,old in PAIRS:
                    a,b=current[new],current[old];value=paired(y[mask],logits[old][mask],logits[new][mask],co[mask])
                    value.update(comparison=new+'_minus_'+old,**context,
                        macro_F1_pp=100*(a['macro_F1']-b['macro_F1']) if mask.any() else None,
                        worst_RX_pp=100*(a['worst_RX']-b['worst_RX']) if mask.any() else None,
                        mean_true_class_margin_over30_delta=a['mean_true_class_margin_over30']-b['mean_true_class_margin_over30'] if mask.any() else None)
                    effects.append(value)
    return metrics,effects


def plot(out,metrics,effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'NOAS30':'#64748b','AS30':'#d97706','RANDOM30':'#16a34a','PREFIX30':'#2563eb'}
    fig,axes=plt.subplots(2,3,figsize=(18,8.5))
    for si,scene in enumerate(SCENES[1:]):
        for di,(dimension,labels) in enumerate((('state',STATES),('lock',LOCKS),('signal_to_added_noise_db',(*QUALITY_LABELS,'MISSING')))):
            ax=axes[si,di];xx=np.arange(len(labels))
            if di<2:
                for i,arm in enumerate(ARMS):
                    lookup={r['stratum']:r for r in metrics if r['arm']==arm and r['scene']==scene and r['draw']==-1 and r['dimension']==dimension}
                    ax.bar(xx+(i-1.5)*.2,[100*lookup[label]['accuracy'] if lookup[label]['accuracy'] is not None else np.nan for label in labels],.2,label=arm,color=colors[arm])
                ax.set(ylabel='Source accuracy (%)')
            else:
                for new,old in PAIRS:
                    key=new+'_minus_'+old;lookup={r['stratum']:r for r in effects if r['comparison']==key and r['scene']==scene and r['draw']==-1 and r['dimension']==dimension}
                    ax.plot(xx,[lookup[label]['accuracy_pp'] if lookup[label]['accuracy_pp'] is not None else np.nan for label in labels],marker='.',label=key)
                ax.axhline(0,color='grey',lw=.7);ax.set(ylabel='Paired accuracy change (pp)')
            counts={r['stratum']:r['count'] for r in metrics if r['arm']=='PREFIX30' and r['scene']==scene and r['draw']==-1 and r['dimension']==dimension}
            ax.set(title=scene+' / '+dimension,xticks=xx,xticklabels=[label+'\nn='+str(counts[label]) for label in labels]);ax.tick_params(axis='x',labelrotation=20);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Descriptive source readout: one model seed; two repeated channel draws; added-noise ratio is not total SNR',fontsize=11);fig.tight_layout()
    files=[]
    for ext in ('png','pdf'):
        name='source_state_quality.'+ext;fig.savefig(out/name,dpi=160);files.append(name)
    plt.close(fig);return files


def run(project,root,out,physics_path=None):
    start=time.perf_counter();out=Path(out);data,physics,inputs=load_inputs(project,root,physics_path)
    out.mkdir(parents=True,exist_ok=False)
    try:
        metrics,effects=analyze(data,physics);csv_write(out/'source_state_quality_metrics.csv',metrics);csv_write(out/'source_state_quality_paired.csv',effects)
        write(out/'source_state_quality_metrics.json',metrics);write(out/'source_state_quality_paired.json',effects);write(out/'input_summary.json',inputs)
        figures=plot(out,metrics,effects)
        summary=dict(status='SOURCE_STRATA_ANALYZED',run_id=RUN,model_seed=2026092701,model_seed_count=1,channel_draws=2,independent_physical_samples=1800,
            metric_records=len(metrics),paired_records=len(effects),missing_quality_fields=inputs['missing_quality_fields'],figures=figures,
            bins=dict(edges=['-inf',3,8,15,'inf'],labels=list(QUALITY_LABELS),intervals='left-closed/right-open; >=15 last; nonfinite/unavailable recorded as MISSING'),
            margin_definition='True-class logit minus maximum alternative class, divided by fixed default cosine scale30; correct-prediction conditional mean separately named',
            fixed_cohort='Original P43 NOAS30 clean-correct V-panel; not the As20 parent cohort',
            observation_condition='Same saved full practical observation; condition0; state/quality labels unchanged across models',
            interpretation='Posthoc descriptive associations; overlapping strata; channel draws are not model seeds; no unique physical-causality claim',
            added_noise_ratio_is_total_SNR=False,source_candidate_selection=False,source_promotion=False,formal_feedback=False,automatic_training=False,automatic_extension=False,
            NN=0,IQ_reads=0,checkpoint_reads=0,truth_reads=0,formal_prediction_reads=0,formal_score_reads=0,wall_seconds=time.perf_counter()-start)
        write(out/'complete.json',summary);return summary
    except BaseException as exc:
        write(out/'failure.json',dict(status='SOURCE_STRATA_ANALYSIS_FAILED',error=str(exc),traceback=traceback.format_exc(),preserve_outputs=True,automatic_retry=False,NN=0,IQ_reads=0,checkpoint_reads=0));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',required=True);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);parser.add_argument('--physics');args=parser.parse_args()
    print(json.dumps(run(args.project,args.root,args.output,args.physics),ensure_ascii=False,allow_nan=False))
