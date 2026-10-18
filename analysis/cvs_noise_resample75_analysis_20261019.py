"""P75 saved-output CPU readout: same-practical independent-noise CE, no inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[_key]='2'
import argparse
import hashlib
import importlib.util
import json
import time
import traceback
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import numpy as np

RUN='20261019-phase1-FcrossRX-noiseResample6-e30-s2026092701-m2-r01'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
SAME='20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
FULL='20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01'
SEED,STEPS,UPDATES,PER_VIEW=2026092701,222,2220,64
ARMS=('NOAS30','AS30','INDEP_FULL_CE6','SAME_NOISE_CE6','INDEP_NOISE_CE6')
NEW_ARMS=ARMS[-2:]
PRIMARY=('INDEP_NOISE_CE6','SAME_NOISE_CE6')
PAIRS=(PRIMARY,('INDEP_NOISE_CE6','INDEP_FULL_CE6'),('SAME_NOISE_CE6','INDEP_FULL_CE6'),
       ('AS30','NOAS30'),('SAME_NOISE_CE6','AS30'),('INDEP_NOISE_CE6','AS30'),
       ('SAME_NOISE_CE6','NOAS30'),('INDEP_NOISE_CE6','NOAS30'),('INDEP_FULL_CE6','AS30'))
METHODS={'SAME_NOISE_CE6':'SAME_NOISE_CE6','INDEP_NOISE_CE6':'INDEP_NOISE_CE6'}
RECORD_METHODS={'SAME_NOISE_CE6':('SAME_CE6','SAME_NOISE_CE6'),
                'INDEP_NOISE_CE6':('NOISE_CE6','INDEP_NOISE_CE6')}
ARM_IDS={'SAME_NOISE_CE6':'C01','INDEP_NOISE_CE6':'N01'}
NOISE_FIELDS=frozenset(('measured_added_noise_power_pre_agc','signal_to_added_noise_db','agc_gain_db','agc_limited','output_metrics'))
SECOND_SUFFIX='/view1-noise-independent'


def load_helpers():
    path=Path(__file__).with_name('cvs_full_channel_pair68_analysis_20261018.py')
    spec=importlib.util.spec_from_file_location('_p75_existing68',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


h=load_helpers();c=h.c
_write=h._write
h.RUN=RUN
PAIR_MEANS=h.PAIR_MEANS


def row_specs(project):
    runs=Path(project)/'runs'
    values=(('NOAS30',REF,'B00'),('AS30',REF,'B01'),('INDEP_FULL_CE6',FULL,'B01'),
            ('SAME_NOISE_CE6',RUN,'C01'),('INDEP_NOISE_CE6',RUN,'N01'))
    rows=[]
    for key,run,arm in values:
        shared=key in ARMS[2:];prun=SAME if shared else run;parm='AS_CE' if shared else arm
        rows.append(dict(key=key,run=run,arm=arm,row_id=arm+'-s'+str(SEED),root=runs/run/(arm+'-s'+str(SEED)),
            reused=key not in NEW_ARMS,parent=runs/prun/(parm+'-s'+str(SEED))/'stage20',
            parent_run=prun,parent_row_id=parm+'-s'+str(SEED),parent_As_KL_off=shared))
    return rows


training_complete=h.training_complete
source_criteria=h.source_criteria
read_source_decision=h.read_source_decision
parse_tail=h.parse_tail


def comparison_context(new,control):
    shared=new in ARMS[2:] and control in ARMS[2:]
    return dict(same_parent=shared,primary_mechanism_comparison=(new,control)==PRIMARY,
        first20_As_KL_history_confounded=(new in ARMS[2:] and control=='AS30'),
        matched_noAs_parent=False if new in ARMS[2:] and control=='NOAS30' else None,
        causal_attribution='Only second existing As64 noise reuse differs' if (new,control)==PRIMARY else
            ('Same AS_CE20 origin; second-view scope differs' if shared else 'Whole-recipe descriptive reference'))


def effects(rows,new,control):
    result={view:{key+'_delta_pp':100*(rows[new][view][key]-rows[control][view][key])
        for key in ('accuracy','macro_f1','worst_RX')} for view in c.VIEWS}
    result['urban_accuracy_mean_delta_pp']=float(np.mean([result[v]['accuracy_delta_pp'] for v in c.VIEWS[-3:]]))
    return result


def enrich_formal(obj):
    rows=obj['rows'];obj.pop('all4_rows_all7_verified_before_truth',None)
    obj['all5_rows_all7_verified_before_truth']=True
    obj['comparison_context']=[dict(new=a,control=b,**comparison_context(a,b)) for a,b in PAIRS]
    obj['comparisons']={a+'__'+b:effects(rows,a,b) for a,b in PAIRS}
    obj['primary']=dict(new=PRIMARY[0],control=PRIMARY[1],same_parent=True,parent_run=SAME,
        parent_row='AS_CE-s'+str(SEED),parent_epoch=20,parent_As_KL_off=True,
        effects=effects(rows,*PRIMARY),by_view={v:effects(rows,*PRIMARY)[v] for v in c.VIEWS})
    for arm in NEW_ARMS:
        obj['stage'][arm].update(NoAs30_is_whole_recipe_reference=True,same_parent_NoAs_control_available=False,
            reproducible_dual_gain=False)
    obj['suburban_gap_from_own_clean_pp']={arm:{view:100*(rows[arm]['clean']['accuracy']-rows[arm][view]['accuracy']) for view in c.VIEWS[1:4]} for arm in ARMS}
    obj['urban_accuracy_mean']={arm:float(np.mean([rows[arm][view]['accuracy'] for view in c.VIEWS[-3:]])) for arm in ARMS}
    obj['Ag_claim']=False;obj['extra_deployment_module']=False;obj['formal_channel_changed']=False


def write(path,obj):
    if Path(path).name=='formal_summary.json':enrich_formal(obj)
    _write(path,obj)


c.RUN,c.NEW_ARMS,c.PAIRS=RUN,NEW_ARMS,PAIRS
c.training_complete,c.row_specs,c.write=training_complete,row_specs,write


def stable_seed(seed,*parts):
    raw=json.dumps([int(seed),*parts],ensure_ascii=False,separators=(',',':'))
    return int.from_bytes(hashlib.sha256(raw.encode('utf-8')).digest()[:8],'little')


def pair_recount(records,physics,arm,expected_steps=UPDATES,packets_per_view=PER_VIEW):
    """Verify saved metadata and actual calls; no IQ, seed draws or waveform synthesis."""
    c.require(arm in NEW_ARMS,'Only the fixed C/N noise pair arms')
    independent=arm==PRIMARY[0];counts=Counter();strata=Counter();transitions=Counter();sums=Counter();rows=[]
    agc_differences=0;native_snapshots=secondary_snapshots=0
    for step,(r,p) in enumerate(zip_longest(records,physics),1):
        n=packets_per_view;second=n if independent else 0
        c.require(r is not None and p is not None and r.get('step')==p.get('step')==step and
            r.get('cumulative_step')==p.get('cumulative_step')==step+4440,'Complete ordered pair records and physics required')
        c.require(r.get('method') in RECORD_METHODS[arm] and r.get('noise_arm')==ARM_IDS[arm] and
            r.get('L_NN')==768 and r.get('extra_NN')==0 and r.get('generated_As0')==n and r.get('generated_As1')==second and
            r.get('As_generated_packets')==r.get('actual_residual_snapshots')==n+second and
            r.get('actual_secondary_snapshots')==r.get('secondary_noise_redraws')==second and
            r.get('physical_latent_draws')==n and r.get('As_forward_packets')==2*n and
            r.get('As_same_IQ') is (not independent) and r.get('same_hardware') is True and
            r.get('ordinary_raw_same_IQ') is True and r.get('all_labeled_CE_retained') is True and
            r.get('two_separate_CE_mean') is True and r.get('cross_channel_KL') is False and
            r.get('bypassed_elements')==0,'Logical CE slots must be separate from actual noise/snapshot generation')
        c.require(p.get('noise_arm')==ARM_IDS[arm] and p.get('actual_residual_snapshots')==n+second and
            p.get('physical_latent_draws')==n and p.get('secondary_noise_redraws')==second,'Physics payload actual costs differ')
        c.require(len(p.get('view0',[]))==len(p.get('view1',[]))==n,'Two complete physical metadata view slots required')
        mse=c.finite(r['As_IQ_pair_MSE'],'As pair MSE');c.require(mse>=0 and (independent or mse==0),'Same-IQ control must have exactly zero pair MSE')
        for key in PAIR_MEANS:sums[key]+=c.finite(r[key],key)
        c.require(abs(r['used_mixed_KL']-.5*r['ordinary_half_KL'])<3e-5 and
            abs(r['removed_satellite_KL']-.5*r['satellite_half_KL'])<3e-5,'Only ordinary mixed KL contributes')
        agc_delta=0.;noise0=noise1=0.
        for a,b in zip(p['view0'],p['view1']):
            pid=a.get('sample_id');namespace=a.get('namespace')
            c.require(isinstance(pid,list) and len(pid)==5 and pid[0]==1 and pid[1] in c.RX and pid[2] in c.DAYS and
                a.get('scene') in c.SCENES and a.get('state_end') in c.STATES and a.get('receiver_lock_status') in c.LOCKS and
                a.get('channel_equalization_applied') is False and a.get('session_id')==f'rx{pid[1]}:day{pid[2]}:eq1',
                'Original complete legal source residual/EQoff observation required')
            c.require(all(a.get(key)==b.get(key) for key in ('sample_id','session_id','scene','state_end','receiver_lock_status')),
                'Only added noise may differ; physical sample/scene/state/lock must replay')
            physical_seed=stable_seed(20261016421,namespace,a['scene'],pid)
            c.require(a.get('seed')==b.get('seed')==physical_seed,'Original root physical seed must replay')
            if independent:
                ns=namespace+SECOND_SUFFIX;noise_seed=stable_seed(20261016421,ns,a['scene'],pid)
                c.require(b.get('namespace')==b.get('noise1_namespace')==ns and b.get('physics_namespace')==namespace and
                    b.get('physics_seed')==physical_seed and b.get('noise1_seed')==noise_seed and
                    noise_seed!=stable_seed(physical_seed,'noise'),'Only second named native noise stream may be replaced')
                for key in ('noise_stream_only_resampled','non_noise_metadata_equal','hardware_object_reused','from_same_raw',
                    'native_noise_chain_retained','native_agc_recomputed'):
                    c.require(b.get(key) is True,'Required conditional-noise evidence missing '+key)
                c.require(b.get('output_white_noise_injection') is False and b.get('exact_noise_energy_calibration') is False and
                    b.get('first_agc_reused') is False and b.get('channel_generated') is True,'Native noise chain/energy/AGC semantics required')
                checked=b.get('non_noise_metadata_fields_checked')
                required=('receiver','geometry','quality_snr_db','residual_scale','estimated_frequency_hz','output_frequency_hz',
                    'phase_tracking_applied','iq_compensation_applied','channel_equalization_applied','residual_kernel_energy','boundary')
                c.require(isinstance(checked,list) and set(required).issubset(checked) and not NOISE_FIELDS.intersection(checked) and
                    all(key in a and key in b and a[key]==b[key] for key in checked),'All saved native non-noise metadata must be equal')
            else:
                c.require(b.get('namespace')==namespace and b.get('channel_generated') is False and
                    b.get('view_kind')=='same_IQ_reuse' and b.get('agc_recomputed') is False and
                    all(key in b and b[key]==value for key,value in a.items()),'C second slot must copy the first complete received IQ metadata')
            for view,obs in (('view0',a),('view1',b)):
                counts[view]+=1;strata[(view,int(pid[1]),int(pid[2]),obs['scene'],obs['state_end'],obs['receiver_lock_status'])]+=1
            difference=c.finite(b['agc_gain_db'],'second AGC')-c.finite(a['agc_gain_db'],'first AGC')
            agc_delta+=difference;agc_differences+=int(difference!=0.)
            noise0+=c.finite(a['measured_added_noise_power_pre_agc'],'first noise power')
            noise1+=c.finite(b['measured_added_noise_power_pre_agc'],'second noise power')
            transitions[(a['scene'],b['scene'],a['state_end'],b['state_end'],a['receiver_lock_status'],b['receiver_lock_status'])]+=1
        native_snapshots+=n+second;secondary_snapshots+=second
        rows.append(dict(row_id=arm,step=step,cumulative_step=step+4440,epoch=(step-1)//STEPS+21,
            As_IQ_pair_MSE=mse,satellite_CE0=r['satellite_CE0'],satellite_CE1=r['satellite_CE1'],
            mean_AGC_delta_db=agc_delta/n,mean_noise0_power_pre_agc=noise0/n,mean_noise1_power_pre_agc=noise1/n,
            actual_snapshots=n+second,logical_As_forward_slots=2*n))
    total=expected_steps*packets_per_view
    c.require(len(rows)==expected_steps and counts==dict(view0=total,view1=total),'Complete C/N logical slot coverage required')
    physical=[dict(view=v,RX=rx,day=day,scene=sc,state=st,lock=lk,count=n) for (v,rx,day,sc,st,lk),n in sorted(strata.items())]
    return dict(status='SAVED_CONDITIONAL_NOISE_PHYSICS_RECOUNTED',arm=arm,steps=len(rows),counts=dict(counts),
        means={k:v/len(rows) for k,v in sums.items()},actual_residual_snapshots=native_snapshots,
        actual_secondary_snapshots=secondary_snapshots,physical_latent_draws=total,logical_second_slots=total,
        logical_As_forward_slots=2*total,AGC_changed_pairs=agc_differences,AGC_recomputed_not_frozen=independent,
        physical_strata=physical,pair_transitions=[dict(scene0=k[0],scene1=k[1],state0=k[2],state1=k[3],lock0=k[4],lock1=k[5],count=v) for k,v in sorted(transitions.items())],
        non_noise_equality='Saved native metadata and fixed root seed; no waveform re-simulation',
        physical_independence_proven_by_draws=False,total_SNR_known=False,NN=0,IQ_reads=0,checkpoint_reads=0),rows


def validate_pair_summary(saved,pair):
    n=pair['steps'];first=pair['counts']['view0'];second=pair['actual_secondary_snapshots']
    expected=dict(method=METHODS[pair['arm']],steps=n,generated_As0=first,generated_As1=second,
        generated_As_total=first+second,As_forward_NN=first*2,L_NN=n*768,physical_latent_draws=first,
        logical_second_slots=first,total_L_CE_scale=1.25,As_CE_coefficient=.25,per_pass_As_CE_coefficient=.125,
        ordinary_raw_KL_unchanged=True,As_KL=False,original_dropout=True,dropout_bypassed_elements=0,
        extra_NN=0,extra_U_NN=0,extra_parameters=0,extra_losses=0,U_truth_read=False,target_access=False,
        arm=ARM_IDS[pair['arm']],first_noisestream_unchanged=True,nonnoise_replayed=True,AGC_per_view_native=True)
    c.require(isinstance(saved,dict) and all(saved.get(k)==v for k,v in expected.items()),'Saved C/N summary conflates slots, true generations or fixed loss recipe')
    for k,v in pair['means'].items():c.require(abs(c.finite(saved.get(k+'_mean'),k)-v)<3e-5,'Saved whole-tail mean differs '+k)


def coverage_tables(result,pair):
    c.require(result.get('L_As_forward_NN')==284160 and result.get('L_As_total')==pair['actual_residual_snapshots'],
        'Actual generated packets and repeated CE slots must be distinguished')
    # Old helper independently recounts both labeled slot tables; adapter only
    # translates its historical L_As_total convention, never the producer result.
    adapted=dict(result,L_As_total=284160)
    return h.coverage_tables(adapted,pair)


def training_analysis(project,specs,out):
    tails={};curves=[];coverage=[];by_key={s['key']:s for s in specs};new_sequences={}
    for spec in specs:
        key=spec['key'];tail,ordinary=parse_tail(spec['root']/'source',key!='NOAS30')
        tail.update(parent_run=spec['parent_run'],parent_row_id=spec['parent_row_id'],parent_As_KL_off=spec['parent_As_KL_off'])
        if key in NEW_ARMS:
            result=c.read(spec['root']/'source/result.json')
            pair,steps=pair_recount(c.lines(spec['root']/'source/full_as_pair_steps.jsonl.gz'),
                c.lines(spec['root']/'source/full_as_pair_physics.jsonl.gz'),key)
            validate_pair_summary(result['noise_pair_summary'],pair)
            c.require(result.get('full_as_pair_summary')==result['noise_pair_summary'],'Compatibility summary alias must match actual noise summary')
            tail.update(pair_physics=pair,noise_pair_summary=result['noise_pair_summary'])
            c.csv_write(out/(key+'_pair_steps.csv'),steps);c.csv_write(out/(key+'_pair_physical_strata.csv'),pair['physical_strata'])
            c.csv_write(out/(key+'_pair_transitions.csv'),pair['pair_transitions'])
            coverage.extend(dict(row_id=key,**r) for r in coverage_tables(result,pair))
            new_sequences[key]=ordinary
        for r in c.epoch_curve(spec['parent'],0)+tail['curve']:
            curves.append(dict(row_id=key,reused=spec['reused'],model_seed=SEED,parent_reused=True,**r))
        tails[key]=tail
    a,b=(by_key[k] for k in PRIMARY)
    physical_sequence=h.same_sequence(a['root']/'source/training_sequence.jsonl.gz',b['root']/'source/training_sequence.jsonl.gz')
    c.require(new_sequences[PRIMARY[0]]==new_sequences[PRIMARY[1]],'Matched original ordinary augmentation stream must remain identical')
    c.csv_write(out/'training_curves.csv',curves);c.csv_write(out/'pair_TX_RX_scene_state_lock.csv',coverage)
    result=dict(status='FULL_AVAILABLE_TRAINING_LOGS_VERIFIED',rows=tails,new_model_epochs=20,new_updates=4440,
        matched_L_U_sequence=physical_sequence,ordinary_augmentation_stream_identical=True,parent_epoch20_reused=True,
        parent_history_difference=True,extra_NN=0,U_truth_read=False,native_U_temporal_mode='batch_neighbor',
        coverage_is_not_identity_performance=True)
    c.write(out/'training_summary.json',result);return result,curves


def costs(project,specs,local_nn=0,smoke_nn=1540):
    rows={};new={};parents={}
    for spec in specs:
        tail=c.native_cost(spec['root']/'source');q=c.read(spec['root']/'source_qualification/complete.json');p=c.read(spec['root']/'prediction/complete.json')
        c.require(q.get('NN')==12600 and p.get('actual_NN')==1176002,'Actual saved source/formal NN differ')
        parent=c.native_cost(spec['parent']);parents[str(spec['parent'])]=parent
        row=dict(parent=parent,tail=tail,source_NN=q['NN'],formal_NN=p['actual_NN'],source_wall_seconds=q.get('wall_seconds'),
            formal_wall_seconds=p.get('wall_seconds'),parent_run=spec['parent_run'],parent_row_id=spec['parent_row_id'])
        if spec['key'] in NEW_ARMS:
            saved=c.read(spec['root']/'source/result.json')['noise_pair_summary'];expected=142080 if spec['key']==NEW_ARMS[0] else 284160
            c.require(saved['generated_As_total']==expected and saved['physical_latent_draws']==142080 and saved['As_forward_NN']==284160,
                'Actual per-row native simulations differ')
            row.update(actual_residual_snapshots=expected,physical_latent_draws=142080,logical_As_forward_slots=284160)
            new[spec['key']]=row
        rows[spec['key']]=row
    smoke=c.read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json')
    c.require(smoke.get('status')=='PASS' and smoke.get('NN')==smoke_nn==1540 and smoke.get('physical_snapshots')==192 and local_nn==0,'Shared actual1540 NN smoke/local0 required')
    c.require(len(new)==2 and all(r['tail']['native_NN']==3162962 for r in new.values()),'Two exact native ten-epoch tails required')
    total=sum(r['tail']['training_total_NN']+r['source_NN']+r['formal_NN'] for r in new.values())+smoke_nn
    c.require(total==8704668,'Observed two-tail/source/formal/shared smoke budget differs')
    return dict(rows=rows,new_native_NN=6325924,new_source_NN=25200,new_formal_NN=2352004,shared_remote_smoke_NN=1540,
        new_total_remote_NN=total,local_verification_NN=0,new_model_epochs=20,new_updates=4440,
        student_parent_epochs=20,student_parent_updates=4440,student_cumulative_epochs=30,student_cumulative_updates=6660,
        parent_training_reused_not_repeated=True,parent_As_KL_off=True,parent_run=SAME,unique_reused_parents=parents,
        historical_reference_model_epochs=90,historical_reference_updates=19980,
        unique_total_model_epochs_including_reused=110,unique_total_updates_including_reused=24420,
        new_training_wall_seconds_sum=sum(r['tail']['wall_seconds'] for r in new.values()),
        new_training_residual_snapshots=426240,new_smoke_residual_snapshots=192,
        per_row_nonnoise_latents=142080,As_forward_slots_per_row=284160,
        extra_student_NN=0,extra_teacher_NN=0,reference_reprediction=0,reference_rescoring=0,
        analysis_NN=0,NN_matching_does_not_imply_equal_wall=True,budget_exhaustion_is_not_convergence=True)


def plots(out,curves,source_rows,formal,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(ARMS,('#64748b','#d97706','#8b5cf6','#16a34a','#2563eb')));files=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=out/(name+'.'+ext);fig.savefig(path,dpi=180);files.append(str(path))
        plt.close(fig)
    if curves:
        fig,axes=plt.subplots(1,3,figsize=(16,4.8))
        for arm in ARMS:
            rr=[r for r in curves if r['row_id']==arm]
            for ax,key,title in zip(axes,('source_val_accuracy_pct','native_loss','grad_backbone'),('Full source V','Native training loss','Backbone gradient norm')):
                ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],label=arm,color=colors[arm]);ax.set(xlabel='Cumulative epoch',title=title);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);save(fig,'training_curves')
    if source_rows:
        fig,axes=plt.subplots(2,3,figsize=(18,8.5))
        for si,scene in enumerate(c.SCENES[1:]):
            for di,dim in enumerate(('state','lock','signal_to_added_noise_db')):
                ax=axes[si,di];rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dim]
                labels=list(dict.fromkeys(r['stratum'] for r in rr));xx=np.arange(len(labels))
                for i,arm in enumerate(ARMS):
                    selected=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels]
                    ax.bar(xx+(i-2)*.15,[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in selected],.15,label=arm,color=colors[arm])
                ax.set(xticks=xx,xticklabels=labels,title=scene+' / '+dim,ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=7);save(fig,'source_quality_state_lock_noise')
    fig,axes=plt.subplots(1,2,figsize=(16,5));xx=np.arange(7)
    for arm in ARMS:axes[0].plot(xx,[100*formal['rows'][arm][v]['accuracy'] for v in c.VIEWS],marker='.',label=arm,color=colors[arm])
    for new,old,label in ((*PRIMARY,'N - C (primary)'),('INDEP_NOISE_CE6','INDEP_FULL_CE6','N - independent full'),('INDEP_NOISE_CE6','NOAS30','N - P43 NoAs (recipe)')):
        axes[1].plot(xx,[100*(formal['rows'][new][v]['accuracy']-formal['rows'][old][v]['accuracy']) for v in c.VIEWS],marker='.',label=label)
    for ax,title,ylabel in zip(axes,('Frozen all7 default-head readout','Paired changes; one model seed'),('Accuracy (%)','Absolute percentage points')):
        ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title,ylabel=ylabel);ax.tick_params(axis='x',labelrotation=30);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    axes[1].axhline(0,color='grey',lw=.7);save(fig,'formal_paired')
    if training:
        fig,axes=plt.subplots(2,3,figsize=(17,8.5))
        for row,arm in enumerate(NEW_ARMS):
            pair=training['rows'][arm]['pair_physics']
            for ax,dim,allowed in zip(axes[row],('scene','state','lock'),(c.SCENES,c.STATES,c.LOCKS)):
                xx=np.arange(len(allowed))
                for i,view in enumerate(('view0','view1')):
                    hist=Counter()
                    for r in pair['physical_strata']:
                        if r['view']==view:hist[r[dim]]+=r['count']
                    ax.bar(xx+(i-.5)*.35,[hist[a] for a in allowed],.35,label=view)
                ax.set(xticks=xx,xticklabels=allowed,title=arm+' / '+dim,ylabel='CE view slots (C view1 reused)');ax.tick_params(axis='x',labelrotation=20);ax.legend();ax.grid(axis='y',alpha=.2)
        save(fig,'pair_physical_coverage')
    return files


def report(out,result):
    formal=result['formal'];rows=formal['rows'];cost=result['cost'];stage=formal['stage'][PRIMARY[0]]
    text=['# P75：同一完整 practical 的独立新增噪声 CE','',
        '两个新模型均从同一 SAT18 AS_CE20 的 student 与 ownEMA 分叉，追加10轮，总E30。C复用第一 satellite IQ；N仅替换第二 native noise RNG，几何/H/状态/相位/RX IQ/补偿与原 seed 保持，AGC按各view信号加噪声重新计算。六次L前向、两次As CE各0.125、原ordinary/raw KL、ORD128 SupCon和native U保留。无新增一致性损失、Ag、分类头或部署分支。','',
        '主比较 N−C。旧P68独立完整信道为同父来源的范围参照；原P43 As20前20轮包含As KL，与新共同父历史不同，P43 NoAs30是同预算、同seed、同数据角色整配方参照，并非同As父NoAs控制。三旧行全部复用，未重预测或重评分。','',
        '| 环境 | NoAs % | As % | 独立完整 % | C % | N % | N−C pp |','|---|---:|---:|---:|---:|---:|---:|']
    for v in c.VIEWS:text.append('| '+v+' | '+' | '.join(f'{100*rows[a][v]["accuracy"]:.6f}' for a in ARMS)+f' | {formal["primary"]["by_view"][v]["accuracy_delta_pp"]:+.6f} |')
    primary=formal['primary']['effects']
    text+=['',f'N相对C：low urban {primary[c.VIEWS[-1]]["accuracy_delta_pp"]:+.6f} pp，mid urban {primary[c.VIEWS[-2]]["accuracy_delta_pp"]:+.6f} pp，urban均值 {primary["urban_accuracy_mean_delta_pp"]:+.6f} pp，clean {primary["clean"]["accuracy_delta_pp"]:+.6f} pp。',
        f'N相对P43 NoAs30：clean {stage["clean_delta_vs_NoAs30_pp"]:+.6f} pp，urban均值 {stage["urban_mean_vs_NoAs30_pp"]:+.6f} pp；clean下降不超过0.1绝对百分点：{stage["clean_within_0p1pp"]}。',
        '只有一个模型seed；两个channel draw不是两个模型seed。不能宣称可重复双提升。全部七环境acc/F1/最差RX、逐RX/day/TX、修复与新增错误、默认头margin/30、固定P43 NoAs30 clean正确集合保持率见CSV。该集合不是共同AS_CE20父模型集合。','',
        '源面板分层包括LOS/PARTIAL/BLOCKED、lock、接收质量代理和新增噪声比。新增噪声SNR不是总SNR；相关分层和保存的native元数据重放检查不单独证明物理因果或随机独立性。EQ关闭保持G=1，但RX IQ校准残差仍可能产生improper noise。N未在输出加白噪声，也未精确归一化噪声能量；AGC允许随新噪声改变。','',
        f'新增20 model-epochs／4440更新；每row总E30/6660更新。新增native NN={cost["new_native_NN"]}、source={cost["new_source_NN"]}、formal={cost["new_formal_NN"]}、共享真实CP smoke={cost["shared_remote_smoke_NN"]}，合计={cost["new_total_remote_NN"]}，本地NN=0。C真实训练快照142080，N284160；每row非噪声latent实现142080，两As前向槽位284160，不能把C复制槽位再算新信道生成。共享smoke物理快照192。训练过程秒数合计={cost["new_training_wall_seconds_sum"]:.3f}；相同NN不意味着相同墙钟。父训练成本单列且复用，不计新增。预算耗尽不等于收敛。','',
        '已存源判定（正式读取前固定，不在本分析重新晋级）：'+json.dumps(result['stored_source_decision'],ensure_ascii=False),'',
        '固定判据：N fullV clean相对原NoAs下降不超过0.1pp，源两draw pooled mid/low相对max(C,原As)至少+0.5pp。源失败仍完成all7，关闭该固定方案，不扫描噪声强度/seed或自动扩大E50。正式分数只用于独立报告。','',
        '未完成／限制：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：']
    text+=['- '+p for p in result['figures']]
    text+=['','本离线分析新增NN、训练更新、IQ/checkpoint读取、重推理与重新评分均为0。']
    (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def run(project,out,truth=None,code_commit=None,local_nn=0,smoke_nn=1540):
    project=Path(project);decision=read_source_decision(project);out=Path(out)
    c.require(not out.exists(),'Immutable new analysis output required');out.mkdir(parents=False)
    c.write(out/'stored_source_decision.json',decision);specs=row_specs(project);started=time.perf_counter()
    missing=[];training=None;curves=[];source=None;source_rows=[]
    if decision['status'] in c.DECISION_PARTIAL:missing.append(dict(component='source_decision',error='Stored source decision PARTIAL; not recomputed'))
    for name in ('training','source'):
        try:
            if name=='training':training,curves=training_analysis(project,specs,out)
            else:source,source_rows=c.source_analysis(project,specs,out,decision)
        except Exception as error:
            failure=dict(component=name,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True)
            missing.append(failure);c.write(out/(name+'_failure.json'),failure)
    truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    formal=c.formal_analysis(project,specs,out,truth);cost=costs(project,specs,local_nn,smoke_nn);c.write(out/'costs.json',cost)
    try:figures=plots(out,curves,source_rows,formal,training)
    except Exception as error:
        figures=[];failure=dict(component='plots',error=repr(error),traceback=traceback.format_exc());missing.append(failure);c.write(out/'plot_failure.json',failure)
    result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,code_commit=code_commit,
        stored_source_decision=decision,source=source,formal=formal,training=training,cost=cost,figures=figures,missing=missing,
        model_seed_count=1,crossseed_confirmed=False,reproducible_dual_gain=False,source_decision_recomputed=False,
        target_feedback=False,goal_complete=False,NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,wall_seconds=time.perf_counter()-started)
    report(out,result);c.write(out/'summary.json',{k:v for k,v in result.items() if k!='training'});c.write(out/'complete.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--truth',type=Path)
    parser.add_argument('--code-commit');parser.add_argument('--local-nn',type=int,default=0);parser.add_argument('--smoke-nn',type=int,default=1540)
    parser.add_argument('--checks',action='store_true');args=parser.parse_args()
    if args.checks:
        path=Path(__file__).with_name('cvs_noise_resample75_analysis_checks_20261019.py')
        spec=importlib.util.spec_from_file_location('_p75_analysis_checks',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.checks();return
    if args.project is None or args.out is None:parser.error('--project and --out required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[r['component'] for r in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
