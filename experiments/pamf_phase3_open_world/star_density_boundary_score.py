"""Independent fixed-GMM boundary scoring; completed source replay precedes truths."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import numpy as np
from . import star_density_core as density
from . import star_class_density_core as conditional
from . import star_class_density_run as parent
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from .star_signal_ssl_score import _pair
from .scoring import _truth as _target_truth

RUN_ID='20261031-phase3-density-boundary-s2026092701-r01'
SCHEMA='pamf_phase3_density_boundary_scores_v1'
STATUS='DENSITY_BOUNDARY_SCORES_COMPLETE'
METHODS=('E0_pooled','E0_predicted','REPLAY_pooled','REPLAY_predicted','BOUNDARY_pooled','BOUNDARY_predicted')
PRIMARY=METHODS[5]
MODEL_NAMES=('E0','REPLAY','BOUNDARY')
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
STEPS=2220
_read,_write,_source_truth=common._read,common._write,common._truth


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _validate_rows(a,n,fields,state,tau):
    if set(a)!=fields or not common._ids(a['ids'],n):raise ValueError('Exact complete public rows required; no private fields')
    if (a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))):
        raise ValueError('All six methods must preserve the same-input original F0 identity')
    if 'baseline_closed' in fields and (a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):
        raise ValueError('Same-DURA original F0 baseline cannot be replaced')
    z=a['signal_features']
    if (z.shape!=(n,3,160) or z.dtype!=np.float32 or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1.,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,6) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Three ordered unit160 representations and six finite evidence columns required')
    raw=np.concatenate([conditional.raw_scores(state,z[:,i],a['closed'])[:,[1,3]] for i in range(3)],axis=1)
    if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):
        raise ValueError('Every model axis must use the same frozen pooled/predicted density and actual threshold route')
    return a


def _finite(value):
    if isinstance(value,dict):return all(_finite(x) for x in value.values())
    if isinstance(value,list):return all(_finite(x) for x in value)
    return math.isfinite(value) if isinstance(value,(int,float)) else True


def _trace_summary(root):
    """Read every saved update and match the two source sampling trajectories."""
    root=Path(root);summaries={};usage={arm:dict(L=set(),OE=set()) for arm in MODEL_NAMES[1:]}
    with ExitStack() as stack:
        streams={arm:stack.enter_context((root/arm/'steps.jsonl').open(encoding='utf-8')) for arm in MODEL_NAMES[1:]}
        for step in range(1,STEPS+1):
            pair={}
            for arm,stream in streams.items():
                line=stream.readline()
                if not line:raise ValueError('Both complete source training traces are required')
                row=json.loads(line)
                if (row.get('step')!=step or row.get('arm')!=arm or not _finite(row) or row.get('source_indices_equal') is not True
                        or row.get('sinc_optimizer_step')!=step or row.get('physical_clip_calls')!=step or row.get('sinc_lr')!=.1
                        or not isinstance(row.get('loss_total'),(int,float)) or not isinstance(row.get('gradient_norm_before_physical_clip'),(int,float))):
                    raise ValueError('Contiguous finite CENTER_UNITS updates and physical clipping counters required')
                for role,count in (('L',192),('OE',64)):
                    indices=row.get(role+'_indices')
                    if (not isinstance(indices,list) or len(indices)!=count or any(type(i) is not int or i<0 or i>=COUNTS[role] for i in indices)
                            or len(set(indices))!=count):raise ValueError('Every source update requires L192/OE64 valid distinct records')
                    usage[arm][role].update(indices)
                summary=summaries.setdefault(arm,dict(actual_trace_rows=0,loss_total_sum=0.,loss_total_first=row['loss_total'],max_gradient_norm=0.))
                summary['actual_trace_rows']+=1;summary['loss_total_sum']+=row['loss_total'];summary['loss_total_last']=row['loss_total']
                summary['max_gradient_norm']=max(summary['max_gradient_norm'],row['gradient_norm_before_physical_clip']);pair[arm]=row
            if any(pair['REPLAY'][k+'_indices']!=pair['BOUNDARY'][k+'_indices'] for k in ('L','OE')):
                raise ValueError('Both arms must use exactly the same source record indices at every step')
        if any(stream.readline() for stream in streams.values()):raise ValueError('Training traces cannot contain extra optimizer updates')
    for arm,s in summaries.items():
        s['loss_total_mean']=s.pop('loss_total_sum')/STEPS
        s.update(source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,actual_unique_L=len(usage[arm]['L']),actual_unique_OE=len(usage[arm]['OE']),all_numeric_values_finite=True)
    return dict(arms=summaries,source_sampling_matched=True,total_optimizer_updates=2*STEPS)


def _identities(c,m,fit,pm):
    root=Path(c['train_config']['output_root']);train=_read(root/'complete.json')
    expected=dict(schema='pamf_phase3_density_boundary_train_v1',status='DENSITY_BOUNDARY_TRAINING_COMPLETE',run_id=RUN_ID,code_commit=m['code_commit'],config=c['train_config'],
        arms=['REPLAY','BOUNDARY'],updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_sampling_matched=True,original_models_unchanged=True,
        all_student_buffers_unchanged=True,fixed_density_state_unchanged=True,point_teacher_unchanged=True,source_only=True,target_U_used=False,
        native_updates=0,identity_head_updates=0,validation_reads=0,novel_validation_reads=0,target_Q_reads=0,target_U_reads=0,original_U_s_reads=0,private_truth_reads=0,U_truth_reads=0)
    if any(train.get(k)!=v for k,v in expected.items()) or set(train.get('rows',{}))!=set(MODEL_NAMES[1:]):raise ValueError('Both independent source-only frozen continuations must complete')
    ids=fit['identities']
    if set(ids)!=set(MODEL_NAMES) or ids['E0']!=pm['model_identity'] or m['model_identities']!=ids:raise ValueError('Three feature axes must have their actual model identities')
    for arm in MODEL_NAMES[1:]:
        identity=ids[arm]
        expected=dict(run_id=RUN_ID,code_commit=m['code_commit'],arm=arm,updates=STEPS,open_signal_updates=STEPS,projection_updates=STEPS,native_updates=0,
            original_parent_updates=11100,parent_signal_updates=2220,signal_feature_cumulative_updates=pm['model_identity']['signal_feature_cumulative_updates']+STEPS,
            projection_cumulative_updates=pm['model_identity']['projection_updates']+STEPS,signal_parameters=334721,original_F0_parameters=232993,
            input_path='cached_P1_IQ_only',identity_head_replaced=False,original_F0_frozen=True,source_only=True,target_U_used=False,target_U_reads=0,
            fixed_density=True,point_anchor_weight=1.,boundary_weight=1. if arm=='BOUNDARY' else 0.)
        if (any(identity.get(k)!=v for k,v in expected.items()) or identity['parent_signal_identity']!=pm['model_identity']
                or identity['parent_F0_identity']!=pm['parent_F0_identity'] or identity['dura_identity']!=pm['dura_identity']):
            raise ValueError('Both new source-only arms must descend from the same original r03 branch')
        row=train['rows'][arm]
        if row['identity']!=identity or train.get('identities',{}).get(arm)!=identity or any(row.get(k)!=v for k,v in dict(schema=train['schema'],status=train['status'],run_id=RUN_ID,code_commit=m['code_commit'],
                arm=arm,updates=STEPS,source_only=True,target_U_used=False,model_buffers_unchanged=True,initial_state_equal_to_parent=True,
                original_models_unchanged=True,fixed_density_state_unchanged=True,point_teacher_unchanged=True,native_updates=0,identity_head_updates=0,
                validation_reads=0,novel_validation_reads=0,target_Q_reads=0,target_U_reads=0,original_U_s_reads=0,private_truth_reads=0,U_truth_reads=0).items()):
            raise ValueError('Trained row completion and model identity differ')
    return train,_trace_summary(root)


def _prediction(root):
    from . import star_density_boundary_run as evaluation
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],all_three_DURA_conditions_complete=True,classification_unchanged=True,
        classification_path='same_DURA_output_original_F0',identity_baseline_path='same_DURA_output_original_F0',score_threshold=0.,source_models_frozen_before_V=True,
        all_thresholds_frozen_before_novel_and_Q=True,all_conditions_through_DURA=True,source_E0_controls_reused_exactly=True,
        source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,states_shared_and_fixed=True,
        source_density_models_fitted=0,new_threshold_sets_fitted=4,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        private_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,original_U_s_reads=0,target_U_reads=0,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All frozen three-model six-column predictions must complete before truth')
    c=evaluation.validate_config(m['config']);pm,state,old_tau,old_cal=evaluation._reference(c)
    if (c['code_commit']!=m['code_commit'] or m['reference_prediction_identity']!=pm or m['parent_F0_identity']!=pm['parent_F0_identity']
            or m['dura_identity']!=pm['dura_identity'] or m['loaded_dura_identity']!=pm['loaded_dura_identity']):raise ValueError('Original class-density model and same-DURA reference identity required')
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json')
    if any(fit.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
            states_shared_and_fixed=True,source_density_models_fitted=0,normalizer_fit=False,source_V_fit=False,novel_fit=False,target_Q_fit=False).items()):
        raise ValueError('One source density must remain fixed for all three neural representations')
    train,traces=_identities(c,m,fit,pm)
    stored=_arrays(root/'models/density.npz');density.validate_state(stored,'density_ratio')
    if stored['schema'].item()!=density.SCHEMA or not evaluation._same(stored,state) or not evaluation._same(_arrays(Path(c['train_config']['output_root'])/'density.npz'),state):
        raise ValueError('Training and all predictions must share the original exact D160 density')
    saved=_arrays(root/'thresholds.npz');tau=saved.get('conditional')
    if (set(saved)!={'conditional'} or tau is None or tau.dtype!=np.float64 or tau.shape!=(6,6) or not np.isfinite(tau).all() or not np.array_equal(tau[:2],old_tau)
            or not np.array_equal(tau,np.asarray(cal['conditional_thresholds'],np.float64))):raise ValueError('Two old exact and four new frozen class-threshold sets required')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,
        group_count=42,known_only=True,unknown_inputs_used=False,target_reads=0,hard_names=list(evaluation.risk.HARD_NAMES),diagnostic_names=list(evaluation.risk.DIAGNOSTIC_NAMES))
    if any(cal.get(k)!=v for k,v in expected.items()) or set(cal.get('methods',{}))!=set(METHODS):raise ValueError('All six source-known12 calibrations and cross diagnostics required')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if report['conditional_thresholds']!=tau[j].tolist() or set(report['groups'])!=set(common.GROUP_NAMES):raise ValueError('Known risk groups and deployed thresholds must agree')
        if j<2 and report!=old_cal['methods'][pm['method_names'][(1,3)[j]]]:raise ValueError('E0 parent calibrations cannot change')
    parent_root=c['train_config']['reference_config']['output_root'];rows={'source_DURA':{}};seen=set()
    for role in ('known','novel'):
        a=evaluation._source_rows(root,'source_DURA',role);_validate_rows(a,COUNTS['V' if role=='known' else 'novel'],evaluation.KNOWN_FIELDS if role=='known' else evaluation.NOVEL_FIELDS,state,tau)
        _exact_E0(a,parent._rows(parent_root,role))
        if seen&set(a['ids'].tolist()):raise ValueError('Source known and novel roles overlap')
        seen.update(a['ids'].tolist());rows['source_DURA'][role]=a
    for condition in CONDITIONS[1:]:
        a=evaluation._target_rows(root,condition);_validate_rows(a,Q_COUNT,evaluation.TARGET_FIELDS,state,tau);_exact_E0(a,parent._target_rows(parent_root,condition))
        if seen&set(a['ids'].tolist()):raise ValueError('Future query cannot overlap source evaluation')
        rows[condition]=a
    m=dict(m,scorer_training_trace=traces,scorer_training_cost={k:train[k] for k in ('total_optimizer_updates','real_student_signal_forward_rows','real_parent_signal_forward_rows',
        'F_head_only_rows','real_F_forward_rows','G_forward_rows','synthetic_F_forward_rows','synthetic_parent_signal_forward_rows','G_synthetic_forward_rows','synthetic_gradient_student_rows')})
    return m,cal,rows


def _exact_E0(a,old):
    for key,value in a.items():
        actual=value[:,:2] if key in ('raw_scores','scores') else value[:,0] if key=='signal_features' else value
        expected=old[key][:,[1,3]] if key in ('raw_scores','scores') else old[key]
        if not np.array_equal(actual,expected):raise ValueError('E0 features, old ratio controls and every shared public field must remain exact')


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent boundary score output required')
    m,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source_DURA']['novel']);truths={base:_target_truth(path,rows[base+'_DURA']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        if not np.isin(t['y'],range(9)).all() or not np.all(t['receiver']==26) or not np.isin(t['scenario'],('clean',) if base=='clean' else common.SCENES).all():raise ValueError('Fixed development RX/condition and private scorer roster required')
    methods={}
    for j,name in enumerate(METHODS):
        paths={'source_DURA':previous._source(rows['source_DURA'],nt,j)}
        paths['source_DURA']['source_calibration_passed']=bool(cal['methods'][name]['constraints_passed'])
        paths.update({base+'_DURA':previous._target(rows[base+'_DURA'],truths[base],base+'_DURA',j) for base in ('clean','star')})
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY)
    main=methods[PRIMARY]['paths'];paired={}
    for reference in (1,3,4):
        by_condition={}
        for condition in CONDITIONS:
            if condition=='source_DURA':
                k,n=rows[condition]['known'],rows[condition]['novel'];km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],nt['scenario']))
                groups={g:_pair(k,n,5,reference,mask,nm[g]) for g,mask in km.items()};uy=nt['tx'];unknown_roster=common.UNKNOWN_TX
            else:
                t=truths[condition.split('_')[0]];y=t['y'];mask=y<6;a=rows[condition];k=dict(y=y[mask],closed=a['closed'][mask],scores=a['scores'][mask]);n=dict(scores=a['scores'][~mask])
                scenes=('clean',) if condition=='clean_DURA' else common.SCENES;groups={g:_pair(k,n,5,reference,t['scenario'][mask]==g,t['scenario'][~mask]==g) for g in scenes};uy=y[~mask];unknown_roster=(6,7,8)
            by_condition[condition]=dict(counts=_pair(k,n,5,reference),groups=groups,delta=previous._delta(main[condition]['overall'],methods[METHODS[reference]]['paths'][condition]['overall']),
                by_known_TX={str(c):_pair(k,n,5,reference,k['y']==c) for c in range(6)},by_unknown_TX={str(c):_pair(k,n,5,reference,nm=uy==c) for c in unknown_roster})
        paired[METHODS[reference]]=by_condition
    source_known=main['source_DURA']['known_12_groups_passed'] and main['source_DURA']['source_calibration_passed']
    no_regression=all(paired[name]['star_DURA']['delta'][key] is not None and paired[name]['star_DURA']['delta'][key]>=-1e-12 for name in ('E0_predicted','REPLAY_predicted') for key in ('A_K','R_U'))
    protection=all(main[c]['goals']['passed'] for c in CONDITIONS[1:]);passed=bool(source_known and protection and no_regression)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,primary_summary={c:main[c]['overall'] for c in CONDITIONS},
        paired_primary_vs_E0_predicted=paired['E0_predicted'],paired_primary_vs_REPLAY_predicted=paired['REPLAY_predicted'],paired_primary_vs_BOUNDARY_pooled=paired['BOUNDARY_pooled'],
        source_known_12_passed=bool(source_known),source_unknown_diagnostic_only=True,star_not_worse_than_both_controls=bool(no_regression),target_protection_passed=bool(protection),development_gate_passed=passed,eligible_primary=passed,
        next_step='DEVELOPMENT_PASSED_FOR_SEPARATE_UDA_PREREGISTRATION' if passed else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,U_truth_reads=0,
        source_only=True,target_U_reads=0,target_DA=False,target_REG=False,classification_unchanged=True,source_density_models_fitted=0,new_source_threshold_sets=4,
        new_signal_optimizer_updates_total=2*STEPS,updates_per_arm=STEPS,source_training_exposures_total=2*256*STEPS,training_trace=m['scorer_training_trace'],training_cost=m['scorer_training_cost'],
        per_record_source_rows_deployed=0,source_bank_deployed=False,formal_target_accessed=False,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False,full_goal_passed=False,
        costs={k:m[k] for k in ('real_F_forward_rows','G_forward_rows','real_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows','synthetic_signal_forward_rows')},
        primary_deployed_neural_parameters=834665,all_three_models_resident_parameters=1504107,scope='Both learned arms source-only; original fixed density and F0 identity. Historical source/Q development exposure, no U or anonymous registration')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_E0_predicted','paired_primary_vs_REPLAY_predicted','paired_primary_vs_BOUNDARY_pooled')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser()
    for key in ('prediction-root','source-novel-truth','clean-truth','star-truth','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
