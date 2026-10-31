"""Independent four-truth scoring for source-ground native energy readers."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import ground_open_run as evaluation
from . import fresh_world_score as fresh
from . import star_consensus_score as common
from . import star_dura_best_score as previous
from .star_physical_open_score import _metrics

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_ground_open_scores_v1'
STATUS='GROUND_OPEN_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=METHODS[2]
CONDITIONS=evaluation.CONDITIONS
_read,_write,_source_truth,_future_truth=fresh._read,fresh._write,fresh._source_truth,fresh._future_truth


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _reconstruct(a,tau):
    logits=a['default_logits'];closed=a['closed']
    if (logits.dtype!=np.float32 or logits.shape!=(len(a['ids']),3,6) or not np.isfinite(logits).all()
            or closed.dtype!=np.int64 or closed.shape!=logits.shape[:2] or not np.array_equal(closed,logits.argmax(2))
            or not np.array_equal(a['baseline_closed'],closed[:,0])):
        raise ValueError('Every native default head and original same-DURA F0 baseline must agree')
    raw=np.column_stack([evaluation.core.energy(logits[:,j]) for j in range(3)])
    margins=np.column_stack([evaluation.core.apply(raw[:,j],closed[:,j],dict(schema=evaluation.core.SCHEMA,conditional_thresholds=tau[j])) for j in range(3)])
    if not np.array_equal(a['raw_scores'],raw) or not np.array_equal(a['scores'],margins):
        raise ValueError('FP64 native energy and each own predicted-class threshold must reconstruct')


def _public_novel(c):
    # Only public identifiers/metadata are needed to join the existing source
    # novel truth. Do not materialize old features, scores, IQ or private labels.
    path=Path(c['reference_config']['output_root'])/'source/source_DURA/novel_predictions.npz'
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('ids','rx_i','day_i')}


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        calibration_known_count=evaluation.CAL_COUNT,source_query_count=evaluation.SOURCE_Q_COUNT,cross_query_count=evaluation.TARGET_Q_COUNT,
        source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,target_U_reads=0,target_U_GT_reads=0,source_U_GT_reads=0,
        source_Us_values_read_during_prediction=0,prediction_optimizer_updates=0,source_density_fit_updates=0,new_threshold_values=18,threshold_bytes=144,
        real_student_forward_rows=2*evaluation.CAL_COUNT+4*evaluation.SOURCE_Q_COUNT+4*evaluation.TARGET_Q_COUNT,real_F_forward_rows=0,G_forward_rows=2*evaluation.TARGET_Q_COUNT,
        source_only=True,ground_clean_training=True,new_target_DA=False,original_models_unchanged=True,student_heads_jointly_trained=True,
        primary_deployed_neural_parameters=499944,primary_threshold_values=6,formal_target_accessed=False,full_goal_status='ACTIVE',final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All twelve frozen ground predictions must complete before any private truth')
    c=evaluation.validate_config(m['config']);d,t=evaluation._metadata(c)
    if c['code_commit']!=m['code_commit'] or m['dataset_identity']!=d or m['model_identities']!=t['identities']:
        raise ValueError('Exact current source-ground data and completed native training identities required')
    if (m['training_costs']!={k:v for k,v in t.items() if isinstance(v,(bool,int,float))}
            or m['data_costs']!={k:v for k,v in d.items() if isinstance(v,(bool,int,float))} or m['trained_optimizer_updates']!=t['total_optimizer_updates']):
        raise ValueError('Recorded data and training costs must remain those of the completed stages')
    for key in ('parent_F0_identity','dura_identity','loaded_dura_identity'):
        if m[key]!=d[key] or t[key]!=d[key]:raise ValueError('Original F0 and frozen same-input DURA ancestry differs')
    if m['data_capsule_id']!=d['capsule_id'] or m['data_split_id']!=d['split_id']:raise ValueError('Once-validated ground capsule/split differs')
    train=evaluation._train();pm,_,_,_=train._reference(c['train_config'])
    for arm in ('SUP','SSL'):
        ground,handles=train._ground_metadata(c['train_config'],pm,with_us=arm=='SSL');i=m['model_identities'][arm]
        expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,updates=train.STEPS,reader_parameters=232993,trainable_parameters=232945,
            default_head_parameters=960,Sinc_frozen=True,identity_head_updates=train.STEPS,identity_head_jointly_trained=True,identity_head_replaced=False,
            projection_updates=0,initial_native_logits_equal_original_F0=True,parent_F0_identity=d['parent_F0_identity'],dura_identity=d['dura_identity'],loaded_dura_identity=d['loaded_dura_identity'],
            ground_data_identity=ground,source_handles=handles,source_only=True,ground_clean_training=True,source_U_s_used=arm=='SSL',source_U_s_reads=train.US_COUNT if arm=='SSL' else 0,
            target_U_used=False,target_U_reads=0,target_Q_reads=0,source_U_GT_reads=0,input_path='G(clean)')
        if any(i.get(k)!=v for k,v in expected.items()):raise ValueError('SUP source-only and SSL source-Us native reader provenance required')
    smoke=_read(root/'smoke.json')
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_ground_students_DURA=True,optimizer_updates=0,real_data_reads=0).items()):
        raise ValueError('Actual frozen students/DURA no-query smoke required')
    if any(m.get(k)!=smoke.get(k) for k in ('synthetic_student_forward_rows','G_synthetic_forward_rows')):
        raise ValueError('Prediction synthetic costs must match the actual no-query smoke')
    cal=_read(root/'calibration/complete.json');saved=_arrays(root/'thresholds.npz');tau=saved.get('conditional')
    if (set(saved)!={'conditional'} or not isinstance(tau,np.ndarray) or tau.shape!=(3,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
                method_names=list(METHODS),primary_method=PRIMARY,calibration_count=evaluation.CAL_COUNT,known_only=True,unknown_inputs_used=False,test_inputs_used=False,target_reads=0,
                neural_models_frozen_before_CAL=True,group_names=list(evaluation.core.GROUP_NAMES)).items())
            or set(cal.get('methods',{}))!=set(METHODS) or cal.get('conditional_thresholds')!=tau.tolist() or cal.get('calibration_manifest')!=d['roles']['CAL']):
        raise ValueError('Three six-class thresholds must be fitted only on the separate source CAL role')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if (any(report.get(k)!=v for k,v in dict(schema=evaluation.core.SCHEMA,conditional_thresholds=tau[j].tolist(),calibration_count=evaluation.CAL_COUNT,
                group_names=list(evaluation.core.GROUP_NAMES),known_only=True,unknown_inputs_used=False,test_inputs_used=False,U_labels_used=False).items())
                or set(report.get('groups',{}))!=set(evaluation.core.GROUP_NAMES) or type(report.get('constraints_passed')) is not bool):
            raise ValueError('All twenty-four known-only calibration groups and own-head thresholds required')
    rows={condition:evaluation._rows(root,condition) for condition in CONDITIONS}
    for a in rows.values():_reconstruct(a,tau)
    if any(rows[c][k].dtype!=np.int64 for c in CONDITIONS[:2] for k in ('rx_i','day_i')):
        raise ValueError('Source public receiver/date fields must retain their exact int64 schema')
    source,target=rows[CONDITIONS[0]],rows[CONDITIONS[2]]
    if any(not np.array_equal(source[k],rows[CONDITIONS[1]][k]) for k in ('ids','rx_i','day_i')) or not np.array_equal(target['ids'],rows[CONDITIONS[3]]['ids']):
        raise ValueError('Clean/star must share ordered physical records and source public metadata')
    if set(source['ids'].tolist())&set(target['ids'].tolist()):raise ValueError('Source test and cross-RX future records overlap')
    for condition in CONDITIONS[2:]:
        old=evaluation._old_target(c,condition)
        if not np.array_equal(rows[condition]['ids'],old['ids']) or not np.array_equal(rows[condition]['default_logits'][:,0],old['default_logits']):
            raise ValueError('Original same-DURA F0 cross-RX baseline must remain exact')
    novel=_public_novel(c)
    if not common._ids(novel['ids'],d['counts']['novel']) or not set(novel['ids'].tolist()).issubset(set(source['ids'].tolist())):
        raise ValueError('Completed source novel public IDs must be contained in the role-blind query')
    evaluation.core.masks(novel['rx_i'],novel['day_i'])
    return m,cal,rows,novel


def _source_known_truth(path):
    a=_arrays(path);fields={'ids','y','rx_i','day_i','star_scenario'}
    if (set(a)!=fields or not common._ids(a['ids'],len(a['ids'])) or any(v.shape!=a['ids'].shape for v in a.values())
            or any(a[k].dtype!=np.int64 for k in ('y','rx_i','day_i')) or not np.isin(a['y'],range(6)).all() or a['star_scenario'].dtype.kind!='U' or not np.isin(a['star_scenario'],common.SCENES).all()):
        raise ValueError('Exact independent source known-test truth required, with six old TX only')
    evaluation.core.masks(a['rx_i'],a['day_i']);return a


def _join_source(a,known,novel,star):
    ki=known['ids'].tolist();ni=novel['ids'].tolist();ids=a['ids'].tolist();K=set(ki);N=set(ni)
    if K&N or K|N!=set(ids) or novel['tx'].dtype.kind!='U' or not np.isin(novel['tx'],common.UNKNOWN_TX).all():
        raise ValueError('Disjoint six-old source TEST and three source-novel truth rosters must exactly cover the query')
    combined={k:np.concatenate((known[k],novel[k])) for k in ('ids','rx_i','day_i')};lookup={v:i for i,v in enumerate(combined['ids'].tolist())};order=np.asarray([lookup[v] for v in ids],np.int64)
    if any(not np.array_equal(combined[k][order],a[k]) for k in ('rx_i','day_i')):raise ValueError('Source truth public RX/day differs')
    y=np.concatenate((known['y'],np.full(len(ni),-1,np.int64)))[order]
    tx=np.concatenate((np.asarray(fresh.TX_BY_LABEL)[known['y']],novel['tx']))[order]
    scene=np.concatenate((known['star_scenario'],novel['scenario']))[order] if star else np.full(len(ids),'clean')
    return dict(ids=a['ids'],y=y,tx=tx,known=y>=0,rx_i=a['rx_i'],day_i=a['day_i'],scenario=scene)


def _passed(row,known_minimum,unknown_minimum):
    return bool(row['coverage']['complete'] and row['C_K'] is not None and row['C_K']>=known_minimum-1e-12
        and row['R_U'] is not None and row['R_U']>=unknown_minimum-1e-12 and row['original_closed_minus_new_open'] is not None and row['original_closed_minus_new_open']<=.02+1e-12)


def _source(a,t,condition,index):
    k=t['known'];u=~k;allrows=np.ones(len(k),bool);clean=condition=='source_clean_DURA';minimum=.95 if clean else .90
    def metric(mask,kr=tuple(range(6)),ur=common.UNKNOWN_TX):
        km=mask&k;um=mask&u
        return _metrics(t['y'][km],a['closed'][km,index],a['scores'][km,index],t['tx'][um],a['scores'][um,index],a['baseline_closed'][km],kr,ur)
    masks=evaluation.core.masks(t['rx_i'],t['day_i']);groups={g:metric(mask) for g,mask in masks.items()};overall=groups['overall']
    scenes=('clean',) if clean else common.SCENES;by_scene={s:metric(t['scenario']==s) for s in scenes}
    scene_masks={'overall':allrows}|{'rx/'+str(r):t['rx_i']==r for r in common.RECEIVERS}|{'scene/'+s:t['scenario']==s for s in scenes}|{'rxscene/'+str(r)+'/'+s:(t['rx_i']==r)&(t['scenario']==s) for r in common.RECEIVERS for s in scenes}
    scene_groups={g:metric(mask) for g,mask in scene_masks.items()}
    known_pass=lambda r:bool(r['coverage']['known_complete'] and r['C_K'] is not None and r['C_K']>=minimum-1e-12 and r['original_closed_minus_new_open'] is not None and r['original_closed_minus_new_open']<=.02+1e-12)
    passed=all(_passed(r,minimum,minimum) for r in groups.values())
    return dict(overall=overall,groups=groups,scene_groups=scene_groups,by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},
        by_day={str(d):groups['day/'+str(d)] for d in evaluation.core.DAYS},by_RX_day={str(r)+'/'+str(d):groups['rxday/'+str(r)+'/'+str(d)] for r in common.RECEIVERS for d in evaluation.core.DAYS},by_scene=by_scene,
        by_day_scene={str(d):{s:metric((t['day_i']==d)&(t['scenario']==s)) for s in scenes} for d in evaluation.core.DAYS},
        by_TX={fresh.TX_BY_LABEL[c]:metric((t['y']==c)|u,(c,)) for c in range(6)}|{c:metric(k|(t['tx']==c),ur=(c,)) for c in common.UNKNOWN_TX},
        confidence_intervals=previous._confidence(overall),known_24_groups_passed=all(known_pass(r) for r in groups.values()),
        known_12_scene_groups_passed=None if clean else all(known_pass(scene_groups[g]) for g in common.GROUP_NAMES[:12]),
        known_scene_groups_passed=all(known_pass(r) for r in scene_groups.values()),known_42_groups_passed=None if clean else all(known_pass(r) for r in scene_groups.values()),
        goals=dict(known_minimum=minimum,unknown_minimum=minimum,original_F0_loss_max=.02,overall_passed=_passed(overall,minimum,minimum),all_24_groups_passed=passed,
            required_group_coverage_complete=all(r['coverage']['complete'] for r in groups.values()),passed=passed),
        known_roster=list(range(6)),unknown_roster=list(common.UNKNOWN_TX),source_test_role='known_TEST13500_disjoint_from_CAL13500',source_RX_seen=True,source_novel_historically_exposed=True,
        scene_groups_diagnostic_only=True,source_star_diagnostic_only=not clean)


def _paired(a,t,index,reference,source):
    k=t['known'] if source else t['y']<6;u=~k
    def known(j):return dict(ids=a['ids'][k],y=t['y'][k],closed=a['closed'][k,j],scores=a['scores'][k,j:j+1])
    def novel(j):return dict(ids=a['ids'][u],scores=a['scores'][u,j:j+1])
    new,old,nn,on=known(index),known(reference),novel(index),novel(reference)
    def pair(mask=None):return previous.paired_counts(new,old,nn,on,0,None if mask is None else mask[k],None if mask is None else mask[u])
    rx=t['rx_i'] if source else t['receiver'];day=t['day_i'] if source else t['day'];scenes=('clean',) if np.all(t['scenario']=='clean') else common.SCENES
    ur=common.UNKNOWN_TX if source else fresh.TARGET_UNKNOWN;uy=t['tx'] if source else t['y']
    groups=evaluation.core.masks(rx,day) if source else {s:t['scenario']==s for s in scenes}
    return dict(primary_column=index,reference_column=reference,counts=pair(),groups={g:pair(mask) for g,mask in groups.items()},
        by_RX={str(r):pair(rx==r) for r in np.unique(rx)},by_day={str(d):pair(day==d) for d in (evaluation.core.DAYS if source else fresh.DAYS)},
        by_RX_day={str(r)+'/'+str(d):pair((rx==r)&(day==d)) for r in np.unique(rx) for d in (evaluation.core.DAYS if source else fresh.DAYS)},
        by_scene={str(s):pair(t['scenario']==s) for s in scenes},
        by_day_scene={str(d):{str(s):pair((day==d)&(t['scenario']==s)) for s in scenes} for d in (evaluation.core.DAYS if source else fresh.DAYS)},
        by_known_TX={fresh.TX_BY_LABEL[c]:previous.paired_counts(new,old,nn,on,0,t['y'][k]==c) for c in range(6)},
        by_unknown_TX={str(c):previous.paired_counts(new,old,nn,on,0,unknown_mask=uy[u]==c) for c in ur})


def _decision(primary,cal):
    source=bool(primary['source_clean_DURA']['goals']['passed'] and cal['methods'][PRIMARY]['constraints_passed'])
    cross=bool(primary['cross_clean_DURA']['goals']['passed'])
    return source,cross,bool(source and cross)


def score(prediction_root,source_known_truth,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable ground-open score root required')
    m,cal,rows,public_novel=_prediction(prediction_root)
    # First private reads: all four conditions and all three readers are fixed.
    kt=_source_known_truth(source_known_truth);nt=_source_truth(source_novel_truth,public_novel)
    truths={condition:_join_source(rows[condition],kt,nt,condition=='source_star_DURA') for condition in CONDITIONS[:2]}
    for condition,path in zip(CONDITIONS[2:],(clean_truth,star_truth)):
        truths[condition]=_future_truth(path,rows[condition]['ids']);scenes=('clean',) if condition=='cross_clean_DURA' else common.SCENES
        if not np.isin(truths[condition]['scenario'],scenes).all():raise ValueError('Future clean/star scorer scenario mismatch')
    if any(not np.array_equal(truths[CONDITIONS[2]][k],truths[CONDITIONS[3]][k]) for k in ('ids','y','tx','receiver','day')):
        raise ValueError('Cross clean/star truth must refer to the same physical records')
    methods={}
    for j,name in enumerate(METHODS):
        paths={c:_source(rows[c],truths[c],c,j) for c in CONDITIONS[:2]}
        for condition in CONDITIONS[2:]:
            a=dict(rows[condition],closed=rows[condition]['closed'][:,j]);paths[condition]=fresh._target(a,truths[condition],condition.removeprefix('cross_'),j)
        source=bool(paths[CONDITIONS[0]]['goals']['passed'] and cal['methods'][name]['constraints_passed']);cross=bool(paths[CONDITIONS[2]]['goals']['passed'])
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,source_CAL_constraints_passed=cal['methods'][name]['constraints_passed'],
            ground_source_success=source,cross_clean_success=cross,ground_training_success=bool(source and cross))
    primary=methods[PRIMARY]['paths'];paired={}
    for j,label in enumerate(METHODS[:2]):
        paired[label]={}
        for condition in CONDITIONS:
            p=_paired(rows[condition],truths[condition],2,j,condition in CONDITIONS[:2]);p['delta']=previous._delta(primary[condition]['overall'],methods[label]['paths'][condition]['overall']);paired[label][condition]=p
    source,cross,success=_decision(primary,cal)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:primary[c]['overall'] for c in CONDITIONS},paired_primary_vs_controls=paired,ground_source_success=source,cross_clean_success=cross,
        ground_training_success=success,scientific_success=success,development_gate_passed=success,source_star_diagnostic_passed=primary[CONDITIONS[1]]['goals']['passed'],cross_star_diagnostic_passed=primary[CONDITIONS[3]]['goals']['passed'],
        next_step='GROUND_STAGE_PASSED_REQUIRES_STAR_ADAPTATION' if success else 'NO_PROMOTION',truth_last=True,all_four_predictions_before_truth=True,all_12_points_before_truth=True,
        source_known_truth_reads=1,source_novel_truth_reads=1,target_truth_reads=2,private_truth_reads=4,U_truth_reads=0,source_U_GT_reads=0,adaptation_success=None,new_target_DA=False,target_REG=False,registration_success=None,A_N=None,H_KN=None,H_OW=None,
        source_only=True,ground_clean_training=True,SUP_source_Us_used=False,SSL_source_Us_used=True,source_CAL_role='separate_known_half_source_V',source_known_test_role='other_half_source_V_not_CAL; source_RX_seen',
        original_F0_unchanged=True,classification_unchanged=False,classification_path='each own complete native default head',identity_baseline_path='same_DURA_output_original_F0',
        new_source_density_models=0,new_source_threshold_sets=3,threshold_values=18,threshold_bytes=144,primary_threshold_values=6,primary_deployed_neural_parameters=499944,
        training_costs=m['training_costs'],data_costs=m['data_costs'],prediction_costs={k:m[k] for k in ('real_student_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_student_forward_rows','G_synthetic_forward_rows')},
        combined_new_real_student_forward_rows=m['training_costs']['real_student_forward_rows']+m['real_student_forward_rows'],
        combined_new_G_forward_rows=m['data_costs']['G_forward_rows']+m['G_forward_rows'],combined_new_real_F_forward_rows=m['data_costs']['real_F_forward_rows']+m['real_F_forward_rows'],
        source_bank_deployed=False,formal_target_accessed=False,source_novel_historically_exposed=True,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        scope='Source-ground G(clean) SUP and ordinary source-Us PL native energy readers; source TEST/NV clean and star plus exposed cross-RX clean/star. Promotion uses fixed SSL source-clean 24-group CK95/RU95 and cross-clean CK90/RU95 with original F0 loss2pp and complete required rosters; star remains diagnostic, full Phase3 goal active')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_controls')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser()
    for flag in ('prediction-root','source-known-truth','source-novel-truth','clean-truth','star-truth','output'):p.add_argument('--'+flag,required=True)
    a=p.parse_args();score(a.prediction_root,a.source_known_truth,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
