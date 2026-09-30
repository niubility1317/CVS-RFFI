"""Hand-built AFFINE_JOINT training archives, no core fit, data, outer score or SSH."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools')]
import collect_d92_affine_joint_training_diagnostics as collector


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows),encoding='utf-8')


def archive(lane, name, **values):
    (lane/'state_arrays').mkdir(exist_ok=True)
    relative='state_arrays/'+name+'.npz'
    with (lane/relative).open('xb') as stream: np.savez_compressed(stream,**values)
    namespace = json.dumps(dict(state='B_AFFINE', run_id='synthetic', row_id=lane.parent.name, scope='support_oof'))
    return dict(path=relative,key=name,namespace=namespace,arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype),nbytes=value.nbytes)
        for key,value in values.items()})


def head(lane, name, classes, *, later=False, tau=1., gamma=2., prior=False):
    c=len(classes); n=c; labels=np.arange(c,dtype=np.int64)
    M=np.zeros((n,c))
    if c==3:
        M[:,:2]=np.asarray([[.8,.1],[.2,.7],[.1,.1]])
        scores=M+np.asarray([[0.,0.,.3],[0.,0.,0.],[0.,0.,.4]])
        if later: scores=scores+np.asarray([[0.,0.,.7],[.1,0.,0.],[0.,0.,.1]])
        q=np.asarray([.5,.5,0.])
    else:
        scores=np.eye(c)*(.8 if later else .6); q=np.ones(n)/n
    if prior: M=np.zeros_like(scores)
    intercept=np.linspace(-.2,.2,c); scores=scores+intercept
    Y=np.eye(c)-1/c; K=np.eye(n)*(1.2 if later else 1.)
    b=np.ones((n,2)); a=np.ones((n,3)); adapted=b+(.05 if later else 0.)
    scalar=lambda value:np.empty(0) if value is None else np.asarray(value)
    values=dict(original_train_b=b,original_train_a=a,original_held_b=b,original_held_a=a,
        adapted_train_b=adapted,adapted_train_a=a,adapted_held_b=adapted,adapted_held_a=a,
        train_labels=labels,held_labels=labels,q=q,raw_train=K,raw_cross=K,
        raw_train_minus_one=K-1,raw_cross_minus_one=K-1,distance=K,cross_distance=K,
        K=K,L=K,chol=np.eye(n),alpha=np.eye(c),Y=Y,M_train=M,M_held=M,E=Y-M,
        train_scores=scores,scores=scores,reference_kernel=np.zeros(n),reference_self=np.asarray(0.),
        center_mean=np.zeros(n),center_grand=np.asarray(0.),tau=scalar(tau),gamma=scalar(gamma),
        s0=np.asarray(3.),actual_trace=np.asarray(float(np.trace(K))),intercept=intercept,
        schur_z=np.linspace(.8,1.1,n),schur_s=np.asarray(float(np.linspace(.8,1.1,n).sum())),
        combined_rhs=np.column_stack((Y-M,np.ones(n))))
    ref=archive(lane,name,**values)
    audit=dict(inner_fold=0,classes=classes,old_classes=classes[:2],held_physical_count=n,
        head_fit_count=1,factorization_count=int(gamma is not None),head_triangular_solve_count=2,
        actual_kernel_trace=float(np.trace(K)),bandwidth_tau=tau,trace_scale=gamma,
        interaction_centered_trace=3.,forward_seconds=.04,adjoint_seconds=.02,
        block_angle_radians=dict(count=n,minimum=0.,mean=.1 if later else 0.,maximum=.2 if later else 0.),
        head_training_loss_data=.3,head_training_loss_ridge=.4,class_sum_max_abs=0.,head_ref=ref,
        head_triangular_rhs_count=2*(c+1)*int(gamma is not None),
        head_triangular_rhs_element_count=2*n*(c+1)*int(gamma is not None),
        head_triangular_dense_work_unit_count=2*n*n*(c+1)*int(gamma is not None),
        intercept_fit_count=1,intercept_addition_count=2*n*c,
        expected_old_reference_mean=(q@M+intercept).tolist())
    return ref,audit,values


def fixture(root):
    summary=root/'verified';summary.mkdir();stages=[];sources=[];archives=[];events_by_row={}
    descriptions=[ [('B_AFFINE','move',1,'first'),('B_AFFINE','move',1,'repeat')],
        [('C_AFFINE_seq','move',2,'first')], [('B_AFFINE','noinfo',0,'first')],
        [('B_AFFINE','zero',3,'first'),('C_AFFINE_seq','exhaust',3,'first'),('B_AFFINE','tau0',3,'degenerate')] ]
    for ri,rows in enumerate(descriptions):
        row_id='row-'+str(ri);lane=root/'run'/row_id/'probe';lane.mkdir(parents=True)
        events=[];logs=[]
        for si,(state,behavior,rank,split) in enumerate(rows):
            prefix=str(si)+'-';mode=collector.MODES[state];info=behavior not in ('noinfo','tau0')
            k=3 if info else 1;scope='support_oof' if info else 'support_full_k1'
            coords=dict(run_id='synthetic', schema=collector.SCHEMA, method=collector.METHOD,
                row_id=row_id,split_id=split,scope=scope,fold=0 if info else None,
                outer_trial=None,state=state,train_k=k)
            classes=['old-a','old-b']+(['new-c'] if mode=='C_seq' else [])
            # Same physical old B binding repeats across different parent IDs.
            physical=['p-a','p-b'];Z=np.zeros((736,rank));W=np.eye(8,rank);H=np.ones((6,8))
            anchor=np.zeros((736,8));anchor[0,0]=.25 if mode=='C_seq' else 0.;U=anchor.copy()
            prep_state='B_prepare' if mode=='B' else 'C_prepare'
            prepared_ref=archive(lane,prefix+'prepared',H=H,W=W,anchor_U=anchor,singular_values=np.arange(8,0,-1,dtype=float))
            prep=dict(coords,state=prep_state,event='AFFINE_JOINT_PREPARED',prepared_state_ref=prepared_ref,
                classes=classes,old_classes=['old-a','old-b'],training_physical_ids=physical,
                latent_rank=rank,rank_estimated=True,rank_energy_threshold=1e-12,whitening_residual=1e-15,
                whitening_tolerance=1e-12,dictionary_rms=1.,no_information=not info,
                no_information_reason=None if info else 'PHYSICAL_K1' if behavior=='noinfo' else 'ALL_INNER_GEOMETRY_DEGENERATE',
                ajlr_preparation_count=1,latent_svd_count=int(rank>0),prior_head_fit_count=int(mode=='C_seq' and info),
                prior_factorization_count=int(mode=='C_seq' and info),prior_triangular_solve_count=2*int(mode=='C_seq' and info),
                dictionary_physical_evaluation_count=6,raw_distance_pair_count=2,future_preparation_count=11,
                preparation_seconds=.2,inner_folds=[],final_problem={'bandwidth_tau':0. if behavior=='tau0' else 1.},prior_folds=[])
            prior_ref=None; prior_values=None
            if mode=='C_seq' and info:
                prior_ref,prior_audit,prior_values=head(lane,prefix+'prior',['old-a','old-b'],prior=True)
                prep['prior_folds']=[dict(inner_fold=0,head_state_ref=prior_ref,head_ref=prior_ref,final_fit=prior_audit)]
                prep.update(prior_triangular_rhs_count=6,prior_triangular_rhs_element_count=12,
                    prior_triangular_dense_work_unit_count=24,prior_intercept_fit_count=1,prior_intercept_addition_count=19)
                prep['final_problem']['prior_ref']=prior_ref
            for key in collector.PREPARATION_COUNTERS:
                if key.startswith('prior_'): prep.setdefault(key,0)
            events.append(prep)
            logs.append(dict(prep,event='AFFINE_PREPARATION',trial=None))
            initial_head,initial_audit,ih=head(lane,prefix+'initial-head',classes)
            initial_audit['prior_ref']=prior_ref
            means=np.arange(1,len(classes)+1,dtype=float);counts=np.arange(1,len(classes)+1,dtype=np.int64)
            def objective(z,hr,ha,*,cached=False,backward=False,better=False):
                risk=float(np.linalg.norm(means)/np.sqrt(len(means)))-(.1 if better else 0.)
                prox=.5*float(np.sum(z*z));fold=dict(ha,head_ref=hr,adjoint_seconds=.02 if backward else 0.)
                return dict(RMSCE=risk,loss_ce=risk,loss_task=risk,loss_proximal=prox,loss_total=risk+prox,
                    class_ce_means=means.tolist(),class_ce_counts=counts.tolist(),class_ce_sums=(means*counts).tolist(),
                    inner_folds=[fold],head_refs=[hr],forward_cache_reused=cached,objective_seconds=.01,
                    inner_objective_evaluation_count=int(not cached),inner_head_fit_count=int(not cached),
                    ce_adjoint_solve_count=int(backward),derivative_triangular_solve_count=2*int(backward),
                    pre_tangent_displacement_mean_squared=2*prox,pre_tangent_displacement_rms=float(np.linalg.norm(z)),
                    coordinate_squared_norm=2*prox,function_coordinate_reconstruction_error=0.,
                    **{key:(0 if cached else ha[key]) for key in
                        ('head_triangular_rhs_count','head_triangular_rhs_element_count','head_triangular_dense_work_unit_count','intercept_fit_count','intercept_addition_count')},
                    derivative_triangular_rhs_count=2*len(classes)*int(backward),
                    derivative_triangular_rhs_element_count=2*len(classes)**2*int(backward),
                    derivative_triangular_dense_work_unit_count=2*len(classes)**3*int(backward))
            def save_state(name,z,u,scores,labels,**extra):
                arrays=dict(Z=z,U=u,anchor_U=anchor,W=W,singular_values=np.arange(8,0,-1,dtype=float),
                    scores=scores,labels=labels,class_ce_means=means,class_ce_counts=counts,class_ce_sums=means*counts,
                    RMSCE=np.asarray(float(np.linalg.norm(means)/np.sqrt(len(means)))),prox=np.asarray(.5*float(np.sum(z*z))))
                arrays.update(extra)
                if 'g_Z' in extra:
                    c=len(classes)
                    arrays.update(fold_0_adjoint_T=np.ones((c,c))*.2,fold_0_adjoint_eta=np.linspace(-.1,.1,c),
                        fold_0_adjoint_g_b=np.linspace(-.3,.3,c),fold_0_adjoint_rhs=np.ones((c,c))*.4)
                return archive(lane,prefix+name,**arrays)
            scores=ih['scores'] if info else np.empty((0,len(classes)));labels=ih['held_labels'] if info else np.empty(0,dtype=np.int64)
            initial_ref=save_state('initial',Z,U,scores,labels)
            obj=objective(Z,initial_head,initial_audit) if info else None
            events.append(dict(coords,event='AFFINE_JOINT_INITIAL',state_ref=initial_ref,objective=obj))
            gradients=[];trials=[];steps=[];final_obj=obj;last_head=initial_head;last_audit=initial_audit
            if info:
                g=np.zeros_like(Z);direction=np.zeros_like(Z)
                if behavior!='zero':g[0,0]=1.;direction[0,0]=-1.
                grad_ref=save_state('gradient',Z,U,scores,labels,g_Z=g,d_Z=direction)
                gr=dict(coords,event='AFFINE_JOINT_GRADIENT',iteration=1,state_ref=grad_ref,
                    objective=objective(Z,initial_head,initial_audit,cached=True,backward=True),
                    gradient_norm=float(np.linalg.norm(g)),direction_norm=float(np.linalg.norm(direction)))
                events.append(gr);gradients.append(gr)
                for trial in range(1,2 if behavior=='move' else 13 if behavior=='exhaust' else 1):
                    step=.125*.5**(trial-1);tz=Z+step*direction;tu=anchor+tz@W.T
                    trial_head,trial_audit,th=head(lane,prefix+'trial-head-'+str(trial),classes,later=True)
                    trial_audit['prior_ref']=prior_ref
                    trial_ref=save_state('trial-'+str(trial),tz,tu,th['scores'],labels,delta_Z=tz-Z,d_Z=direction)
                    accepted=behavior=='move';to=objective(tz,trial_head,trial_audit,better=accepted)
                    if not accepted:to['loss_total']=obj['loss_total']+.1
                    tr=dict(coords,event='AFFINE_JOINT_TRIAL',iteration=1,trial=trial,step_size=step,state_ref=trial_ref,
                        gradient_state_ref=grad_ref,objective=to,update_norm=float(np.linalg.norm(tz-Z)),
                        loss_before=obj['loss_total'],loss_after=to['loss_total'],gradient_dot_delta=-step,
                        comparison_tolerance=1e-13,armijo_pass=accepted,objective_nonincrease_pass=accepted,accepted=accepted)
                    events.append(tr);trials.append(tr)
                    if accepted:
                        Z,U=tz,tu;scores=th['scores'];last_head,last_audit=trial_head,trial_audit
                        final_obj=objective(Z,last_head,last_audit,cached=True,better=True)
                        sr=dict(tr,event='AFFINE_JOINT_STEP',step=1,learning_rate=step,step_seconds=.05)
                        events.append(sr);steps.append(sr)
            final_obj_ref=save_state('final-objective',Z,U,scores,labels) if info else None
            full_ref,full_audit,full_arrays=head(lane,prefix+'final-head',classes,later=bool(steps),
                tau=0. if behavior=='tau0' else None if behavior=='noinfo' else 1.,
                gamma=None if behavior=='noinfo' else 2.)
            # Hand-built head arrays exercise the collector, not a numeric core fit.
            with np.load(lane/full_ref['path'],allow_pickle=False) as original:final_arrays={name:np.array(original[name]) for name in original.files}
            final_arrays.update(Z=Z,U=U,anchor_U=anchor,W=W,H=H,singular_values=np.arange(8,0,-1,dtype=float))
            if prior_values is not None:
                final_arrays.update({'prior_B_'+key:value for key,value in dict(prior_values,U=anchor).items()})
            final_ref=archive(lane,prefix+'final',**final_arrays)
            work=dict.fromkeys(collector.STAGE_COUNTERS,0)
            work.update(ajlr_stage_count=1,optimizer_steps=len(steps),optimizer_iterations=int(info),
                backward_evaluation_count=int(info),accepted_trial_count=len(steps),rejected_trial_count=len(trials)-len(steps),
                trial_count=len(trials),trial_attempt_count=len(trials),inner_objective_evaluation_count=int(info)*(1+len(trials)),
                inner_head_fit_count=int(info)*(1+len(trials)),inner_factorization_count=int(info)*(1+len(trials)),
                ce_adjoint_solve_count=int(info),derivative_triangular_solve_count=2*int(info),
                final_head_fit_count=1,final_factorization_count=int(behavior!='noinfo'),future_solver_count=7)
            for key in ('head_triangular_rhs_count','head_triangular_rhs_element_count','head_triangular_dense_work_unit_count',
                        'intercept_fit_count','intercept_addition_count'):
                work[key]=int(info)*(1+len(trials))*initial_audit[key]+full_audit[key]
            for suffix,power in (('rhs_count',1),('rhs_element_count',2),('dense_work_unit_count',3)):
                work['derivative_triangular_'+suffix]=2*len(classes)**power*int(info)
            final=dict(prep,**work)
            final.update(coords,event='AFFINE_JOINT_FINAL',mode=mode,classes=classes,old_classes=['old-a','old-b'],
                training_physical_ids=physical,no_information=not info,
                initialization_state_ref=initial_ref,final_state_ref=final_ref,final_objective_state_ref=final_obj_ref,
                final_objective=final_obj,final_fit=full_audit,
                stop_reason='TRIAL_BUDGET_EXHAUSTED' if behavior=='exhaust' else 'ZERO_GRADIENT' if behavior=='zero' else
                    'MAX_ITERATIONS' if info else prep['no_information_reason'],
                coordinate_parameter_count=736*rank,trainable_parameter_count=736*rank if info else 0,
                trained_parameter_count=736*rank if steps else 0,fit_seconds=.3,persistent_state_bytes=1000,
                deployment_numeric_state_bytes=900,prior_state_bytes=500 if mode=='C_seq' else 0,
                serialized_deployment_bytes=None,incremental_transmission_bytes=None,gpu_memory_bytes=None)
            if final_obj: final.update({name:final_obj[name] for name in collector.FUNCTION_FIELDS})
            events.append(final)
            logs.append(dict(final,event='CANDIDATE_FIT',trial=None,score_seconds=.02,
                score_workload={'residual':{'raw_distance_pair_count':17},'prior':{'raw_distance_pair_count':13}}))
            stages.append(dict(coords,trial=None,k=5 if info else 1,new_count=2 if mode=='C_seq' else si*2,
                model_seed=100+ri,cohort='rx',receiver='r'+str(ri),scenario='practical_high',
                initialization_state_ref=initial_ref,final_state_ref=final_ref,
                gradients=[{}]*len(gradients),trials=[{}]*len(trials),steps=[{}]*len(steps)))
        # Use the actual entry compactor, copied structurally here to avoid core imports.
        def compact(value):
            result=collector.scalars(value)
            for key,item in value.items():
                if key in ('state_ref','head_ref','head_refs','prior_ref') or key.endswith(('_state_ref','_head_ref')):result[key]=item
            for key in ('objective','initial_objective','final_objective','final_fit'):
                if isinstance(value.get(key),dict):result[key]=compact(value[key])
            for key in ('class_ce_means','class_ce_sums','class_ce_counts','held_ce_sums','held_ce_counts','classes','old_classes'):
                if key in value:result[key]=value[key]
            if isinstance(value.get('inner_folds'),list):result['inner_folds']=[compact(v) for v in value['inner_folds']]
            return result
        jsonl(lane/'training_events.jsonl',events);jsonl(lane/'training_events_compact.jsonl',[compact(e) for e in events])
        jsonl(lane/'fit_stages.jsonl',[compact(v) for v in logs]);events_by_row[row_id]=events
        (lane/'fit_trace.jsonl').write_text('DO_NOT_OPEN_OUTER_SCORES',encoding='utf-8')
        sources.append(dict(row_id=row_id,compact_training_events=str(lane/'training_events_compact.jsonl'),
            full_training_events=str(lane/'training_events.jsonl'),fit_trace=str(lane/'fit_trace.jsonl')))
        files=list((lane/'state_arrays').glob('*.npz'));size=sum(p.stat().st_size for p in files)
        archives.append(dict(row_id=row_id,root=str(lane),file_count=len(files),file_bytes=size,
            by_phase={'synthetic':dict(file_count=len(files),file_bytes=size,numeric_array_bytes=0,archive_seconds=0.)}))
    metadata=dict(status=collector.INPUT_STATUS,scope=collector.SCOPE,schema=collector.SCHEMA,method=collector.METHOD,
        run_id='synthetic',release_commit='same',
        coverage={'ajlr_stage_count':7,'future_run_count':101},training_stage_count=7,raw_training_sources=sources,
        state_archives=archives,state_archive_file_count=sum(s['file_count'] for s in archives),
        state_archive_file_bytes=sum(s['file_bytes'] for s in archives),query_rows_used=0,source_rows_used=0,
        algorithm={'schema':'d92_affine_joint_local_ridge_v1'},resources={'outer_prior_raw_distance_pair_count_sum':13,
            'outer_residual_raw_distance_pair_count_sum':17,'incremental_transmission_bytes':None},
        resource_statistics={'by_row':[{'row_id':'row-1','outer_prior_raw_distance_pair_count_sum':13}]},
        statistics={'SECRET_OUTER_SCORE':123})
    (summary/'summary.json').write_text(json.dumps(metadata),encoding='utf-8');jsonl(summary/'training_objectives.jsonl',stages)
    return summary,events_by_row


class CollectorTests(unittest.TestCase):
    def derive(self,root):
        summary,_=fixture(root)
        with patch.object(collector,'EXACT',{'ajlr_stage_count':7}),patch.object(collector,'EXPECTED_STAGES',7):
            snap=collector.snapshot(summary);value=collector.extract(snap)
        return value,snap,summary

    def test_complete_streams_actual_dynamic_work_rank_repetitions_and_costs(self):
        with tempfile.TemporaryDirectory() as directory:value,snap,_=self.derive(Path(directory))
        self.assertEqual(value['totals']['actual_stages'],7)
        self.assertEqual(value['totals']['information_stages'],5);self.assertEqual(value['totals']['updated_stages'],3)
        self.assertEqual(value['totals']['zero_update_information_stages'],2)
        self.assertEqual(value['totals']['preparation_records'],7);self.assertEqual(value['totals']['prior_head_records'],2)
        self.assertEqual(value['totals']['repeated_B_actual_contexts'],1)
        self.assertEqual(value['totals']['unique_B_physical_bindings'],4)
        self.assertEqual(sorted(set(s['coordinates']['latent_rank'] for s in value['stages'])),[0,1,2,3])
        self.assertEqual(value['stage_count_totals']['future_solver_count'],49)
        self.assertEqual(value['preparation_count_totals']['future_preparation_count'],77)
        for stage in value['stages']:
            self.assertNotIn('future_preparation_count',stage['counts'])
            self.assertEqual(stage['all_event_counters']['future_preparation_count'],11)
        noinfo=next(s for s in value['stages'] if s['coordinates']['latent_rank']==0)
        self.assertIsNone(noinfo['initial']['RMSCE']);self.assertEqual(noinfo['initial']['Z_coordinate_count'],0)
        self.assertIsNone(noinfo['initial']['Z_min']);self.assertEqual(noinfo['counts']['final_head_fit_count'],1)
        self.assertNotIn('outer_prior_raw_distance_pair_count_sum',json.dumps(value))
        self.assertAlmostEqual(value['resources']['candidate_fit_seconds_sum'],2.1)
        self.assertTrue(all(s['outer_inference_work'] is None for s in value['stages']))
        self.assertNotIn('SECRET_OUTER_SCORE',json.dumps(value));self.assertNotIn('SECRET_OUTER_SCORE',json.dumps(snap))

    def test_C_registration_and_adapter_are_paired_inner_effects_full_Z0_is_NA(self):
        with tempfile.TemporaryDirectory() as directory:value,_,_=self.derive(Path(directory))
        c=next(s for s in value['stages'] if s['mode']=='C_seq' and s['updated_stage'])
        registration=c['registration_Z0_inner_effect']
        self.assertTrue(registration['available'])
        self.assertEqual(registration['Z0_residual_effect']['new_wrong_to_correct_count'],1)
        self.assertEqual(registration['Z0_residual_effect']['old_winner_change_count'],0)
        self.assertEqual(c['later_adapter_inner_effect']['old_correct_to_wrong_count'],1)
        self.assertIsNone(registration['full_support_Z0_head_effect'])
        self.assertIsNone(c['final_full_support_head']['K_change_from_initial'])
        self.assertEqual(c['final_full_support_head']['initial_comparison_unmeasured_reason'],'NO_INITIAL_HEAD_REFERENCE')
        self.assertAlmostEqual(c['initial']['old_CE_RMS'],np.sqrt(2.5))
        self.assertAlmostEqual(c['initial']['new_CE_RMS'],3.)
        self.assertEqual(c['initial']['class_ce_counts'],[1,2,3])
        self.assertEqual(c['initial']['inner_fold_mechanisms'][0]['block_angle_radians']['mean'],0.)
        self.assertEqual(c['final']['inner_fold_mechanisms'][0]['block_angle_radians']['mean'],.1)

    def test_gradient_risk_decomposition_uses_Z_not_Z_divided_N_empty_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            lane=Path(directory);z=np.zeros((736,2));z[0,1]=.6;g=np.zeros_like(z);g[0,1]=.2
            ref=archive(lane,'g',Z=z,g_Z=g,d_Z=-g/np.linalg.norm(g))
            result=collector.gradient_metrics({'state_ref':ref,'train_physical_count':200},collector.Arrays(lane))
            empty=np.empty((736,0));ref=archive(lane,'empty',Z=empty,g_Z=empty,d_Z=empty)
            zero=collector.gradient_metrics({'state_ref':ref},collector.Arrays(lane))
        self.assertAlmostEqual(result['CE_gradient_norm'],.4);self.assertAlmostEqual(result['proximal_gradient_norm'],.6)
        self.assertTrue(result['CE_vs_proximal_conflict']);self.assertAlmostEqual(result['CE_vs_proximal_cosine'],-1.)
        self.assertEqual(result['normalized_direction_deviation_norm'],0.)
        self.assertEqual(zero['CE_gradient_norm'],0.);self.assertIsNone(zero['CE_vs_proximal_cosine'])
        self.assertIsNone(zero['Z_CE_gradient_min'])

    def test_affine_arrays_companion_thirteen_counts_and_actual_B_inheritance_are_descriptive(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collector.np.linalg,'solve',side_effect=AssertionError('NO_SOLVE')), \
                 patch.object(collector.np.linalg,'svd',side_effect=AssertionError('NO_SVD')), \
                 patch.object(collector.np.linalg,'cholesky',side_effect=AssertionError('NO_FIT')):
                value,_,_=self.derive(Path(directory))
        merged=set(value['stage_count_totals'])|set(value['preparation_count_totals'])
        self.assertTrue(set(collector.RHS_COUNTERS+collector.INTERCEPT_COUNTERS)<=merged)
        self.assertEqual(value['preparation_count_totals']['prior_triangular_rhs_count'],12)
        self.assertEqual(value['preparation_count_totals']['prior_intercept_fit_count'],2)
        self.assertEqual(value['preparation_count_totals']['prior_intercept_addition_count'],38)
        c=next(s for s in value['stages'] if s['mode']=='C_seq' and s['updated_stage'])
        h=c['initial']['inner_fold_mechanisms'][0]
        self.assertEqual(h['intercept_values'],[-.2,0.,.2]);self.assertEqual(h['combined_rhs_shape'],[3,4])
        self.assertEqual(h['combined_rhs_bytes'],96);self.assertAlmostEqual(h['Schur_s'],2.85)
        self.assertEqual(h['analytic_intercept_contrast_count'],2)
        self.assertTrue(c['actual_B_to_C_state']['available'])
        self.assertEqual(c['actual_B_to_C_state']['inherited_original_U_distance'],0.)
        self.assertEqual(c['actual_B_to_C_state']['prior_intercept_values'],[-.2,.2])
        point=next(p for p in value['curves'] if p['mode']=='C_seq' and p['kind']=='gradient')
        companion=point['metrics']['affine_companion_statistics'][0]
        self.assertEqual(companion['adjoint_rhs_shape'],[3,3])
        self.assertEqual(companion['adjoint_g_b_coordinate_count'],3)
        self.assertIsNone(c['registration_Z0_inner_effect']['full_support_Z0_head_effect'])

    def test_tau0_and_missing_scale_are_exact_measured_boundaries_not_epsilon(self):
        with tempfile.TemporaryDirectory() as directory:value,_,_=self.derive(Path(directory))
        degenerate=next(s for s in value['stages'] if s['stop_reason']=='ALL_INNER_GEOMETRY_DEGENERATE')
        forward=degenerate['final_full_support_head']
        self.assertEqual(forward['tau'],0.)
        self.assertEqual(forward['adapted_distance_unmeasured_reason'],'ZERO_BANDWIDTH_USES_ORIGINAL_EQUIVALENCE')
        self.assertIsNone(forward['adapted_only_distance_change']);self.assertIsNone(forward['tangent_over_kappa'])
        missing=next(s for s in value['stages'] if s['coordinates']['latent_rank']==0)['final_full_support_head']
        self.assertIsNone(missing['tau']);self.assertIsNone(missing['gamma'])
        self.assertEqual(missing['nuisance_unmeasured_reason'],'NO_OLD_KERNEL_INFORMATION')

    def test_all_twelve_rejections_and_final_cache_keep_last_accepted_measurement(self):
        with tempfile.TemporaryDirectory() as directory:value,_,_=self.derive(Path(directory))
        stage=next(s for s in value['stages'] if s['stop_reason']=='TRIAL_BUDGET_EXHAUSTED')
        trials=[p for p in value['curves'] if p['row_id']==stage['row_id'] and p['state']==stage['state'] and p['kind']=='trial']
        self.assertEqual([p['trial'] for p in trials],list(range(1,13)))
        self.assertEqual(stage['rejection_reasons'],{'ARMIJO_AND_OBJECTIVE_INCREASE':12})
        self.assertFalse(stage['updated_stage']);self.assertEqual(stage['final']['Z_norm'],0.)
        self.assertEqual(stage['final']['RMSCE'],stage['initial']['RMSCE'])
        self.assertEqual(stage['counts']['inner_objective_evaluation_count'],13)
        self.assertTrue(all(p['metrics']['armijo_slack']<0 for p in trials))

    def test_snapshot_has_no_array_reads_and_incomplete_gate_opens_no_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);summary,_=fixture(root)
            with patch.object(collector,'Arrays') as arrays,patch.object(collector,'EXACT',{'ajlr_stage_count':7}),patch.object(collector,'EXPECTED_STAGES',7):
                captured=collector.snapshot(summary);arrays.assert_not_called()
            self.assertEqual(captured['status'],collector.SNAPSHOT_STATUS)
            self.assertTrue(all('events' not in lane and 'stage_logs' not in lane for lane in captured['lanes']))
            self.assertTrue(all('gradients' not in stage and 'trials' not in stage and 'steps' not in stage for stage in captured['summary_stages']))
            (summary/'summary.json').write_text('{"status":"RUNNING"}',encoding='utf-8')
            with patch.object(collector,'jsonlines') as streams,patch.object(collector,'Arrays') as arrays:
                with self.assertRaisesRegex(ValueError,'complete independently verified'):collector.snapshot(summary)
                streams.assert_not_called();arrays.assert_not_called()

    def test_selected_metadata_never_deserializes_outer_statistics(self):
        text='{"status":"safe","statistics":{"outer":NOT_JSON},"resources":{"outer":NOT_JSON},"outer_stats":NOT_JSON,"coverage":{"n":1}}'
        self.assertEqual(collector.selected_json(text,collector.META_FIELDS),{'status':'safe','coverage':{'n':1}})
        text='{"event":"CANDIDATE_FIT","fit_seconds":1,"score_seconds":NOT_JSON,"score_workload":NOT_JSON,"outer_stats":NOT_JSON}'
        self.assertEqual(collector.selected_json(text,collector.training_log_key),{'event':'CANDIDATE_FIT','fit_seconds':1})

    def test_captured_reference_allowlist_blocks_outer_and_unlisted_npz_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            value,snap,_=self.derive(Path(directory));changed=deepcopy(snap)
            ref=changed['lanes'][0]['training_refs'][0];namespace=json.loads(ref['namespace'])
            namespace['state']='OUTER_SUPPORT_HELD';ref['namespace']=json.dumps(namespace)
            with patch.object(collector.np,'load') as loading,patch.object(collector,'EXACT',{'ajlr_stage_count':7}),patch.object(collector,'EXPECTED_STAGES',7):
                with self.assertRaisesRegex(ValueError,'current-run training'):collector.extract(changed)
                loading.assert_not_called()
            lane=snap['lanes'][0]
            with patch.object(collector.np,'load') as loading:
                with self.assertRaisesRegex(ValueError,'captured training references'):
                    collector.Arrays(lane['lane'],lane['training_refs'])({'path':'state_arrays/unlisted.npz'})
                loading.assert_not_called()

    def test_missing_event_and_npz_path_escape_are_rejected_without_math_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            value,snap,_=self.derive(Path(directory));changed=deepcopy(snap)
            stream=Path(changed['lanes'][0]['compact_training_events'])
            records=list(collector.jsonlines(stream));jsonl(stream,[e for e in records if not
                (e['event']=='AFFINE_JOINT_FINAL' and e['split_id']=='first')])
            with patch.object(collector,'EXACT',{'ajlr_stage_count':7}),patch.object(collector,'EXPECTED_STAGES',7):
                with self.assertRaisesRegex(ValueError,'stream changed'):collector.extract(changed)
            with self.assertRaisesRegex(ValueError,'outside training archive'):
                collector.Arrays(Path(directory))({'path':'state_arrays/../../other.npz'})

    def test_exclusive_local_exports_preserve_original_refs_NA_strata_and_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value,_,_=self.derive(root);out=root/'derived';collector.write_outputs(out,value)
            summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
            self.assertNotIn('curves',summary);self.assertNotIn('stages',summary)
            self.assertEqual(len(list(collector.jsonlines(out/'curves.jsonl'))),value['totals']['curve_records'])
            with (out/'curves.csv').open(encoding='utf-8',newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),value['totals']['curve_records'])
            self.assertTrue((out/'prior_heads.csv').is_file());self.assertTrue((out/'resources_by_row.csv').is_file())
            self.assertTrue((out/'by_scope_k_new_count.csv').is_file());self.assertTrue((out/'archive_by_phase.csv').is_file())
            self.assertIn('N/A',(out/'stages.csv').read_text(encoding='utf-8'))
            self.assertTrue(all(p['state_ref']['path'].startswith('state_arrays/') for p in value['curves']))
            with self.assertRaises(FileExistsError):collector.write_outputs(out,value)

    def test_remote_phases_send_script_and_snapshot_without_remote_writes(self):
        args=SimpleNamespace(ssh_host='n607',ssh_config='explicit-config',remote_python='/remote/ssr/python',
            summary_root='/remote/verified summary',run_root='/remote/run')
        response={'status':collector.SNAPSHOT_STATUS}
        with patch.object(collector.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(response),stderr='')) as ssh:
            collector.remote_operation(args,'--snapshot-stdout')
        self.assertEqual(collector.shlex.split(ssh.call_args.args[0][-1]),
            ['/remote/ssr/python','-','--snapshot-stdout','--summary-root','/remote/verified summary','--run-root','/remote/run'])
        self.assertEqual(ssh.call_args.kwargs['input'],Path(collector.__file__).read_text(encoding='utf-8'))
        request={'status':collector.SNAPSHOT_STATUS,'literal':'`$()中文\\n'}
        with patch.object(collector.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='{}',stderr='')) as ssh:
            collector.remote_operation(args,'--extract-stdout',request)
        self.assertIn('SNAPSHOT_REQUEST = json.loads(',ssh.call_args.kwargs['input'])
        self.assertNotIn('--output',ssh.call_args.args[0][-1])


if __name__=='__main__':unittest.main()
