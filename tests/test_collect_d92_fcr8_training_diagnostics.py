"""Synthetic complete FCR8 training products; no real run or outer score reads."""
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
import collect_d92_fcr8_training_diagnostics as collector


def save_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows), encoding='utf-8')


def archive(lane, name, **arrays):
    (lane/'state_arrays').mkdir(exist_ok=True)
    relative = 'state_arrays/'+name+'.npz'
    with (lane/relative).open('xb') as stream: np.savez_compressed(stream, **arrays)
    return dict(path=relative, key=name, arrays={name: dict(shape=list(v.shape), dtype=str(v.dtype), nbytes=v.nbytes)
        for name, v in arrays.items()})


def objective(Z, task=1., keep=.1, cached=False, backward=False, zero_bandwidth=False):
    prox = .5*float(np.sum(Z*Z))
    stats = dict(count=4, minimum=0., mean=.1, maximum=.2)
    return dict(loss_task=task, loss_keep=keep, loss_data=task+keep, loss_proximal=prox, loss_total=task+keep+prox,
        objective_seconds=.01, pre_tangent_displacement_mean_squared=2*prox,
        pre_tangent_displacement_rms=float(np.linalg.norm(Z)), coordinate_squared_norm=2*prox,
        function_coordinate_reconstruction_error=0., forward_cache_reused=cached,
        inner_folds=[dict(inner_fold=0, held_physical_count=6, held_training_correct_count=3,
            held_training_margin_mean=.2, head_fit_count=int(not cached), forward_seconds=.04,
            task_adjoint_seconds=.03 if backward else 0., keep_adjoint_seconds=.02 if backward else 0.,
            original_bandwidth_tau=0. if zero_bandwidth else None, bandwidth_tau=0. if zero_bandwidth else 1.,
            block_angle_radians=stats, tangent_over_kappa=stats, joint_distance_relative_change=stats,
            adapted_distance_relative_change=None if zero_bandwidth else stats,
            adapted_distance_unmeasured_reason='ZERO_BANDWIDTH_BYPASSES_ADAPTED_DISTANCE' if zero_bandwidth else None,
            original_zero_distance_pair_count=2, kernel_frobenius_norm=2., kernel_change_from_initial=.1,
            kernel_relative_change_from_initial=.05, kernel_change_unmeasured_reason=None,
            held_score_change_rms=.02, held_winner_change_count=1, identity_forward=zero_bandwidth)])


def fixture(root):
    summary = root/'verified'; summary.mkdir(); stages=[]; sources=[]; archives=[]; all_events={}
    # Repeated physical B contexts retain real work. C sequential/reset share one
    # preparation. Empty rank-zero coordinates and ranks one/three are all used.
    rows = [[('B_FCR8','move',1,'parent'), ('B_FCR8','move',1,'parent-repeat')],
        [('C_FCR8_seq','zero',3,'parent'), ('C_reset_init','reject',3,'parent')],
        [('B_FCR8','noinfo',0,'parent')], [('B_FCR8','move',3,'parent')]]
    for row_index, descriptions in enumerate(rows):
        row_id='row-'+str(row_index); lane=root/'run'/row_id/'probe'; lane.mkdir(parents=True)
        events=[]; shared_prep=None
        for index, (state, behavior, rank, split) in enumerate(descriptions):
            mode=collector.MODES[state]; k=1 if behavior=='noinfo' else 3; noinfo=behavior=='noinfo'
            coords=dict(row_id=row_id, split_id=split, scope='support_oneshot_proxy' if noinfo else 'support_oof',
                fold=None if noinfo else 0, outer_trial=0 if noinfo else None, state=state, train_k=k)
            prefix=str(index)+'-'; Z=np.zeros((736,rank)); W=np.eye(8,rank); H=np.zeros((6,8)); H[:,:rank]=1.
            U=np.zeros((736,8)); anchor_U=U.copy()
            if mode=='C_seq': anchor_U[0,0]=.25; U=anchor_U.copy()
            if shared_prep is None or mode=='B':
                coordinate_ref=archive(lane,prefix+'coordinates',H=H,W=W,singular_values=np.ones(8))
                prep=dict(coordinate_state_ref=coordinate_ref, latent_rank=rank, rank_estimated=not noinfo,
                    rank_energy_threshold=1e-12, whitening_residual=None if noinfo else 1e-15,
                    whitening_tolerance=None if noinfo else 1e-12, dictionary_rms=1., singular_values=None if noinfo else [1.]*8,
                    no_information_reason='PHYSICAL_K1' if noinfo else None,
                    optimizer_coordinate_scope='ALL_CURRENT_OUTER_TRAIN_UNLABELLED',
                    fcr_preparation_count=1, latent_svd_count=int(not noinfo), dictionary_physical_evaluation_count=6,
                    inner_head_fit_count=int(not noinfo), prepared_distance_evaluation_count=2,
                    fcr_forward_evaluation_count=int(not noinfo), future_preparation_count=11)
                if not noinfo:
                    q=np.asarray([0.,0.,.1,.25,.5,.75,1.])
                    scores=np.asarray([[0.,1.],[0.,0.],[.1,0.],[.25,0.],[.5,0.],[.75,0.],[1.2,0.]])
                    teacher_ref=archive(lane,prefix+'teacher',teacher_q=q,teacher_scores=scores,held_old_mask=np.ones(7))
                    events.append(dict(coords,event='FCR_INNER_PREPARED',state='B_prepare' if mode=='B' else 'C_prepare',inner_fold=0,
                        teacher=dict(available=True,source='INITIAL_R0_CACHE' if mode=='B' else 'FROZEN_B_ADAPTER_OLD_INNER_HEAD',
                            state_ref=teacher_ref,head_audit=dict(original_bandwidth_tau=None,bandwidth_tau=1.))))
                shared_prep=prep
            else: prep=shared_prep; coordinate_ref=prep['coordinate_state_ref']
            initial_ref=archive(lane,prefix+'initial',Z=Z,U=U,anchor_U=anchor_U,W=W,singular_values=np.ones(8))
            if not noinfo:
                events.append(dict(coords,event='FCR_INITIAL',state_ref=initial_ref,objective=objective(Z),keep_limit=.11))
                g=np.zeros_like(Z);keep=np.zeros_like(Z);direction=np.zeros_like(Z)
                if behavior!='zero':g[0,0]=1.;keep[0,0]=-1.;direction[0,0]=-.5
                grad_ref=archive(lane,prefix+'gradient',Z=Z,U=U,anchor_U=anchor_U,W=W,singular_values=np.ones(8),
                    g_Z=g,keep_g_Z=keep,d_Z=direction)
                events.append(dict(coords,event='FCR_GRADIENT',iteration=1,state_ref=grad_ref,
                    objective=objective(Z,cached=True,backward=True),train_physical_count=6,
                    gradient_norm=float(np.linalg.norm(g)),keep_gradient_norm=float(np.linalg.norm(keep)),
                    direction_norm=float(np.linalg.norm(direction)),guard_active=behavior!='zero',
                    guard_dot_before=1.,guard_dot_after=.5,guard_bound=.5))
            trials=[];steps=[]
            if behavior in ('move','reject'):
                for trial in range(1,2 if behavior=='move' else 4):
                    size=.125*.5**(trial-1); tz=Z+size*direction; tu=anchor_U+tz@W.T
                    ref=archive(lane,prefix+'trial-'+str(trial),Z=tz,U=tu,anchor_U=anchor_U,W=W,singular_values=np.ones(8))
                    accepted=behavior=='move';obj=objective(tz,task=.8 if accepted else .4,keep=.105 if accepted else .5,
                        zero_bandwidth=behavior=='reject')
                    event=dict(coords,event='FCR_TRIAL',iteration=1,trial=trial,step_size=size,state_ref=ref,gradient_state_ref=grad_ref,
                        objective=obj,loss_before=1.1,loss_after=obj['loss_total'],armijo_rhs=1.09,armijo_tolerance=1e-13,
                        objective_acceptance_bound=1.09,armijo_pass=True,objective_nonincrease_pass=True,
                        keep_pass=accepted,accepted=accepted,rejection_reason=None if accepted else 'KEEP',
                        keep_risk_trial=obj['loss_keep'],keep_limit=.11,keep_tolerance=1e-13,
                        keep_violation=max(0.,obj['loss_keep']-.11),update_norm=float(np.linalg.norm(tz-Z)))
                    events.append(event);trials.append(event)
                    if accepted:
                        Z,U=tz,tu; step=dict(event,event='FCR_STEP',step=1,learning_rate=size,step_seconds=.05)
                        events.append(step);steps.append(step)
            final_ref=archive(lane,prefix+'final',Z=Z,U=U,anchor_U=anchor_U,W=W,singular_values=np.ones(8))
            count=dict.fromkeys(collector.COUNTERS,0)
            count.update(optimizer_steps=len(steps),optimizer_iterations=int(not noinfo),backward_evaluation_count=int(not noinfo),
                accepted_trial_count=len(steps),rejected_trial_count=len(trials)-len(steps),trial_count=len(trials),trial_attempt_count=len(trials),
                inner_objective_evaluation_count=0 if noinfo else 1+len(trials),inner_head_fit_count=0 if noinfo else 1+len(trials),
                inner_factorization_count=0 if noinfo else 1+len(trials),task_derivative_triangular_solve_count=0 if noinfo else 2,
                keep_derivative_triangular_solve_count=0 if noinfo else 2,derivative_triangular_solve_count=0 if noinfo else 4,
                nonzero_projected_update_count=len(steps),fcr_stage_count=1,future_solver_count=7)
            final_obj=None if noinfo else objective(Z,.8,.105,cached=True) if behavior=='move' else objective(Z,cached=True)
            final=dict(coords,event='FCR_FINAL',mode=mode,no_information=noinfo,
                stop_reason='PHYSICAL_K1' if noinfo else 'ZERO_GRADIENT' if behavior=='zero' else 'MAX_ITERATIONS' if behavior=='move' else 'ARMIJO_OR_KEEP_BUDGET_EXHAUSTED',
                final_objective=final_obj,preparation=prep,final_fit=dict(original_bandwidth_tau=None,bandwidth_tau=None),
                final_state_ref=final_ref,initialization_state_ref=initial_ref,keep_available=True,
                keep_anchor_risk=None if noinfo else .1,keep_slack=None if noinfo else .01,keep_limit=None if noinfo else .11,
                classes=['old-a','old-b'],training_physical_ids=['a','b'],fit_seconds=.2,
                head_state_bytes=1000,adapter_state_bytes=94208,lineage_state_bytes=2000,persistent_state_bytes=97208,
                optimizer_state_bytes=0,trainable_parameter_count=736*rank,maximum_trainable_parameter_count=5888,
                latent_svd_count=prep['latent_svd_count'],future_preparation_count=prep['future_preparation_count'],**count)
            if final_obj:final.update({name:final_obj[name] for name in collector.FUNCTION_FIELDS})
            events.append(final)
            stages.append(dict(coords,trial=coords['outer_trial'],k=5,new_count=2+index,coordinate_state_ref=coordinate_ref,
                initialization_state_ref=initial_ref,final_state_ref=final_ref,
                gradients=[] if noinfo else [{}],trials=[{}]*len(trials),steps=[{}]*len(steps)))
        save_jsonl(lane/'training_events.jsonl',events);all_events[row_id]=events
        (lane/'fit_trace.jsonl').write_text('DO_NOT_OPEN_OUTER_SCORES',encoding='utf-8')
        sources.append(dict(row_id=row_id,full_training_events=str(lane/'training_events.jsonl'),fit_trace=str(lane/'fit_trace.jsonl')))
        files=list((lane/'state_arrays').glob('*.npz'));total=sum(p.stat().st_size for p in files)
        archives.append(dict(row_id=row_id,root=str(lane),file_count=len(files),file_bytes=total,
            by_phase={'synthetic':dict(file_count=len(files),file_bytes=total,numeric_array_bytes=0,archive_seconds=0.)}))
    metadata=dict(status=collector.INPUT_STATUS,run_id='synthetic',release_commit='same',coverage={'fcr_stage_count':6,'future_run_count':123},
        training_stage_count=6,raw_training_sources=sources,state_archives=archives,
        state_archive_file_count=sum(a['file_count'] for a in archives),state_archive_file_bytes=sum(a['file_bytes'] for a in archives),
        query_rows_used=0,source_rows_used=0,resources={'run_wall_seconds':1.},algorithm={'task_weight':1,'keep_weight':1},
        statistics={'secret_outer_metric':'DO_NOT_DESERIALIZE_OR_REPORT'})
    (summary/'summary.json').write_text(json.dumps(metadata),encoding='utf-8');save_jsonl(summary/'training_objectives.jsonl',stages)
    return summary,all_events


class CollectorTests(unittest.TestCase):
    def collect_fixture(self, root):
        summary,events=fixture(root)
        with patch.object(collector,'EXACT',{'fcr_stage_count':6}),patch.object(collector,'EXPECTED_STAGES',6):value=collector.collect(summary)
        return value,summary,events

    def test_complete_dynamic_rank_curves_teacher_and_distinct_accounting(self):
        with tempfile.TemporaryDirectory() as directory:value,_,_=self.collect_fixture(Path(directory))
        self.assertEqual(value['totals']['actual_stages'],6);self.assertEqual(value['totals']['information_stages'],5)
        self.assertEqual(value['totals']['updated_stages'],3);self.assertEqual(value['totals']['zero_update_information_stages'],2)
        self.assertEqual(value['totals']['preparation_records'],5)
        self.assertEqual(value['totals']['unique_B_physical_bindings'],3);self.assertEqual(value['totals']['repeated_B_actual_contexts'],1)
        self.assertEqual(sorted(set(s['coordinates']['latent_rank'] for s in value['stages'])),[0,1,3])
        shared=next(p for p in value['preparations'] if p['state']=='C_prepare')
        self.assertEqual(shared['actual_consuming_stages'],2)
        self.assertEqual(value['preparation_count_totals']['future_preparation_count'],55)
        self.assertEqual(value['coverage']['future_run_count'],123)
        for stage in value['stages']:
            self.assertEqual(stage['counts']['future_solver_count'],7)
            self.assertNotIn('future_preparation_count',stage['counts'])
            self.assertEqual(stage['all_event_counters']['future_preparation_count'],11)
        zero=next(s for s in value['stages'] if s['mode']=='C_seq')
        self.assertFalse(zero['updated_stage']);self.assertEqual(zero['counts']['backward_evaluation_count'],1)
        noinfo=next(s for s in value['stages'] if not s['information_stage'])
        self.assertIsNone(noinfo['initial']['loss_total']);self.assertEqual(noinfo['initial']['Z_coordinate_count'],0)
        self.assertIsNone(noinfo['initial']['Z_min']);self.assertIsNone(noinfo['coordinates']['singular_values'])
        self.assertIsNone(noinfo['final_forward_mechanisms']['original_bandwidth_tau'])
        teacher=value['teachers'][0]
        self.assertEqual(teacher['wrong_or_tied_zero_q_count'],2);self.assertEqual(teacher['guaranteed_wrong_unique_winner_count'],1)
        self.assertEqual(teacher['ambiguous_zero_q_tied_winner_count'],1);self.assertIsNone(teacher['teacher_exact_wrong_count'])
        self.assertNotIn('DO_NOT_',json.dumps(value));self.assertNotIn('V_gradient',json.dumps(value))
        self.assertNotIn('ball_projection',json.dumps(value))

    def test_task_gradient_removes_Z_without_N_and_zero_rank_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            lane=Path(directory);Z=np.zeros((736,2));Z[0,1]=.6
            g=np.zeros_like(Z);g[0,1]=1.;keep=np.zeros_like(Z);keep[0,1]=.8
            ref=archive(lane,'g',Z=Z,g_Z=g,keep_g_Z=keep,d_Z=-g)
            result=collector.gradient_metrics(dict(state_ref=ref,train_physical_count=200),collector.Arrays(lane))
            empty=np.empty((736,0));zero_ref=archive(lane,'empty',Z=empty,g_Z=empty,keep_g_Z=empty,d_Z=empty)
            zero=collector.gradient_metrics(dict(state_ref=zero_ref),collector.Arrays(lane))
        self.assertFalse(result['total_vs_keep_conflict']);self.assertTrue(result['task_vs_keep_conflict'])
        self.assertAlmostEqual(result['task_gradient_norm'],.4);self.assertAlmostEqual(result['task_vs_keep_cosine'],-1.)
        self.assertEqual(result['task_gradient_reconstruction'],'total_minus_keep_minus_Z')
        self.assertEqual(zero['task_gradient_norm'],0.);self.assertIsNone(zero['Z_task_gradient_min'])
        self.assertIsNone(zero['total_vs_keep_cosine'])

    def test_zero_bandwidth_mechanisms_remain_NA_while_angles_kernel_and_winners_survive(self):
        with tempfile.TemporaryDirectory() as directory:value,_,_=self.collect_fixture(Path(directory))
        trials=[p for p in value['curves'] if p['mode']=='C_reset_init' and p['kind']=='trial']
        self.assertEqual([p['trial'] for p in trials],[1,2,3])
        for point in trials:
            measured=point['metrics'];fold=measured['inner_fold_mechanisms'][0]
            self.assertEqual(fold['original_bandwidth_tau'],0.)
            self.assertIsNone(fold['adapted_distance_relative_change'])
            self.assertEqual(fold['adapted_distance_unmeasured_reason'],'ZERO_BANDWIDTH_BYPASSES_ADAPTED_DISTANCE')
            self.assertIsNone(measured['adapted_distance_relative_change_mean'])
            self.assertEqual(measured['adapted_distance_relative_change_unmeasured_fold_count'],1)
            self.assertEqual(measured['block_angle_radians_count'],4)
            self.assertEqual(measured['held_winner_change_count'],1)
            self.assertEqual(fold['kernel_relative_change_from_initial'],.05)

    def test_metadata_skips_outer_statistics_without_deserializing_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json'
            path.write_text('{"status":"safe","statistics":{"score":NOT_JSON},"coverage":{"n":1}}',encoding='utf-8')
            value=collector.read_metadata(path)
        self.assertEqual(value,{'status':'safe','coverage':{'n':1}})

    def test_incomplete_summary_never_opens_training_events_or_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'summary.json').write_text('{"status":"RUNNING"}',encoding='utf-8')
            with patch.object(collector,'jsonlines') as events,patch.object(collector,'Arrays') as arrays:
                with self.assertRaisesRegex(ValueError,'complete independently verified'):collector.collect(root)
                events.assert_not_called();arrays.assert_not_called()

    def test_missing_final_event_is_detected_without_rerunning_math(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);summary,events=fixture(root);events['row-3'].pop()
            save_jsonl(root/'run'/'row-3'/'probe'/'training_events.jsonl',events['row-3'])
            with patch.object(collector,'EXACT',{'fcr_stage_count':6}),patch.object(collector,'EXPECTED_STAGES',6):
                with self.assertRaisesRegex(ValueError,'final training event'):collector.collect(summary)

    def test_compact_outputs_preserve_curve_rows_original_refs_and_exclusive_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value,_,_=self.collect_fixture(root);out=root/'derived';collector.write_outputs(out,value)
            delivered=json.loads((out/'summary.json').read_text(encoding='utf-8'))
            self.assertNotIn('stages',delivered);self.assertNotIn('preparations',delivered)
            points=list(collector.jsonlines(out/'curves.jsonl'));self.assertEqual(len(points),value['totals']['curve_records'])
            self.assertTrue(all(p['state_ref']['path'].startswith('state_arrays/') for p in points))
            with (out/'curves.csv').open(encoding='utf-8',newline='') as stream:self.assertEqual(len(list(csv.DictReader(stream))),len(points))
            self.assertTrue((out/'preparations.csv').is_file());self.assertTrue((out/'archive_by_phase.csv').is_file())
            self.assertIn('N/A',(out/'stages.csv').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):collector.write_outputs(out,value)

    def test_readonly_ssh_stdin_interface_uses_explicit_numpy_python(self):
        result=dict(status='SYNTHETIC',totals={},state_archive_file_count=0)
        args=['collector','--summary-root','/remote/verified summary','--run-root','/remote/run',
            '--output','local-new','--ssh-host','n607','--ssh-config','config-path','--remote-python','/remote/ssr/python']
        with patch.object(sys,'argv',args),patch.object(collector.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')) as ssh, \
             patch.object(collector,'write_outputs') as writer,patch('builtins.print'):
            collector.main()
        self.assertEqual(collector.shlex.split(ssh.call_args.args[0][-1]),['/remote/ssr/python','-','--summary-root','/remote/verified summary','--collect-stdout','--run-root','/remote/run'])
        self.assertEqual(ssh.call_args.kwargs['input'],Path(collector.__file__).read_text(encoding='utf-8'))
        writer.assert_called_once_with('local-new',result)


if __name__=='__main__':unittest.main()
