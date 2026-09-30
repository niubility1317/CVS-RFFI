"""Compact synthetic training products; no actual pilot or held scores are read."""
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
import collect_d92_mc_training_diagnostics as collector


def save_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows), encoding='utf-8')


def archive(lane, name, **arrays):
    directory = lane/'state_arrays'; directory.mkdir(exist_ok=True)
    relative = 'state_arrays/'+name+'.npz'
    with (lane/relative).open('xb') as stream: np.savez_compressed(stream, **arrays)
    return dict(path=relative, key=name, arrays={k:dict(shape=list(v.shape), dtype=str(v.dtype), nbytes=v.nbytes) for k,v in arrays.items()})


def objective(task=1., keep=.1, prox=0., cached=False, backward=False):
    return dict(loss_task=task, loss_keep=keep, loss_data=task+keep, loss_proximal=prox, loss_total=task+keep+prox,
        objective_seconds=.01, inner_folds=[dict(held_physical_count=6, held_training_correct_count=3,
            held_training_margin_mean=.2, head_fit_count=int(not cached), forward_seconds=.04,
            task_adjoint_seconds=.03 if backward else 0., keep_adjoint_seconds=.02 if backward else 0.)])


def fixture(root):
    summary = root/'verified'; summary.mkdir(); stages = []; sources = []; archives = []; lane_events = {}
    for index, (state, behavior) in enumerate((('B_MC','move'), ('C_MC_seq','zero'), ('C_reset_init','reject'), ('B_MC','noinfo'))):
        row_id = 'row-'+str(index); lane = root/'run'/row_id/'probe'; lane.mkdir(parents=True)
        mode = collector.MODES[state]; k = 1 if behavior == 'noinfo' else 3
        coords = dict(row_id=row_id, split_id='parent', scope='support_oneshot_proxy' if k == 1 else 'support_oof',
            fold=None if k == 1 else 0, outer_trial=0 if k == 1 else None, state=state, train_k=k)
        U = np.zeros((736,8)); V = collector.initial_V(); anchor = dict(anchor_U=U.copy(), anchor_V=V.copy())
        initial_ref = archive(lane, 'initial', U=U, V=V, **anchor); events = []
        if behavior != 'noinfo':
            q = np.asarray([0.,0.,.1,.25,.5,.75,1.]); scores = np.asarray([[0.,1.],[0.,0.],[.1,0.],[.25,0.],[.5,0.],[.75,0.],[1.2,0.]])
            teacher_ref = archive(lane, 'teacher', teacher_q=q, teacher_scores=scores, held_old_mask=np.ones(7))
            events.append(dict(coords, state='B_prepare' if mode == 'B' else 'C_prepare', event='MC_INNER_PREPARED',
                inner_fold=0, teacher=dict(available=True, source='INITIAL_R0_CACHE' if mode == 'B' else 'FROZEN_B_ADAPTER_OLD_INNER_HEAD', state_ref=teacher_ref)))
            events.append(dict(coords, event='MC_INITIAL', state_ref=initial_ref, objective=objective(), keep_limit=.11))
            gU=np.zeros_like(U); gV=np.zeros_like(V); aU=np.zeros_like(U); aV=np.zeros_like(V); dU=np.zeros_like(U); dV=np.zeros_like(V)
            if behavior != 'zero': gU[0,0]=1.; aU[0,0]=-1.; dU[0,0]=-.5
            grad_ref = archive(lane, 'gradient', U=U,V=V,g_U=gU,g_V=gV,keep_g_U=aU,keep_g_V=aV,d_U=dU,d_V=dV)
            events.append(dict(coords, event='MC_GRADIENT', iteration=1, state_ref=grad_ref, objective=objective(cached=True,backward=True),
                train_physical_count=6, gradient_norm=float(np.linalg.norm(gU)), keep_gradient_norm=float(np.linalg.norm(aU)),
                direction_norm=float(np.linalg.norm(dU)), guard_active=behavior != 'zero', guard_dot_before=1., guard_dot_after=.5, guard_bound=.5))
        trials=[]; steps=[]
        if behavior in ('move','reject'):
            for trial in range(1,2 if behavior == 'move' else 4):
                size=.125*.5**(trial-1); candidate=U+size*dU; ref=archive(lane,'trial-'+str(trial),U=candidate,V=V)
                accepted=behavior == 'move'; obj=objective(task=.8 if accepted else .4,keep=.105 if accepted else .5)
                event=dict(coords,event='MC_TRIAL',iteration=1,trial=trial,step_size=size,state_ref=ref,gradient_state_ref=grad_ref,
                    objective=obj,loss_before=1.1,loss_after=obj['loss_total'],armijo_rhs=1.09,armijo_tolerance=1e-13,
                    objective_acceptance_bound=1.09,armijo_pass=True,objective_nonincrease_pass=True,keep_pass=accepted,accepted=accepted,
                    rejection_reason=None if accepted else 'KEEP',keep_risk_trial=obj['loss_keep'],keep_limit=.11,keep_tolerance=1e-13,
                    keep_violation=max(0.,obj['loss_keep']-.11),update_norm=float(np.linalg.norm(candidate-U)))
                events.append(event);trials.append(event)
                if accepted:
                    U=candidate; step=dict(event,event='MC_STEP',step=1); events.append(step);steps.append(step)
        final_ref=archive(lane,'final',U=U,V=V,**anchor); noinfo=behavior == 'noinfo'
        count={name:0 for name in collector.COUNTERS}
        count.update(optimizer_steps=len(steps),optimizer_iterations=int(not noinfo),backward_evaluation_count=int(not noinfo),
            accepted_trial_count=len(steps),rejected_trial_count=len(trials)-len(steps),trial_count=len(trials),trial_attempt_count=len(trials),
            inner_objective_evaluation_count=0 if noinfo else 1+len(trials),inner_head_fit_count=0 if noinfo else 1+len(trials),
            inner_factorization_count=0 if noinfo else 1+len(trials),task_derivative_triangular_solve_count=0 if noinfo else 2,
            keep_derivative_triangular_solve_count=0 if noinfo else 2,derivative_triangular_solve_count=0 if noinfo else 4)
        final_obj=None if noinfo else objective(.8,.105,cached=True) if behavior=='move' else objective(cached=True)
        reason='PHYSICAL_K1' if noinfo else 'ZERO_GRADIENT' if behavior=='zero' else 'MAX_ITERATIONS' if behavior=='move' else 'ARMIJO_OR_KEEP_BUDGET_EXHAUSTED'
        final=dict(coords,event='MC_FIT',mode=mode,no_information=noinfo,stop_reason=reason,final_objective=final_obj,
            final_state_ref=final_ref,initialization_state_ref=initial_ref,keep_available=True,keep_anchor_risk=None if noinfo else .1,
            keep_slack=None if noinfo else .01,keep_limit=None if noinfo else .11,classes=['old-a','old-b'],training_physical_ids=['a','b'],
            fit_seconds=.2,head_state_bytes=1000,adapter_state_bytes=94208,lineage_state_bytes=2000,persistent_state_bytes=97208,
            optimizer_state_bytes=0,trainable_parameter_count=11776,**count)
        # Repeated giant source fields are deliberately discarded by thin_event.
        final['preparation']={'do_not_copy':['unused']*1000}; events.append(final)
        stage=dict(coords,trial=coords['outer_trial'],k=5,new_count=2,no_information=noinfo,
            initialization_state_ref=initial_ref,final_state_ref=final_ref,gradients=[] if noinfo else [{}],trials=[{}]*len(trials),steps=[{}]*len(steps))
        stages.append(stage);save_jsonl(lane/'training_events.jsonl',events);lane_events[row_id]=events
        # Collector must not open this unparseable result trace.
        (lane/'fit_trace.jsonl').write_text('DO_NOT_OPEN_OUTER_SCORES',encoding='utf-8')
        sources.append(dict(row_id=row_id,full_training_events=str(lane/'training_events.jsonl'),fit_trace=str(lane/'fit_trace.jsonl')))
        files=list((lane/'state_arrays').glob('*.npz')); total=sum(p.stat().st_size for p in files)
        archives.append(dict(row_id=row_id,root=str(lane),file_count=len(files),file_bytes=total,by_phase={state:dict(file_count=len(files),file_bytes=total,numeric_array_bytes=0,archive_seconds=0.)}))
    metadata=dict(status=collector.INPUT_STATUS,run_id='synthetic',release_commit='same',coverage={'mc_stage_count':4},
        training_stage_count=4,raw_training_sources=sources,state_archives=archives,
        state_archive_file_count=sum(a['file_count'] for a in archives),state_archive_file_bytes=sum(a['file_bytes'] for a in archives),
        query_rows_used=0,source_rows_used=0,resources={'run_wall_seconds':1.},algorithm={'task_weight':1,'keep_weight':1},
        statistics={'secret_outer_metric':'DO_NOT_DESERIALIZE_OR_REPORT'})
    (summary/'summary.json').write_text(json.dumps(metadata),encoding='utf-8');save_jsonl(summary/'training_objectives.jsonl',stages)
    return summary, lane_events


class CollectorTests(unittest.TestCase):
    def collect_fixture(self, root):
        summary, events=fixture(root)
        with patch.object(collector,'EXACT',{'mc_stage_count':4}),patch.object(collector,'EXPECTED_STAGES',4): value=collector.collect(summary)
        return value,summary,events

    def test_complete_curves_zero_update_rejections_and_teacher_error_bounds(self):
        with tempfile.TemporaryDirectory() as directory: value,_,_=self.collect_fixture(Path(directory))
        self.assertEqual(value['totals']['actual_stages'],4);self.assertEqual(value['totals']['information_stages'],3)
        self.assertEqual(value['totals']['updated_stages'],1);self.assertEqual(value['totals']['zero_update_information_stages'],2)
        zero=next(s for s in value['stages'] if s['mode']=='C_seq')
        self.assertFalse(zero['updated_stage']);self.assertEqual(zero['counts']['backward_evaluation_count'],1)
        self.assertIsNone(next(s for s in value['stages'] if not s['information_stage'])['initial']['loss_total'])
        trials=[p for p in value['curves'] if p['mode']=='C_reset_init' and p['kind']=='trial']
        self.assertEqual([p['trial'] for p in trials],[1,2,3]);self.assertTrue(all(not p['accepted'] for p in trials))
        teacher=value['teachers'][0]
        self.assertEqual(teacher['wrong_or_tied_zero_q_count'],2);self.assertEqual(teacher['guaranteed_wrong_unique_winner_count'],1)
        self.assertEqual(teacher['ambiguous_zero_q_tied_winner_count'],1);self.assertIsNone(teacher['teacher_exact_wrong_count'])
        self.assertEqual(teacher['positive_margin_correct_count'],5)
        self.assertNotIn('DO_NOT_',json.dumps(value));self.assertNotIn('do_not_copy',json.dumps(value))

    def test_total_and_task_gradient_conflict_are_distinct_and_proximal_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            lane=Path(directory);(lane/'state_arrays').mkdir();U=np.zeros((736,8));U[0,0]=.6;V=collector.initial_V()
            gu=np.zeros_like(U);gu[0,0]=1.;au=np.zeros_like(U);au[0,0]=.8;zv=np.zeros_like(V)
            ref=archive(lane,'g',U=U,V=V,g_U=gu,g_V=zv,keep_g_U=au,keep_g_V=zv)
            result=collector.gradient_metrics(dict(state_ref=ref,train_physical_count=2),collector.Arrays(lane),
                dict(anchor_U=np.zeros_like(U),anchor_V=V))
        self.assertFalse(result['total_vs_keep_conflict']);self.assertTrue(result['task_vs_keep_conflict'])
        self.assertAlmostEqual(result['task_gradient_norm'],.1)
        self.assertAlmostEqual(result['task_vs_keep_cosine'],-1.)

    def test_metadata_skips_outer_statistics_without_deserializing_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json';path.write_text('{"status":"safe","statistics":{"score":NOT_JSON},"coverage":{"n":1}}',encoding='utf-8')
            value=collector.read_metadata(path)
        self.assertEqual(value,{'status':'safe','coverage':{'n':1}})

    def test_incomplete_summary_never_opens_events_or_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'summary.json').write_text('{"status":"RUNNING"}',encoding='utf-8')
            with patch.object(collector,'jsonlines') as events,patch.object(collector,'Arrays') as arrays:
                with self.assertRaisesRegex(ValueError,'complete independently verified'):collector.collect(root)
                events.assert_not_called();arrays.assert_not_called()

    def test_missing_tail_event_is_detected_by_complete_curve_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);summary,events=fixture(root);lane=root/'run'/'row-2'/'probe'
            events['row-2'].pop();save_jsonl(lane/'training_events.jsonl',events['row-2'])
            with patch.object(collector,'EXACT',{'mc_stage_count':4}),patch.object(collector,'EXPECTED_STAGES',4):
                with self.assertRaisesRegex(ValueError,'final training event'):collector.collect(summary)

    def test_compact_outputs_preserve_all_curve_rows_refs_and_exclusive_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value,_,_=self.collect_fixture(root);out=root/'derived';collector.write_outputs(out,value)
            delivered=json.loads((out/'summary.json').read_text(encoding='utf-8'))
            self.assertNotIn('stages',delivered);self.assertNotIn('curves',delivered)
            self.assertEqual(len(list(collector.jsonlines(out/'curves.jsonl'))),value['totals']['curve_records'])
            with (out/'curves.csv').open(encoding='utf-8',newline='') as source: self.assertEqual(len(list(csv.DictReader(source))),value['totals']['curve_records'])
            self.assertTrue((out/'archive_by_phase.csv').is_file());self.assertIn('N/A',(out/'stages.csv').read_text(encoding='utf-8'))
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
