import unittest
from copy import deepcopy
from unittest.mock import patch
from experiments.pamf_channel_pl import receiver_balance_evaluation as e


def completed_source(scores=None):
    s=e.source;scores=scores or {m:dict(raw_correct=26568,H3_correct=62109) for m in s.METHODS}
    decisions=[s.source_decision(m,scores[m],dict(raw_correct=26552,H3_correct=62148)) for m in s.METHODS]
    valid=[r for r in decisions if r['source_pass']]
    chosen=max(valid,key=lambda r:(r['raw_correct'],r['H3_correct'],-s.METHODS.index(r['method'])))['method'] if valid else None
    return dict(status='RECEIVER_BALANCE_SOURCE_COMPLETE',run_id=s.RUN,model_seed=2026092701,U_truth_reads=0,target_reads=0,
        source_scores=scores,reference=dict(raw_correct=26552,H3_correct=62148),decisions=decisions,source_selected=chosen)


def prediction_config(method=None,seed=2026092701):
    method=method or e.source.METHODS[0];rid='F0-'+method+'-s'+str(seed);project='/project'
    source_run=e.source.RUN if seed==e.SEEDS[0] else e.CONFIRM_RUN
    return dict(schema='pamf_clean_h_raw_eval_v1',run_id=e.CONFIRM_RUN,row_id=rid,protocol_id='F0',purpose='selected_method',method=method,model_seed=seed,
        source_days=[1,2,3],classes=list('abcdef'),tests={t:dict(views_root='views_'+t) for t in e.TASK_COUNTS},readouts=['raw'],output_root='predictions',
        project=project,selection_source=project+'/runs/'+e.source.RUN+'/complete.json',source_run_id=source_run,source_row_id=rid,
        source_output=project+'/runs/'+source_run+'/F/'+rid+'/source',source_contract='roles',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')})


def scoring_config():
    p=prediction_config();c={k:v for k,v in p.items() if k not in e.shared.PREDICT_ONLY}
    c.update(prediction_output=p['output_root'],output_root='scores',tests={t:dict(views_root='views_'+t,truth='truth_'+t,kind=t) for t in e.TASK_COUNTS})
    return c


class ReceiverBalanceEvaluationTests(unittest.TestCase):
    def test_source_gate_recomputed_raw_then_H_then_RX_tie(self):
        c=dict(selection_source='/project/runs/'+e.source.RUN+'/complete.json',purpose='source_selection')
        for raw,h,winner in [(26568,62109,e.source.METHODS[0]),(26569,62109,e.source.METHODS[1]),(26568,62110,e.source.METHODS[1])]:
            scores={m:dict(raw_correct=26568,H3_correct=62109) for m in e.source.METHODS};scores[e.source.METHODS[1]]=dict(raw_correct=raw,H3_correct=h)
            with patch.object(e,'read',return_value=completed_source(scores)):self.assertEqual(e.selected_method(c),winner)

    def test_incomplete_failed_wrong_decision_or_winner_cannot_select(self):
        c=dict(selection_source='/project/runs/'+e.source.RUN+'/complete.json',purpose='source_selection');good=completed_source()
        failed=completed_source({m:dict(raw_correct=26567,H3_correct=62109) for m in e.source.METHODS})
        examples=[dict(good,status='RUNNING'),dict(good,target_reads=1),dict(good,U_truth_reads=1),dict(good,source_scores={}),dict(good,decisions=[]),dict(good,source_selected=e.source.METHODS[1]),failed]
        for value in examples:
            with patch.object(e,'read',return_value=value),self.assertRaises(ValueError):e.selected_method(c)

    def test_nonwinning_method_denied_even_with_complete_source(self):
        c=prediction_config(e.source.METHODS[1])
        with patch.object(e,'read',return_value=completed_source()),self.assertRaises(ValueError):e.selected_method(c)

    def test_missing_source_pass_rejected_before_activation_or_target_IO(self):
        c=prediction_config();failed=completed_source({m:dict(raw_correct=26567,H3_correct=62109) for m in e.source.METHODS})
        with patch.object(e,'read',return_value=failed),patch.object(e.source.base,'activate') as activate,patch.object(e.np,'load') as load:
            with self.assertRaises(ValueError):e.predict(c)
        activate.assert_not_called();load.assert_not_called()

    def test_only_selected_rows_three_populations_raw_and_F4_no_truth(self):
        for seed in e.SEEDS:e.validate(prediction_config(seed=seed))
        for changes in [dict(method='SIG1-PL'),dict(purpose='baseline_cross_date'),dict(readouts=['raw','DURA']),dict(target_truth='forbidden'),dict(tests={}),dict(source_run_id='foreign')]:
            with self.assertRaises(ValueError):e.validate(dict(prediction_config(),**changes))
        c=prediction_config();c['tests']['cross_rx']['truth']='forbidden'
        with self.assertRaises(ValueError):e.validate(c)
        c=prediction_config();c['frozen_sources']['G']='forbidden'
        with self.assertRaises(ValueError):e.validate(c)

    def test_scorer_has_no_F_handles_and_requires_frozen_winner(self):
        c=scoring_config();e.validate(c,scoring=True)
        self.assertTrue(e.shared.PREDICT_ONLY.isdisjoint(c))
        for end in [dict(parameters_and_buffers_unchanged=False,source_frozen_verified=True,source_selected=c['method']),dict(parameters_and_buffers_unchanged=True,source_frozen_verified=True,source_selected=e.source.METHODS[1])]:
            with patch.object(e,'read',return_value=end),patch('experiments.cric_dura_comparison.evaluation.score') as score:
                with self.assertRaises(ValueError):e.score(c)
            score.assert_not_called()

    def test_truth_side_delegates_only_after_checks_and_restores_task_scope(self):
        from experiments.cric_dura_comparison import evaluation as scorer
        original=scorer.tasks;c=scoring_config();end=dict(parameters_and_buffers_unchanged=True,source_frozen_verified=True,source_selected=c['method'])
        def cpu(actual):
            self.assertEqual(actual,c);self.assertEqual(scorer.tasks(c),list(e.TASK_COUNTS));raise RuntimeError('scorer failure')
        with patch.object(e,'read',return_value=end),patch.object(scorer,'score',side_effect=cpu),patch.object(e.source.base,'activate') as activate:
            with self.assertRaises(RuntimeError):e.score(c)
        self.assertIs(scorer.tasks,original);activate.assert_not_called()


if __name__=='__main__':unittest.main()
