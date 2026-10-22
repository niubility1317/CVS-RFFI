import copy
import unittest

from experiments.cric_dura_comparison.completeness import (
    CONTROLS, GFACTOR, GEOMETRY, TIMING, build_plan, validate,
    audit_prediction_coverage, main_source,
)


class CompletenessTests(unittest.TestCase):
    def snapshot(self):
        rows = []
        for pid in build_plan()['protocols']:
            for regime in ('native_cric', 'matched_sat_supervised', 'matched_sat_plain_pl'):
                for seed in (2026092701, 2026092702, 2026092703):
                    readouts = ['raw', 'DURA-LU']
                    if regime == 'matched_sat_supervised':
                        readouts.append('DURA-L')
                    rows.append(dict(scope='main', protocol_id=pid, regime=regime,
                        model_seed=seed, readouts=readouts))
        return dict(rows=rows, truth_or_score_reads=False,
            current_owner='existing-owner', effective_counts={'RUNNING':25})

    def test_off_diagonal_only_and_replacement_source(self):
        audit = audit_prediction_coverage(self.snapshot())
        self.assertEqual(audit['main_logical_rows'], 153)
        self.assertEqual(audit['missing_count'], 51)
        self.assertEqual({r['regime'] for r in audit['missing_readout_cells']}, {'matched_sat_plain_pl'})
        self.assertEqual({r['readout'] for r in audit['missing_readout_cells']}, {'DURA-L'})
        self.assertEqual(main_source('F0','matched_sat_plain_pl',2026092701)['row_id'],
                         'F0-matched_sat_plain_pl-s2026092701')

    def test_reject_duplicate_or_missing_effective_main(self):
        snapshot = self.snapshot()
        snapshot['rows'].append(copy.deepcopy(snapshot['rows'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicated'):
            audit_prediction_coverage(snapshot)
        snapshot = self.snapshot();snapshot['rows'].pop()
        with self.assertRaisesRegex(ValueError,'17x3x3'):
            audit_prediction_coverage(snapshot)

    def test_reject_target_feedback_input(self):
        snapshot=self.snapshot();snapshot['truth_or_score_reads']=True
        with self.assertRaisesRegex(ValueError,'target scores'):
            audit_prediction_coverage(snapshot)

    def test_no_G_training_and_immutable_scope(self):
        plan=build_plan();validate(plan)
        self.assertEqual(plan['total_new_stage_jobs'],211)
        self.assertEqual(plan['new_G_training'],0)
        self.assertEqual(plan['new_F_training'],15)
        self.assertEqual(len(plan['runs'][GFACTOR]),51)
        self.assertEqual(len(plan['runs'][GEOMETRY]),15)
        self.assertEqual(plan['runs'][TIMING][0]['case_rounds'],150)
        self.assertEqual(plan['status'],'PLANNED_NOT_PUBLISHED')

    def test_reject_widened_permission_budget_or_excluded_scan(self):
        plan=build_plan();plan['runs'][CONTROLS][0]['U_IQ_reads']=1
        with self.assertRaisesRegex(ValueError,'permissions'):
            validate(plan)
        plan=build_plan();plan['runs'][CONTROLS][0]['optimizer_updates']=44400
        with self.assertRaisesRegex(ValueError,'budget'):
            validate(plan)
        plan=build_plan();plan['protocols']['m1-natural']['source_count']=5
        with self.assertRaisesRegex(ValueError,'m5'):
            validate(plan)


if __name__ == '__main__':
    unittest.main()
