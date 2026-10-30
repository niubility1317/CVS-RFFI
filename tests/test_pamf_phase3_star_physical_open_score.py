"""Independent synthetic truth-last physical-regularized PAMF-OW metrics and coverage regressions."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_pamf_phase3_star_physical_open_eval import run_fixture


ROOT=Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'experiments/pamf_phase3_open_world/star_physical_open_score.py').is_file(), 'physical-regularized PAMF-OW scorer is not implemented'
    return importlib.import_module('experiments.pamf_phase3_open_world.star_physical_open_score')


def truth_files(folder,roles,qt,missing=False):
    novel=roles['novel'];n=len(novel['ids'])
    nt=dict(ids=novel['ids'][::-1],tx=np.asarray(['19-15','19-16','7-1']*(n//3))[::-1],
        rx_i=novel['metadata']['rx_i'][::-1],day_i=novel['metadata']['day_i'][::-1],
        scenario=np.tile(np.asarray(module().common.SCENES),n//6)[::-1])
    q={k:v[::-1].copy() for k,v in qt.items()}
    if missing:q['y'][(q['y']==6)&(q['scenario']==module().common.SCENES[0])]=7
    np.savez(folder/'novel_truth.npz',**nt);np.savez(folder/'Q_truth.npz',**q)
    return folder/'novel_truth.npz',folder/'Q_truth.npz'


class StarPhysicalOpenScoreTests(unittest.TestCase):
    def test_score_complete_preserves_launcher_terminal_contract(self):
        s=module()
        with tempfile.TemporaryDirectory() as temp,run_fixture(Path(temp)) as (e,c,b,roles,events,qt):
            e.evaluate(c,'cpu');nt,tt=truth_files(Path(temp),roles,qt)
            result=s.score(c['output_root'],nt,tt,Path(temp)/'rows/P_LOE/score')
            self.assertIs(result['eligible_primary'],False);self.assertEqual(result['next_step'],'NO_PROMOTION')
            consumed=s._read(Path(temp)/'rows/P_LOE/score/complete.json')
            self.assertEqual(consumed['status'],'STAR_PHYSICAL_OPEN_SCORES_COMPLETE')
            self.assertEqual(consumed['run_id'],e.RUN_ID);self.assertEqual(consumed['arm'],'P_LOE')
            self.assertTrue(consumed['truth_last']);self.assertFalse(consumed['formal_target_accessed'])
            self.assertEqual(consumed['eligible_primary'],result['eligible_primary'])
            self.assertEqual(consumed['next_step'],result['next_step'])

    def test_only_P_LOE_can_qualify_even_if_both_full_target_conditions_pass(self):
        s=module()
        for arm in ('P_L','P_LOE'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as temp:
                with run_fixture(Path(temp),arm=arm) as (e,c,b,roles,events,qt):
                    e.evaluate(c,'cpu');nt,tt=truth_files(Path(temp),roles,qt);root=Path(c['output_root'])
                    with np.load(root/'target_predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    a['relation_logits'][qt['y']>=6]=-2.;tau=np.load(root/'thresholds.npy',allow_pickle=False)
                    a['scores']=a['relation_logits'].max(1).astype(np.float64)-tau[a['closed']]
                    np.savez(root/'target_predictions.npz',**a)
                    result=s.score(root,nt,tt,Path(temp)/'score')
                    self.assertTrue(result['target_overall_and_six_scenes_passed'])
                    self.assertIs(result['eligible_primary'],arm=='P_LOE')
                    self.assertEqual(result['next_step'],'SOURCE_OPEN_WORLD_QUALIFIED_FOR_SEPARATE_UDA' if arm=='P_LOE' else 'NO_PROMOTION')
                    self.assertEqual(result['reference_mode'],'constant')
                    self.assertEqual(result['physical_mode'],'L_only' if arm=='P_L' else 'L_and_OE')
                    self.assertEqual(result['training_input_paths'],['cached_P1','same_record_cached_P0'])
                    self.assertEqual(result['evaluation_input_path'],'cached_P1_only')
                    self.assertEqual(result['received_cache_values_read'],0)
                    self.assertEqual(result['identifier_parameters'],315106);self.assertEqual(result['open_world_parameters'],82113)
                    self.assertEqual(result['training_container_parameters'],982835);self.assertEqual(result['unused_native_parameters'],667729)

    def test_incomplete_prediction_does_not_open_either_truth(self):
        s=module()
        with tempfile.TemporaryDirectory() as temp,patch.object(s.common,'_truth',side_effect=AssertionError('novel truth opened')), \
             patch.object(s,'target_truth_load',side_effect=AssertionError('Q truth opened')):
            with self.assertRaises((FileNotFoundError,ValueError)):
                s.score(Path(temp)/'incomplete','novel-private','Q-private',Path(temp)/'scores')

    def test_actual_completed_prediction_gate_reconstructs_calibration_before_truth(self):
        s=module()
        with tempfile.TemporaryDirectory() as temp,run_fixture(Path(temp)) as (e,c,b,roles,events,qt):
            e.evaluate(c,'cpu');root=Path(c['output_root']);done,cal,rows=s._prediction(root)
            self.assertEqual(done['method'],'relation_max');self.assertEqual(cal['group_count'],42)
            self.assertEqual(rows['target']['ids'].tolist(),qt['ids'].tolist())
            with np.load(root/'target_predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            a['y']=qt['y'];np.savez(root/'target_predictions.npz',**a)
            with patch.object(s.common,'_truth',side_effect=AssertionError('private truth touched')), \
                 patch.object(s,'target_truth_load',side_effect=AssertionError('private truth touched')):
                with self.assertRaises(ValueError):s.score(root,'not-opened','not-opened',Path(temp)/'score')

    def test_real_synthetic_truth_score_has_macro_TX_RX_day_scene_and_no_DA_REG_claim(self):
        s=module()
        with tempfile.TemporaryDirectory() as temp,run_fixture(Path(temp)) as (e,c,b,roles,events,qt):
            e.evaluate(c,'cpu');nt,tt=truth_files(Path(temp),roles,qt)
            result=s.score(c['output_root'],nt,tt,Path(temp)/'scores')
            row=result['target']['overall'];self.assertEqual(row['known_macro_acceptance'],1.)
            self.assertEqual(row['known_macro_open_accuracy'],1.);self.assertEqual(row['unknown_macro_rejection'],0.)
            self.assertEqual(row['H_OS'],0.);self.assertTrue(row['coverage']['complete'])
            self.assertEqual(len(result['target']['by_scene']),6);self.assertEqual(set(result['target']['by_RX']),{'26'})
            self.assertEqual(set(result['target']['by_day']),{'2021_03_15'})
            self.assertEqual(set(result['target']['by_TX']),{str(i) for i in range(9)})
            self.assertEqual(len(result['source']['groups']),42);self.assertEqual(len(result['source']['by_day']),3)
            self.assertEqual(result['primary_arm'],'P_LOE');self.assertIsNone(result['A_N']);self.assertIsNone(result['H_OW'])
            self.assertFalse(result['target_DA']);self.assertFalse(result['target_REG']);self.assertFalse(result['final_phase3_complete'])
            self.assertFalse(result['source_unknown_success_target_entry_gate'])
            with self.assertRaises(FileExistsError):s.score(c['output_root'],nt,tt,Path(temp)/'scores')

    def test_missing_required_unknown_scene_stays_NA_not_observed_only_success(self):
        s=module()
        with tempfile.TemporaryDirectory() as temp,run_fixture(Path(temp)) as (e,c,b,roles,events,qt):
            e.evaluate(c,'cpu');nt,tt=truth_files(Path(temp),roles,qt,missing=True)
            # All unknown rows now reject; only the required missing class
            # prevents certification of the first scene.
            root=Path(c['output_root'])
            with np.load(root/'target_predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            a['relation_logits'][qt['y']>=6]=-2.;tau=np.load(root/'thresholds.npy',allow_pickle=False)
            a['scores']=a['relation_logits'].max(1).astype(np.float64)-tau[a['closed']]
            np.savez(root/'target_predictions.npz',**a)
            result=s.score(root,nt,tt,Path(temp)/'scores');scene=result['target']['by_scene'][e.SCENES[0]]
            self.assertFalse(scene['coverage']['unknown_complete']);self.assertIsNone(scene['unknown_macro_rejection'])
            self.assertIsNone(scene['H_OS']);self.assertFalse(result['target']['all_six_scenes_passed'])
            self.assertEqual(result['target']['overall']['unknown_macro_rejection'],1.)

    def test_closed_damage_and_rejection_damage_are_separate_true_TX_macro(self):
        s=module();y=np.array([0,1,2]);closed=np.array([0,1,0]);baseline=y.copy()
        row=s._metrics(y,closed,np.array([1.,-1.,1.]),np.array([6]),np.array([-1.]),baseline,(0,1,2),(6,))
        self.assertAlmostEqual(row['candidate_macro_closed_accuracy'],2/3)
        self.assertAlmostEqual(row['original_F0_minus_candidate_closed'],1/3)
        self.assertAlmostEqual(row['candidate_closed_minus_open'],1/3)
        self.assertAlmostEqual(row['original_closed_minus_new_open'],2/3)
        self.assertEqual(row['known_closed_correct'],2);self.assertEqual(row['known_correct_accepted'],1)
        self.assertEqual(row['known_per_class'][1]['accepted'],0)


if __name__=='__main__':unittest.main()
