"""Truth-last source nuisance scoring with synthetic identities only."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarNuisanceScoreTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_nuisance_score')

    def fixture(self, root, main_good=True, control_good=False):
        from experiments.pamf_phase3_open_world import star_nuisance_source as s
        rows=[(y,r,scene) for y in range(6) for r in s.RECEIVERS for scene in s.SCENES]
        y=np.array([x[0] for x in rows],np.int64);n=len(y)
        known=dict(ids=np.array(['k'+str(i) for i in range(n)]),y=y,rx_i=np.array([x[1] for x in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scenario=np.array([x[2] for x in rows]),
            scores=np.ones((n,12),np.float64),closed=np.column_stack((y,y)),baseline_closed=np.column_stack((y,y)))
        rows=[(tx,r,scene) for tx in ('19-15','19-16','7-1') for r in s.RECEIVERS for scene in s.SCENES]
        n=len(rows);us=np.ones((n,12),np.float64)
        if control_good:us[:,:]=-1
        us[:,s.CANDIDATES.index('rxday::RMD::P1')]=-1 if main_good else 1
        novel=dict(ids=np.array(['u'+str(i) for i in range(n)]),rx_i=np.array([x[1] for x in rows],np.int64),
            day_i=(np.arange(n)%3+1).astype(np.int64),scores=us,closed=np.zeros((n,2),np.int64))
        truth=dict(ids=novel['ids'][::-1],tx=np.array([x[0] for x in rows])[::-1],rx_i=novel['rx_i'][::-1],
            day_i=novel['day_i'][::-1],scenario=np.array([x[2] for x in rows])[::-1])
        source=root/'source';source.mkdir();(source/'source/calibration').mkdir(parents=True)
        s._npz(source/'known_predictions.npz',known);s._npz(source/'novel_predictions.npz',novel)
        np.save(source/'source/calibration/thresholds.npy',np.zeros(12))
        calibration=dict(status=s.CALIBRATION_STATUS,schema=s.SCHEMA,run_id=s.RUN_ID,candidate_names=list(s.CANDIDATES),
            group_count=42,constraints_passed=True,source_only=True,target_reads=0,unknown_inputs_used=False,
            source_V_metric_fit=False,source_V_sigma_fit=False,methods={name:dict(threshold=0.,constraints_passed=True) for name in s.CANDIDATES})
        s._write(source/'source/calibration/complete.json',calibration)
        marker=dict(status=s.STATUS,schema=s.SCHEMA,run_id=s.RUN_ID,code_commit='a'*40,candidate_names=list(s.CANDIDATES),
            path_names=list(s.PATHS),variant_names=list(s.VARIANTS),method_names=list(s.METHODS),primary_candidate=s.PRIMARY,
            source_only=True,target_reads=0,source_novel_truth_reads=0,source_U_cache_reads=0,new_neural_updates=0,
            model_parameters_buffers_unchanged=True,L_aggregates_unchanged_during_V=True,
            all_thresholds_frozen_before_novel=True,known_count=len(y),novel_count=n,baseline='same-input same-path original F0 complete head')
        s._write(source/'complete.json',marker)
        tp=root/'truth.npz';s._npz(tp,truth)
        return source,tp,known,novel,truth

    def test_truthlast_main_gate_and_macro_metrics(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src,truth,*_=self.fixture(root)
            result=m.score(src,truth,root/'score')
            self.assertTrue(result['source_gate_passed']);self.assertFalse(result['final_phase3_complete'])
            self.assertEqual(result['next_step'],'SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION')
            self.assertEqual(result['eligible_candidates'],['rxday::RMD::P1'])
            row=result['candidates']['rxday::RMD::P1']
            self.assertEqual(len(row['groups']),42);self.assertEqual(row['groups']['overall']['unknown_macro_rejection'],1.)
            self.assertAlmostEqual(row['groups']['overall']['auroc_known_macro'],1.)
            self.assertEqual(len(row['by_unknown_TX']),3)
            with self.assertRaises(FileExistsError):m.score(src,truth,root/'score')

    def test_good_ablation_cannot_replace_failed_main(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src,truth,*_=self.fixture(root,main_good=False,control_good=True)
            result=m.score(src,truth,root/'score')
            self.assertFalse(result['source_gate_passed']);self.assertEqual(result['next_step'],'NO_PROMOTION')
            self.assertEqual(result['eligible_candidates'],[])

    def test_no_truth_open_before_all_predictions_complete(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src,truth,*_=self.fixture(root);(src/'complete.json').unlink()
            with patch.object(m,'_truth',side_effect=AssertionError('truth opened')),self.assertRaises(FileNotFoundError):
                m.score(src,truth,root/'score')

    def test_reject_duplicate_missing_ids_truth_metadata_and_private_prediction_fields(self):
        m=self.module()
        for defect in ('duplicate','missing','rx','tx','public_scene'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);src,tp,k,u,t=self.fixture(root)
                if defect=='duplicate':t['ids'][0]=t['ids'][1]
                elif defect=='missing':t={key:value[:-1] for key,value in t.items()}
                elif defect=='rx':t['rx_i'][0]=8 if t['rx_i'][0]!=8 else 1
                elif defect=='tx':t['tx'][0]='forbidden'
                else:
                    u['scenario']=np.array(['private']*len(u['ids']));(src/'novel_predictions.npz').unlink()
                    from experiments.pamf_phase3_open_world.star_nuisance_source import _npz
                    _npz(src/'novel_predictions.npz',u)
                if defect!='public_scene':np.savez(tp,**t)
                with self.assertRaises(ValueError):m.score(src,tp,root/'score')
                self.assertFalse((root/'score/complete.json').exists())

    def test_missing_class_cannot_pass_and_RXscene_unknown_is_only_diagnostic(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src,tp,k,u,t=self.fixture(root)
            # One missing novel TX in a single RX-scene need not fail the
            # predefined overall/RX/scene gate; that cell remains N/A.
            keep=~((t['tx']=='19-15')&(t['rx_i']==1)&(t['scenario']==m.SCENES[0]))
            ids=set(t['ids'][keep]);uk=np.array([sid in ids for sid in u['ids']]);u={key:value[uk] for key,value in u.items()}
            np.savez(tp,**{key:value[keep] for key,value in t.items()});np.savez(src/'novel_predictions.npz',**u)
            marker=json.loads((src/'complete.json').read_text());marker['novel_count']=len(u['ids'])
            (src/'complete.json').write_text(json.dumps(marker))
            result=m.score(src,tp,root/'score')
            self.assertTrue(result['source_gate_passed'])
            group=result['candidates']['rxday::RMD::P1']['groups']['rxscene/1/'+m.SCENES[0]]
            self.assertIsNone(group['unknown_macro_rejection'])

    def test_known_RXscene_protection_and_missing_known_coverage_cannot_hide(self):
        m=self.module()
        for defect in ('reject','missing_known_class'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);src,tp,k,u,t=self.fixture(root)
                mask=(k['y']==0)&(k['rx_i']==1)&(k['scenario']==m.SCENES[0])
                if defect=='reject':k['scores'][mask,-1]=-1
                else:k={key:value[~mask] for key,value in k.items()}
                np.savez(src/'known_predictions.npz',**k)
                marker=json.loads((src/'complete.json').read_text());marker['known_count']=len(k['ids'])
                (src/'complete.json').write_text(json.dumps(marker))
                result=m.score(src,tp,root/'score')
                self.assertFalse(result['source_gate_passed']);self.assertEqual(result['next_step'],'NO_PROMOTION')


if __name__=='__main__':unittest.main()
