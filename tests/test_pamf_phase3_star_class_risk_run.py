"""Calibration-only orchestration over synthetic frozen parent artifacts."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarClassRiskRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_class_risk_run')
    def fixture(self,root):
        m=self.module()
        from test_pamf_phase3_star_consensus_run import StarConsensusRunTests
        from test_pamf_phase3_star_paired_run import StarPairedRunTests
        from experiments.pamf_phase3_open_world import star_paired_run as parent
        parentroot=root/m.PARENT_RUN/'prediction';(parentroot/'predict').mkdir(parents=True);(parentroot/'calibration').mkdir()
        roles={name:StarConsensusRunTests().role(name) for name in ('L','OE','V','novel')};counts={k:len(v['ids']) for k,v in roles.items()}
        p=dict(status=parent.STATUS,schema=parent.SCHEMA,run_id=m.PARENT_RUN,code_commit=m.PARENT_COMMIT,
            method_names=list(parent.METHODS),primary_method=parent.PRIMARY_METHOD,classification_path='P1',
            rejection_input_paths=['P1','P0'],control_p0_zeroed=True,known_count=counts['V'],novel_count=counts['novel'],counts=counts,
            parameters=232993,model_classes=6,parent_F0_identity=roles['L']['manifest']['parent_F0_identity'],
            dura_identity=roles['L']['manifest']['dura_identity'],cache_handles={k:v['manifest'] for k,v in roles.items()},
            source_only=True,target_reads=0,source_novel_truth_reads=0,new_encoder_updates=0,new_identity_head_updates=0,
            encoder_updates=0,identity_head_updates=0,optimizer_updates_F=0,G_forward_rows=0,G_updates=0,
            model_parameters_buffers_unchanged=True,L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,
            readout_steps_per_arm=1000,readout_optimizer_updates_total=2000,checkpoint_steps=[200,1000],snapshot_count=4,
            readout_parameter_counts={name:3233 for name in parent.LEARNED_METHODS},
            known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz')
        parent._write(parentroot/'complete.json',p)
        state=StarPairedRunTests().state(parent);state['training'].update(known_count=counts['L'],oe_count=counts['OE'],domain_count=15,model_seed=2026092701)
        parent._save_readouts(parentroot/'readouts.npz',state);parent._write(parentroot/'readouts.json',dict(training=state['training']))
        t=np.zeros(5,np.float64);np.save(parentroot/'thresholds.npy',t)
        cal=dict(status=parent.CALIBRATION_STATUS,schema=parent.SCHEMA,run_id=m.PARENT_RUN,code_commit=m.PARENT_COMMIT,
            method_names=list(parent.METHODS),group_count=42,known_only=True,unknown_inputs_used=False,
            methods={name:dict(threshold=0.,constraints_passed=True) for name in parent.METHODS})
        parent._write(parentroot/'calibration/complete.json',cal)
        known=roles['V'];y=known['metadata']['y'];closed=y.copy()
        kd=dict(ids=known['ids'],y=y,rx_i=known['metadata']['rx_i'],day_i=known['metadata']['day_i'],scenario=known['metadata']['scenario'],
            scores=np.tile(np.arange(5,dtype=np.float64),(len(y),1)),closed=closed,baseline_closed=closed.copy())
        novel=roles['novel'];nd=dict(ids=novel['ids'],rx_i=novel['metadata']['rx_i'],day_i=novel['metadata']['day_i'],
            scores=np.tile(np.arange(5,dtype=np.float64),(len(novel['ids']),1)),closed=(np.arange(len(novel['ids']))%6).astype(np.int64))
        parent._npz(parentroot/'predict/known_predictions.npz',kd);parent._npz(parentroot/'predict/novel_predictions.npz',nd)
        c=dict(run_id=m.RUN_ID,code_commit='a'*40,parent_prediction_root=str(parentroot),output_root=str(root/'new'))
        return c,p,kd,nd,counts
    def calibration(self,m,failed=False):
        state=dict(schema=m.core.STATE_SCHEMA,global_threshold=0.,conditional_thresholds=np.arange(6,dtype=np.float64))
        report=dict(methods={name:dict(constraints_passed=not failed) for name in m.METHODS},constraints_passed=not failed)
        return state,report

    def test_strict_four_fields_and_invalid_parent_before_checkpoint_or_data(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c,p,k,u,counts=self.fixture(Path(td));self.assertIs(m.validate_config(c),c)
            for key in ('truth','novel_truth','device','target_root','source_cache'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(c,**{key:'x'}))
            for defect in ('commit','classification','budget','methods','source'):
                bad=copy.deepcopy(p)
                if defect=='commit':bad['code_commit']='b'*40
                elif defect=='classification':bad['classification_path']='P0'
                elif defect=='budget':bad['readout_optimizer_updates_total']=2400
                elif defect=='methods':bad['method_names']=bad['method_names'][:-1]
                else:bad['source_only']=False
                path=Path(c['parent_prediction_root'])/'complete.json';path.write_text(json.dumps(bad))
                with self.subTest(defect=defect),patch.object(m,'COUNTS',counts),patch.object(m,'_smoke',side_effect=AssertionError('parent invalid before weights')),\
                     self.assertRaises(ValueError):m.run(c)

    def test_readout_smoke_is_numpy_feature_only_and_does_not_open_predictions(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c,p,k,u,counts=self.fixture(Path(td))
            with patch.object(m,'COUNTS',counts),patch.object(m,'_load_known',side_effect=AssertionError('No known rows in smoke')):
                state,report=m._smoke(Path(c['parent_prediction_root']),p)
            self.assertEqual(report['status'],'PASS');self.assertEqual(report['synthetic_pair_rows'],2)
            self.assertEqual(report['F0_forward_rows'],0);self.assertEqual(report['G_forward_rows'],0)
            self.assertEqual(report['source_prediction_rows_read'],0);self.assertTrue(report['feature_smoke'])
            self.assertEqual(len(state['models']),4)

    def test_smoke_known_calibration_freeze_then_novel_copy_identity_and_margin(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c,p,k,u,counts=self.fixture(Path(td));real_k=m._load_known;real_n=m._load_novel;real_s=m._smoke;real_write=m._write;writes=[]
            def smoke(*a):events.append('smoke');return real_s(*a)
            def known(*a):events.append('known');return real_k(*a)
            def novel(*a):
                events.append('novel');self.assertTrue((Path(c['output_root'])/'calibration/complete.json').exists())
                self.assertTrue((Path(c['output_root'])/'thresholds.npz').exists());return real_n(*a)
            with patch.object(m,'COUNTS',counts),patch.object(m,'_smoke',side_effect=smoke),patch.object(m,'_load_known',side_effect=known),\
                 patch.object(m,'_load_novel',side_effect=novel),patch.object(m.core,'calibrate_known',return_value=self.calibration(m)),\
                 patch.object(m,'_write',side_effect=lambda path,value:writes.append(str(path)) or real_write(path,value)):
                result=m.run(c)
            self.assertEqual(events,['smoke','known','novel']);self.assertEqual(result['status'],m.STATUS)
            self.assertEqual(result['new_readout_updates'],0);self.assertEqual(result['parent_code_commit'],m.PARENT_COMMIT)
            self.assertEqual(writes[-1],str(Path(c['output_root'])/'complete.json'))
            with np.load(Path(c['output_root'])/'predict/known_predictions.npz') as z:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS);np.testing.assert_array_equal(z['closed'],k['closed']);np.testing.assert_array_equal(z['baseline_closed'],k['baseline_closed'])
                np.testing.assert_array_equal(z['scores'][:,0],k['scores'][:,4]);np.testing.assert_array_equal(z['scores'][:,1],k['scores'][:,4]-k['closed'])
            with np.load(Path(c['output_root'])/'predict/novel_predictions.npz') as z:
                self.assertEqual(set(z.files),m.NOVEL_FIELDS);self.assertFalse({'y','tx','scenario'}&set(z.files))
            from experiments.pamf_phase3_open_world.star_class_risk_score import _prediction
            checked=_prediction(c['output_root']);self.assertEqual(checked[0]['status'],m.STATUS)
            with self.assertRaises(FileExistsError):m.run(c)

    def test_parent_global_threshold_mismatch_blocks_novel(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c,p,k,u,counts=self.fixture(Path(td));state,report=self.calibration(m);state['global_threshold']=.5
            with patch.object(m,'COUNTS',counts),patch.object(m.core,'calibrate_known',return_value=(state,report)),\
                 patch.object(m,'_load_novel',side_effect=AssertionError('threshold differs before novel')),self.assertRaisesRegex(ValueError,'global'):
                m.run(c)
            self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_failed_source_risk_still_predicts_and_illegal_identity_private_novel_rejected(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c,p,k,u,counts=self.fixture(Path(td))
            with patch.object(m,'COUNTS',counts),patch.object(m.core,'calibrate_known',return_value=self.calibration(m,failed=True)):
                result=m.run(c)
            self.assertEqual(result['status'],m.STATUS)
            cal=json.loads((Path(c['output_root'])/'calibration/complete.json').read_text());self.assertFalse(cal['methods']['predicted_class']['constraints_passed'])
        for defect in ('baseline','private_novel'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                c,p,k,u,counts=self.fixture(Path(td));parent=Path(c['parent_prediction_root'])
                if defect=='baseline':k['baseline_closed'][0]=(k['closed'][0]+1)%6;np.savez(parent/'predict/known_predictions.npz',**k)
                else:u['tx']=np.array(['private']*len(u['ids']));np.savez(parent/'predict/novel_predictions.npz',**u)
                with patch.object(m,'COUNTS',counts),patch.object(m.core,'calibrate_known',return_value=self.calibration(m)),self.assertRaises(ValueError):m.run(c)
                self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_routing_uses_wrong_frozen_head_class_instead_of_true_label(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c,p,k,u,counts=self.fixture(Path(td));k['closed']=(k['y']+1)%6;k['baseline_closed']=k['closed'].copy()
            np.savez(Path(c['parent_prediction_root'])/'predict/known_predictions.npz',**k)
            with patch.object(m,'COUNTS',counts),patch.object(m.core,'calibrate_known',return_value=self.calibration(m)):
                m.run(c)
            with np.load(Path(c['output_root'])/'predict/known_predictions.npz') as z:
                np.testing.assert_array_equal(z['scores'][:,1],4.-k['closed'])
                self.assertFalse(np.array_equal(z['scores'][:,1],4.-k['y']))

    def test_real_core_global_parity_and_complete_tie_acceptance(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c,p,k,u,counts=self.fixture(Path(td));parent=Path(c['parent_prediction_root'])
            values=np.zeros(5);values[4]=4.;np.save(parent/'thresholds.npy',values)
            cal=json.loads((parent/'calibration/complete.json').read_text());cal['methods'][m.PARENT_METHODS[4]]['threshold']=4.
            (parent/'calibration/complete.json').write_text(json.dumps(cal))
            with patch.object(m,'COUNTS',counts):result=m.run(c)
            self.assertTrue(result['global_control_matches_parent'])
            with np.load(Path(c['output_root'])/'predict/known_predictions.npz') as z:
                np.testing.assert_array_equal(z['scores'][:,0],np.zeros(len(k['ids'])))
                np.testing.assert_array_equal(z['scores'][:,0]>=0,k['scores'][:,4]>=values[4])


if __name__=='__main__':unittest.main()
