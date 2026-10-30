"""Synthetic frozen physical-learning geometry diagnostics; no real data or models."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'experiments/pamf_phase3_open_world/star_physical_geometry_score.py').is_file(), 'Physical geometry scorer is not implemented'
    return importlib.import_module('experiments.pamf_phase3_open_world.star_physical_geometry_score')


@contextmanager
def export_fixture(root):
    s=module();n=45;root.mkdir();y=np.repeat(np.arange(3),15)
    rx=np.tile(np.repeat(np.asarray(s.common.RECEIVERS,np.int64),3),3);day=np.tile(np.arange(1,4,dtype=np.int64),15)
    ids=np.asarray(['source-'+str(i).zfill(3) for i in range(n)]);meta=dict(ids=ids,rx_i=rx,day_i=day)
    native=dict(base_release='original-release',source_output='original-source',source_contract='original-contract')
    f0=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,**native)
    parents=dict(F0=f0)
    for arm in s.ARMS:
        parents[arm]=dict(run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,arm=arm,checkpoint='train/'+arm+'/final.pt',
            updates=2220,cumulative_updates=13320,identifier_parameters=315106,open_world_parameters=82113,
            training_container_parameters=982835,identity_head_replaced=False,parent_F0_identity=f0,
            original_parent_identity=dict(parent_epoch=50,parent_updates=11100,model_seed=2026092701,**native),
            input_path='cached_P1_and_same_record_P0',identity_classification_path='P1',reference_mode='constant',
            membership_task='rotating_five',physical_mode='L_only' if arm=='P_L' else 'L_and_OE',
            training_input_paths=['cached_P1','same_record_cached_P0'])
    manifest=dict(schema='pamf_phase3_star_representation_cache_role_v1',status='STAR_REPRESENTATION_ROLE_COMPLETE',
        role='novel',count=n,iq_shape=[n,2,256],source_only=True,all_new_IQ_satellite=True,
        restored_input_path='P1',parent_F0_identity=f0,dura_identity=dict(deployment_id='dura_v1'))
    baseline=dict(schema=s.BASELINE_SCHEMA,status=s.BASELINE_STATUS,run_id=s.BASELINE_RUN,code_commit=s.BASELINE_COMMIT,
        source_count=n,source_role='novel',parent_identities=dict(F0=f0),source_manifest=manifest)
    done=dict(schema=s.EXPORT_SCHEMA,status=s.EXPORT_STATUS,run_id=s.RUN_ID,code_commit='b'*40,source_count=n,
        optimizer_updates=0,target_reads=0,private_truth_reads=0,model_states_unchanged=True,source_role='novel',
        model_updates=0,threshold_updates=0,source_L_reads=0,source_V_reads=0,source_OE_reads=0,original_U_s_reads=0,
        G_forward_rows=0,input_path='cached_P1_only',received_cache_values_read=0,
        baseline_F0_features_reused=True,actual_frozen_checkpoints=list(s.ARMS),real_F_forward_rows=2*n,synthetic_F_forward_rows=4,
        baseline_original_export_identity=baseline,source_manifest=manifest,parent_identities=parents,representations=s.REPRESENTATIONS)
    (root/'complete.json').write_text(json.dumps(done),encoding='utf-8')
    np.savez(root/'F0.npz',**meta,fused160=np.eye(160,dtype=np.float32)[y])
    for arm in s.ARMS:
        np.savez(root/(arm+'.npz'),**meta,fused160=np.eye(160,dtype=np.float32)[y],open64=np.eye(64,dtype=np.float32)[y])
    with patch.object(s,'SOURCE_COUNT',n):yield s,done,meta,y


class StarPhysicalGeometryScoreTests(unittest.TestCase):
    def test_new_source_identity_and_exact_baseline_reuse_are_required_before_truth(self):
        defects=('old_parent','old_counts','reference','physical_mode','train_path','head','ancestor','parent_F0',
                 'baseline_code','baseline_manifest','reuse','loaded_F0','role','source_V','target','G','updates')
        for defect in defects:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as temp:
                with export_fixture(Path(temp)/'features') as (s,d,meta,y):
                    p=d['parent_identities']['P_LOE']
                    if defect=='old_parent':p['code_commit']='a'*40
                    elif defect=='old_counts':p['identifier_parameters']=284834
                    elif defect=='reference':p['reference_mode']='reference'
                    elif defect=='physical_mode':p['physical_mode']='L_only'
                    elif defect=='train_path':p['input_path']='cached_P1_only'
                    elif defect=='head':p['identity_head_replaced']=True
                    elif defect=='ancestor':p['original_parent_identity']['parent_updates']=11099
                    elif defect=='parent_F0':p['parent_F0_identity']=dict(p['parent_F0_identity'],parameters=1)
                    elif defect=='baseline_code':d['baseline_original_export_identity']['code_commit']='a'*40
                    elif defect=='baseline_manifest':d['baseline_original_export_identity']['source_manifest']=dict(d['source_manifest'],restored_input_path='P0')
                    elif defect=='reuse':d['baseline_F0_features_reused']=False
                    elif defect=='loaded_F0':d['actual_frozen_checkpoints']=['F0','P_L','P_LOE']
                    elif defect=='role':d['source_role']='V'
                    else:d[dict(source_V='source_V_reads',target='target_reads',G='G_forward_rows',updates='optimizer_updates')[defect]]=1
                    folder=Path(temp)/'features';(folder/'complete.json').write_text(json.dumps(d),encoding='utf-8')
                    with patch.object(s.common,'_truth',side_effect=AssertionError('Truth opened early')) as opened:
                        with self.assertRaises(ValueError):s.score(folder,'private-not-opened',Path(temp)/'score')
                        opened.assert_not_called()

    def test_all_five_feature_arrays_are_public_finite_unit_and_metadata_aligned(self):
        for defect in ('nan','ids','rx','day','float64','norm','y','open_raw64','missing_open64'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as temp:
                with export_fixture(Path(temp)/'features') as (s,d,meta,y):
                    folder=Path(temp)/'features';path=folder/'P_LOE.npz'
                    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='nan':a['open64'][0,0]=np.nan
                    elif defect=='ids':a['ids']=a['ids'][::-1]
                    elif defect=='rx':a['rx_i'][0]=3
                    elif defect=='day':a['day_i'][0]=4
                    elif defect=='float64':a['open64']=a['open64'].astype(np.float64)
                    elif defect=='norm':a['open64'][0]*=2.
                    elif defect=='y':a['y']=y
                    elif defect=='open_raw64':a['open_raw64']=a['open64'].copy()
                    else:del a['open64']
                    np.savez(path,**a)
                    with patch.object(s.common,'_truth',side_effect=AssertionError('Truth opened early')) as opened:
                        with self.assertRaises(ValueError):s.score(folder,'private-not-opened',Path(temp)/'score')
                        opened.assert_not_called()

    def test_all_five_four_mask_choices_freeze_before_sole_truth_then_report_fixed_primary(self):
        with tempfile.TemporaryDirectory() as temp,export_fixture(Path(temp)/'features') as (s,d,meta,y):
            truth=dict(ids=meta['ids'][::-1],rx_i=meta['rx_i'][::-1],day_i=meta['day_i'][::-1],
                tx=np.asarray(s.common.UNKNOWN_TX)[y][::-1],scenario=np.tile(np.asarray(s.common.SCENES),8)[:len(y)][::-1])
            np.savez(Path(temp)/'truth.npz',**truth);selected=[];nearest=s._nearest_indices;truth_load=s.common._truth
            def choose(*args):
                result=nearest(*args);self.assertEqual(tuple(result),s.CONDITIONS);selected.append(result);return result
            def load(path,novel):self.assertEqual(len(selected),5);return truth_load(path,novel)
            with patch.object(s,'_nearest_indices',choose),patch.object(s.common,'_truth',side_effect=load) as opened:
                result=s.score(Path(temp)/'features',Path(temp)/'truth.npz',Path(temp)/'score');opened.assert_called_once()
            self.assertEqual(result['schema'],s.SCHEMA);self.assertEqual(result['status'],s.STATUS)
            self.assertEqual(tuple(result['representations']),tuple(s.REPRESENTATIONS));self.assertEqual(result['primary_arm'],'P_LOE')
            for row in result['representations'].values():
                for mode in s.CONDITIONS:self.assertEqual(row['nearest_neighbor'][mode]['overall']['macro_hit_rate'],1.)
                self.assertEqual(row['between_group_variance_ratio']['TX'],1.)
            self.assertTrue(result['baseline_F0_features_reused']);self.assertEqual(result['actual_frozen_checkpoints'],list(s.ARMS))
            self.assertEqual(result['target_reads'],0);self.assertEqual(result['optimizer_updates'],0)
            self.assertIsNone(result['A_N']);self.assertIsNone(result['H_OW']);self.assertFalse(result['eligible_primary'])
            self.assertFalse(result['full_goal_passed']);self.assertEqual(result['full_goal_status'],'ACTIVE')
            self.assertEqual(result['next_step'],'DIAGNOSTIC_ONLY')
            self.assertTrue((Path(temp)/'score/complete.json').is_file())
            with self.assertRaises(FileExistsError):s.score(Path(temp)/'features','not-opened',Path(temp)/'score')

    def test_reused_helpers_preserve_zero_collapse_missing_neighbors_and_opaque_ties(self):
        s=module();x=np.zeros((4,160),np.float32);ids=np.asarray(['z','b','a','c'])
        rx=np.ones(4,np.int64);day=np.ones(4,np.int64);nearest=s._nearest_indices(x,ids,rx,day)
        self.assertEqual(nearest['all_other'][0],2)
        self.assertTrue(np.all(nearest['cross_RX_and_day']==-1))
        row=s._retrieval(nearest,np.asarray(['a','a','b','b']),rx,day,('a','b'))
        self.assertIsNone(row['cross_RX_and_day']['overall']['macro_hit_rate'])
        self.assertFalse(row['cross_RX_and_day']['overall']['coverage_complete'])
        self.assertEqual(s._spectrum(x)['effective_rank'],0.);self.assertIsNone(s._between_ratio(x,np.arange(4)))
        self.assertIsNone(s._cosine_gap(x,np.arange(4),rx,day)['gap'])


if __name__=='__main__':unittest.main()
