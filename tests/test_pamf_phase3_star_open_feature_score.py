"""Synthetic source-only geometry diagnostics; no IQ, model or real truth."""
from contextlib import contextmanager
import importlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'experiments/pamf_phase3_open_world/star_open_feature_score.py').is_file(), 'Open-feature scorer is not implemented'
    return importlib.import_module('experiments.pamf_phase3_open_world.star_open_feature_score')


@contextmanager
def export_fixture(root):
    s=module();n=45;root.mkdir();y=np.repeat(np.arange(3),15)
    rx=np.tile(np.repeat(np.asarray(s.common.RECEIVERS,np.int64),3),3);day=np.tile(np.arange(1,4,dtype=np.int64),15)
    ids=np.asarray(['source-'+str(i).zfill(3) for i in range(n)]);meta=dict(ids=ids,rx_i=rx,day_i=day)
    f0=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701)
    parents=dict(F0=f0)
    for arm in ('S0','S1','M'):
        parents[arm]=dict(run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,arm=arm,updates=2220,cumulative_updates=13320,
            identifier_parameters=284834,open_world_parameters=51841,training_container_parameters=952563,
            identity_head_replaced=False,parent_F0_identity=f0)
    manifest=dict(schema='pamf_phase3_star_representation_cache_role_v1',status='STAR_REPRESENTATION_ROLE_COMPLETE',
        role='novel',count=n,iq_shape=[n,2,256],source_only=True,all_new_IQ_satellite=True,restored_input_path='P1',parent_F0_identity=f0)
    done=dict(schema=s.EXPORT_SCHEMA,status=s.EXPORT_STATUS,run_id=s.RUN_ID,code_commit='b'*40,source_count=n,
        optimizer_updates=0,target_reads=0,private_truth_reads=0,model_states_unchanged=True,source_role='novel',
        model_updates=0,threshold_updates=0,source_L_reads=0,source_V_reads=0,source_OE_reads=0,original_U_s_reads=0,
        G_forward_rows=0,input_path='cached_P1_only',received_cache_values_read=0,
        source_manifest=manifest,parent_identities=parents,representations=s.REPRESENTATIONS)
    (root/'complete.json').write_text(json.dumps(done),encoding='utf-8')
    np.savez(root/'F0.npz',**meta,fused160=np.eye(160,dtype=np.float32)[y])
    for arm in ('S0','S1','M'):np.savez(root/(arm+'.npz'),**meta,fused160=np.eye(160,dtype=np.float32)[y],open64=np.eye(64,dtype=np.float32)[y])
    with patch.object(s,'SOURCE_COUNT',n):yield s,done,meta,y


class StarOpenFeatureScoreTests(unittest.TestCase):
    def test_neighbor_choice_is_label_blind_self_excluded_domain_filtered_and_ID_tied(self):
        s=module();x=np.tile(np.asarray([[1.,0.]]),(4,1));ids=np.asarray(['z','b','a','c'])
        rx=np.asarray([1,3,1,3]);day=np.asarray([1,1,2,2]);neighbors=s._nearest_indices(x,ids,rx,day)
        self.assertEqual(neighbors['all_other'][0],2);self.assertEqual(neighbors['cross_RX'][0],1)
        self.assertEqual(neighbors['cross_day'][0],2);self.assertEqual(neighbors['cross_RX_and_day'][0],3)
        self.assertTrue(all(np.all(v!=np.arange(4)) for v in neighbors.values()))
        self.assertEqual(tuple(inspect.signature(s._nearest_indices).parameters),('features','ids','rx','day'))
        perm=np.array([3,1,0,2]);changed=s._nearest_indices(x[perm],ids[perm],rx[perm],day[perm])
        for mode,v in neighbors.items():np.testing.assert_array_equal(ids[v][perm],ids[perm][changed[mode]])
        first=s._retrieval(neighbors,np.asarray(['A','A','B','B']),rx,day,('A','B'))
        second=s._retrieval(neighbors,np.asarray(['A','B','B','A']),rx,day,('A','B'))
        self.assertNotEqual(first['cross_RX_and_day']['overall']['hits'],second['cross_RX_and_day']['overall']['hits'])

    def test_macro_uses_equal_TX_weight_and_missing_candidates_are_NA(self):
        s=module();tx=np.asarray(['a','a','a','a','b']);rx=np.ones(5,int);day=np.ones(5,int)
        neighbors={mode:np.asarray([1,0,0,0,0]) if mode=='all_other' else np.full(5,-1,np.int64) for mode in s.CONDITIONS}
        rows=s._retrieval(neighbors,tx,rx,day,('a','b'))
        self.assertEqual(rows['all_other']['overall']['macro_hit_rate'],.5)
        self.assertEqual(rows['all_other']['overall']['pooled_hit_rate'],.8)
        for mode in s.CONDITIONS[1:]:
            self.assertIsNone(rows[mode]['overall']['macro_hit_rate']);self.assertIsNone(rows[mode]['overall']['pooled_hit_rate'])
            self.assertFalse(rows[mode]['overall']['coverage_complete']);self.assertEqual(rows[mode]['overall']['missing_candidate_count'],5)

    def test_streamed_cosine_gap_covariance_ranks_and_variance_ratios_match_manual(self):
        s=module();x=np.asarray([[1.,0.],[1.,0.],[-1.,0.],[-1.,0.]])
        tx=np.asarray(['a','a','b','b']);rx=np.asarray([1,3,1,3]);day=np.asarray([1,2,1,2])
        gap=s._cosine_gap(x,tx,rx,day)
        self.assertEqual(gap['same_TX_pair_count'],4);self.assertEqual(gap['different_TX_pair_count'],4)
        self.assertEqual(gap['same_TX_mean'],1.);self.assertEqual(gap['different_TX_mean'],-1.);self.assertEqual(gap['gap'],2.)
        self.assertEqual(s._between_ratio(x,tx),1.);self.assertEqual(s._between_ratio(x,rx),0.)
        self.assertEqual(s._between_ratio(x,day),0.)
        rank=s._spectrum(np.asarray([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]))
        self.assertAlmostEqual(rank['effective_rank'],2.);self.assertAlmostEqual(rank['participation_rank'],2.)
        zero=s._spectrum(np.zeros((4,2)));self.assertEqual(zero['effective_rank'],0.)
        self.assertEqual(zero['participation_rank'],0.);self.assertIsNone(s._between_ratio(np.zeros((4,2)),tx))
        constant=np.tile(np.asarray([[1.,2.,3.]]),(45,1))
        self.assertEqual(s._spectrum(constant)['effective_rank'],0.)
        self.assertIsNone(s._between_ratio(constant,np.arange(45)%3))
        neighbors=s._nearest_indices(np.zeros((4,2)),np.asarray(['d','b','a','c']),np.ones(4,int),np.ones(4,int))
        self.assertTrue(np.all(neighbors['cross_RX_and_day']==-1));self.assertTrue(np.all(neighbors['all_other']>=0))

    def test_all_seven_features_and_all_model_markers_pass_before_truth_opens(self):
        s=module()
        for defect in ('nan_last','order_last','private_field','norm','updates','target','parent','V_read','G_forward'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as temp:
                with export_fixture(Path(temp)/'features') as (s,done,meta,y):
                    folder=Path(temp)/'features'
                    if defect in ('nan_last','order_last','norm','private_field'):
                        with np.load(folder/'M.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                        if defect=='nan_last':a['open64'][0,0]=np.nan
                        elif defect=='order_last':a['ids']=a['ids'][::-1]
                        elif defect=='norm':a['open64'][0]*=2.
                        else:a['tx']=np.asarray(['forbidden']*len(y))
                        np.savez(folder/'M.npz',**a)
                    else:
                        if defect=='updates':done['optimizer_updates']=1
                        elif defect=='target':done['target_reads']=864
                        elif defect=='V_read':done['source_V_reads']=27000
                        elif defect=='G_forward':done['G_forward_rows']=3150
                        else:done['parent_identities']['M']['code_commit']='a'*40
                        (folder/'complete.json').write_text(json.dumps(done),encoding='utf-8')
                    with patch.object(s.common,'_truth',side_effect=AssertionError('Private truth opened early')) as opened:
                        with self.assertRaises(ValueError):s.score(folder,'private-not-opened',Path(temp)/'score')
                        opened.assert_not_called()

    def test_full_source_score_aligns_private_truth_after_all_feature_choices_and_never_promotes(self):
        s=module()
        with tempfile.TemporaryDirectory() as temp,export_fixture(Path(temp)/'features') as (s,done,meta,y):
            truth=dict(ids=meta['ids'][::-1],rx_i=meta['rx_i'][::-1],day_i=meta['day_i'][::-1],
                tx=np.asarray(s.common.UNKNOWN_TX)[y][::-1],scenario=np.tile(np.asarray(s.common.SCENES),8)[:len(y)][::-1])
            np.savez(Path(temp)/'truth.npz',**truth)
            selected=[];nearest=s._nearest_indices;truth_load=s.common._truth
            def choose(*args):selected.append(1);return nearest(*args)
            def load(path,novel):self.assertEqual(len(selected),7);return truth_load(path,novel)
            with patch.object(s,'_nearest_indices',choose),patch.object(s.common,'_truth',load):
                result=s.score(Path(temp)/'features',Path(temp)/'truth.npz',Path(temp)/'score')
            self.assertEqual(tuple(result['representations']),tuple(s.REPRESENTATIONS));self.assertEqual(result['primary_arm'],'M')
            for row in result['representations'].values():
                for mode in s.CONDITIONS:self.assertEqual(row['nearest_neighbor'][mode]['overall']['macro_hit_rate'],1.)
            self.assertIsNone(result['A_N']);self.assertIsNone(result['H_OW']);self.assertFalse(result['final_phase3_complete'])
            self.assertEqual(result['full_goal_status'],'ACTIVE');self.assertEqual(result['target_reads'],0)
            with self.assertRaises(FileExistsError):s.score(Path(temp)/'features',Path(temp)/'truth.npz',Path(temp)/'score')


if __name__=='__main__':unittest.main()
