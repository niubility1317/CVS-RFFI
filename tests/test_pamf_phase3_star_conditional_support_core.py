"""Predicted-old-class routing on fixed source banks, with synthetic features."""
import copy
import importlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np


class StarConditionalSupportCoreTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_conditional_support_core')
    def banks(self):
        y=np.repeat(np.arange(6,dtype=np.int64),8);L=np.eye(160,dtype=np.float32)[y]
        OE=np.tile(np.eye(160,dtype=np.float32)[0],(40,1));return L,y,OE
    def unit(self,x):
        x=np.asarray(x,np.float64);return (x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32)

    def test_hand_distances_route_by_closed_prediction_not_closest_or_true_class(self):
        m=self.module();state=m.fit(*self.banks());q=np.zeros((4,160),np.float32)
        q[0,0]=1;q[1,0]=1;q[2,0]=-1;q[3,1]=1;closed=np.array([1,0,0,0],np.int64);eps=1e-6
        actual=m.score(q,closed,state)
        expected=np.array([[0.,np.log(eps)-np.log(2+eps)],[0.,0.],
            [np.log(4+eps)-np.log(2+eps),0.],[np.log(2+eps)-np.log(eps),0.]])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
        self.assertEqual(actual.dtype,np.float64);self.assertEqual(m.METHODS,('pooled_relative','predicted_identity_relative'))
        self.assertEqual((m.K,m.EPS,m.BLOCK),(8,1e-6,128))

    def test_exact_eight_in_predicted_subbank_and_complete_equal_distance_ties(self):
        m=self.module();L,y,OE=self.banks();L[:]=0;L[:,1]=1;L[:7,1]=0;L[:7,0]=1
        L[8:16,1]=0;L[8:16,0]=1;OE[:]=0;OE[:,1]=1
        q=np.eye(160,dtype=np.float32)[:1];state=m.fit(L,y,OE);actual=m.score(q,np.zeros(1,np.int64),state)
        np.testing.assert_allclose(actual,[[np.log(2+1e-6)-np.log(1e-6),np.log(2+1e-6)-np.log(.25+1e-6)]],rtol=0,atol=1e-14)
        order=np.random.default_rng(6).permutation(len(y))
        np.testing.assert_array_equal(actual,m.score(q,np.zeros(1,np.int64),m.fit(L[order],y[order],OE[::-1])))

    def test_pooled_control_matches_previous_bitwise_and_block_row_permutations(self):
        m=self.module();old=importlib.import_module('experiments.pamf_phase3_open_world.star_local_support_core')
        rng=np.random.default_rng(74);L=self.unit(rng.normal(size=(48,160)));OE=self.unit(rng.normal(size=(40,160)))
        q=self.unit(rng.normal(size=(259,160)));y=np.repeat(np.arange(6,dtype=np.int64),8);closed=np.arange(len(q),dtype=np.int64)%6
        state=m.fit(L,y,OE);calls=[];distance=m._distances
        with patch.object(m,'_distances',side_effect=lambda a,b:calls.append((len(a),len(b))) or distance(a,b)):actual=m.score(q,closed,state)
        self.assertEqual(calls,[(128,48),(128,40),(128,48),(128,40),(3,48),(3,40)])
        rx=np.asarray([1,3,4,6,8],np.int64)
        previous=old.score(q,old.fit(L,np.resize(rx,48),OE,np.repeat(rx,8)))[:,1]
        np.testing.assert_array_equal(actual[:,0],previous)
        each=np.concatenate([m.score(q[i:i+1],closed[i:i+1],state) for i in range(len(q))]);np.testing.assert_array_equal(each,actual)
        order=rng.permutation(len(q));np.testing.assert_array_equal(m.score(q[order],closed[order],state),actual[order])
        library=rng.permutation(len(y));np.testing.assert_array_equal(m.score(q,closed,m.fit(L[library],y[library],OE[::-1])),actual)

    def test_working_state_copies_source_and_score_preserves_state_and_inputs(self):
        m=self.module();rng=np.random.default_rng(22);L=self.unit(rng.normal(size=(48,160)));OE=self.unit(rng.normal(size=(40,160)))
        y=np.repeat(np.arange(6,dtype=np.int64),8);before=[v.copy() for v in (L,y,OE)];state=m.fit(L,y,OE)
        self.assertEqual(set(state),{'L','L_y','OE'});self.assertEqual(state['L_y'].dtype,np.int64)
        self.assertEqual(state['L'].dtype,np.float64);self.assertEqual(state['OE'].dtype,np.float64)
        np.testing.assert_allclose(np.linalg.norm(state['L'],axis=1),1.,rtol=0,atol=3e-16)
        snapshot=copy.deepcopy(state);q=L[:3].copy();closed=np.arange(3,dtype=np.int64);qbefore=q.copy();m.score(q,closed,state)
        for k in state:np.testing.assert_array_equal(state[k],snapshot[k])
        for old,new in zip(before,(L,y,OE)):np.testing.assert_array_equal(old,new)
        np.testing.assert_array_equal(q,qbefore);np.testing.assert_array_equal(closed,np.arange(3))
        L[0]=0;y[0]=5;OE[0]=0
        for k in state:np.testing.assert_array_equal(state[k],snapshot[k])

    def test_invalid_source_labels_class_coverage_vectors_and_OE_capacity(self):
        m=self.module()
        for defect in ('negative','seventh','float_y','missing','seven_in_class','shape','dtype','norm','nan','OE7'):
            L,y,OE=[v.copy() for v in self.banks()]
            if defect=='negative':y[0]=-1
            elif defect=='seventh':y[0]=6
            elif defect=='float_y':y=y.astype(np.float64)
            elif defect=='missing':y[y==5]=4
            elif defect=='seven_in_class':L=L[:-1];y=y[:-1]
            elif defect=='shape':L=L[:,:159]
            elif defect=='dtype':OE=OE.astype(np.float64)
            elif defect=='norm':L[0]*=2
            elif defect=='nan':OE[0,0]=np.nan
            else:OE=OE[:7]
            with self.subTest(defect=defect),self.assertRaises(ValueError):m.fit(L,y,OE)

    def test_no_query_truth_RX_interface_and_invalid_prediction_or_state(self):
        m=self.module();state=m.fit(*self.banks());q=self.banks()[0][:2].copy();closed=np.array([0,1],np.int64)
        self.assertEqual(list(inspect.signature(m.fit).parameters),['L_features','L_y','OE_features'])
        self.assertEqual(list(inspect.signature(m.score).parameters),['query_features','closed','state'])
        with self.assertRaises(TypeError):m.score(q,closed,state,query_y=closed)
        with self.assertRaises(TypeError):m.score(q,closed,state,query_rx=np.ones(2))
        for bad in (np.array([-1,1]),np.array([0,6]),np.array([0]),np.array([0.,1.])):
            with self.subTest(closed=bad),self.assertRaises(ValueError):m.score(q,bad,state)
        with self.assertRaises(ValueError):m.score(q,closed,dict(state,query_bank=q))
        with self.assertRaises(ValueError):m.score(q*2,closed,state)


if __name__=='__main__':unittest.main()
