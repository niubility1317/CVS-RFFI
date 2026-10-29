"""Fixed local source-bank evidence; synthetic unit features only."""
import copy
import importlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np


class StarLocalSupportCoreTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_local_support_core')
    def banks(self):
        rx=np.repeat(np.asarray([1,3,4,6,8],np.int64),8)
        L=np.zeros((40,160),np.float32);L[:8,0]=1;L[8:,1]=1
        OE=np.zeros_like(L);OE[:8,0]=1;OE[8:,0]=-1
        return L,rx,OE,rx.copy()
    def unit(self,values):
        x=np.asarray(values,np.float64);return (x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32)

    def test_hand_distances_pooled_and_RX_consensus_are_distinct(self):
        m=self.module();state=m.fit(*self.banks());q=np.zeros((3,160),np.float32);q[0,0]=1;q[1,1]=1;q[2,0]=-1
        actual=m.score(q,state);e=1e-6
        expected=np.array([[0.,0.,np.log(4+e)-np.log(2+e)],
            [0.,np.log(2+e)-np.log(e),np.log(2+e)-np.log(e)],
            [-2.,np.log(e)-np.log(2+e),np.log(e)-np.log(2+e)]],np.float64)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
        self.assertEqual(actual.dtype,np.float64);self.assertEqual(m.METHODS,('pooled_known','pooled_relative','rx_consensus_relative'))
        self.assertEqual((m.K,m.EPS,m.RECEIVERS),(8,1e-6,(1,3,4,6,8)))

    def test_exact_eight_neighbors_and_equal_distance_ties_not_one_or_all(self):
        m=self.module();L,rx,OE,orx=self.banks();L[:]=0;L[:,1]=1;L[:7,1]=0;L[:7,0]=1;OE[:]=0;OE[:,1]=1
        q=np.zeros((1,160),np.float32);q[0,0]=1;state=m.fit(L,rx,OE,orx)
        actual=m.score(q,state)
        np.testing.assert_allclose(actual,[[-.25,np.log(2+1e-6)-np.log(.25+1e-6),0.]],rtol=0,atol=1e-14)
        order=np.random.default_rng(38).permutation(40)
        np.testing.assert_array_equal(actual,m.score(q,m.fit(L[order],rx[order],OE[order],orx[order])))

    def test_FP64_working_units_copy_inputs_and_scoring_does_not_mutate_bank(self):
        m=self.module();rng=np.random.default_rng(28);L=self.unit(rng.normal(size=(40,160)));OE=self.unit(rng.normal(size=(40,160)))
        rx=self.banks()[1];original=[v.copy() for v in (L,rx,OE)];state=m.fit(L,rx,OE,rx)
        self.assertEqual(set(state),{'L','OE','L_rx','OE_rx'})
        for key in ('L','OE'):
            self.assertEqual(state[key].dtype,np.float64);np.testing.assert_allclose(np.linalg.norm(state[key],axis=1),1.,rtol=0,atol=3e-16)
        self.assertEqual(state['L_rx'].dtype,np.int64);before=copy.deepcopy(state)
        q=L[:3].copy();q_before=q.copy();m.score(q,state)
        for key in state:np.testing.assert_array_equal(before[key],state[key])
        for values,old in zip((L,rx,OE),original):np.testing.assert_array_equal(values,old)
        np.testing.assert_array_equal(q,q_before)
        L[0]=0;rx[0]=26;np.testing.assert_array_equal(state['L'],before['L']);np.testing.assert_array_equal(state['L_rx'],before['L_rx'])

    def test_128_blocks_row_independence_and_query_and_bank_permutations(self):
        m=self.module();rng=np.random.default_rng(53);rx=self.banks()[1]
        L=self.unit(rng.normal(size=(40,160)));OE=self.unit(rng.normal(size=(40,160)));q=self.unit(rng.normal(size=(259,160)))
        state=m.fit(L,rx,OE,rx);calls=[];real=m._distances
        with patch.object(m,'_distances',side_effect=lambda x,b:calls.append((len(x),len(b))) or real(x,b)):
            expected=m.score(q,state)
        self.assertEqual(calls,[(128,40),(128,40),(128,40),(128,40),(3,40),(3,40)])
        each=np.concatenate([m.score(q[i:i+1],state) for i in range(len(q))]);np.testing.assert_array_equal(each,expected)
        order=rng.permutation(len(q));np.testing.assert_array_equal(m.score(q[order],state),expected[order])
        bank=rng.permutation(40);np.testing.assert_array_equal(m.score(q,m.fit(L[bank],rx[bank],OE[bank],rx[bank])),expected)

    def test_invalid_source_vector_and_RX_contracts_fail(self):
        m=self.module();data=self.banks()
        for defect in ('shape','dtype','zero','norm','nan','receiver','missing_rx','fewer_than_eight'):
            L,rx,OE,orx=[v.copy() for v in data]
            if defect=='shape':L=L[:,:159]
            elif defect=='dtype':L=L.astype(np.float64)
            elif defect=='zero':L[0]=0
            elif defect=='norm':L[0]*=2
            elif defect=='nan':L[0,0]=np.nan
            elif defect=='receiver':rx[0]=26
            elif defect=='missing_rx':rx[rx==8]=6
            else:L=L[:-1];rx=rx[:-1]
            with self.subTest(defect=defect),self.assertRaises(ValueError):m.fit(L,rx,OE,orx)

    def test_score_is_row_local_with_no_RX_labels_K_or_state_extension_interface(self):
        m=self.module();state=m.fit(*self.banks());q=self.banks()[0][:1].copy()
        self.assertEqual(list(inspect.signature(m.fit).parameters),['L_features','L_rx','OE_features','OE_rx'])
        self.assertEqual(list(inspect.signature(m.score).parameters),['query_features','state'])
        with self.assertRaises(TypeError):m.score(q,state,query_rx=np.ones(1))
        with self.assertRaises(TypeError):m.fit(*self.banks(),labels=np.zeros(40))
        for bad in (q.astype(np.float64),q*2,np.full_like(q,np.nan),np.zeros((0,160),np.float32)):
            with self.subTest(shape=bad.shape),self.assertRaises(ValueError):m.score(bad,state)
        with self.assertRaises(ValueError):m.score(q,dict(state,labels=np.zeros(40)))


if __name__=='__main__':unittest.main()
