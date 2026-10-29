"""Small synthetic unlabeled neighborhoods; no IQ, models or true U labels."""
import copy
import importlib
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


class StarNeighborhoodTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_neighborhood_core.py').is_file(),
                        'Unlabeled neighborhood core is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_neighborhood_core')

    @staticmethod
    def data(n=16,negative=False):
        z=np.zeros((2*n,688),np.float32)
        z[:n,0]=-3+np.linspace(-.1,.1,n);z[n:,0]=3+np.linspace(-.1,.1,n)
        logits=np.zeros((2*n,6),np.float32);logits[:n,1]=12;logits[n:,3]=6
        margin=np.full(2*n,-.8 if negative else .6,np.float64)
        ids=np.asarray(['opaque-'+str(i).zfill(3) for i in range(2*n)])
        anchors=np.zeros((90,688),np.float32);anchors[:,0]=100
        am=np.zeros(90,np.float64)
        return z,logits,margin,ids,anchors,am

    def test_all_positive_U_participates_in_actual_clustering_without_pre_rejection(self):
        m=self.module();args=self.data();before=[a.copy() for a in args]
        state,diag=m.fit(*args)
        self.assertEqual(diag['selected_cluster_count'],2);self.assertTrue(diag['all_U_rows_used'])
        self.assertFalse(diag['pre_rejection_applied']);self.assertEqual(len(state['anonymous_indices']),0)
        self.assertTrue(np.all(diag['cluster_labels']>=0));self.assertEqual(diag['cluster_sizes'].tolist(),[16,16])
        self.assertEqual([row['K'] for row in diag['candidates']],list(range(2,6)))
        for old,new in zip(before,args):np.testing.assert_array_equal(old,new)
        self.assertNotIn('y',inspect.signature(m.fit).parameters)
        with self.assertRaises(TypeError):m.fit(*args,y=np.zeros(32))

    def test_ID_sorting_makes_aggregates_and_inference_invariant_to_input_row_order(self):
        m=self.module();args=self.data(negative=True);first,diag=m.fit(*args)
        order=np.random.default_rng(4).permutation(len(args[0]))
        second,other=m.fit(*(a[order] for a in args[:4]),*args[4:])
        self.assertEqual(set(first),{'schema','cluster_embedding','cluster_logits','cluster_margin','cluster_radius','anonymous_indices','anonymous_penalty'})
        for key in first:
            if isinstance(first[key],np.ndarray):np.testing.assert_array_equal(first[key],second[key])
            else:self.assertEqual(first[key],second[key])
        for key in ('cluster_labels','assignments','candidate_mask'):
            np.testing.assert_array_equal(other[key][np.argsort(order)],diag[key])
        self.assertEqual(diag['candidates'],other['candidates'])
        q=np.zeros((1,688),np.float32);q[0,0]=-3
        logits=np.zeros((1,6),np.float32);margin=np.array([-.3],np.float64)
        a=m.infer(q,logits,margin,first,True,True);b=m.infer(q,logits,margin,second,True,True)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_registration_requires_both_ID_alternating_halves_and_uses_LOO_radius(self):
        m=self.module();args=list(self.data(negative=True));args[2][:16:2]=.2;args[2][1:16:2]=-1.2
        state,diag=m.fit(*args)
        self.assertEqual(state['anonymous_indices'].tolist(),[1]);self.assertEqual(int(diag['candidate_mask'].sum()),16)
        self.assertTrue(np.all(diag['assignments'][:16]==-1));self.assertTrue(np.all(diag['assignments'][16:]==0))
        self.assertGreater(diag['clusters'][0]['half_mean_margins'][0],0)
        z=args[0][:16].astype(np.float64);loo=(z.sum(0)-z)/15
        expected=max(float(np.quantile(np.linalg.norm(z-loo,axis=1),.95)),1e-6)
        self.assertEqual(state['cluster_radius'][0],expected)
        args[2][:16:2]=0.  # A zero half-mean is not strictly negative.
        boundary,_=m.fit(*args);self.assertEqual(boundary['anonymous_indices'].tolist(),[1])
        small,_=m.fit(*self.data(n=15,negative=True));self.assertEqual(len(small['anonymous_indices']),0)

    def test_each_anonymous_penalty_uses_all_source_anchors_and_their_tanh_margins(self):
        m=self.module();args=list(self.data(negative=True));args[4][0,0]=-3;args[5][0]=-1.
        anchor_before=args[4].copy();margin_before=args[5].copy();state,_=m.fit(*args)
        for slot,cluster in enumerate(state['anonymous_indices']):
            distance=np.linalg.norm(args[4].astype(np.float64)-state['cluster_embedding'][cluster],axis=1)
            expected=.05+max(0.,float(np.max(1-distance/state['cluster_radius'][cluster]-np.tanh(args[5]))))
            self.assertEqual(state['anonymous_penalty'][slot],expected)
        self.assertGreater(state['anonymous_penalty'][0],1.)
        np.testing.assert_array_equal(args[4],anchor_before);np.testing.assert_array_equal(args[5],margin_before)

    def test_DA_fuses_only_inside_nearest_cluster_radius_and_keeps_state_frozen(self):
        m=self.module();state,_=m.fit(*self.data());before=copy.deepcopy(state)
        q=np.zeros((2,688),np.float32);q[:,0]=[-3,1000]
        logits=np.zeros((2,6),np.float32);logits[:,0]=10.;margin=np.full(2,-.2,np.float64)
        raw=m.infer(q,logits,margin,state,False,False);adapt=m.infer(q,logits,margin,state,True,False)
        np.testing.assert_array_equal(m.fusion_mask(q,state),[True,False])
        self.assertEqual(raw['prediction'].tolist(),[-1,-1]);self.assertEqual(adapt['closed'].tolist(),[1,0])
        self.assertEqual(adapt['prediction'].tolist(),[1,-1]);self.assertAlmostEqual(adapt['knownness'][0],np.tanh(.2))
        np.testing.assert_array_equal(adapt['joint_margins'][1],raw['joint_margins'][1])
        np.testing.assert_array_equal(adapt['joint_margins'].max(1),adapt['knownness'])
        for key in state:
            if isinstance(state[key],np.ndarray):np.testing.assert_array_equal(state[key],before[key])

    def test_REG1_all_old_and_anonymous_compete_with_zero_acceptance_and_old_tie_priority(self):
        m=self.module();state,_=m.fit(*self.data(negative=True))
        q=np.zeros((3,688),np.float32);q[:,0]=[-3,-3,1000]
        logits=np.zeros((3,6),np.float32);logits[:,4]=30
        margin=np.array([10.,-10.,-10.],np.float64)
        out=m.infer(q,logits,margin,state,False,True)
        self.assertEqual(out['joint_margins'].shape,(3,8));self.assertEqual(out['prediction'].tolist(),[4,6,-1])
        reg0=m.infer(q,logits,margin,state,False,False)
        self.assertEqual(reg0['joint_margins'].shape,(3,6));self.assertEqual(reg0['prediction'].tolist(),[4,-1,-1])
        tied=dict(schema=m.STATE_SCHEMA,cluster_embedding=np.zeros((1,688),np.float64),cluster_logits=np.zeros((1,6),np.float64),
            cluster_margin=np.zeros(1,np.float64),cluster_radius=np.ones(1,np.float64),anonymous_indices=np.array([0],np.int64),anonymous_penalty=np.ones(1,np.float64))
        out=m.infer(np.zeros((1,688),np.float32),np.zeros((1,6),np.float32),np.zeros(1,np.float64),tied,False,True)
        self.assertEqual(out['prediction'].tolist(),[0]);self.assertEqual(out['joint_margins'].max(),0.)

    def test_silhouette_exact_tie_prefers_smaller_K_and_no_legal_partition_is_identity(self):
        m=self.module();args=self.data(n=32)
        with patch.object(m,'_silhouette_score',return_value=.25):state,diag=m.fit(*args)
        self.assertEqual(diag['selected_cluster_count'],2)
        self.assertGreaterEqual(sum(row['eligible'] for row in diag['candidates']),2)
        small=tuple(a[:7] for a in args[:4])+args[4:];empty,diagnostic=m.fit(*small)
        self.assertEqual(diagnostic['selected_cluster_count'],0);self.assertEqual(len(empty['cluster_radius']),0)
        q=np.zeros((2,688),np.float32);l=np.zeros((2,6),np.float32);l[:,2]=1
        raw=np.array([0.,-1.],np.float64);out=m.infer(q,l,raw,empty,True,True)
        self.assertEqual(out['prediction'].tolist(),[2,-1]);np.testing.assert_array_equal(out['knownness'],np.tanh(raw))

    def test_nonfinite_shape_dtype_duplicate_ID_and_bad_state_are_rejected(self):
        m=self.module()
        for defect in ('embedding','logits','margin','ids','anchors','anchor_margin'):
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                args=list(self.data())
                if defect=='embedding':args[0]=args[0][:,:687]
                elif defect=='logits':args[1]=args[1].astype(np.float64)
                elif defect=='margin':args[2][0]=np.nan
                elif defect=='ids':args[3][0]=args[3][1]
                elif defect=='anchors':args[4]=args[4][:89]
                else:args[5]=args[5].astype(np.float32)
                m.fit(*args)
        state,_=m.fit(*self.data());q=np.zeros((1,688),np.float32);l=np.zeros((1,6),np.float32);margin=np.zeros(1,np.float64)
        with self.assertRaises(ValueError):m.infer(q,l,margin,state,1,False)
        changed=copy.deepcopy(state);changed['cluster_radius'][0]=-1
        with self.assertRaises(ValueError):m.infer(q,l,margin,changed,True,False)
        with self.assertRaises(TypeError):m.infer(q,l,margin,state,False,False,y=np.array([0]))


if __name__=='__main__':unittest.main()
