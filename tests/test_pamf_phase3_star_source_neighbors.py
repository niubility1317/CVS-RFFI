"""Pure synthetic direct-edge mining and source-only distance calibration."""
import copy
import importlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'experiments/pamf_phase3_open_world/star_source_neighbors.py').is_file(), 'Source neighbor core is not implemented'
    return importlib.import_module('experiments.pamf_phase3_open_world.star_source_neighbors')


def vectors(rows):
    x=np.zeros((len(rows),160),np.float32);x[:,:len(rows[0])]=np.asarray(rows,np.float32);return x


def calibration_fixture(blocks=7,early_blocks=None):
    # Five receiver nodes per block, three dates. Only valid cross-day edges;
    # max degree four, like a per-other-RX mutual graph, never clique labels.
    receivers=(1,3,4,6,8);days=(1,2,3,1,2);stride=5*blocks+2
    y=[];rx=[];pairs=[];distances=[]
    for label in range(6):
        y.extend([label]*stride);rx.extend(list(receivers)*blocks+[1,3])
        for block in range(blocks):
            offset=label*stride+block*5
            for i in range(5):
                for j in range(i+1,5):
                    if days[i]!=days[j]:
                        pairs.append((offset+i,offset+j));distances.append(.1 if early_blocks is None or block<early_blocks else .3)
    return dict(pair_indices=np.asarray(pairs,np.int64),distances=np.asarray(distances,np.float64),rule='synthetic_source_edges'),np.asarray(y,np.int64),np.asarray(rx,np.int64),stride


def add_wrong(candidate,edges,distance=.2):
    result=copy.deepcopy(candidate);result['pair_indices']=np.vstack((result['pair_indices'],np.sort(np.asarray(edges,np.int64),axis=1)))
    result['distances']=np.r_[result['distances'],np.full(len(edges),distance,np.float64)];return result


class StarSourceNeighborTests(unittest.TestCase):
    def test_cross_RX_cross_day_per_RX_mutuality_and_two_view_consistency(self):
        m=module();ids=np.asarray(['z','a','b','c']);rx=np.asarray([1,3,1,3]);day=np.asarray([1,2,3,1])
        z1=vectors([[1,0],[1,0],[0,1],[-1,0]]);before=z1.copy()
        result=m.candidates(ids,z1,rx,day);np.testing.assert_array_equal(result['pair_indices'],[[0,1]])
        self.assertEqual(result['distances'].dtype,np.float64);self.assertEqual(result['pair_indices'].dtype,np.int64)
        z0=z1.copy();z0[1]=z1[2]
        self.assertEqual(len(m.candidates(ids,z1,rx,day,z0=z0)['pair_indices']),0)
        np.testing.assert_array_equal(m.candidates(ids,z1,rx,day,z0=z1)['pair_indices'],[[0,1]])
        np.testing.assert_array_equal(z1,before)
        triangle=m.candidates(np.asarray(['a','b','c']),vectors([[1,0]]*3),np.asarray([1,3,4]),np.asarray([1,2,3]))
        self.assertEqual(len(triangle['pair_indices']),3)  # Per-RX, not one global nearest neighbor.
        self.assertEqual(m.candidates(ids,z1,rx,np.ones(4,int))['pair_indices'].shape,(0,2))

    def test_pair_distance_is_worst_view_and_ties_sort_by_opaque_ID_under_row_permutation(self):
        m=module();p1=vectors([[1,0],[.8,.6]]);p0=vectors([[1,0],[.6,.8]])
        c=m.candidates(np.asarray(['a','b']),p1,np.asarray([1,3]),np.asarray([1,2]),z0=p0)
        self.assertAlmostEqual(c['distances'][0],.4,places=7)
        ids=np.asarray(['z','a','b','c','d','e']);rx=np.asarray([1,1,1,3,3,3]);day=np.asarray([1,1,2,2,2,1])
        x=vectors([[1,0]]*6);first=m.candidates(ids,x,rx,day,block_size=2)
        semantic=lambda data,index:sorted(tuple(sorted(ids[index][edge])) for edge in data['pair_indices'])
        self.assertEqual(semantic(first,np.arange(6)),[('a','c'),('b','e')])
        permutation=np.asarray([5,0,4,2,1,3]);second=m.candidates(ids[permutation],x[permutation],rx[permutation],day[permutation],block_size=1)
        self.assertEqual(semantic(first,np.arange(6)),semantic(second,permutation))
        self.assertTrue(np.all(first['pair_indices'][:,0]<first['pair_indices'][:,1]))
        self.assertEqual([tuple(sorted(ids[p])) for p in first['pair_indices']],[('a','c'),('b','e')])

    def test_complete_distance_ties_and_all_TX_precision_constraints_choose_max_legal_threshold(self):
        m=module();base,y,rx,stride=calibration_fixture()
        wrong=add_wrong(base,[(5*7,stride+5*7+1),(5*7+1,stride+5*7)])
        result=m.calibrate(wrong,y,rx)
        self.assertEqual(result['threshold'],.1);self.assertTrue(result['constraints_passed'])
        self.assertEqual(len(m.apply(wrong,result['threshold'])),len(base['pair_indices']))
        full=result['full_candidate_groups'];self.assertTrue(full['global']['constraints_passed'])
        self.assertTrue(all(full['RX/'+str(r)]['constraints_passed'] for r in m.RECEIVERS))
        self.assertFalse(full['TX/0']['constraints_passed']);self.assertFalse(full['TX/1']['constraints_passed'])
        self.assertEqual(result['summary']['edge_count'],len(base['pair_indices']));json.dumps(result,allow_nan=False)

    def test_RX_group_constraints_cannot_hide_behind_global_or_TX_precision(self):
        m=module();base,y,rx,stride=calibration_fixture();end=35
        edges=[(a*stride+end,b*stride+end+1) for a,b in ((0,1),(2,3),(4,5))]
        result=m.calibrate(add_wrong(base,edges),y,rx);full=result['full_candidate_groups']
        self.assertEqual(result['threshold'],.1);self.assertTrue(full['global']['constraints_passed'])
        self.assertTrue(all(full['TX/'+str(c)]['constraints_passed'] for c in range(6)))
        self.assertFalse(full['RX/1']['constraints_passed'])

    def test_nonmonotone_precision_recovery_and_no_feasible_threshold_never_accept_all(self):
        m=module();base,y,rx,stride=calibration_fixture(blocks=16,early_blocks=4);end=80
        candidate=add_wrong(base,[(end,stride+end+1),(end+1,stride+end)])
        result=m.calibrate(candidate,y,rx);self.assertEqual(result['threshold'],.3)
        base,y,rx,stride=calibration_fixture();bad=add_wrong(base,[(35,stride+36),(36,stride+35)],distance=.1)
        report=m.calibrate(bad,y,rx);self.assertIsNone(report['threshold']);self.assertFalse(report['constraints_passed'])
        self.assertEqual(m.apply(bad,report['threshold']).shape,(0,2))
        empty=dict(pair_indices=np.empty((0,2),np.int64),distances=np.empty(0,np.float64),rule='empty')
        self.assertIsNone(m.calibrate(empty,y,rx)['threshold']);self.assertEqual(m.apply(empty,.1).shape,(0,2))

    def test_summary_reports_direct_edge_coverage_degrees_and_components_as_diagnostics_only(self):
        m=module();ids=np.asarray(['a','b','c']);rx=np.asarray([1,3,4]);day=np.asarray([1,2,3]);pairs=np.asarray([[0,1],[1,2]],np.int64)
        row=m.summary(ids,rx,day,pairs)
        self.assertEqual(row['edge_count'],2);self.assertEqual(row['covered_node_fraction'],1.)
        self.assertEqual(row['by_RX']['1']['covered_node_fraction'],1.)
        self.assertEqual(row['degree']['maximum'],2);self.assertAlmostEqual(row['degree']['mean'],4/3)
        self.assertEqual(row['component_sizes'],[3]);self.assertTrue(row['all_edges_cross_RX']);self.assertTrue(row['all_edges_cross_day'])
        self.assertFalse(row['components_are_identity_labels'])
        empty=m.summary(ids,rx,day,np.empty((0,2),np.int64));self.assertEqual(empty['covered_node_fraction'],0.)
        self.assertEqual(empty['component_sizes'],[1,1,1]);self.assertEqual(empty['by_RX']['3']['covered_node_fraction'],0.)

    def test_input_validation_and_OE_interfaces_have_no_label_or_trueK_parameter(self):
        m=module();ids=np.asarray(['a','b']);x=vectors([[1,0],[0,1]]);rx=np.asarray([1,3]);day=np.asarray([1,2])
        for defect in ('duplicate','nan','norm','dtype','dimension','rx','day','z0','block'):
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                a=ids.copy();z=x.copy();r=rx.copy();d=day.copy();z0=None;block=128
                if defect=='duplicate':a[:]='a'
                elif defect=='nan':z[0,0]=np.nan
                elif defect=='norm':z[0]*=2
                elif defect=='dtype':z=z.astype(np.float64)
                elif defect=='dimension':z=z[:,:159]
                elif defect=='rx':r[0]=26
                elif defect=='day':d[0]=4
                elif defect=='z0':z0=x[:1]
                else:block=0
                m.candidates(a,z,r,d,z0=z0,block_size=block)
        candidate=m.candidates(ids,x,rx,day);before=copy.deepcopy(candidate)
        accepted=m.apply(candidate,2.);accepted[:]=0
        np.testing.assert_array_equal(candidate['pair_indices'],before['pair_indices'])
        with self.assertRaises(TypeError):m.candidates(ids,x,rx,day,y=np.zeros(2))
        with self.assertRaises(TypeError):m.apply(candidate,.2,y=np.zeros(2))
        self.assertEqual(tuple(inspect.signature(m.candidates).parameters),('ids','z1','rx','day','z0','block_size'))
        with self.assertRaises(ValueError):m.calibrate(candidate,np.asarray([0,6]),rx)


if __name__=='__main__':unittest.main()
