"""Synthetic pure-NumPy evidence for irreversible source-domain projections."""
import copy
import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.pamf_phase3_open_world import source_open_readout as geometry_core
from experiments.pamf_phase3_open_world import star_relative_core as relative
from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS


class StarNuisanceCoreTests(unittest.TestCase):
    def module(self):
        p=Path(__file__).resolve().parents[1]/'experiments/pamf_phase3_open_world/star_nuisance_core.py'
        self.assertTrue(p.is_file(),'Source nuisance projection core is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_nuisance_core')

    def fixture(self,kind='random'):
        records=[(c,r,d,j) for c in range(6) for r in (1,3,4,6,8) for d in (1,2,3) for j in range(3)]
        y=np.array([x[0] for x in records],np.int64);rx=np.array([x[1] for x in records],np.int64);day=np.array([x[2] for x in records],np.int64)
        geometry=dict(schema=geometry_core.GEOMETRY_SCHEMA,class_count=6,branches={name:dict(mean=np.zeros(n),scale=np.ones(n),
            prototypes=np.eye(6,n),whitener=np.eye(n)) for name,n in DIMENSIONS.items()})
        values={}
        for name,n in DIMENSIONS.items():
            z=np.zeros((len(y),n),np.float32);z[np.arange(len(y)),y]=1.
            if kind=='random':
                z+=np.random.default_rng(n+29).normal(0,.08,z.shape).astype(np.float32)
                z[:,7]+=np.where(rx<=3,-.2,.2)
            elif kind=='domain':z[:,7]=np.where(rx<=3,-.4,.4)
            elif kind=='day':z[:,7]=np.where(day==1,-.4,.4)
            elif kind!='constant':raise AssertionError(kind)
            if name=='reference29':z[:,27]=0
            values[name]=z
        return values,y,rx,day,geometry

    def test_control_is_exact_existing_MD_RMD_and_does_not_mutate_inputs(self):
        m=self.module();b,y,rx,day,g=self.fixture();before=copy.deepcopy((b,g))
        old=relative.fit_metric(b,y,g)
        with patch('builtins.open',side_effect=AssertionError('Pure core must not read files')):
            state=m.fit_metric(b,y,rx,day,g,'control')
            actual=m.metric_scores(b,g,state)
        np.testing.assert_array_equal(actual,relative.metric_scores(b,g,old))
        for key in ('score_mean','score_scale'):np.testing.assert_array_equal(state[key],old[key])
        for name in relative.BRANCHES:
            np.testing.assert_array_equal(state['branches'][name]['projection'],np.eye(DIMENSIONS[name]))
            for key in old['branches'][name]:np.testing.assert_array_equal(state['branches'][name][key],old['branches'][name][key])
        for name in b:np.testing.assert_array_equal(b[name],before[0][name])
        for name in g['branches']:
            for key in g['branches'][name]:np.testing.assert_array_equal(g['branches'][name][key],before[1]['branches'][name][key])

    def test_pure_RX_direction_removed_and_between_class_span_preserved(self):
        m=self.module();b,y,rx,day,g=self.fixture('domain');state=m.fit_metric(b,y,rx,day,g,'rx')
        units=relative._units(b,g)
        for name in relative.BRANCHES:
            p=state['branches'][name]['projection'];u=units[name]
            self.assertEqual(state['diagnostics'][name]['rank'],1)
            self.assertAlmostEqual(state['diagnostics'][name]['energy_coverage'],1.,places=12)
            np.testing.assert_allclose(p@p,p,atol=2e-12);np.testing.assert_allclose(p,p.T,atol=2e-12)
            domain=np.eye(len(p))[7];self.assertLess(np.linalg.norm(domain@p),1e-12)
            means=np.array([u[y==c].mean(0) for c in range(6)]);centered=means-means.mean(0)
            np.testing.assert_allclose(centered@p,centered,atol=2e-12)
            q=u@p;q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
            for c in range(6):self.assertLess(np.max(np.linalg.norm(q[y==c]-q[y==c].mean(0),axis=1)),1e-12)
            self.assertLess(state['diagnostics'][name]['between_class_damage'],1e-12)
            self.assertEqual(state['diagnostics'][name]['remaining_unit_norm_anomalies'],0)

    def test_day_only_signal_is_not_confused_with_RX_only(self):
        m=self.module();b,y,rx,day,g=self.fixture('day')
        a=m.fit_metric(b,y,rx,day,g,'rx');c=m.fit_metric(b,y,rx,day,g,'rxday')
        for name in relative.BRANCHES:
            self.assertEqual(a['diagnostics'][name]['rank'],0)
            self.assertEqual(c['diagnostics'][name]['rank'],1)

    def test_class_label_permutation_keeps_projection_and_scores(self):
        m=self.module();b,y,rx,day,g=self.fixture()
        original=m.fit_metric(b,y,rx,day,g,'rxday')
        permutation=np.array([2,5,1,4,0,3]);other=m.fit_metric(b,permutation[y],rx,day,g,'rxday')
        for name in relative.BRANCHES:
            np.testing.assert_allclose(original['branches'][name]['projection'],other['branches'][name]['projection'],rtol=0,atol=2e-12)
        np.testing.assert_allclose(m.metric_scores(b,g,original),m.metric_scores(b,g,other),rtol=2e-10,atol=2e-10)

    def test_zero_domain_residual_is_exact_identity_not_an_arbitrary_SVD_basis(self):
        m=self.module();b,y,rx,day,g=self.fixture('constant');reference=relative.fit_metric(b,y,g)
        for variant in ('rx','rxday'):
            with self.subTest(variant=variant):
                state=m.fit_metric(b,y,rx,day,g,variant)
                for name in relative.BRANCHES:
                    self.assertEqual(state['diagnostics'][name]['rank'],0)
                    self.assertIsNone(state['diagnostics'][name]['energy_coverage'])
                    np.testing.assert_array_equal(state['branches'][name]['projection'],np.eye(DIMENSIONS[name]))
                np.testing.assert_array_equal(m.metric_scores(b,g,state),relative.metric_scores(b,g,reference))

    def test_rank_cap_energy_and_no_source_rows_in_flat_roundtrip(self):
        m=self.module();b,y,rx,day,g=self.fixture();state=m.fit_metric(b,y,rx,day,g,'rxday')
        for diag in state['diagnostics'].values():
            self.assertLessEqual(diag['rank'],16)
            if diag['energy_coverage'] is not None:
                self.assertAlmostEqual(diag['captured_energy']/diag['residual_energy'],diag['energy_coverage'],places=12)
                self.assertTrue(diag['energy_coverage']>=.9-1e-12 or diag['rank']==16)
        arrays=m.to_arrays(state);loaded=m.from_arrays(arrays)
        self.assertFalse(any(x in key for key in arrays for x in ('source_rows','labels','features','ids')))
        self.assertFalse(any(a.dtype.kind=='O' for a in arrays.values()))
        np.testing.assert_array_equal(m.metric_scores(b,g,state),m.metric_scores(b,g,loaded))
        bad=dict(arrays,source_rows=np.zeros((len(y),160)))
        with self.assertRaises(ValueError):m.from_arrays(bad)

    def test_bad_shapes_labels_domains_and_nonfinite_values_rejected(self):
        m=self.module()
        for mutation in ('variant','y_unknown','y_shape','rx_target','rx_float','day_unknown','day_nan','branch_nan','geometry_nan','missing_branch','missing_class_domain'):
            with self.subTest(mutation=mutation):
                b,y,rx,day,g=self.fixture();variant='rxday'
                if mutation=='variant':variant='oracle'
                elif mutation=='y_unknown':y[0]=6
                elif mutation=='y_shape':y=y[:-1]
                elif mutation=='rx_target':rx[0]=26
                elif mutation=='rx_float':rx=rx.astype(float)
                elif mutation=='day_unknown':day[0]=4
                elif mutation=='day_nan':day=day.astype(float);day[0]=np.nan
                elif mutation=='branch_nan':b['fused160'][0,0]=np.nan
                elif mutation=='geometry_nan':g['branches']['time160']['scale'][0]=np.nan
                elif mutation=='missing_branch':b.pop('pa160')
                else:
                    keep=~((y==0)&(rx==1)&(day==1));b={k:v[keep] for k,v in b.items()};y,rx,day=y[keep],rx[keep],day[keep]
                with self.assertRaises(ValueError):m.fit_metric(b,y,rx,day,g,variant)


if __name__=='__main__':unittest.main()
