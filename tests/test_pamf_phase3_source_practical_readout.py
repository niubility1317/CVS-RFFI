"""Synthetic-only contracts for the four-view source rejection experiment."""
import contextlib
from fractions import Fraction
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


def module():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_practical_readout')


def state():
    from experiments.pamf_phase3_open_world import source_open_readout as core
    return dict(schema=core.READOUT_SCHEMA, descriptor_dim=100, method_names=list(core.METHODS),
        mean=np.zeros(100), scale=np.ones(100), models={
            'linear_oe': dict(weight=np.ones((1, 100), np.float32), bias=np.zeros(1, np.float32)),
            'mlp16_oe': dict(weight1=np.ones((16, 100), np.float32) / 100,
                bias1=np.zeros(16, np.float32), weight2=np.ones((1, 16), np.float32),
                bias2=np.zeros(1, np.float32))}, training={})


def geometry():
    from experiments.pamf_phase3_open_world import source_open_readout as core
    from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
    return dict(schema=core.GEOMETRY_SCHEMA, class_count=6, branches={
        name: dict(mean=np.zeros(d), scale=np.ones(d), prototypes=np.zeros((6, d)),
            whitener=np.eye(d)) for name, d in DIMENSIONS.items()})


def calibration_data(n_per_class=10):
    ids=[]; y=[]; rx=[]; views=[]; score=[]
    for v in range(4):
        for r in (1, 3, 4, 6, 8):
            for k in range(6):
                for q in range(n_per_class):
                    ids.append(f'r{r}-k{k}-q{q}')
                    y.append(k); rx.append(r); views.append(v)
                    score.append(float((q * 3 + k + r + v) % 13))
    y=np.asarray(y, np.int64)
    return dict(ids=np.asarray(ids), y=y, closed_new=y.copy(), closed_original=y.copy(),
        knownness=np.repeat(np.asarray(score)[:, None], 7, axis=1),
        rx_i=np.asarray(rx, np.int64), view_i=np.asarray(views, np.int64))


class SourcePracticalReadoutTests(unittest.TestCase):
    def test_exact_global_threshold_ties_and_all_cross_groups_match_bruteforce(self):
        m=module(); x=calibration_data()
        # Some false closed predictions have no correct-rejection cost, while
        # the original baseline agrees; feasibility must retain exact budgets.
        x['closed_new'][::71]=(x['y'][::71]+1)%6
        result=m.calibrate_known(**x)
        masks=[np.ones(len(x['y']),bool)]
        masks += [x['rx_i']==r for r in m.RECEIVERS]
        masks += [x['view_i']==v for v in range(4)]
        masks += [(x['rx_i']==r)&(x['view_i']==v) for r in m.RECEIVERS for v in range(4)]
        def legal(t):
            for mask in masks:
                frr=Fraction(); loss=Fraction()
                for k in range(6):
                    a=mask&(x['y']==k); n=int(a.sum())
                    rejected=a&(x['knownness'][:,0]<t)
                    frr += Fraction(int(rejected.sum()),n)/6
                    old=int((a&(x['closed_original']==k)).sum())
                    opened=int((a&(x['closed_new']==k)&~rejected).sum())
                    loss += Fraction(old-opened,n)/6
                if frr>Fraction(1,20) or loss>Fraction(1,50): return False
            return True
        levels=np.unique(x['knownness'][:,0])
        candidates=np.r_[levels,np.nextafter(levels[-1],np.inf)]
        safe=[t for t in candidates if legal(t)]
        expected=max(safe) if safe else min(levels)
        self.assertTrue(np.all(result['thresholds']==expected))
        row=result['methods']['mlp16_oe']
        self.assertEqual(len(row['by_receiver_view']),5)
        self.assertEqual(sum(len(a) for a in row['by_receiver_view'].values()),20)
        self.assertFalse(row['next_candidate_passed'])
        # A tie is either fully rejected or fully accepted at every group.
        for t in levels:
            accepted=x['knownness'][x['knownness'][:,0]==t,0]>=expected
            self.assertEqual(len(np.unique(accepted)),1)

    def test_zero_rejection_infeasible_keeps_all_and_does_not_claim_pass(self):
        m=module(); x=calibration_data()
        x['closed_new']=(x['y']+1)%6
        out=m.calibrate_known(**x)
        self.assertFalse(out['constraints_passed'])
        for z in out['methods'].values():
            self.assertFalse(z['no_rejection_feasible'])
            self.assertEqual(z['threshold'],float(x['knownness'].min()))
            self.assertEqual(z['overall']['rejected_count'],0)
            self.assertFalse(z['overall']['constraints_passed'])

    def test_cross_cell_can_bind_global_threshold_and_nonempty_legal_rejection(self):
        m=module();x=calibration_data(100)
        x['knownness'][:]=10.
        # Two correct rows in one RX/view each have mass 1/600. Both are
        # admissible; a following 20-row tie exceeds that cell's 2pp budget,
        # despite being harmless in the pooled overall group.
        mask=(x['rx_i']==1)&(x['view_i']==0)
        positions=np.flatnonzero(mask)
        x['knownness'][positions[:2]]=0.
        x['knownness'][positions[2:22]]=1.
        out=m.calibrate_known(**x)
        np.testing.assert_array_equal(out['thresholds'],np.ones(7))
        row=out['methods']['mlp16_oe']
        self.assertTrue(row['no_rejection_feasible']);self.assertTrue(row['constraints_passed'])
        self.assertEqual(row['overall']['rejected_count'],2)
        self.assertFalse(row['next_candidate_passed'])
        self.assertEqual(row['by_receiver_view']['1'][m.VIEWS[0]]['rejected_count'],2)

    def test_encoder_half_point_is_diagnostic_not_a_new_hard_gate(self):
        m=module(); x=calibration_data(100)
        # Exactly 1% encoder damage in each group: above .5pp, below 2pp.
        mask=np.arange(len(x['y']))%100==0
        x['closed_new'][mask]=(x['y'][mask]+1)%6
        x['knownness'][:]=0.
        out=m.calibrate_known(**x)
        self.assertTrue(out['constraints_passed'])
        self.assertFalse(out['encoder_protection_diagnostic_passed'])

    def test_calibration_rejects_unknown_labels_missing_cross_cells_and_extra_truth(self):
        m=module()
        for which in ('unknown','receiver','view','duplicate'):
            with self.subTest(which=which):
                x=calibration_data()
                if which=='unknown':x['y'][0]=6
                elif which=='receiver':x['rx_i'][0]=26
                elif which=='view':x['view_i'][0]=4
                else:x['ids'][1]=x['ids'][0]
                with self.assertRaises(ValueError):m.calibrate_known(**x)
        with self.assertRaises(TypeError):m.calibrate_known(**calibration_data(),unknown_count=3)

    def test_compact_scoring_fixed_seven_order_and_no_state_mutation(self):
        m=module(); s=state(); before={k:v.copy() for k,v in s.items() if isinstance(v,np.ndarray)}
        d=np.ones((3,100),np.float32); static=np.arange(15,dtype=np.float64).reshape(3,5)
        z=m.score_compact(d,static,s)
        self.assertEqual(z.shape,(3,7)); np.testing.assert_array_equal(z[:,:5],static)
        for k,v in before.items():np.testing.assert_array_equal(s[k],v)
        for bad in (np.zeros((3,99),np.float32),np.full((3,100),np.nan,np.float32)):
            with self.subTest(shape=bad.shape),self.assertRaises(ValueError):m.score_compact(bad,static,s)

    def _fake_export(self,m):
        roles={}; n=30
        for role in m.COUNTS:
            roles[role]={}
            for v,view in enumerate(m.VIEWS):
                y=np.tile(np.arange(6),5).astype(np.int64)
                ids=np.asarray([f'{role}-{i}' for i in range(n)])
                logits=np.eye(6,dtype=np.float32)[y]*2
                value=dict(ids=ids,descriptor=np.full((n,100),v+(1 if role=='source_L' else -1),np.float32),
                    static_scores=np.full((n,5),v+1.,np.float64),logits=logits,
                    baseline_closed=y.copy() if role in ('source_V','novel_validation') else np.full(n,-2,np.int64),
                    rx_i=np.repeat(np.asarray(m.RECEIVERS,np.int64),6),day_i=np.zeros(n,np.int64))
                if role in ('source_L','source_V'):value['y']=y
                roles[role][view]=value
        marker=dict(status=m.EXPORT_STATUS,schema=m.EXPORT_SCHEMA,counts={k:n for k in m.COUNTS},
            view_order=list(m.VIEWS),source_only=True,target_reads=0,descriptor_schema=m.DESCRIPTOR_SCHEMA,
            encoder_state_identity={'run_id':'synthetic-E51'},parent_export_identity={'baseline':'synthetic-F0'},
            data_refs={r:{v:f'{r}/{v}/data.npz' for v in m.VIEWS} for r in m.COUNTS})
        return marker,roles

    def _patch_export(self,m,marker,roles,events):
        stack=contextlib.ExitStack()
        stack.enter_context(mock.patch.object(m,'COUNTS',marker['counts']))
        stack.enter_context(mock.patch.object(m,'_export_marker',return_value=marker))
        def load(root,role,view,include_y=False):
            events.append(('load',role,view));return {k:v.copy() for k,v in roles[role][view].items()}
        stack.enter_context(mock.patch.object(m,'_load_role_view',side_effect=load))
        return stack

    def test_fit_only_L_OE_copies_control_and_freezes_roundtrip_bundle(self):
        m=module(); marker,roles=self._fake_export(m); events=[]; oldstate=state(); geo=geometry()
        oldmarker=dict(encoder_state_identity=marker['encoder_state_identity'],parent_export_identity=marker['parent_export_identity'])
        def fitted(a,b,seed):
            self.assertEqual(a.shape,(120,100));self.assertEqual(b.shape,(120,100))
            self.assertEqual(seed,2026092701);events.append(('fit',))
            out=state();out['mean']=np.r_[np.ones(50),np.zeros(50)]
            out['training']=dict(steps=200,known_count=120,oe_count=120,
                parameter_counts={'linear_oe':101,'mlp16_oe':1633},losses={name:[.5]*200 for name in ('linear_oe','mlp16_oe')})
            return out
        with tempfile.TemporaryDirectory() as td,self._patch_export(m,marker,roles,events), \
                mock.patch.object(m,'_old_source',return_value=(oldmarker,geo,oldstate)), \
                mock.patch.object(m.core,'fit_readouts',side_effect=fitted):
            root=Path(td); m.fit('export','old-fit',root/'fit')
            self.assertEqual({e[1] for e in events if e[0]=='load'},{'source_L','exposure_train'})
            self.assertEqual(sum(e[0]=='fit' for e in events),1)
            self.assertTrue((root/'fit/complete.json').exists())
            m.calibrate('export',root/'fit',root/'calibration')
            self.assertFalse(any(e[0]=='load' and e[1]=='novel_validation' for e in events))
            before=list(events)
            for arm in m.ARMS:
                bundle=m.load_bundle(root/'fit',root/'calibration',arm)
                self.assertEqual(bundle['thresholds'].shape,(7,))
                np.testing.assert_array_equal(bundle['geometry']['branches']['fused160']['mean'],geo['branches']['fused160']['mean'])
                expected=oldstate['mean'] if arm=='calibration_only' else np.r_[np.ones(50),np.zeros(50)]
                np.testing.assert_array_equal(bundle['state']['mean'],expected)
            self.assertEqual(events,before)  # aggregate-only deployment loader

    def test_unknown_role_label_is_rejected_and_source_mutation_blocks_complete(self):
        m=module(); marker,roles=self._fake_export(m); events=[]
        roles['exposure_train'][m.VIEWS[0]]['y']=np.zeros(30,np.int64)
        with tempfile.TemporaryDirectory() as td,self._patch_export(m,marker,roles,events), \
                mock.patch.object(m,'_old_source',return_value=(dict(encoder_state_identity=marker['encoder_state_identity'],parent_export_identity=marker['parent_export_identity']),geometry(),state())), \
                mock.patch.object(m.core,'fit_readouts') as fitted:
            with self.assertRaises(ValueError):m.fit('export','old',Path(td)/'fit')
            fitted.assert_not_called()
            self.assertFalse((Path(td)/'fit/complete.json').exists())

    def test_source_geometry_mutation_prevents_fit_completion(self):
        m=module();marker,roles=self._fake_export(m);events=[];geo=geometry()
        def mutates(a,b,seed):
            geo['branches']['time160']['mean'][0]=1.
            s=state();s['training']=dict(steps=200,known_count=len(a),oe_count=len(b),
                parameter_counts={'linear_oe':101,'mlp16_oe':1633},losses={k:[.5]*200 for k in ('linear_oe','mlp16_oe')})
            return s
        with tempfile.TemporaryDirectory() as td,self._patch_export(m,marker,roles,events), \
                mock.patch.object(m,'_old_source',return_value=(dict(encoder_state_identity=marker['encoder_state_identity'],parent_export_identity=marker['parent_export_identity']),geo,state())), \
                mock.patch.object(m.core,'fit_readouts',side_effect=mutates):
            with self.assertRaisesRegex(ValueError,'changed'):m.fit('export','old',Path(td)/'fit')
            self.assertFalse((Path(td)/'fit/complete.json').exists())

    def test_novel_waits_for_both_arm_calibration_and_truth_waits_for_all_views(self):
        m=module()
        with tempfile.TemporaryDirectory() as td, mock.patch.object(m,'_load_role_view') as load, \
                mock.patch.object(m,'_novel_truth') as truth:
            root=Path(td)
            with self.assertRaises((OSError,ValueError)):m.predict_source('export',root/'fit',root/'cal',root/'pred')
            load.assert_not_called()
            with self.assertRaises((OSError,ValueError)):m.score_source(root/'pred','private-truth',root/'score')
            truth.assert_not_called()

    def test_all_views_predictions_then_independent_score_and_no_target_claim(self):
        m=module(); marker,roles=self._fake_export(m); events=[]
        def fitted(a,b,seed):
            s=state();s['training']=dict(steps=200,known_count=len(a),oe_count=len(b),
                parameter_counts={'linear_oe':101,'mlp16_oe':1633},losses={k:[.5]*200 for k in ('linear_oe','mlp16_oe')})
            return s
        with tempfile.TemporaryDirectory() as td,self._patch_export(m,marker,roles,events), \
                mock.patch.object(m,'_old_source',return_value=(dict(encoder_state_identity=marker['encoder_state_identity'],parent_export_identity=marker['parent_export_identity']),geometry(),state())), \
                mock.patch.object(m.core,'fit_readouts',side_effect=fitted):
            root=Path(td);m.fit('export','old',root/'fit');m.calibrate('export',root/'fit',root/'cal')
            pred=m.predict_source('export',root/'fit',root/'cal',root/'pred')
            self.assertEqual(pred['completed_arms'],list(m.ARMS));self.assertEqual(pred['view_order'],list(m.VIEWS))
            truthfile=root/'truth.npz'
            np.savez(truthfile,ids=roles['novel_validation'][m.VIEWS[0]]['ids'],tx=np.asarray(['N0','N1','N2']*10))
            # Removing even one view must keep private truth closed.
            path=root/'pred/augmented_readout'/m.VIEWS[-1]/'predictions.npz';saved=path.read_bytes();path.unlink()
            with mock.patch.object(m,'_novel_truth') as private:
                with self.assertRaises((OSError,ValueError)):m.score_source(root/'pred',truthfile,root/'badscore')
                private.assert_not_called()
            path.write_bytes(saved)
            scored=m.score_source(root/'pred',truthfile,root/'score')
            self.assertEqual(scored['primary_candidate'],'augmented_readout::mlp16_oe')
            self.assertFalse(scored['final_phase3_complete']);self.assertEqual(scored['target_reads'],0)
            self.assertEqual(set(scored['arms']),set(m.ARMS))
            for arm in m.ARMS:
                self.assertEqual(set(scored['arms'][arm]['methods']),set(m.METHODS))
                goals=scored['arms'][arm]['methods']['mlp16_oe']['goals']
                self.assertIn('overall_known',goals)
                self.assertEqual(set(goals['by_receiver_known']),set(map(str,m.RECEIVERS)))
                overall=scored['arms'][arm]['methods']['mlp16_oe']['overall']
                self.assertIsNone(overall['confidence_intervals95'])
                self.assertTrue(overall['pooled_views_CI_not_reported'])

    def test_cli_first_line_actual_process_and_no_unknown_fit_argument(self):
        m=module()
        with mock.patch.object(m,'fit',return_value={'status':'SYNTHETIC'}),contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(m.main(['fit','--export-root','export','--source-fit-root','old','--output','new']),0)
        event=json.loads(stream.getvalue().splitlines()[0])
        self.assertEqual(event['event'],'SOURCE_PRACTICAL_READOUT_PROCESS')
        self.assertTrue(event['pid']);self.assertTrue(event['cwd']);self.assertIsInstance(event['argv'],list)


if __name__=='__main__':unittest.main()
