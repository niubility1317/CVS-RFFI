"""Synthetic source-only runner checks; no real IQ or checkpoint."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarNuisanceSourceTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_nuisance_source')

    def config(self, root):
        from experiments.pamf_phase3_open_world import star_relative_source as old
        mixture = root/'mixture.json'
        base = Path(__file__).resolve().parents[1]
        mixture.write_text((base/'configs/pamf_phase2_practical_mixture_20261028.json').read_text(), encoding='utf-8')
        nested = dict(run_id='20261029-phase3-star-nuisance-s2026092701-r01', code_commit='a'*40,
            reader_config=dict(base_release='F0-release', frozen_sources={k:k for k in ('F','AF','BASE','FOLD','G')},
                               source_output='F0/source', model_seed=2026092701),
            source_cache='source-cache', source_contract='source-contract', raw_mainsig='raw.pkl',
            mixture_config_path=str(mixture), episode=copy.deepcopy(old.EPISODE),
            dura_checkpoint='G/final.pt', dura_release='G-release', output_root=str(root/'out'), batch_size=128)
        return dict(run_id=nested['run_id'], code_commit='a'*40, source_config=nested,
                    novel_public_root='novel/public', novel_capsule_id='capsule', novel_split_id='split',
                    output_root=nested['output_root'], batch_size=128)

    @staticmethod
    def metadata(prefix):
        from experiments.pamf_phase3_open_world.star_relative_source import RECEIVERS, SCENES
        rows = [(y, r, s) for y in range(6) for r in RECEIVERS for s in range(len(SCENES))]
        return dict(ids=np.array([prefix+str(i) for i in range(len(rows))]),
                    y=np.array([x[0] for x in rows], np.int64), rx_i=np.array([x[1] for x in rows], np.int64),
                    day_i=np.array([x[2]%3+1 for x in rows], np.int64),
                    scenario=np.array([SCENES[x[2]] for x in rows]))

    def patches(self, m, config, events, fail_novel=False, mutate=False):
        from contextlib import ExitStack
        meta={r:self.metadata(r) for r in ('L_s','V')}
        stack=ExitStack()
        def observations(raw, row, *args):
            role='L_s' if str(row['ids'][0]).startswith('L_s') else 'V'
            events.append(('observations',role))
            return np.column_stack((row['y'], np.arange(len(row['y'])))).astype(np.float32),row['scenario']
        def encode(backend, x, path, device):
            events.append(('encode',len(x),path))
            return {'fused160':x.copy()}, np.eye(6,dtype=np.float32)[x[:,0].astype(int)%6]*5
        def fit(branches,y,rx,day,geometry,variant):
            events.append(('fit',variant,len(y)))
            return dict(variant=variant,center=np.array([float(len(y))]))
        def scores(branches,geometry,metric):
            if mutate and len(branches['fused160']) != 180: metric['center'][0]+=1
            return np.column_stack((branches['fused160'][:,0]*0+1,branches['fused160'][:,0]*0+2)).astype(np.float64)
        def load_novel(*args):
            events.append(('novel',args))
            self.assertTrue((Path(config['output_root'])/'source/calibration/complete.json').is_file())
            if fail_novel: raise ValueError('synthetic public failure')
            x=np.column_stack((np.arange(90)%6,np.arange(90))).astype(np.float32)
            return dict(count=90),np.array(['u'+str(i) for i in range(90)]),x,dict(rx_i=np.repeat([1,3,4,6,8],18).astype(np.int64),day_i=(np.arange(90)%3+1).astype(np.int64))
        patches=[patch.object(m,'COUNTS',dict(L_s=180,V=180)),patch.object(m,'NOVEL_COUNT',90),
            patch.object(m,'source_metadata',side_effect=lambda c:({},meta)),patch.object(m,'_backend',return_value={}),
            patch.object(m,'_snapshot',return_value=[]),patch.object(m,'_check_models'),
            patch.object(m,'_smoke',side_effect=lambda *a:events.append(('smoke',)) or dict(status='PASS')),
            patch.object(m,'_load_raw',side_effect=lambda *a:events.append(('raw',)) or {}),
            patch.object(m,'_observations',side_effect=observations),patch.object(m,'_encode',side_effect=encode),
            patch.object(m,'_branches',side_effect=lambda backend,x,path,device,bs:encode(backend,x,path,device)),
            patch.object(m.geometry_core,'fit_geometry',side_effect=lambda b,y:dict(n=len(y))),
            patch.object(m,'_save_geometry',side_effect=lambda p,g:m._npz(p,dict(n=np.asarray(g['n'])))),
            patch.object(m.core,'fit_metric',side_effect=fit),patch.object(m.core,'metric_scores',side_effect=scores),
            patch.object(m.core,'to_arrays',side_effect=lambda b:dict(center=b['center'])),
            patch.object(m,'_load_novel',side_effect=load_novel)]
        for p in patches:stack.enter_context(p)
        return stack

    def test_exact_config_and_nested_forbidden_inputs(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));self.assertIs(m.validate_config(c),c)
            for key in ('target_root','truth','U_root','unknown_count'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(c,**{key:'x'}))
            for key in ('target_root','truth','unknown_count'):
                bad=copy.deepcopy(c);bad['source_config'][key]='x'
                with self.subTest(nested=key),self.assertRaises(ValueError):m.validate_config(bad)
            bad=copy.deepcopy(c);bad['source_config']['output_root']='old-root'
            with self.assertRaises(ValueError):m.validate_config(bad)

    def test_fit_only_L_freeze_all_thresholds_then_novel(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            with self.patches(m,c,events):result=m.run(c,'cpu')
            self.assertEqual(events[0],('smoke',));self.assertEqual(events[1],('raw',))
            v=events.index(('observations','V'));fits=[i for i,e in enumerate(events) if e[0]=='fit']
            self.assertEqual(len(fits),6);self.assertTrue(all(i<v for i in fits))
            self.assertEqual(result['status'],m.STATUS);self.assertEqual(result['source_novel_truth_reads'],0)
            root=Path(c['output_root'])
            with np.load(root/'known_predictions.npz') as z:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS);self.assertEqual(z['scores'].shape,(180,12))
                np.testing.assert_array_equal(z['baseline_closed'],z['closed'])
            with np.load(root/'novel_predictions.npz') as z:
                self.assertEqual(set(z.files),m.NOVEL_FIELDS);self.assertEqual(z['scores'].shape,(90,12))
                self.assertFalse({'scenario','tx','y'} & set(z.files))
            with self.assertRaises(FileExistsError):m.run(c,'cpu')

    def test_failure_keeps_frozen_source_without_final_completion(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            with self.patches(m,c,events,fail_novel=True),self.assertRaises(ValueError):m.run(c,'cpu')
            root=Path(c['output_root'])
            self.assertTrue((root/'source/calibration/complete.json').is_file())
            self.assertFalse((root/'complete.json').exists())

    def test_mutating_L_metric_during_V_prevents_novel_or_completion(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            with self.patches(m,c,events,mutate=True),self.assertRaisesRegex(ValueError,'changed after fitting'):
                m.run(c,'cpu')
            self.assertFalse(any(e[0]=='novel' for e in events))
            self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_calibration_42groups_scalar_and_ties(self):
        m=self.module();row=self.metadata('v')
        # 50 observations per class/RX/scene permit exactly 2% extra errors.
        repeat=50;indices=np.repeat(np.arange(180),repeat)
        y=row['y'][indices];rx=row['rx_i'][indices];scene=row['scenario'][indices]
        closed=np.column_stack((y,y));scores=np.ones((len(y),12),np.float64);scores[::repeat]=0
        ids=np.array(['id'+str(i) for i in range(len(y))])
        t,r=m.calibrate_known(ids,y,closed,closed.copy(),scores,rx,scene)
        np.testing.assert_array_equal(t,np.ones(12));self.assertTrue(r['constraints_passed'])
        self.assertEqual(r['group_count'],42)
        for method in r['methods'].values():self.assertFalse(method['next_candidate_passed'])
        mask=~((y==0)&(rx==1)&(scene==m.SCENES[0]))
        t,r=m.calibrate_known(ids[mask],y[mask],closed[mask],closed[mask],scores[mask],rx[mask],scene[mask])
        self.assertFalse(r['constraints_passed']);np.testing.assert_array_equal(t,np.zeros(12))

    def test_baseline_must_be_same_path_original_F0(self):
        m=self.module();row=self.metadata('v');y=row['y'];closed=np.column_stack((y,(y+1)%6))
        baseline=closed.copy();scores=np.ones((len(y),12),np.float64)
        _,r=m.calibrate_known(row['ids'],y,closed,baseline,scores,row['rx_i'],row['scenario'])
        self.assertTrue(r['constraints_passed'])
        baseline[:,1]=y
        with self.assertRaises(ValueError):m.calibrate_known(row['ids'],y,closed,baseline,scores,row['rx_i'],row['scenario'])

    def test_actual_numpy_core_codec_and_candidate_column_order(self):
        m=self.module();meta=self.metadata('L')
        from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
        rng=np.random.default_rng(22);n=len(meta['ids'])
        branches={name:rng.normal(0,.1,(n,d)).astype(np.float32) for name,d in DIMENSIONS.items()}
        for value in branches.values():value[np.arange(n),meta['y']]+=2
        branches['reference29'][:,27]=0
        geometry=m.geometry_core.fit_geometry(branches,meta['y'])
        metrics={v:m.core.fit_metric(branches,meta['y'],meta['rx_i'],meta['day_i'],geometry,v) for v in m.VARIANTS}
        for variant in metrics:
            arrays=m.core.to_arrays(metrics[variant]);self.assertTrue(all(v.dtype.kind!='O' for v in arrays.values()))
            metrics[variant]=m.core.from_arrays(arrays)
        states={p:dict(geometry=geometry,metrics=metrics) for p in m.PATHS}
        logits=np.eye(6,dtype=np.float32)[meta['y']]
        def encode(backend, indices, path, device):
            take=indices.astype(int)
            return {k:v[take] for k,v in branches.items()},np.roll(logits[take],int(path=='P1'),axis=1)
        with patch.object(m,'_encode',side_effect=encode):values,closed=m._score_paths({},np.arange(n),states,'cpu',128)
        np.testing.assert_array_equal(closed[:,0],meta['y']);np.testing.assert_array_equal(closed[:,1],(meta['y']+1)%6)
        for variant in m.VARIANTS:
            expected=m.core.metric_scores(branches,geometry,metrics[variant])
            for j,method in enumerate(m.METHODS):
                for path in m.PATHS:np.testing.assert_allclose(values[:,m.CANDIDATES.index(variant+'::'+method+'::'+path)],expected[:,j],rtol=1e-11,atol=1e-11)


if __name__=='__main__':unittest.main()
