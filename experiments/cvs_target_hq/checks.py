import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
from . import design as d
from .views import generate
from .score import predictions,metric
from experiments.cvs_urban_sigtime.channel import imports

class Checks(unittest.TestCase):
    def test_sampling_is_label_blind_order_independent_and_bounded(self):
        ids=[f'opaque-{i}' for i in range(40)];chosen={ids[i] for i in d.sample_positions(ids,12)}
        reverse=list(reversed(ids));self.assertEqual(chosen,{reverse[i] for i in d.sample_positions(reverse,12)})
        self.assertEqual(len(chosen),12)
        self.assertTrue(d.configuration()['parent_checkpoint'].startswith('/home/'))
        self.assertNotIn('\\',d.configuration()['parent_checkpoint'])
        for values,count in ((['a','a'],1),(['a'],2),([''],1)):
            with self.assertRaises(ValueError):d.sample_positions(values,count)

    def fixture(self):
        ch,residual=imports();cfg=ch.Config(fs_hz=25e6,scenario=d.SCENE,processing_route='residual',mode='post_sync',
            equalization_enabled=False,input_processing_state='WiSig_equalized1_center256_unit_rms')
        manifest=dict(seed=392005,namespace='phase1_full_six_fixed_20261002_v1',receiver_seed=2027,
            session='virtual_target_session0',evidence=[dict(scene=d.SCENE,config=asdict(cfg))])
        clean=np.random.default_rng(81).normal(size=(2,256)).astype(np.float32)
        x,_=ch.normalize_input(clean[0].astype(float)+1j*clean[1].astype(float))
        receiver=ch.receiver_for_session(cfg,2027,manifest['session']);seed=ch.stable_seed(392005,manifest['namespace'],d.SCENE,'opaque-1')
        full,meta=residual.ResidualChannel(cfg,seed,receiver).process(x)
        return clean,np.stack((full.real,full.imag)),manifest

    def test_canonical_full_is_unchanged_and_independent_dose_is_paired(self):
        clean,full,manifest=self.fixture();old=full.copy();waves,meta=generate(clean,full,'opaque-1',manifest)
        self.assertTrue(np.array_equal(full,old));self.assertTrue(np.array_equal(waves['full'],full))
        self.assertLessEqual(meta['full_replay_max_abs_error'],2e-6)
        self.assertTrue(meta['receiver_actions_and_kernels_frozen']);self.assertFalse(meta['output_RMS_renormalized'])
        self.assertAlmostEqual(meta['independent_HQ_added_noise_snr_db']-meta['independent_samelevel_added_noise_snr_db'],6.,places=10)
        self.assertGreater(float(abs(waves[d.VIEWS[1]]-full).max()),1e-5)

    def test_replay_mismatch_and_weakened_config_are_rejected(self):
        clean,full,manifest=self.fixture()
        with self.assertRaises(ValueError):generate(clean,full+0.01,'opaque-1',manifest)
        bad=copy.deepcopy(manifest);bad['evidence'][0]['config']['equalization_enabled']=True
        with self.assertRaises(ValueError):generate(clean,full,'opaque-1',bad)

    def test_all_predictions_required_before_truth_stage(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(d,'COUNT',4):
            p=Path(temp);ids=np.asarray(['a','b','c','d'])
            done=dict(status='PREDICTIONS_COMPLETE',config=d.configuration(),count=4,views=list(d.VIEWS),truth_read=False,query_fit=False,
                model_state_unchanged=True,all_parameters_frozen=True,optimizer_updates=0,EMA_updates=0,head_fit_updates=0)
            (p/'complete.json').write_text(json.dumps(done),encoding='utf-8')
            for view in d.VIEWS:np.savez(p/(view+'.npz'),ids=ids,logits=np.ones((4,6),np.float32))
            self.assertEqual(set(predictions(p,ids)[0]),set(d.VIEWS))
            for update in ({'truth_read':True},{'head_fit_updates':1},{'views':list(d.VIEWS[:-1])}):
                (p/'complete.json').write_text(json.dumps(dict(done,**update)),encoding='utf-8')
                with self.assertRaises(ValueError):predictions(p,ids)
            (p/'complete.json').write_text(json.dumps(done),encoding='utf-8')
            np.savez(p/(d.VIEWS[-1]+'.npz'),ids=ids[::-1],logits=np.ones((4,6),np.float32))
            with self.assertRaises(ValueError):predictions(p,ids)
            bad=np.ones((4,6),np.float32);bad[0,0]=np.nan
            np.savez(p/(d.VIEWS[-1]+'.npz'),ids=ids,logits=bad)
            with self.assertRaises(ValueError):predictions(p,ids)
            (p/(d.VIEWS[-1]+'.npz')).unlink()
            with self.assertRaises(FileNotFoundError):predictions(p,ids)

    def test_independent_metric_counts_and_macro_f1(self):
        y=np.arange(6);good=np.eye(6);self.assertEqual(metric(y,good)['accuracy'],1.)
        self.assertEqual(metric(y,good)['macro_f1'],1.)
        wrong=np.roll(good,1,axis=1);self.assertEqual(metric(y,wrong)['accuracy'],0.)
        self.assertEqual(metric(y,wrong)['macro_f1'],0.)

if __name__=='__main__':unittest.main()
