from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'code/worktrees/standard_ce_baselines_20261010/experiments/pamf_phase2_practical/data.py'


def reference():
    name='_phase2_star_numeric_reference'
    if name+'.data' not in sys.modules:
        package=ModuleType(name);package.__path__=[str(REFERENCE.parent)];sys.modules[name]=package
        spec=importlib.util.spec_from_file_location(name+'.data',REFERENCE)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return sys.modules[name+'.data']


def mixture():
    return json.loads((ROOT/'configs/pamf_phase2_practical_mixture_20261028.json').read_text(encoding='utf-8-sig'))


class StarChannelTests(unittest.TestCase):
    def module(self):
        name='experiments.pamf_phase3_open_world.star_channel'
        self.assertTrue((ROOT/(name.replace('.','/')+'.py')).is_file())
        return importlib.import_module(name)

    def raw(self):
        t=np.arange(256,dtype=np.float64)
        return np.asarray([3.7*np.cos(.13*t)+.1*np.sin(.37*t),2.9*np.sin(.13*t)+.2*np.cos(.23*t)])

    def episode(self,seed=0,scenario=None):
        ep=dict(id='MIX01',scene_seed=2026103001+seed,channel_seed=2026104001+seed)
        if scenario is not None:ep['scenario']=scenario
        return ep

    def test_reference_geometry_exact_cospsi_area_all_exclusive_bands(self):
        module=self.module();ref=reference();channel,_=module._channel_api()
        cfg=channel.Config(fs_hz=25e6,scenario='practical_mid_urban')
        for band in ((10.,30.),(30.,45.),(45.,80.)):
            for seed in (3,77,2026103001):
                with self.subTest(band=band,seed=seed):
                    actual=module.geometry(cfg,band,np.random.default_rng(seed))
                    expected=ref.geometry(cfg,band,np.random.default_rng(seed))
                    self.assertEqual(asdict(actual),asdict(expected))
                    self.assertGreaterEqual(actual.elevation_deg,band[0]);self.assertLessEqual(actual.elevation_deg,band[1])

    def test_actual_reference_numerical_parity_MIX_and_three_native_pure_multiseed(self):
        module=self.module();ref=reference();mix=mixture();raw=self.raw();before=deepcopy((mix,raw))
        for scenario in (None,'practical_low_suburban','practical_mid_urban','practical_low_urban'):
            for seed in (0,2,19):
                with self.subTest(scenario=scenario,seed=seed):
                    episode=self.episode(seed,scenario)
                    sid=module.identity(['WiSig/Full/eq1','opaque-synthetic','24-1','2021_03_08',seed])
                    self.assertEqual(sid,ref.identity(['WiSig/Full/eq1','opaque-synthetic','24-1','2021_03_08',seed]))
                    value,meta=module.observe(raw,physical_id=sid,receiver='24-1',day='2021_03_08',episode=episode,mixture=mix)
                    expected,emeta=ref.observe(raw,mix['scenarios'],sid=sid,session='rx:24-1/day:2021_03_08',pool=episode,mixture=mix)
                    self.assertTrue(np.array_equal(value,expected));self.assertEqual(meta,emeta)
                    self.assertEqual((value.dtype,value.shape),(np.dtype('float32'),(2,256)))
                    self.assertEqual(set(meta),{'scenario','overlay_seed','input_rms','geometry'})
                    if scenario is not None:
                        lo,hi=(20.,45.) if scenario=='practical_mid_urban' else (10.,30.)
                        self.assertGreaterEqual(meta['geometry']['elevation_deg'],lo);self.assertLessEqual(meta['geometry']['elevation_deg'],hi)
        self.assertEqual(mix,before[0]);self.assertTrue(np.array_equal(raw,before[1]))

    def test_frequency_route_exact_session_one_input_RMS_and_no_output_RMS(self):
        module=self.module();channel,residual=module._channel_api();raw=self.raw();captured={}
        actual_receiver=channel.receiver_for_session
        def receiver(cfg,seed,session):
            captured.update(receiver_seed=seed,session=session);return actual_receiver(cfg,seed,session)
        class FakeResidual:
            def __init__(self,cfg,seed,rx,geometry=None):captured.update(cfg=cfg,seed=seed,geometry=geometry)
            def process(self,z):
                captured['normalized']=z.copy();return z*2.,dict(geometry=None)
        with patch.object(channel,'normalize_input',wraps=channel.normalize_input) as normalize, \
             patch.object(channel,'receiver_for_session',side_effect=receiver), \
             patch.object(residual,'ResidualChannel',FakeResidual):
            output,meta=module.observe(raw,physical_id='opaque-synthetic',receiver='24-1',day='2021_03_08',
                episode=self.episode(scenario='practical_mid_urban'),mixture=mixture())
        cfg=captured['cfg'];self.assertEqual(normalize.call_count,1)
        self.assertEqual((cfg.fs_hz,cfg.fc_hz,cfg.altitude_m,cfg.reference_snr_db,cfg.boundary),(25e6,2.45e9,600000.,30.,'reflect'))
        self.assertEqual((cfg.processing_route,cfg.mode,cfg.equalization_enabled),('residual','post_sync',False))
        self.assertEqual(cfg.input_processing_state,'WiSig_equalized1_center256_unit_rms')
        self.assertEqual((captured['receiver_seed'],captured['session']),(2027,'rx:24-1/day:2021_03_08'))
        self.assertIsNone(captured['geometry'])  # Pure MU takes native 20..45, not mixed 30..45.
        self.assertAlmostEqual(np.mean(abs(captured['normalized'])**2),1.,places=12)
        self.assertAlmostEqual(float(np.sqrt(np.mean(output.astype(np.float64)**2)*2)),2.,places=6)
        self.assertGreater(meta['input_rms'],1.)

    def test_validator_ignores_historical_design_text_but_checks_required_route_and_precise_probabilities(self):
        module=self.module();mix=mixture();self.assertIs(module.validate_mixture(mix),mix)
        self.assertEqual(mix['status'],'DESIGN_ONLY')
        module.validate_mixture(dict(mix,status='historical prose',capsule_built=False,extra_note='not a gate'))
        mutations=[]
        for key,value in (('fc_hz',2462e6),('fs_hz',20e6),('altitude_m',550000.),('mode','pre_sync'),('equalization_enabled',True)):
            bad=deepcopy(mix);bad['route'][key]=value;mutations.append(bad)
        bad=deepcopy(mix);bad['receiver_seed']=2026102912;mutations.append(bad)
        bad=deepcopy(mix);bad['scenarios'][1]['probability']=.0732;mutations.append(bad)
        bad=deepcopy(mix);bad['scenarios'][4]['mixed_conditional_elevation_range_deg']=[20.,45.];mutations.append(bad)
        bad=deepcopy(mix);bad['scenarios']=bad['scenarios'][::-1];mutations.append(bad)
        for index,bad in enumerate(mutations):
            with self.subTest(index=index),patch.object(module,'_channel_api',side_effect=AssertionError('No channel after invalid mixture')), \
                 self.assertRaises(ValueError):
                module.observe(self.raw(),physical_id='x',receiver='24-1',day='2021_03_08',episode=self.episode(),mixture=bad)

    def test_false_role_fields_and_bad_raw_rejected_before_channel(self):
        module=self.module();mix=mixture();kwargs=dict(physical_id='x',receiver='24-1',day='2021_03_08',episode=self.episode(),mixture=mix)
        module.validate_episode(dict(self.episode(),condition='MIX'))
        with patch.object(module,'_channel_api',side_effect=AssertionError('Invalid input cannot run channel')):
            for key in ('truth','role','TX','unknown_count'):
                with self.subTest(key=key),self.assertRaises(TypeError):module.observe(self.raw(),**dict(kwargs,**{key:'forbidden'}))
            with self.assertRaises(ValueError):module.validate_episode(dict(self.episode(),role='known'))
            with self.assertRaises(ValueError):module.observe(self.raw().T,**kwargs)
            bad=self.raw();bad[0,0]=np.nan
            with self.assertRaises(ValueError):module.observe(bad,**kwargs)

    def test_order_independence_and_fresh_episode_realization_difference(self):
        module=self.module();mix=mixture();raw=self.raw();episode=self.episode()
        def call(sid,ep=episode):return module.observe(raw,physical_id=sid,receiver='24-1',day='2021_03_15',episode=ep,mixture=mix)
        first={sid:call(sid) for sid in ('opaque-a','opaque-b')}
        reverse={sid:call(sid) for sid in ('opaque-b','opaque-a')}
        for sid in first:self.assertTrue(np.array_equal(first[sid][0],reverse[sid][0]));self.assertEqual(first[sid][1],reverse[sid][1])
        changed,meta=call('opaque-a',self.episode(1))
        self.assertFalse(np.array_equal(first['opaque-a'][0],changed))
        self.assertNotEqual(first['opaque-a'][1]['overlay_seed'],meta['overlay_seed'])


if __name__=='__main__':unittest.main()
