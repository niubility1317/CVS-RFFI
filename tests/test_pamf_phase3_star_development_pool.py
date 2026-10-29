"""Synthetic-only protocol tests for one G10 development observation per record."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
NEW = ['19-15', '19-16', '7-1']
EPISODE = dict(id='MIX01', condition='MIX', scene_seed=2026103001, channel_seed=2026104001)
MEMBERS = {
    '2021_03_08': ('pkl_wifi_eq_2021_03_08-20261028T143859Z-1-002.zip', 'pkl_wifi_eq_2021_03_08/dataset_2021_03_08_node24-1.pkl'),
    '2021_03_15': ('pkl_wifi_eq_2021_03_15-20261028T144306Z-1-004.zip', 'pkl_wifi_eq_2021_03_15/dataset_2021_03_15_node24-1.pkl'),
}


class StarDevelopmentPoolTests(unittest.TestCase):
    def module(self):
        path = ROOT / 'experiments/pamf_phase3_open_world/star_development_pool.py'
        self.assertTrue(path.is_file(), 'G10 star development builder is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_development_pool')

    def fixture(self, root):
        full, parent, validation = root/'full', root/'parent', root/'validation'
        full.mkdir(); parent.mkdir(); validation.mkdir()
        legacy = json.loads((ROOT/'configs/pamf_phase3_development_pool_20261029.json').read_text(encoding='utf-8'))
        legacy.update(full_root=str(full), validation_root=str(validation), compact_receiver_evidence='already-verified.json')
        old_config = root/'legacy.json'; old_config.write_text(json.dumps(legacy), encoding='utf-8')
        mixture = root/'mixture.json'
        mixture.write_text((ROOT/'configs/pamf_phase2_practical_mixture_20261028.json').read_text(encoding='utf-8'), encoding='utf-8')
        (validation/'summary.json').write_text(json.dumps(dict(status='VERIFIED', dataset_directory=str(full), members_checked=158,
            expected_nonempty_RX_date_members=158, count_mismatch_cells=0, errors=[])), encoding='utf-8')
        complete = dict(status='PHASE3_DEVELOPMENT_POOL_VALIDATED', data_status='VALIDATED_ONCE', development_only=True,
            discovery_count=384, evaluation_count=864, unique_canonical_count=1248, canonical_physical_unique=True,
            single_observation=True, formal_target_reads=0, model_reads=0, model_predictions=0, optimizer_updates=0)
        report = dict(schema='pamf_phase3_development_builder_v1', config=legacy, development_only=True,
            validation_reused=dict(root=str(validation), status='VERIFIED', members_checked=158, count_mismatch_cells=0),
            formal_target_reads=0, canonical_physical_unique=True, scene_and_role_canonical_disjoint=True, mapping={})
        payloads = []
        for role,day,n,txs in [('discovery','2021_03_08',48,OLD+NEW[:2]), ('evaluation','2021_03_15',96,OLD+NEW)]:
            archive,member=MEMBERS[day]
            with zipfile.ZipFile(full/archive,'w') as z:
                z.writestr(member,b'SYNTHETIC PATCHED PAYLOAD')
                z.writestr('formal24-18.pkl',b'DO NOT READ')
            arrays=[]
            for j,tx in enumerate(OLD+NEW):
                a=np.ones((100,256,2),dtype=np.float64)
                a[:,:,0] = j+1+np.arange(100)[:,None]/1000
                a[0]=np.nan  # Never selected and must never cause quality reselection.
                arrays.append(a)
            payloads.append(dict(data=arrays,node_list=OLD+NEW))
            records=[dict(opaque_id=f'old-{role}-{j}-{i}',canonical_record=['FullWiSigEq',tx,'24-1',day,1,i+1],
                          source_archive=archive,source_member=member) for j,tx in enumerate(txs) for i in range(n)]
            records=records[1::2]+list(reversed(records[::2]))
            report['mapping'][role]=records
            public=parent/role/'public'; public.mkdir(parents=True)
            manifest=dict(schema='phase3_min_v1',data_status='VALIDATED_ONCE',capsule_id='old-'+role,split_id='old-split-'+role,
                count=len(records),iq_shape=[len(records),2,256],registered_classes=OLD,processing_route='residual',mode='post_sync',
                equalization_enabled=False,fs_hz=25000000,single_observation=True)
            (public/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            np.save(public/'ids.npy',np.asarray([r['opaque_id'] for r in records]))
            (public/'received.npy').write_bytes(b'OLD IQ MUST NOT OPEN')
            truth=parent/role/'truth';truth.mkdir();(truth/'truth.npz').write_bytes(b'OLD TRUTH MUST NOT OPEN')
            complete[role]=dict(count=len(records),capsule_id=manifest['capsule_id'],split_id=manifest['split_id'])
        (parent/'complete.json').write_text(json.dumps(complete),encoding='utf-8')
        (parent/'builder_report.json').write_text(json.dumps(report),encoding='utf-8')
        return dict(legacy_pool_root=str(parent),legacy_config_path=str(old_config),mixture_config_path=str(mixture),
                    episode=EPISODE.copy(),full_root=str(full),validation_root=str(validation)),report,payloads

    @staticmethod
    def observer(raw, **kw):
        from experiments.pamf_phase3_open_world.star_channel import identity
        assert raw.shape == (2,256) and raw.dtype == np.float64
        assert set(kw)=={'physical_id','receiver','day','episode','mixture'}
        assert set(kw['episode'])=={'id','scene_seed','channel_seed'}
        assert kw['receiver']=='24-1'
        value=raw.astype(np.float32)
        value/=np.sqrt(np.mean(np.sum(value.astype(np.float64)**2,axis=0)))
        return value*.7,dict(scenario='practical_high',overlay_seed=int(identity(kw['physical_id']),16)%100000,
                            input_rms=1.23,geometry=dict(elevation_deg=60.))

    def test_exact_role_canonical_order_same_formula_and_no_parent_truth_or_scene(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,parent,payloads=self.fixture(root);calls=[]
            def observe(raw,**kw):calls.append(kw.copy());return self.observer(raw,**kw)
            original=np.load
            def guarded(path,*a,**kw):
                if Path(path).is_relative_to(Path(cfg['legacy_pool_root'])) and Path(path).name!='ids.npy':
                    raise AssertionError('No old IQ/truth')
                return original(path,*a,**kw)
            with patch.object(m,'_read_numpy_pickle',side_effect=payloads),patch.object(m,'_observe',side_effect=observe),patch.object(np,'load',side_effect=guarded):
                result=m.build(cfg,root/'out')
            self.assertEqual(result['status'],'PHASE3_STAR_G10_POOL_VALIDATED')
            self.assertEqual((result['unique_canonical_count'],result['new_observation_count']),(1248,1248))
            self.assertEqual(len(calls),1248)
            from experiments.pamf_phase3_open_world.star_channel import identity
            report=json.loads((root/'out/builder_report.json').read_text())
            for role,count in [('discovery',384),('evaluation',864)]:
                old=parent['mapping'][role];new=report['mapping'][role]
                self.assertEqual([r['canonical_record'] for r in new],[r['canonical_record'] for r in old])
                self.assertTrue(all(r['scenario']=='practical_high' for r in new))  # No quota balancing.
                public=root/'out'/role/'public'
                self.assertEqual({p.name for p in public.iterdir()},{'manifest.json','ids.npy','received.npy'})
                manifest=json.loads((public/'manifest.json').read_text())
                self.assertEqual(manifest['schema'],'phase3_min_v1')
                self.assertNotIn('scenario',manifest);self.assertNotIn('episode',manifest)
                ids=np.load(public/'ids.npy');self.assertTrue(all(str(x).startswith('p3dev-star-g10-') for x in ids))
                self.assertFalse(set(ids)&{r['opaque_id'] for r in old})
                with np.load(root/'out'/role/'truth/truth.npz') as truth:
                    self.assertEqual(set(truth.files),{'ids','y','receiver','scenario'})
                    np.testing.assert_array_equal(truth['ids'],ids)
                    np.testing.assert_array_equal(truth['y'],[(OLD+NEW).index(r['canonical_record'][1]) for r in old])
                    self.assertTrue(np.all(truth['receiver']==26))
                received=np.load(public/'received.npy')
                np.testing.assert_allclose(np.mean(np.sum(received.astype(np.float64)**2,axis=1),axis=1),.49,rtol=2e-7)
                for r in new:
                    _,tx,rx,day,_,index=r['canonical_record']
                    self.assertEqual(r['physical_id'],identity(['WiSig/Full/eq1',tx,rx,day,index]))
            for key in ('formal_target_reads','model_reads','model_predictions','optimizer_updates','foreign_processes_stopped'):
                self.assertEqual(result[key],0)
            self.assertFalse(report['old_received_read']);self.assertFalse(report['old_truth_read'])
            self.assertEqual((report['archive_reads'],report['pickle_reads']),(2,2))

    def test_invalid_handles_mapping_and_controls_fail_before_full_read(self):
        m=self.module()
        for mutation in ('wrong_status','validation','ids_order','duplicate','formal','wrong_role_day','member','unknown_count','episode'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg,r,payloads=self.fixture(root);parent=Path(cfg['legacy_pool_root'])
                if mutation=='wrong_status':
                    path=parent/'complete.json';x=json.loads(path.read_text());x['status']='RUNNING';path.write_text(json.dumps(x))
                elif mutation=='validation':cfg['validation_root']=str(root/'different')
                elif mutation=='ids_order':
                    path=parent/'discovery/public/ids.npy';np.save(path,np.load(path)[::-1])
                elif mutation=='duplicate':r['mapping']['discovery'][1]['canonical_record']=r['mapping']['discovery'][0]['canonical_record']
                elif mutation=='formal':r['mapping']['evaluation'][0]['canonical_record'][2]='24-18'
                elif mutation=='wrong_role_day':r['mapping']['evaluation'][0]['canonical_record'][3]='2021_03_08'
                elif mutation=='member':r['mapping']['discovery'][0]['source_member']='formal24-18.pkl'
                elif mutation=='unknown_count':cfg['true_unknown_count']=2
                elif mutation=='episode':cfg['episode']['condition']='CLEAN'
                (parent/'builder_report.json').write_text(json.dumps(r),encoding='utf-8')
                with patch.object(m,'_read_numpy_pickle',side_effect=AssertionError('No IQ before handles')),self.assertRaises(ValueError):m.build(cfg,root/'out')
                self.assertFalse((root/'out').exists())

    def test_only_two_indexed_members_no_archive_rescan_or_new_selection(self):
        m=self.module()
        from experiments.pamf_phase3_open_world import development_pool
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_,payloads=self.fixture(root);opened=[];original=zipfile.ZipFile.open
            def read(z,name,*a,**kw):
                if str(z.filename).endswith('.zip'):opened.append((Path(z.filename).name,name))
                return original(z,name,*a,**kw)
            with patch.object(zipfile.ZipFile,'open',read),patch.object(zipfile.ZipFile,'namelist',side_effect=AssertionError('No rescan')),\
                 patch.object(development_pool,'_select_rows',side_effect=AssertionError('No selection')),\
                 patch.object(m,'_read_numpy_pickle',side_effect=payloads),patch.object(m,'_observe',side_effect=self.observer):
                m.build(cfg,root/'out')
            self.assertEqual(opened,list(MEMBERS.values()))

    def test_absent_selected_member_bad_packet_and_output_collision_preserve_history(self):
        m=self.module()
        for mutation in ('missing_archive','missing_member','bad_packet','existing_output'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg,r,payloads=self.fixture(root);output=root/'out'
                if mutation=='missing_archive':(Path(cfg['full_root'])/MEMBERS['2021_03_15'][0]).unlink()
                elif mutation=='missing_member':
                    with zipfile.ZipFile(Path(cfg['full_root'])/MEMBERS['2021_03_15'][0],'w') as z:z.writestr('other',b'wrong')
                elif mutation=='bad_packet':
                    r['mapping']['evaluation'][0]['canonical_record'][5]=1000
                    (Path(cfg['legacy_pool_root'])/'builder_report.json').write_text(json.dumps(r))
                else:output.mkdir();(output/'keep').write_text('keep')
                with patch.object(m,'_read_numpy_pickle',side_effect=payloads),patch.object(m,'_observe',side_effect=self.observer),self.assertRaises((ValueError,FileExistsError)):
                    m.build(cfg,output)
                self.assertFalse((output/'complete.json').exists())
                if mutation=='existing_output':self.assertEqual((output/'keep').read_text(),'keep')

    def test_real_adapter_one_small_synthetic_channel_matches_contract(self):
        m=self.module()
        mixture=json.loads((ROOT/'configs/pamf_phase2_practical_mixture_20261028.json').read_text())
        raw=np.random.default_rng(4).normal(size=(2,256))
        value,meta=m._observe(raw,physical_id='synthetic-no-target-record',receiver='24-1',day='2021_03_08',
                              episode={k:v for k,v in EPISODE.items() if k!='condition'},mixture=mixture)
        self.assertEqual((value.shape,value.dtype),((2,256),np.dtype('float32')))
        self.assertTrue(np.isfinite(value).all())
        self.assertIn(meta['scenario'],[x['id'] for x in mixture['scenarios']])
        self.assertGreater(meta['input_rms'],0);self.assertIn('overlay_seed',meta);self.assertIn('geometry',meta)

    def test_cli_reports_actual_process_before_config_io(self):
        m=self.module()
        with io.StringIO() as text,contextlib.redirect_stdout(text),patch.object(m,'_read_json',return_value={}),\
             patch.object(m,'build',return_value=dict(status='PHASE3_STAR_G10_POOL_VALIDATED',discovery_count=384,evaluation_count=864)):
            m.main(['--config','synthetic.json','--output','synthetic-out'])
            first=json.loads(text.getvalue().splitlines()[0])
        self.assertEqual(first['pid'],os.getpid());self.assertEqual(Path(first['cwd']),Path.cwd())
        self.assertEqual(first['argv'],['--config','synthetic.json','--output','synthetic-out'])


if __name__=='__main__':unittest.main()
