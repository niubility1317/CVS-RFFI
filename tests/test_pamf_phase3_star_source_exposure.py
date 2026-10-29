"""Synthetic-only checks for exact-record, truth-free source OE G10 data."""
from collections import Counter
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


class StarSourceExposureTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_source_exposure.py').is_file(),
                        'G10 source exposure builder is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_source_exposure')

    def fixture(self, root):
        from experiments.pamf_phase3_open_world import source_oe_data as source, source_oe_diverse_data as diverse
        from experiments.pamf_phase3_open_world.data import _identity
        full, parent = root/'full', root/'parent'; full.mkdir(); parent.mkdir()
        previous = dict(full_root=str(full), validation_root='DO-NOT-REVALIDATE', compact_memberships='DO-NOT-REOPEN',
            known_classes=source.KNOWN, reserved_target_classes=source.RESERVED, receivers=source.RECEIVERS,
            dates=source.DATES, exposure_train_classes=diverse.CLASSES, excluded_novel_classes=source.NOVEL,
            per_cell=32, selection_seed=2026102906)
        small = np.array([0]*95+[4]*32+[22]+[33]*127); np.random.default_rng(7).shuffle(small)
        cells, position = [], 0
        for tx in diverse.CLASSES:
            for day in source.DATES:
                for rx in source.RECEIVERS:
                    available = 33 if tx in source.TRAIN else int(small[position])
                    if tx not in source.TRAIN: position += 1
                    cells.append(dict(tx=tx, receiver=rx, date=day, available=available, selected=min(32, available)))
        lookup = {(c['tx'], c['receiver'], c['date']): c for c in cells}
        members, mapping, payloads = [], [], []
        for day in source.DATES:
            for rx in source.RECEIVERS:
                archive=f'pkl_wifi_eq_{day}-synthetic-{rx}.zip'; member=f'pkl_wifi_eq_{day}/dataset_{day}_node{rx}.pkl'
                members.append(dict(archive=archive, member=member, receiver=rx, date=day))
                with zipfile.ZipFile(full/archive,'w') as stored:
                    stored.writestr(member,b'SYNTHETIC NUMPY PAYLOAD'); stored.writestr('not-permitted.pkl',b'DO NOT OPEN')
                arrays, names = [], []
                for j, tx in enumerate(diverse.CLASSES):
                    cell=lookup[tx,rx,day]
                    if cell['selected'] == 0: continue
                    raw=np.ones((cell['available']+1,256,2),np.float64)
                    raw[:,:,0] = j+2+np.arange(len(raw))[:,None]/1000
                    raw[0]=np.nan; arrays.append(raw); names.append(tx)
                    for index in range(1,cell['selected']+1):
                        canonical=['FullWiSigEq',tx,rx,day,1,index]
                        mapping.append(dict(opaque_id='p3oe-'+_identity(['source_oe_physical',canonical]),
                            canonical_record=canonical, source_archive=archive, source_member=member))
                payloads.append(dict(data=arrays[::-1],node_list=names[::-1]))
        mapping.sort(key=lambda row:row['opaque_id']); self.assertEqual(len(mapping),6134)
        ids=np.asarray([row['opaque_id'] for row in mapping])
        rx=np.asarray([source.MAIN_RX[source.RECEIVERS.index(row['canonical_record'][2])] for row in mapping],np.int64)
        day=np.asarray([source.DATES.index(row['canonical_record'][3])+1 for row in mapping],np.int64)
        public=parent/'exposure_train/public'; public.mkdir(parents=True)
        np.save(public/'ids.npy',ids); np.savez(public/'metadata.npz',rx_i=rx,day_i=day,domain=np.searchsorted(source.MAIN_RX,rx)*3+day-1)
        (public/'iq.npy').write_bytes(b'OLD NORMALIZED IQ MUST NOT OPEN')
        truth=parent/'exposure_train/truth'; truth.mkdir(); (truth/'truth.npz').write_bytes(b'TRUTH MUST NOT OPEN')
        handle=dict(public_root=str(public),truth_path=str(truth/'truth.npz'),count=6134,
            capsule_id='old-diverse-capsule',split_id='old-diverse-split',data_status='VALIDATED_ONCE',iq_shape=[6134,2,256],iq_dtype='float32')
        complete=dict(status='SOURCE_OE_DIVERSE_DATA_VALIDATED',data_status='VALIDATED_ONCE',count=6134,
            counts={'exposure_train':6134},exposure_train_count=6134,exposure_train=handle,source_only=True,
            canonical_physical_unique=True,unique_canonical_count=6134,target_reads=0,original_L_s_reads=0,
            original_U_s_reads=0,original_V_reads=0,novel_validation_IQ_reads=0,model_reads=0,model_predictions=0,optimizer_updates=0)
        allocation=dict(count=6134,nonempty_cells=220,zero_cells=95,cells=cells)
        report=dict(schema='phase3_source_oe_diverse_builder_v1',config=previous,allocation=allocation,
                    mapping=mapping,members=members,samples_replicated=False)
        (parent/'complete.json').write_text(json.dumps(complete),encoding='utf-8')
        (parent/'builder_report.json').write_text(json.dumps(report),encoding='utf-8')
        mix=root/'mixture.json'; mix.write_text((ROOT/'configs/pamf_phase2_practical_mixture_20261028.json').read_text(encoding='utf-8'),encoding='utf-8')
        config=dict(run_id='20261029-phase3-star-source-exposure-g10-r01',parent_root=str(parent),full_root=str(full),
            mixture_config_path=str(mix),episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),code_commit='a'*40)
        return config, report, payloads

    @staticmethod
    def observer(raw, **kwargs):
        assert raw.dtype == np.float64 and raw.shape == (2,256)
        assert set(kwargs)=={'physical_id','receiver','day','episode','mixture'}
        assert np.sqrt(np.mean(np.sum(raw**2,axis=0)))>2
        value=raw.astype(np.float32); value/=np.sqrt(np.mean(np.sum(value.astype(np.float64)**2,axis=0)))
        return value*.4,dict(scenario='practical_mid_urban',overlay_seed=9,input_rms=3.,geometry={})

    def test_exact_original_order_nonuniform_domains_and_only_six_outside_supervision(self):
        m=self.module()
        from experiments.pamf_phase3_open_world import star_source_novel as raw_reader, source_oe_diverse_data as diverse
        from experiments.pamf_phase3_open_world.star_channel import identity
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,parent,payloads=self.fixture(root); calls=[]; original=np.load
            def guarded(path,*args,**kwargs):
                if Path(path).is_relative_to(Path(config['parent_root'])):
                    self.assertIn(Path(path).name,{'ids.npy','metadata.npz'})
                return original(path,*args,**kwargs)
            def observe(raw,**kwargs):calls.append((raw[0,0],kwargs));return self.observer(raw,**kwargs)
            with patch.object(raw_reader,'_read_numpy_pickle',side_effect=payloads),patch.object(m,'_observe',side_effect=observe),patch.object(np,'load',side_effect=guarded):
                done=m.build(config,root/'out')
            self.assertEqual((done['status'],done['data_status'],done['count']),('STAR_SOURCE_EXPOSURE_VALIDATED','VALIDATED_ONCE',6134))
            manifest,ids,received,meta=m.load_public(root/'out/public',done['capsule_id'],done['split_id'])
            np.testing.assert_array_equal(ids,[row['opaque_id'] for row in parent['mapping']])
            self.assertEqual(received.shape,(6134,2,256));self.assertFalse(received.flags.writeable)
            np.testing.assert_allclose(np.mean(np.sum(received.astype(np.float64)**2,axis=1),axis=1),.16,rtol=3e-7)
            received._mmap.close()
            self.assertEqual(manifest['role'],'source_auxiliary_training');self.assertIs(manifest['outside_registered_six'],True)
            self.assertEqual(set(manifest),m.PUBLIC_FIELDS);self.assertEqual(set(meta),{'rx_i','day_i','domain'})
            self.assertEqual(len(set(Counter(meta['domain']).values()))>1,True)
            report=json.loads((root/'out/builder_report.json').read_text())
            self.assertEqual([r['canonical_record'] for r in report['mapping']],[r['canonical_record'] for r in parent['mapping']])
            self.assertEqual(report['historical_old_OE4_physical_overlap_count'],537)
            self.assertFalse(report['historical_overlap_recomputed'])
            expected={identity(['WiSig/Full/eq1',r['canonical_record'][1],r['canonical_record'][2],r['canonical_record'][3],r['canonical_record'][5]]):r['canonical_record'] for r in parent['mapping']}
            self.assertEqual(len(calls),6134);self.assertEqual({kw['physical_id'] for _,kw in calls},set(expected))
            for value,kw in calls:
                _,tx,rx,day,_,index=expected[kw['physical_id']]
                self.assertEqual(value,diverse.CLASSES.index(tx)+2+index/1000)
                self.assertEqual((kw['receiver'],kw['day']),(rx,day))
            self.assertFalse((root/'out/truth').exists())
            for key in ('target_reads','formal_target_reads','original_L_s_reads','original_U_s_reads','original_V_reads',
                        'novel_validation_IQ_reads','model_reads','model_predictions','optimizer_updates'):
                self.assertEqual(done[key],0)

    def test_bad_controls_parent_alignment_or_private_fields_fail_before_raw_read(self):
        m=self.module()
        for mutation in ('config_truth','seed','commit','status','ids','target_rx','member','cell','metadata_tx','domain'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg,r,_=self.fixture(root);parent=Path(cfg['parent_root'])
                if mutation=='config_truth':cfg['truth']='not-allowed'
                elif mutation=='seed':cfg['episode']['scene_seed']+=1
                elif mutation=='commit':cfg['code_commit']='unknown'
                elif mutation=='status':
                    p=parent/'complete.json';c=json.loads(p.read_text());c['status']='PARTIAL';p.write_text(json.dumps(c))
                elif mutation=='ids':
                    p=parent/'exposure_train/public/ids.npy';np.save(p,np.load(p)[::-1])
                elif mutation=='target_rx':r['mapping'][0]['canonical_record'][2]='24-18'
                elif mutation=='member':r['mapping'][0]['source_member']='other.pkl'
                elif mutation=='cell':r['allocation']['cells'][0]['selected']=31
                else:
                    p=parent/'exposure_train/public/metadata.npz'
                    with np.load(p) as old:meta={k:old[k] for k in old.files}
                    if mutation=='metadata_tx':meta['tx']=np.full(6134,'not-public')
                    else:meta['domain'][0]=99
                    np.savez(p,**meta)
                (parent/'builder_report.json').write_text(json.dumps(r))
                with patch.object(m,'_read_member',side_effect=AssertionError('No raw before parent gate')),self.assertRaises(ValueError):m.build(cfg,root/'out')
                self.assertFalse((root/'out').exists())

    def test_only_indexed_fifteen_members_no_reselection_or_full_validation(self):
        m=self.module()
        from experiments.pamf_phase3_open_world import star_source_novel as raw_reader, source_oe_diverse_data as diverse, source_oe_data as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,r,payloads=self.fixture(root);opened=[]; original,names=zipfile.ZipFile.open,zipfile.ZipFile.namelist
            def opening(z,name,*args,**kwargs):
                if str(z.filename).endswith('.zip'):opened.append((Path(z.filename).name,name))
                return original(z,name,*args,**kwargs)
            def no_scan(z):
                if str(z.filename).endswith('.zip'):raise AssertionError('No Full scan')
                return names(z)
            writes=[];write=m._write_json
            def writing(path,value):
                self.assertFalse((root/'out/complete.json').exists());writes.append(Path(path).name);return write(path,value)
            with patch.object(raw_reader,'_read_numpy_pickle',side_effect=payloads),patch.object(m,'_observe',side_effect=self.observer), \
                 patch.object(zipfile.ZipFile,'open',opening),patch.object(zipfile.ZipFile,'namelist',no_scan), \
                 patch.object(diverse,'_plan',side_effect=AssertionError('No new validation')),patch.object(diverse,'_selected',side_effect=AssertionError('No new selection')), \
                 patch.object(old,'_locations',side_effect=AssertionError('No validation index reread')),patch.object(m,'_write_json',side_effect=writing):
                m.build(cfg,root/'out')
            self.assertEqual(opened,[(x['archive'],x['member']) for x in r['members']]);self.assertEqual(writes[-1],'complete.json')

    def test_bad_raw_or_output_collision_cannot_complete_or_replace(self):
        m=self.module()
        from experiments.pamf_phase3_open_world import star_source_novel as raw_reader
        for mutation in ('index','dtype','nonfinite','output'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg,_,payloads=self.fixture(root);output=root/'out'
                if mutation=='index':payloads[0]['data'][0]=payloads[0]['data'][0][:1]
                elif mutation=='dtype':payloads[0]['data'][0]=payloads[0]['data'][0].astype(np.float32)
                elif mutation=='nonfinite':payloads[0]['data'][0][1]=np.nan
                else:output.mkdir();(output/'keep').write_text('history')
                with patch.object(raw_reader,'_read_numpy_pickle',side_effect=payloads),patch.object(m,'_observe',side_effect=self.observer),self.assertRaises((ValueError,FileExistsError)):
                    m.build(cfg,output)
                self.assertFalse((output/'complete.json').exists())
                if mutation=='output':self.assertEqual((output/'keep').read_text(),'history')

    def test_public_loader_handle_and_tx_k_leak_rejection(self):
        m=self.module()
        from experiments.pamf_phase3_open_world import star_source_novel as raw_reader
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_,payloads=self.fixture(root)
            with patch.object(raw_reader,'_read_numpy_pickle',side_effect=payloads),patch.object(m,'_observe',side_effect=self.observer):done=m.build(cfg,root/'out')
            public=root/'out/public';cap,split=done['capsule_id'],done['split_id']
            self.assertEqual({p.name for p in public.iterdir()},m.PUBLIC_FILES)
            with self.assertRaises(ValueError):m.load_public(public,'wrong',split)
            with self.assertRaises(ValueError):m.load_public(public,cap,'wrong')
            leak=public/'truth.npz';leak.write_bytes(b'forbidden')
            with self.assertRaises(ValueError):m.load_public(public,cap,split)
            leak.unlink();p=public/'manifest.json';original=json.loads(p.read_text())
            for key,value in (('tx',['hidden']),('true_K',21)):
                p.write_text(json.dumps(dict(original,**{key:value})))
                with self.assertRaises(ValueError):m.load_public(public,cap,split)

    def test_fixed_config_and_cli_runtime_header(self):
        m=self.module();path=ROOT/'configs/pamf_phase3_star_source_exposure_20261029.json';self.assertTrue(path.is_file())
        m.validate_config(json.loads(path.read_text(encoding='utf-8')))
        with io.StringIO() as out,contextlib.redirect_stdout(out),patch.object(m,'_read_json',return_value={}), \
             patch.object(m,'build',return_value={'status':'STAR_SOURCE_EXPOSURE_VALIDATED','count':6134}):
            self.assertEqual(m.main(['--config','synthetic.json','--output','out']),0);header=json.loads(out.getvalue().splitlines()[0])
        self.assertEqual(header['pid'],os.getpid());self.assertEqual(Path(header['cwd']),Path.cwd())
        self.assertEqual(header['argv'],['--config','synthetic.json','--output','out'])


if __name__=='__main__':unittest.main()
