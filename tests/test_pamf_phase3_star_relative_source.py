"""Synthetic source-only tests: no real IQ, neural checkpoint, or target input."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ['14-10','14-7','20-15','20-19','6-15','8-20']
RX = [1,3,4,6,8]
RX_NAMES = ['1-1','1-19','14-7','18-2','19-2','2-1','2-19','20-1','3-19','7-14','7-7','8-8']
SCENES = ['practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']


class StarRelativeSourceTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_relative_source.py').is_file(), 'Star-only F0 relative source module is missing')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_relative_source')

    def fixture(self, root):
        cache=root/'cache';cache.mkdir()
        contract=dict(classes=OLD,source_rxs=RX,source_days=[1,2,3],num_classes=6,split_seed=392005,equalized=1,out_len=256,normalize=True,
                      counts=dict(L_s=720,U_s=56700,V=1080),role_ids={})
        metadata={}
        for role,packets in [('L_s',range(8)),('V',range(16,28))]:
            rows=[(y,rx,day,sig) for y in range(6) for rx in RX for day in (1,2,3) for sig in packets]
            fields=dict(y=np.array([r[0] for r in rows],np.int64),rx_i=np.array([r[1] for r in rows],np.int64),
                        day_i=np.array([r[2] for r in rows],np.int64),sig_i=np.array([r[3] for r in rows],np.int64),
                        eq_i=np.ones(len(rows),np.int64),base_index=np.arange(len(rows),dtype=np.int64)+100000,
                        domain=np.array([RX.index(r[1])*3+r[2]-1 for r in rows],np.int64),ids=np.array([f'{role}-{i}' for i in range(len(rows))]))
            contract['role_ids'][role]=fields['ids'].tolist();metadata[role]=fields
            np.savez(cache/(role+'_metadata.npz'),**fields)
            (cache/(role+'_x.npy')).write_bytes(b'ALREADY RMS IQ MUST NOT OPEN')
        (cache/'U_s_metadata.npz').write_bytes(b'U MUST NOT OPEN');(cache/'U_s_x.npy').write_bytes(b'U MUST NOT OPEN')
        cp=root/'contract.json';cp.write_text(json.dumps(contract),encoding='utf-8')
        raw=root/'ManySig.pkl';raw.write_bytes(b'RAW LOADER PATCHED; NO REAL IQ')
        mixture=root/'mixture.json';mixture.write_text((ROOT/'configs/pamf_phase2_practical_mixture_20261028.json').read_text(),encoding='utf-8')
        cfg=dict(run_id='20261029-phase3-star-relative-s2026092701-r01',code_commit='a'*40,reader_config=dict(base_release='original-release',
            frozen_sources={k:'frozen-'+k for k in ('F','AF','BASE','FOLD','G')},source_output='original-F0/source',model_seed=2026092701),
            source_cache=str(cache),source_contract=str(cp),raw_mainsig=str(raw),mixture_config_path=str(mixture),
            episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),dura_checkpoint='fixed-G/final.pt',
            dura_release='fixed-G-release',output_root=str(root/'source'),batch_size=128)
        # Reverse raw TX order: y is a named model class, not a raw TX index.
        txs=list(reversed(OLD)); data=[]
        for tx in txs:
            tx_data=[[None,None,None,None] for _ in RX_NAMES]
            for rx in RX:
                for day in (1,2,3):
                    records=np.ones((28,256,2),np.float64)
                    for sig in range(28):
                        records[sig,:,0]=np.sin(np.arange(256)*.07)
                        records[sig,0,0]=OLD.index(tx)+1;records[sig,1,0]=sig
                        records[sig,2,0]=rx;records[sig,3,0]=day
                    tx_data[rx][day]=[None,records]
            data.append(tx_data)
        payload=dict(tx_list=txs,rx_list=RX_NAMES,capture_date_list=['2021_03_01','2021_03_08','2021_03_15','2021_03_23'],equalized_list=[0,1],data=data)
        return cfg,contract,metadata,payload

    @staticmethod
    def geometry():
        from experiments.pamf_phase3_open_world import source_open_readout as g
        from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
        return dict(schema=g.GEOMETRY_SCHEMA,class_count=6,branches={k:dict(mean=np.zeros(d),scale=np.ones(d),
            prototypes=np.eye(6,d),whitener=np.eye(d)) for k,d in DIMENSIONS.items()})

    def backend(self):
        import torch
        reader=torch.nn.Linear(1,1,bias=False).eval().requires_grad_(False)
        model=torch.nn.Linear(1,1,bias=False).eval().requires_grad_(False)
        return dict(reader=reader,dura=(model,None),reader_identity=dict(model='original_F0_E50',parameters=232993),
                    dura_identity=dict(deployment_id='dura_v1',parameters=266951))

    @staticmethod
    def observe(raw,**kwargs):
        assert set(kwargs)=={'physical_id','receiver','day','episode','mixture'}
        value=np.asarray(raw,dtype=np.float32).copy();value[:,255]=999
        return value,dict(scenario=SCENES[int(raw[0,1])%6],overlay_seed=7,input_rms=1.,geometry={})

    @staticmethod
    def encode(backend, values, path, device):
        from experiments.pamf_phase3_open_world.branch_export import DIMENSIONS
        assert np.all(values[:,:,255]==999), 'Raw clean source may not reach a model'
        labels=np.rint(values[:,0,0]).astype(int)-1
        sig=values[:,0,1].astype(int)
        blocks={}
        for k,d in DIMENSIONS.items():
            z=np.zeros((len(values),d),np.float32)
            z[np.arange(len(z)),labels]=2.
            z[:,6:]=np.random.default_rng(d+int(path=='P1')).normal(0,.03,size=(len(z),d-6))
            z[:,6]+=sig*.001
            if k=='reference29':z[:,27]=0.
            blocks[k]=z
        logits=np.eye(6,dtype=np.float32)[labels]*5
        return blocks,logits

    def test_raw_records_use_named_TX_sig_and_strict_original_layout(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            cfg,contract,metadata,payload=self.fixture(Path(tmp))
            rows=list(m.raw_records(payload,metadata['L_s'],contract))
            self.assertEqual(len(rows),720)
            raw,rx,day,sid=rows[0]
            self.assertEqual((raw.shape,raw.dtype),((2,256),np.dtype('float64')))
            self.assertEqual(raw[0,0],1);self.assertEqual(raw[0,1],0)
            self.assertEqual((rx,day,sid),('1-19','2021_03_08','L_s-0'))
            bad=metadata['L_s'].copy();bad['sig_i']=bad['sig_i'].copy();bad['sig_i'][0]=100000
            with self.assertRaises(ValueError):list(m.raw_records(payload,bad,contract))
            for shape in ((128,2),(257,2),(2,128)):
                with self.subTest(shape=shape),self.assertRaises(ValueError):m.raw_packet(np.ones(shape))

    def test_only_L_V_metadata_before_backend_and_no_unknown_controls(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            cfg,contract,meta,payload=self.fixture(Path(tmp))
            with patch.object(m,'COUNTS',dict(L_s=720,V=1080)):
                got,roles=m.source_metadata(cfg)
            self.assertEqual(set(roles),{'L_s','V'})
            np.testing.assert_array_equal(roles['L_s']['ids'],meta['L_s']['ids'])
            for key in ('target_root','oe_data_root','unknown_count','source_U'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(cfg,**{key:'forbidden'}))
            cfg['reader_config']['readout']='E51'
            with self.assertRaises(ValueError):m.validate_config(cfg)

    def test_original_F0_contract_does_not_require_later_external_contract_fields(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            cfg,contract,_,_=self.fixture(Path(tmp))
            # Actual archived F0 schema keeps these three in cache metadata,
            # unlike later counted/external contracts. Do not invent fields.
            for key in ('equalized','out_len','normalize'):contract.pop(key)
            Path(cfg['source_contract']).write_text(json.dumps(contract),encoding='utf-8')
            with patch.object(m,'COUNTS',dict(L_s=720,V=1080)):
                actual,roles=m.source_metadata(cfg)
            self.assertEqual(actual['split_seed'],392005);self.assertEqual(set(roles),{'L_s','V'})

    def test_source_pipeline_smoke_fit_freeze_then_V_and_aggregate_roundtrip(self):
        m=self.module();events=[];expected_sigma=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,contract,meta,payload=self.fixture(root);backend=self.backend()
            from experiments.pamf_phase3_open_world import source_open_readout as geo
            from experiments.pamf_phase3_open_world import star_relative_core as core
            original_fit=core.fit_metric
            original_scores=core.all_scores
            def fit(branches,y,geometry):events.append(('fit',len(y)));return original_fit(branches,y,geometry)
            def scores(branches,logits,geometry,metric):
                result=original_scores(branches,logits,geometry,metric)
                if len(logits)==720:expected_sigma.append(np.maximum(result.std(axis=0),1e-6))
                return result
            def smoke(*args):events.append(('smoke',0));return dict(status='PASS',synthetic_rows=2,source_IQ_reads=0)
            def load(*args):events.append(('raw',0));return payload
            def observe(raw,**kw):events.append(('observe',kw['physical_id']));return self.observe(raw,**kw)
            with patch.object(m,'COUNTS',dict(L_s=720,V=1080)),patch.object(m,'_backend',return_value=backend),\
                 patch.object(m,'_smoke',side_effect=smoke),patch.object(m,'_load_raw',side_effect=load),\
                 patch.object(m,'_observe',side_effect=observe),patch.object(m,'_encode',side_effect=self.encode),\
                 patch.object(geo,'fit_geometry',side_effect=lambda b,y:self.geometry()),patch.object(core,'fit_metric',side_effect=fit),\
                 patch.object(core,'all_scores',side_effect=scores):
                result=m.run(cfg,'cpu')
            self.assertEqual(events[0][0],'smoke');self.assertEqual(events[1][0],'raw')
            fits=[i for i,x in enumerate(events) if x[0]=='fit'];first_V=next(i for i,x in enumerate(events) if x==('observe','V-0'))
            self.assertEqual(len(fits),2);self.assertTrue(all(i<first_V for i in fits))
            self.assertTrue(all(events[i][1]==720 for i in fits))
            self.assertEqual(result['status'],'STAR_RELATIVE_SOURCE_FROZEN')
            self.assertEqual(result['source_U_cache_reads'],0);self.assertEqual(result['new_neural_updates'],0)
            self.assertEqual(sum(x[0]=='observe' for x in events),1800)
            original_load=np.load
            def aggregate_only(path,*a,**kw):
                self.assertNotIn('source_V',str(path));self.assertNotIn('metadata',str(path))
                return original_load(path,*a,**kw)
            for index,path in enumerate(('P0','P1')):
                with patch.object(np,'load',side_effect=aggregate_only):bundle=m.load_source(root/'source',path)
                self.assertEqual(set(bundle),{'marker','geometry','metric','thresholds','statistics'})
                self.assertEqual(bundle['thresholds'].shape,(7,));self.assertEqual(bundle['statistics']['anchor_units'].shape,(90,160))
                self.assertEqual(set(bundle['statistics']),{'eps','radius','sigma','anchor_units','anchor_scores','group_keys'})
                self.assertEqual(bundle['metric']['score_mean'].shape,(2,5))
                np.testing.assert_array_equal(bundle['statistics']['sigma'],expected_sigma[index])
                with np.load(root/'source/source_V'/path/'data.npz') as z:
                    self.assertEqual(z['scores'].shape,(1080,7));np.testing.assert_array_equal(z['closed'],z['baseline_closed'])
            (root/'source/by_path/P0/calibration/complete.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):m.load_source(root/'source','P0')

    def test_shared_backend_reads_contract_and_weights_only_without_source_cache(self):
        m=self.module()
        from experiments.pamf_phase3_open_world import runner
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_,_,_=self.fixture(root)
            backend_config={k:cfg[k] for k in ('reader_config','source_contract')}
            # No source_cache/raw/mixture handle is even passed to this API.
            reader=self.backend()['reader']
            with patch.object(m,'COUNTS',dict(L_s=720,V=1080)),patch.object(runner,'_load_model',return_value=reader) as load:
                result=m._backend(backend_config,'cpu',with_dura=False)
            self.assertIs(result['reader'],reader);self.assertIsNone(result['dura'])
            self.assertEqual(set(load.call_args.args[0]),m.READER_FIELDS)
            self.assertEqual(result['reader_identity']['updates'],11100)

    def test_known_calibration_42groups_intact_ties_and_global_maximum(self):
        m=self.module();records=[(y,rx,scene,k) for y in range(6) for rx in RX for scene in range(6) for k in range(50)]
        y=np.array([r[0] for r in records]);rx=np.array([r[1] for r in records]);scene=np.array([SCENES[r[2]] for r in records])
        scores=np.array([[0. if r[3]==0 else 1. for _ in range(7)] for r in records],np.float64)
        thresholds,report=m.calibrate_known(np.array([str(i) for i in range(len(y))]),y,y,scores,rx,scene)
        np.testing.assert_array_equal(thresholds,np.ones(7))
        self.assertTrue(report['constraints_passed']);self.assertEqual(report['group_count'],42)
        for row in report['methods'].values():
            self.assertTrue(row['no_rejection_feasible']);self.assertFalse(row['next_candidate_passed'])
            self.assertAlmostEqual(row['groups']['overall']['original_closed_minus_new_open'],.02)
        # Absent class in one RX x scene is uncertifiable, never silently PASS.
        keep=~((y==0)&(rx==1)&(scene==SCENES[0]))
        t,r=m.calibrate_known(np.array([str(i) for i in range(keep.sum())]),y[keep],y[keep],scores[keep],rx[keep],scene[keep])
        self.assertFalse(r['constraints_passed']);np.testing.assert_array_equal(t,np.zeros(7))
        bad=y.copy();bad[0]=6
        with self.assertRaises(ValueError):m.calibrate_known(np.array([str(i) for i in range(len(y))]),bad,y,scores,rx,scene)

    def test_missing_role_handle_rejected_before_weights_or_raw(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_,_,_=self.fixture(root);(root/'cache/V_metadata.npz').unlink()
            with patch.object(m,'COUNTS',dict(L_s=720,V=1080)),patch.object(m,'_backend',side_effect=AssertionError('No weights')),\
                 patch.object(m,'_load_raw',side_effect=AssertionError('No raw')),self.assertRaises((ValueError,FileNotFoundError)):
                m.run(cfg,'cpu')
            self.assertFalse((root/'source/complete.json').exists())

    def test_frozen_model_change_cannot_produce_complete(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_,_,payload=self.fixture(root);backend=self.backend()
            def encode(*args):
                import torch
                with torch.no_grad():backend['reader'].weight.add_(1)
                return self.encode(*args)
            with patch.object(m,'COUNTS',dict(L_s=720,V=1080)),patch.object(m,'_backend',return_value=backend),\
                 patch.object(m,'_smoke',return_value=dict(status='PASS')),patch.object(m,'_load_raw',return_value=payload),\
                 patch.object(m,'_observe',side_effect=self.observe),patch.object(m,'_encode',side_effect=encode),self.assertRaises(ValueError):
                m.run(cfg,'cpu')
            self.assertFalse((root/'source/complete.json').exists())

    def test_cli_process_identity_and_no_numpy_torch_bridge(self):
        m=self.module()
        with io.StringIO() as text,contextlib.redirect_stdout(text),patch.object(m,'_read',return_value={}),patch.object(m,'run',return_value=dict(status='STAR_RELATIVE_SOURCE_FROZEN')):
            m.main(['run','--config','synthetic.json','--device','cpu'])
            row=json.loads(text.getvalue().splitlines()[0])
        self.assertEqual(row['pid'],os.getpid());self.assertEqual(Path(row['cwd']),Path.cwd())
        from experiments.pamf_phase3_open_world.source_open_export import _tensor
        import torch
        with patch.object(torch,'as_tensor',side_effect=AssertionError('No NumPy ABI bridge')):
            t=_tensor(np.ones((2,2,256),np.float32),'cpu')
        self.assertEqual(tuple(t.shape),(2,2,256))


if __name__=='__main__':unittest.main()
