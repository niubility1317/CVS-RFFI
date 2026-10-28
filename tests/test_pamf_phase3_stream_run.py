import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


def bundle():
    p=np.zeros((4,160));p[0,0]=1;p[1,0]=-1;p[2,1]=1;p[3,1]=-1
    return dict(schema='phase3_stream_geometry_v1',feature_dim=160,class_count=4,source_count=3360,
        source_counts=np.full(4,840,dtype=np.int64),calibration_count=14400,prototypes=p,
        covariance=np.eye(160),chol=np.eye(160),whitener=np.eye(160),anchors=p.copy(),lambda_det=.5,lambda_cluster=.125)


def marker():
    return dict(status='SOURCE_STREAM_INPUTS_EXPORTED',source_only=True,target_reads=0,optimizer_updates=0,
        head_changed=False,train_count=3360,calibration_count=14400,discovery_count=1050,future_count=5400,
        parent_epochs=20,parent_updates=540)


def dump(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


class StreamRunTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.stream_run')

    def setup_paths(self,root):
        module=self.module();export=root/'export';export.mkdir();dump(export/'complete.json',marker())
        p=bundle()['prototypes'].astype(np.float32)
        np.savez(export/'source_inputs.npz',train_features=np.repeat(p,840,axis=0),
            train_y=np.repeat(np.arange(4,dtype=np.int64),840),calibration_features=np.repeat(p,3600,axis=0))
        calibration=root/'calibration'
        with patch.object(module.geometry,'fit_source',return_value=bundle()):
            module.calibrate(export/'source_inputs.npz',calibration)
        for role,count in [('discovery',1050),('evaluation',5400)]:
            pub=export/role/'public';pub.mkdir(parents=True)
            ids=np.array([role+'-opaque-'+str(i) for i in range(count)])
            f=np.tile(p,(int(np.ceil(count/4)),1))[:count].copy()
            logits=np.zeros((count,4),np.float32);logits[np.arange(count),np.arange(count)%4]=3
            np.savez(pub/'features.npz',ids=ids,features=f,logits=logits)
        config=root/'model.json';dump(config,{'fixture_model':True})
        return dict(source=export/'source_inputs.npz',bundle=calibration/'bundle.npz',
            discovery=export/'discovery/public/features.npz',future=export/'evaluation/public/features.npz',
            model_config=config,output=root/'predict')

    def state(self,adapt):
        b=bundle();return dict(schema=b['schema'],feature_dim=160,class_count=4,adapt_domain=adapt,
            b=np.zeros(160),old_centers=b['anchors'].copy(),free_centers=np.empty((0,160)),
            free_ids=np.array([],dtype='U1'),qualified=np.array([],bool),anonymous_ids=np.array([],np.int64),
            U_ids=np.array(['discovery-opaque-'+str(i) for i in range(1050)]),assignments=np.zeros(1050,np.int64),
            converged=True,cap_reached=False,promotion_eligible=False,rounds=2)

    def test_calibration_is_aggregate_only_and_rejects_missing_export_first(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=self.setup_paths(root)
            with np.load(paths['bundle'],allow_pickle=False) as z:
                self.assertNotIn('train_features',z.files);self.assertNotIn('train_y',z.files)
                self.assertEqual(z['calibration_count'].item(),14400)
            done=json.loads((root/'calibration/complete.json').read_text())
            self.assertEqual(done['status'],'SOURCE_STREAM_CALIBRATED')
            self.assertEqual(done['U_reads'],0);self.assertEqual(done['future_reads'],0)
            (root/'export/complete.json').unlink()
            with patch.object(m,'_npz',side_effect=AssertionError('source opened too early')):
                with self.assertRaises(ValueError):m.calibrate(paths['source'],root/'missing')

    def test_states_are_frozen_before_future_opens_and_no_source_io(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=self.setup_paths(root);events=[];original=m._npz
            def loader(path,keys):
                path=Path(path)
                if path==paths['future']:
                    self.assertTrue((paths['output']/'states.json').is_file())
                    self.assertTrue((paths['output']/'registry.npz').is_file())
                    events.append('future')
                self.assertNotEqual(path,paths['source'])
                return original(path,keys)
            def fit(b,f,l,ids,adapt_domain):
                events.append('D1' if adapt_domain else 'D0');return self.state(adapt_domain)
            with patch.object(m,'_npz',side_effect=loader),patch.object(m.geometry,'fit_stream',side_effect=fit),\
                    patch.object(m,'_load_frozen_head',return_value=object()),\
                    patch.object(m,'_head_logits',side_effect=lambda model,f: np.tile(np.array([[3,0,0,0]],np.float32),(len(f),1))):
                result=m.predict(**{k:paths[k] for k in ('bundle','discovery','future','model_config','output')})
            self.assertEqual(events,['D0','D1','future'])
            self.assertEqual(result['status'],'ALL_STREAM_PREDICTIONS_COMPLETE')
            self.assertFalse(result['truth_read']);self.assertFalse(result['query_fit'])
            with np.load(paths['output']/'future_predictions.npz',allow_pickle=False) as p:
                self.assertEqual(p['prediction'].shape,(5400,5))
                np.testing.assert_array_equal(p['closed_prediction'][:,0],p['baseline_closed'])
                np.testing.assert_array_equal(p['closed_prediction'][:,1],p['baseline_closed'])
                np.testing.assert_array_equal(p['closed_prediction'][:,3],p['baseline_closed'])
                np.testing.assert_array_equal(p['closed_prediction'][:,2],p['closed_prediction'][:,4])
            with np.load(paths['output']/'registry.npz',allow_pickle=False) as r:
                self.assertEqual(len(r.files),8)

    def test_future_U_overlap_rejects_without_completion_and_output_cannot_overwrite(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=self.setup_paths(root)
            with np.load(paths['future'],allow_pickle=False) as z:values={k:z[k] for k in z.files}
            values['ids'][0]='discovery-opaque-0';np.savez(paths['future'],**values)
            with patch.object(m.geometry,'fit_stream',side_effect=lambda *a,**kw:self.state(kw['adapt_domain'])),\
                    patch.object(m,'_load_frozen_head',return_value=object()):
                with self.assertRaises(ValueError):m.predict(**{k:paths[k] for k in ('bundle','discovery','future','model_config','output')})
            self.assertFalse((paths['output']/'complete.json').exists())
            with self.assertRaises(FileExistsError):m.predict(**{k:paths[k] for k in ('bundle','discovery','future','model_config','output')})

    def test_private_feature_fields_reject_before_fit_and_future_is_not_opened(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=self.setup_paths(root)
            with np.load(paths['discovery'],allow_pickle=False) as z:values={k:z[k] for k in z.files}
            np.savez(paths['discovery'],**values,y=np.zeros(1050,np.int64))
            with patch.object(m.geometry,'fit_stream',side_effect=AssertionError('illegal fit')):
                with self.assertRaises(ValueError):m.predict(**{k:paths[k] for k in ('bundle','discovery','future','model_config','output')})
            self.assertFalse((paths['output']/'complete.json').exists())

    def test_nonzero_domain_shift_is_sent_to_same_head_and_shared_by_R0_R1(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=self.setup_paths(root)
            def fit(*args,adapt_domain):
                state=self.state(adapt_domain)
                if adapt_domain:state['b'][0]=.25
                return state
            def logits(model,f):
                self.assertEqual(f.shape,(5400,160));self.assertAlmostEqual(float(f[0,0]),.75)
                return np.tile(np.array([[0,0,0,3]],np.float32),(5400,1))
            with patch.object(m.geometry,'fit_stream',side_effect=fit),\
                    patch.object(m,'_load_frozen_head',return_value=object()) as load,\
                    patch.object(m,'_head_logits',side_effect=logits) as head:
                m.predict(**{k:paths[k] for k in ('bundle','discovery','future','model_config','output')})
            self.assertEqual(load.call_count,1);self.assertEqual(head.call_count,1)
            with np.load(paths['output']/'future_predictions.npz',allow_pickle=False) as p:
                np.testing.assert_array_equal(p['closed_prediction'][:,2],np.full(5400,3))
                np.testing.assert_array_equal(p['closed_prediction'][:,4],p['closed_prediction'][:,2])
                self.assertTrue(np.any(p['baseline_closed']!=p['closed_prediction'][:,2]))

    def test_synthetic_real_export_geometry_and_truth_last_score_integrate(self):
        from test_pamf_phase3_stream_export import StreamExportTests
        from experiments.pamf_phase3_open_world import stream_export,stream_scoring
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config,contract,meta,selected,model,x=StreamExportTests().fixture(root)
            with patch.object(stream_export.branch_export,'build_model',return_value=model),\
                    patch.object(stream_export,'_load_labeled',return_value=(meta['L_s'],x)):
                stream_export.export(config,'cpu')
            export=root/'output';cal=root/'calibration';pred=root/'predictions';score=root/'score'
            m.calibrate(export/'source_inputs.npz',cal)
            with patch.object(m,'_load_frozen_head',return_value=model):
                m.predict(cal/'bundle.npz',export/'discovery/public/features.npz',
                    export/'evaluation/public/features.npz',root/'unused-model.json',pred)
            result=stream_scoring.score(pred/'future_predictions.npz',pred/'discovery_predictions.npz',
                pred/'registry.npz',export/'discovery/truth/truth.npz',export/'evaluation/truth/truth.npz',score)
            self.assertEqual(result['status'],'SCORED_SOURCE_STREAM_COMPLETE')
            self.assertEqual(result['future_count'],5400);self.assertEqual(result['discovery_count'],1050)
            self.assertEqual(set(result['states']),set(m.STATES))
            self.assertTrue(result['truth_last']);self.assertFalse(result['final_six_class_method_selected'])


if __name__=='__main__':unittest.main()
