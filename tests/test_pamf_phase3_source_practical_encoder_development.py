from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.pamf_phase3_open_world import source_open_readout as core
from experiments.pamf_phase3_open_world import source_practical_development as parent


class Wrapper(torch.nn.Module):
    def __init__(self, identity):
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.encoder = torch.nn.Linear(1, 1)
        self.eval().requires_grad_(False)
        self._phase3_encoder_identity = deepcopy(identity)


class FrozenG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('factor', torch.tensor(2., dtype=torch.float32))
        self.eval().requires_grad_(False)

    def forward(self, x):
        return x * self.factor


class EncoderDevelopmentTests(unittest.TestCase):
    def module(self):
        name = 'experiments.pamf_phase3_open_world.source_practical_encoder_development'
        self.assertTrue((Path(__file__).resolve().parents[1] / (name.replace('.', '/') + '.py')).is_file())
        return importlib.import_module(name)

    def fixture(self, root, condition='C0', arm='practical_increment'):
        run = '20261029-phase3-source-practical-encoder-s2026092701-r01'
        train = root / run / 'rows' / arm / 'train'
        train.mkdir(parents=True)
        c = dict(run_id=run, code_commit='a'*40, arm=arm, condition=condition,
            view_family='clean_cross_receiver' if condition=='C0' else 'practical_cross_receiver',
            Q_public_root=str(root/'data/evaluation/public'), Q_capsule_id='capsule-'+condition,
            Q_split_id='split-'+condition, Q_expected_count=864, train_root=str(train),
            source_root=str(train.parent/'source'), parent_prediction_root=str(root/parent.RUN/'rows'/condition/'predict'),
            output_root=str(train.parent/'development'/condition/'predict'), batch_size=128,
            dura_checkpoint=None if condition=='C0' else str(root/'R69/final.pt'),
            dura_release=None if condition=='C0' else str(root/'frozen-G'))
        source_row = str(root/'20261029-phase3-source-embedding-readout-s2026092701-r01/rows/uniform_oe')
        risk_root = str(root/'20261029-phase3-source-baseline-risk-s2026092701-r01/rows/uniform_oe/predict')
        oldid = dict(run_id='20261029-phase3-source-embedding-increment-s2026092701-r02', arm='uniform_oe',
            code_commit='28d53ea9b38c19f642c59e31f178a3f65f40894d', updates=222, cumulative_updates=11322,
            checkpoint='original-E51/final.pt')
        identity = dict(run_id=run, arm=arm, code_commit=c['code_commit'], checkpoint=str(train/'final.pt'),
            updates=222, cumulative_updates=11544, parent_encoder_identity=oldid)
        dura = dict(deployment_id='dura_v1', checkpoint=str(root/'R69/final.pt'), release_root=str(root/'frozen-G'),
            arm='stage2', epoch=20, updates=4940, training_commit='b2a2ce937a64654e17e84c2aa817b9da4257a48f',
            parameters=266951, reuse=True, graph=True, native=False, precision='FP32', TF32=False)
        trainconfig = dict(run_id=run, arm=arm, code_commit=c['code_commit'], output_root=str(train),
            source_row_root=source_row, risk_prediction_root=risk_root)
        training = dict(status='SOURCE_PRACTICAL_ENCODER_FROZEN', schema='pamf_phase3_source_practical_encoder_v1',
            config=trainconfig, encoder_identity=identity, dura_identity=dura, updates=222, cumulative_updates=11544)
        (train/'complete.json').write_text(json.dumps(training), encoding='utf-8')
        geometry = dict(schema=core.GEOMETRY_SCHEMA, class_count=6, branches={})
        for name, dimension in core.DIMENSIONS.items():
            geometry['branches'][name] = dict(mean=np.zeros(dimension), scale=np.ones(dimension),
                prototypes=np.eye(6, dimension), whitener=np.eye(dimension))
        state = dict(schema=core.READOUT_SCHEMA, descriptor_dim=100, method_names=list(core.METHODS),
            mean=np.zeros(100), scale=np.ones(100), training={}, models=dict(
                linear_oe=dict(weight=np.zeros((1,100),np.float32),bias=np.ones(1,np.float32)),
                mlp16_oe=dict(weight1=np.zeros((16,100),np.float32),bias1=np.zeros(16,np.float32),
                    weight2=np.zeros((1,16),np.float32),bias2=np.ones(1,np.float32))))
        common = dict(schema='pamf_phase3_source_practical_encoder_evaluation_v1', run_id=run, arm=arm,
            code_commit=c['code_commit'], encoder_identity=identity, dura_identity=dura, source_only=True,
            method_names=list(parent.METHODS), view_order=['clean','practical_low_suburban','practical_mid_urban','practical_low_urban'])
        bundle = dict(geometry=geometry, state=state, thresholds=np.full(7,.5,np.float64), encoder_identity=identity,
            dura_identity=dura, fit_marker=dict(common,status='SOURCE_PRACTICAL_ENCODER_READOUTS_FROZEN'),
            calibration_marker=dict(common,status='SOURCE_PRACTICAL_ENCODER_THRESHOLDS_FROZEN'),
            source_marker=dict(common,status='SOURCE_PRACTICAL_ENCODER_SOURCE_COMPLETE',
                config=dict(train_root=c['train_root'],output_root=c['source_root']),
                frozen_parameters_buffers_unchanged=True,frozen_own_source_arrays_unchanged=True))
        oldsource = dict(marker=dict(encoder_state_identity=oldid), geometry=deepcopy(geometry), state={},
            thresholds=np.zeros(7,np.float64), source_row_root=source_row, risk_prediction_root=risk_root)
        ids = np.asarray(['p3dev-'+('%024x'%j) for j in range(864)])
        path = root/'synthetic-IQ.npy';np.save(path,np.zeros((864,2,256),np.float32))
        self.parent_predictions(c, oldsource, ids)
        return c,bundle,oldsource,ids,path

    def parent_predictions(self,c,source,ids):
        root=Path(c['parent_prediction_root']);root.mkdir(parents=True,exist_ok=True);n=len(ids)
        marker=dict(status='SOURCE_PRACTICAL_DEVELOPMENT_PREDICTIONS_COMPLETE',run_id=parent.RUN,
            condition=c['condition'],view_family=c['view_family'],count=n,state_names=list(parent.STATES),
            method_names=list(parent.METHODS),primary_state='augmented_readout',primary_method='mlp16_oe',
            query_fit=False,truth_read=False,source_arms_frozen_before_Q=True,
            frozen_parameters_buffers_unchanged=True,frozen_source_aggregates_unchanged=True,
            public_capsule_id=c['Q_capsule_id'],public_split_id=c['Q_split_id'],
            encoder_state_identity=source['marker']['encoder_state_identity'],source_row_root=source['source_row_root'],
            fit_root=str(root.parents[2]/'source/fit'),calibration_root=str(root.parents[2]/'source/calibration'))
        marker.update({k:0 for k in ('target_U_reads','formal_target_reads','source_IQ_reads','new_encoder_updates',
            'new_identity_head_updates','new_readout_updates','new_threshold_calibrations')})
        (root/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        values=np.stack([np.full((n,7),.25),np.full((n,7),.75),np.full((n,7),-1.)])
        thresholds=np.full((3,7),.5,np.float64);closed=np.zeros((3,n),np.int64);accepted=values>=thresholds[:,None,:]
        np.savez(root/'predictions.npz',ids=ids,state_names=np.asarray(parent.STATES),method_names=np.asarray(parent.METHODS),
            knownness=values,thresholds=thresholds,closed_prediction=closed,baseline_closed_prediction=closed,
            accepted=accepted,prediction=np.where(accepted,closed[:,:,None],-1).astype(np.int64))

    def encoded(self,reader,x):
        self.assertFalse(torch.is_grad_enabled())
        values={'features__'+name:np.ones((len(x),dimension),np.float32) for name,dimension in core.DIMENSIONS.items()}
        values['features__reference29'][:,27]=0.
        values['logits']=np.zeros((len(x),6),np.float32);values['logits'][:,1]=2.
        return values

    @contextmanager
    def backend(self,module,c,bundle,oldsource,ids,path,events):
        model=Wrapper(bundle['encoder_identity']);g=FrozenG()
        def public(base):
            events.append('Q')
            self.assertEqual(base['public_root'],c['Q_public_root']);self.assertEqual(base['input_paths'],[c['condition']])
            return {'count':864},ids,np.load(path,mmap_mode='r')
        def read_bundle(root):events.append('source_frozen');return bundle
        def readers(source,device):source['train_config']={'source_contract':'aggregate-only'};return Wrapper({}),Wrapper({})
        with patch.object(module,'_load_bundle',side_effect=read_bundle), \
             patch.object(module.transfer,'_source_bundle',return_value=oldsource), \
             patch.object(module.native,'load_trained',return_value=model), \
             patch.object(module.transfer,'_load_readers',side_effect=readers), \
             patch.object(module.transfer,'_load_dura',return_value=(g,lambda x:(g(x),))) as loadg, \
             patch.object(module.transfer,'_public',side_effect=public), \
             patch.object(module.transfer,'_encode_new',side_effect=self.encoded) as encode, \
             patch.object(core,'fit_readouts',side_effect=AssertionError('Q cannot fit')), \
             patch.object(core,'fit_geometry',side_effect=AssertionError('Q cannot fit geometry')):
            yield model,g,loadg,encode

    def test_two_states_reuse_augmented_parent_and_own_new_head_all_conditions(self):
        module=self.module()
        for arm,condition in ((a,c) for a in ('clean_increment','practical_increment') for c in ('C0','P1')):
            with self.subTest(arm=arm,condition=condition),tempfile.TemporaryDirectory() as tmp:
                c,bundle,oldsource,ids,path=self.fixture(Path(tmp),condition,arm);events=[]
                with self.backend(module,c,bundle,oldsource,ids,path,events) as (model,g,loadg,encode), \
                     patch.object(torch.Tensor,'numpy',side_effect=AssertionError('No Torch NumPy ABI')):
                    result=module.predict(c,'cpu')
                    self.assertLess(events.index('source_frozen'),events.index('Q'))
                    self.assertEqual(loadg.call_count,int(condition=='P1'))
                    self.assertEqual(sum(len(call.args[1]) for call in encode.call_args_list),866)
                    done,pred=module._predictions(c['output_root'])
                self.assertEqual(done['state_names'],['parent_E51','increment'])
                self.assertTrue(np.array_equal(pred['ids'],ids));self.assertTrue(np.all(pred['knownness'][0]==-1.))
                self.assertTrue(np.all(pred['thresholds'][0]==.5));self.assertTrue(np.all(pred['closed_prediction'][1]==1))
                self.assertTrue(np.all(pred['knownness'][1,:,6]==1.));self.assertTrue(np.all(pred['baseline_closed_prediction']==0))
                self.assertEqual(result['increment_closed_diff_from_parent_count'],864)
                self.assertEqual(result['source_IQ_reads'],0);self.assertFalse(result['query_fit'])

    def test_illegal_config_and_output_collision_never_load_sources_or_models(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            c,*_=self.fixture(Path(tmp))
            with patch.object(module,'_load_bundle',side_effect=AssertionError('No source load')):
                for change in ({'U_public_root':'forbidden'},{'truth':'forbidden'},{'unknown_count':3},
                        {'batch_size':64},{'Q_expected_count':863},{'condition':'P0'},
                        {'dura_checkpoint':'forbidden'},{'arm':'other'}):
                    with self.subTest(change=change),self.assertRaises(ValueError):module.predict(dict(c,**change),'cpu')
                Path(c['output_root']).mkdir(parents=True)
                with self.assertRaises(FileExistsError):module.predict(c,'cpu')

    def test_parent_reference_uses_actual_previous_source_fit_directory(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            c,bundle,oldsource,ids,path=self.fixture(Path(tmp))
            done,selected=module._reference(c,oldsource)
            self.assertEqual(Path(done['fit_root']).parent.name,'source')
            self.assertTrue(np.array_equal(selected['ids'],ids))
            self.assertTrue(np.all(selected['knownness']==-1.))
            marker=Path(c['parent_prediction_root'])/'complete.json'
            wrong=dict(done,fit_root=str(Path(done['fit_root']).parent.parent/'fit'))
            marker.write_text(json.dumps(wrong),encoding='utf-8')
            with self.assertRaises(ValueError):module._reference(c,oldsource)

    def test_bad_source_or_parent_identity_is_rejected_before_Q(self):
        module=self.module()
        for failure in ('source_not_frozen','encoder_arm','parent_encoder','parent_run','parent_capsule','parent_fit'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                c,bundle,oldsource,ids,path=self.fixture(Path(tmp));events=[]
                if failure=='source_not_frozen':bundle['fit_marker']['status']='PARTIAL'
                elif failure=='encoder_arm':bundle['encoder_identity']['arm']='clean_increment'
                elif failure=='parent_encoder':oldsource['marker']['encoder_state_identity']={'other':'encoder'}
                else:
                    marker=Path(c['parent_prediction_root'])/'complete.json';value=json.loads(marker.read_text())
                    value[{'parent_run':'run_id','parent_capsule':'public_capsule_id','parent_fit':'fit_root'}[failure]]='wrong'
                    marker.write_text(json.dumps(value),encoding='utf-8')
                with self.backend(module,c,bundle,oldsource,ids,path,events),self.assertRaises(ValueError):module.predict(c,'cpu')
                self.assertNotIn('Q',events);self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_smoke_mutation_and_query_alignment_cannot_complete(self):
        module=self.module()
        for failure in ('model_smoke','aggregate_smoke','ids_order'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                c,bundle,oldsource,ids,path=self.fixture(Path(tmp));events=[]
                actual=module._smoke
                def smoke(model,*args):
                    result=actual(model,*args)
                    if failure=='model_smoke':next(model.parameters()).add_(1.)
                    elif failure=='aggregate_smoke':bundle['thresholds'][0]+=.25
                    return result
                qids=ids[::-1].copy() if failure=='ids_order' else ids
                with self.backend(module,c,bundle,oldsource,qids,path,events), \
                     patch.object(module,'_smoke',side_effect=smoke),self.assertRaises(ValueError):module.predict(c,'cpu')
                if failure!='ids_order':self.assertNotIn('Q',events)
                self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_compact_actual_readout_API_never_requires_ids_or_old_score(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            c,bundle,old,ids,path=self.fixture(Path(tmp));model=Wrapper(bundle['encoder_identity'])
            with patch.object(module.transfer,'_encode_new',side_effect=self.encoded), \
                 patch.object(module.transfer,'_scores',side_effect=AssertionError('Old score API forbidden')):
                values,closed=module._compact(model,torch.zeros((2,2,256)),bundle)
            self.assertEqual((values.shape,values.dtype),((2,7),np.dtype('float64')))
            self.assertEqual(closed.tolist(),[1,1]);self.assertTrue(np.all(values[:,5:]==1.))

    def test_incomplete_or_inconsistent_prediction_never_reads_truth(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(module.transfer,'_truth',side_effect=AssertionError('No truth on partial')):
                with self.assertRaises(ValueError):module.score(root,root/'truth',root/'score',root/'targets')
            c,bundle,oldsource,ids,path=self.fixture(root);events=[]
            with self.backend(module,c,bundle,oldsource,ids,path,events):module.predict(c,'cpu')
            predpath=Path(c['output_root'])/'predictions.npz'
            with np.load(predpath,allow_pickle=False) as stored:p={k:stored[k].copy() for k in stored.files}
            p['accepted'][0,0,0]=~p['accepted'][0,0,0];np.savez(predpath,**p)
            with patch.object(module.transfer,'_truth',side_effect=AssertionError('No truth on inconsistent decisions')):
                with self.assertRaises(ValueError):module.score(c['output_root'],root/'truth',root/'score',root/'targets')

    def test_actual_truth_dictionary_clean95_practical90_and_later_diagnostic(self):
        module=self.module();targets=Path(__file__).resolve().parents[1]/'configs/pamf_phase3_targets_20261029.json'
        for condition,later_only in (('C0',False),('P1',False),('C0',True)):
            with self.subTest(condition=condition,later_only=later_only),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);c,bundle,oldsource,ids,path=self.fixture(root,condition);events=[]
                with self.backend(module,c,bundle,oldsource,ids,path,events):module.predict(c,'cpu')
                pp=Path(c['output_root'])/'predictions.npz'
                with np.load(pp,allow_pickle=False) as stored:p={k:stored[k].copy() for k in stored.files}
                y=np.tile(np.arange(9,dtype=np.int64),96);closed=np.minimum(y,5)
                good=y<6
                for label in ([8] if later_only else [6,7,8]):good[np.flatnonzero(y==label)[- (10 if later_only else 6):]]=True
                values=np.broadcast_to(good[None,:,None],(2,864,7)).astype(np.float64).copy()
                p.update(knownness=values,accepted=values>=p['thresholds'][:,None,:],
                    closed_prediction=np.tile(closed,(2,1)),baseline_closed_prediction=np.tile(closed,(2,1)))
                p['prediction']=np.where(p['accepted'],p['closed_prediction'][:,:,None],-1).astype(np.int64)
                np.savez(pp,**p)
                scenes=np.full(864,'clean') if condition=='C0' else np.repeat(['practical_low_suburban','practical_mid_urban','practical_low_urban'],288)
                # Spread the six false accepts two per practical scene.
                if condition=='P1':
                    values[:]=np.broadcast_to((y<6)[None,:,None],values.shape)
                    for label in (6,7,8):
                        for scene in np.unique(scenes):values[:,np.flatnonzero((y==label)&(scenes==scene))[-2:],:]=1.
                    p['knownness']=values;p['accepted']=values>=p['thresholds'][:,None,:]
                    p['prediction']=np.where(p['accepted'],p['closed_prediction'][:,:,None],-1).astype(np.int64);np.savez(pp,**p)
                truth=root/'private.npz';np.savez(truth,ids=ids[::-1],y=y[::-1],receiver=np.full(864,26,np.int64),scenario=scenes[::-1])
                result=module.score(c['output_root'],truth,root/'score',targets)
                self.assertEqual(result['condition_unknown_rejection_min'],.95 if condition=='C0' else .90)
                self.assertEqual(result['primary_pass'],condition=='P1' or later_only)
                self.assertIn('increment_minus_parent_E51',result['deltas'])
                self.assertFalse(result['anonymous_registration_performed']);self.assertEqual(result['target_U_reads'],0)
                if later_only:self.assertFalse(result['states']['increment']['later_unknown_risk']['at_condition_target'])
                np.savez(truth,ids=ids,y=y,receiver=np.full(864,27,np.int64),scenario=scenes)
                with self.assertRaises(ValueError):module.score(c['output_root'],truth,root/'bad_receiver',targets)
                np.savez(truth,ids=ids,y=y+1,receiver=np.full(864,26,np.int64),scenario=scenes)
                with self.assertRaises(ValueError):module.score(c['output_root'],truth,root/'bad_mapping',targets)


if __name__=='__main__':unittest.main()
