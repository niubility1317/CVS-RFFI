"""Frozen-F0 source consensus orchestration with synthetic role fixtures."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarConsensusRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_consensus_run')
    def config(self,root):
        from experiments.pamf_phase3_open_world.source_open_embedding_train import CLASSES
        run='20261029-phase3-star-consensus-s2026092701-r01';output=str(root/'out')
        source=dict(run_id=run,row_id='star-consensus-s2026092701',model_seed=2026092701,base_release='release-F0',
            frozen_sources={k:k for k in ('F','AF','BASE','FOLD','G')},source_output='F0/source',source_contract='contract.json',
            source_cache='old-clean-cache-MUST-NOT-OPEN',registered_classes=CLASSES.copy(),code_commit='a'*40,batch_size=128,output_root=output)
        return dict(run_id=run,code_commit='a'*40,source_config=source,cache_root=str(root/'cache'),output_root=output,batch_size=128)
    def role(self,role):
        from experiments.pamf_phase3_open_world import star_relative_source as old
        n=180 if role=='V' else 30
        rows=[(y,r,s) for y in range(6) for r in old.RECEIVERS for s in range(6)] if role=='V' else None
        rx=np.array([r[1] for r in rows],np.int64) if rows else np.array(old.RECEIVERS,np.int64)[np.arange(n)%15//3]
        day=np.array([r[2]%3+1 for r in rows],np.int64) if rows else np.arange(n,dtype=np.int64)%3+1
        y=np.array([r[0] for r in rows],np.int64) if rows else np.arange(n,dtype=np.int64)%6
        metadata=dict(rx_i=rx,day_i=day,domain=np.searchsorted(old.RECEIVERS,rx)*3+day-1)
        if role in ('L','V'):metadata['y']=y
        if role=='V':metadata['scenario']=np.array([old.SCENES[r[2]] for r in rows])
        x=np.zeros((n,2,256),np.float32);x[:,0,0]=y;x[:,0,1]=np.arange(n)
        if role=='novel':x[:,0,2]=-10
        manifest=dict(role=role,count=n,source_only=True,received_input_path='P0',restored_input_path='P1',
            parent_F0_identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,source_output='F0/source'),
            dura_identity=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940))
        result=dict(manifest=manifest,ids=np.array([role+str(i) for i in range(n)]),received=x,restored=x.copy(),metadata=metadata)
        if role=='L':result['teacher_logits']=np.zeros((n,6),np.float32)
        if role=='V':result['baseline_closed']=np.column_stack((y,(y+1)%6)).astype(np.int64)
        return result
    def patches(self,m,c,events,fail_novel=False,mutate=False):
        from contextlib import ExitStack
        stack=ExitStack();roles={r:self.role(r) for r in ('L','OE','V','novel')}
        def load(root,role):
            events.append(('load',role))
            if role=='V':self.assertEqual(sum(e[0]=='fit' for e in events),2)
            if role=='novel':
                for path in m.PATHS:self.assertTrue((Path(c['output_root'])/'by_path'/path/'calibration/complete.json').exists())
                if fail_novel:raise ValueError('bad public novel')
            return roles[role]
        def encode(reader,x,device):
            y=x[:,0,0].astype(int);shift=int(np.all(x[:,1,255]==1))
            return {'fused160':x[:,:1,:3].reshape(len(x),3).copy()},np.eye(6,dtype=np.float32)[(y+shift)%6]*5
        # Mark restored P1 while retaining the declared same physical rows.
        for row in roles.values():row['restored'][:,1,255]=1
        def desc(branches,logits,geometry):return np.tile(np.arange(100,dtype=np.float32),(len(logits),1))
        def fit(a,b,kd,od,model_seed):
            events.append(('fit',len(a),len(b)))
            return dict(value=np.array([1.]),training=dict(steps=200,known_count=len(a),oe_count=len(b),regime='source_role_domain_balanced_v1',
                parameter_counts={'linear_oe':101,'mlp16_oe':1633},losses={k:[.5]*200 for k in ('linear_oe','mlp16_oe')}))
        def score(b,l,g,state):
            if mutate:state['value'][0]+=1
            return np.tile(b['fused160'][:,2,None]+1.,(1,7)).astype(np.float64)
        def calibrate(ids,y,closed,baseline,scores,rx,scene):
            events.append(('calibrate',len(ids)))
            np.testing.assert_array_equal(closed,baseline)
            return np.zeros(7),dict(group_count=42,known_only=True,unknown_inputs_used=False,
                constraints_passed=True,methods={k:dict(threshold=0.,constraints_passed=True) for k in m.METHODS})
        def save(output,geometry,state):
            m._npz(output/'geometry.npz',{'value':geometry['value']});m._npz(output/'readouts.npz',{'value':state['value']})
        for p in [patch.object(m,'COUNTS',{'L':30,'OE':30,'V':180,'novel':30}),
            patch.object(m,'_backend',return_value=({'reader':object(),'dura':None,'reader_identity':roles['L']['manifest']['parent_F0_identity']},{})),
            patch.object(m,'_snapshot',return_value=[]),patch.object(m,'_check_models'),
            patch.object(m,'_smoke',side_effect=lambda *a:events.append(('smoke',)) or {'status':'PASS'}),
            patch.object(m,'load_role',side_effect=load),patch.object(m,'close_role'),patch.object(m,'_encode',side_effect=encode),
            patch.object(m,'_branches',side_effect=lambda reader,x,device,batch:encode(reader,x,device)),
            patch.object(m.core,'fit_geometry',side_effect=lambda b,y:{'value':np.array([float(len(y))])}),
            patch.object(m.core,'descriptor',side_effect=desc),patch.object(m.core,'fit_readouts',side_effect=fit),
            patch.object(m.core,'all_scores',side_effect=score),patch.object(m.core,'calibrate_known',side_effect=calibrate),
            patch.object(m,'_save',side_effect=save)]:stack.enter_context(p)
        return stack

    def test_strict_six_field_config_blocks_truth_target_increment_and_old_output(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));self.assertIs(m.validate_config(c),c)
            for key in ('truth','target_root','checkpoint','unknown_count','novel_truth'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(c,**{key:'x'}))
            for key in ('truth','trained_checkpoint','retention'):
                bad=copy.deepcopy(c);bad['source_config'][key]='x'
                with self.subTest(nested=key),self.assertRaises(ValueError):m.validate_config(bad)
            bad=copy.deepcopy(c);bad['output_root']=str(Path(c['cache_root'])/'L/new');bad['source_config']['output_root']=bad['output_root']
            with self.assertRaises(ValueError):m.validate_config(bad)

    def test_both_fits_then_V_then_both_thresholds_then_novel_and_marker_last(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));writes=[];real=m._write
            with self.patches(m,c,events),patch.object(m,'_write',side_effect=lambda p,v:writes.append(str(p)) or real(p,v)):
                result=m.run(c,'cpu')
            self.assertEqual(events[0],('smoke',));self.assertEqual([e[1] for e in events if e[0]=='load'],['L','OE','V','novel'])
            self.assertEqual(result['status'],m.STATUS);self.assertEqual(result['primary_path'],'P1');self.assertEqual(result['primary_method'],'mlp16_oe')
            self.assertEqual(result['new_encoder_updates'],0);self.assertEqual(result['source_novel_truth_reads'],0)
            self.assertEqual(writes[-1],str(Path(c['output_root'])/'complete.json'))
            with np.load(Path(c['output_root'])/'predict/known_predictions.npz') as z:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS);self.assertEqual(z['scores'].shape,(180,2,7))
                np.testing.assert_array_equal(z['baseline_closed'],z['closed']);self.assertFalse(np.array_equal(z['closed'][:,0],z['closed'][:,1]))
            with np.load(Path(c['output_root'])/'predict/novel_predictions.npz') as z:
                self.assertEqual(set(z.files),m.NOVEL_FIELDS);self.assertFalse({'tx','y','scenario'}&set(z.files))
            for path in m.PATHS:
                report=json.loads((Path(c['output_root'])/'by_path'/path/'readouts.json').read_text())
                self.assertEqual(len(report['training']['losses']['mlp16_oe']),200)
            with self.assertRaises(FileExistsError):m.run(c,'cpu')

    def test_novel_failure_retains_all_frozen_source_states_without_complete(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            with self.patches(m,c,events,fail_novel=True),self.assertRaises(ValueError):m.run(c,'cpu')
            self.assertFalse((Path(c['output_root'])/'complete.json').exists())
            for path in m.PATHS:self.assertTrue((Path(c['output_root'])/'by_path'/path/'calibration/complete.json').exists())

    def test_aggregate_mutation_during_V_blocks_novel(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            with self.patches(m,c,events,mutate=True),self.assertRaisesRegex(ValueError,'aggregate'):
                m.run(c,'cpu')
            self.assertNotIn(('load','novel'),events)

    def test_uncertifiable_known_risk_still_completes_fixed_predictions(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            report=dict(constraints_passed=False,methods={name:dict(threshold=0.,constraints_passed=False) for name in m.METHODS})
            with self.patches(m,c,events),patch.object(m.core,'calibrate_known',return_value=(np.zeros(7),report)):
                result=m.run(c,'cpu')
            self.assertEqual(result['status'],m.STATUS);self.assertIn(('load','novel'),events)
            saved=json.loads((Path(c['output_root'])/'by_path/P1/calibration/complete.json').read_text())
            self.assertFalse(saved['constraints_passed'])

    def test_samepath_cached_F0_baseline_mismatch_stops_before_novel(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            with self.patches(m,c,events),patch.object(m,'_score',return_value=(np.zeros((180,7)),np.zeros(180,np.int64))),\
                 self.assertRaisesRegex(ValueError,'baseline'):
                m.run(c,'cpu')
            self.assertNotIn(('load','novel'),events);self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_old_F0_backend_uses_only_frozen_loader_not_increment_or_G(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));reader=object();parent={'config':{'run_id':m.PARENT_RUN,'row_id':m.PARENT_ROW}}
            with patch.object(m.native,'_contract',return_value={}),patch.object(m.runner,'_load_model',return_value=reader) as loader,\
                 patch.object(m.runner,'_read',return_value=parent):backend,contract=m._backend(c['source_config'],'cpu')
            self.assertIs(backend['reader'],reader);self.assertIsNone(backend['dura']);self.assertEqual(loader.call_count,1)
            parent['config']['run_id']='old-E51'
            with patch.object(m.native,'_contract',return_value={}),patch.object(m.runner,'_load_model',return_value=reader),\
                 patch.object(m.runner,'_read',return_value=parent),self.assertRaises(ValueError):m._backend(c['source_config'],'cpu')

    def test_synthetic_smoke_passes_observe_before_frozen_F_without_G(self):
        m=self.module();events=[]
        from experiments.pamf_phase3_open_world import star_channel
        def observe(raw,**kwargs):events.append(('observe',kwargs));return np.asarray(raw,np.float32)*.5,{}
        def encode(reader,x,device):
            events.append(('F',len(x)));return {k:np.zeros((2,d),np.float32) for k,d in m.DIMENSIONS.items()},np.ones((2,6),np.float32)
        with patch.object(star_channel,'observe',side_effect=observe),patch.object(m,'_encode',side_effect=encode):
            result=m._smoke({'reader':object()},'cpu')
        self.assertEqual([e[0] for e in events],['observe','observe','F']);self.assertEqual(result['G_forward_rows'],0)
        self.assertEqual(result['real_cache_IQ_reads'],0);self.assertEqual(result['status'],'PASS')

    def test_real_core_and_existing_aggregate_codec_keep_source_state_and_scores(self):
        m=self.module()
        from experiments.pamf_phase3_open_world import source_consensus_run as codec
        from experiments.pamf_phase3_open_world import source_open_readout
        rng=np.random.default_rng(46);n=30;y=np.arange(n,dtype=np.int64)%6;domain=np.arange(n,dtype=np.int64)%15
        branches={key:rng.normal(0,.1,(n,d)).astype(np.float32) for key,d in m.DIMENSIONS.items()}
        for value in branches.values():value[np.arange(n),y]+=2
        branches['reference29'][:,27]=0;logits=np.eye(6,dtype=np.float32)[y]*4
        g=m.core.fit_geometry(branches,y);a=m.core.descriptor(branches,logits,g)
        b=a+rng.normal(0,.1,a.shape).astype(np.float32)
        state=m.core.fit_readouts(a,b,domain,domain,model_seed=2026092701)
        self.assertEqual(set(state['training']['losses']),{'linear_oe','mlp16_oe'})
        for values in state['training']['losses'].values():self.assertEqual(len(values),200)
        before=copy.deepcopy(dict(g=g,state=state));scores=m.core.all_scores(branches,logits,g,state)
        with tempfile.TemporaryDirectory() as td:
            m._save(Path(td),g,state);g2,state2=codec._load_new(td)
            np.testing.assert_array_equal(scores,m.core.all_scores(branches,logits,g2,state2))
            for file in ('geometry.npz','readouts.npz'):
                with np.load(Path(td)/file,allow_pickle=False) as z:self.assertTrue(all(z[k].dtype.kind!='O' for k in z.files))
        self.assertTrue(m.previous._same(g,before['g']));self.assertTrue(m.previous._same(state,before['state']))


if __name__=='__main__':unittest.main()
