"""Protocol and orchestration checks with synthetic U/Q, never real data."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarNeighborhoodRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_neighborhood_run')
    def source_config(self,root):
        from test_pamf_phase3_star_consensus_run import StarConsensusRunTests
        c=StarConsensusRunTests().config(root);s=c['source_config'];s.update(run_id='20261029-phase3-star-neighborhood-s2026092701-r01',row_id='star-neighborhood-source-s2026092701')
        return dict(run_id=s['run_id'],code_commit='a'*40,source_config=s,L_cache_root=str(root/'cache/L'),
            paired_prediction_root=str(root/'paired'),class_risk_prediction_root=str(root/'class'),output_root=s['output_root'],batch_size=128)
    def target_config(self,root):
        return dict(run_id='20261029-phase3-star-neighborhood-s2026092701-r01',code_commit='a'*40,source_root=str(root/'source'),
            U_public_root=str(root/'pool/discovery/public'),U_capsule_id='Ucap',U_split_id='Usplit',
            Q_public_root=str(root/'pool/evaluation/public'),Q_capsule_id='Qcap',Q_split_id='Qsplit',
            output_root=str(root/'target'),batch_size=128,dura_checkpoint='G/final.pt',dura_release='G-release')
    def prior(self,m,config):
        return dict(geometry={'P0':{},'P1':{}},readouts={'value':np.array([1.])},
            thresholds=dict(schema='pamf_phase3_star_class_risk_state_v1',global_threshold=0.,conditional_thresholds=np.zeros(6)),
            marker=dict(status=m.SOURCE_STATUS,run_id=m.RUN_ID,code_commit='a'*40,model_config=config['source_config'],
                parent_F0_identity=dict(model='original_F0_E50',parameters=232993,source_output='F0/source'),
                dura_identity=dict(deployment_id='dura_v1',parameters=266951,checkpoint='G/final.pt',release_root='G-release'),source_known_42_passed=True),
            anchors=dict(embedding=np.zeros((90,688),np.float32),margin=np.ones(90),group_keys=np.array([(y,r,d) for y in range(6) for r in (1,3,4,6,8) for d in (1,2,3)],np.int64)))
    def state_diag(self,m,n=16):
        state=dict(schema=m.STATE_SCHEMA,cluster_embedding=np.zeros((2,688)),cluster_logits=np.zeros((2,6)),
            cluster_margin=np.array([1.,-1.]),cluster_radius=np.ones(2),anonymous_indices=np.array([1],np.int64),anonymous_penalty=np.array([.1]))
        labels=np.arange(n,dtype=np.int64)//8;assigned=np.where(labels==1,0,-1).astype(np.int64)
        diag=dict(cluster_labels=labels,assignments=assigned,candidate_mask=assigned>=0,cluster_sizes=np.array([8,8],np.int64),
            selected_cluster_count=2,selected_silhouette=.2,all_U_rows_used=True,clusters=[])
        return state,diag
    def target_patches(self,m,c,events,*,mutation=False):
        from contextlib import ExitStack
        stack=ExitStack();sc=self.source_config(Path(c['source_root']).parent);prior=self.prior(m,sc);state,diag=self.state_diag(m)
        def opened(config,role):
            events.append(('open',role))
            if role=='Q':self.assertTrue((Path(c['output_root'])/'adaptation/complete.json').exists())
            n=16 if role=='U' else 8;values=np.zeros((n,2,256),np.float32);values[:,0,0]=np.arange(n)
            return {},np.array([role+str(i) for i in range(n)]),values
        def evidence(backend,values,prior,device):
            index=values[:,0,0];embedding=np.tile(index[:,None],(1,688)).astype(np.float32)
            return embedding,np.eye(6,dtype=np.float32)[index.astype(int)%6]*5,np.where(index%2==0,.3,-.3).astype(np.float64)
        def fit(*args):events.append(('fit',len(args[0])));return state,diag
        def mask(embedding,state):return embedding[:,0]%2==0
        def infer(embedding,logits,margin,state,adapt,registered):
            if mutation:state['cluster_margin'][0]+=1
            l=logits.copy();active=mask(embedding,state)
            if adapt:l[active]=np.roll(l[active],1,axis=1)
            known=np.tanh(margin);old=(l-l.max(axis=1,keepdims=True))/30+known[:,None]
            joint=np.column_stack((old,np.full(len(old),-.2))) if registered else old
            return dict(prediction=np.where(joint.max(1)>=0,joint.argmax(1),-1).astype(np.int64),closed=l.argmax(1).astype(np.int64),
                knownness=known,joint_margins=joint.astype(np.float64))
        for p in [patch.object(m,'U_COUNT',16),patch.object(m,'Q_COUNT',8),patch.object(m,'load_source',return_value=prior),
            patch.object(m,'_target_backend',return_value={'reader':object(),'dura':(object(),object())}),
            patch.object(m,'_snapshot',return_value=[]),patch.object(m,'_check_models'),
            patch.object(m,'_target_smoke',side_effect=lambda *a:events.append(('smoke',)) or {'status':'PASS'}),
            patch.object(m,'_open',side_effect=opened),patch.object(m,'_target_evidence',side_effect=evidence),
            patch.object(m.core,'fit',side_effect=fit),patch.object(m.core,'infer',side_effect=infer),patch.object(m.core,'fusion_mask',side_effect=mask)]:stack.enter_context(p)
        return stack

    def test_exact_source_target_configs_and_role_boundaries(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);s=self.source_config(root);t=self.target_config(root)
            m.validate_source_config(s);m.validate_target_config(t)
            for key in ('truth','source_cache','unknown_count','support'):
                with self.subTest(target=key),self.assertRaises(ValueError):m.validate_target_config(dict(t,**{key:'x'}))
            bad=copy.deepcopy(t);bad['Q_public_root']=bad['U_public_root']
            with self.assertRaises(ValueError):m.validate_target_config(bad)
            for key in ('OE_root','V_root','novel_truth','Q_public_root'):
                with self.subTest(source=key),self.assertRaises(ValueError):m.validate_source_config(dict(s,**{key:'x'}))

    def test_source_only_L_aggregates_mean90_no_rowbank(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.source_config(Path(td));prior=self.prior(m,c);keys=prior['anchors']['group_keys'];n=90
            role=dict(ids=np.array(['L'+str(i) for i in range(n)]),metadata=dict(y=keys[:,0],rx_i=keys[:,1],day_i=keys[:,2]),manifest={})
            def evidence(backend,data,start,end,prior,device):
                ix=np.arange(start,end);return np.tile(ix[:,None],(1,688)).astype(np.float32),np.eye(6,dtype=np.float32)[ix%6],ix.astype(np.float64)
            def save(folder,value):folder.mkdir(parents=True);(folder/'stub').write_text('aggregate fixture')
            with patch.object(m,'SOURCE_L_COUNT',90),patch.object(m,'_source_prior',return_value=prior),\
                 patch.object(m,'_source_backend',return_value=({'reader':object(),'dura':None,'reader_identity':prior['marker']['parent_F0_identity']},{})),\
                 patch.object(m,'_snapshot',return_value=[]),patch.object(m,'_check_models'),\
                 patch.object(m,'_source_smoke',side_effect=lambda *a:events.append('smoke') or {'status':'PASS'}),\
                 patch.object(m,'_load_L',side_effect=lambda *a:events.append('L') or role),patch.object(m,'close_role'),\
                 patch.object(m,'_cache_evidence',side_effect=evidence),patch.object(m,'_save_prior',side_effect=save):
                result=m.source(c,'cpu')
            self.assertEqual(events,['smoke','L']);self.assertEqual(result['status'],m.SOURCE_STATUS)
            with np.load(Path(c['output_root'])/'anchors.npz') as z:
                self.assertEqual(set(z.files),{'embedding','margin','group_keys'});self.assertEqual(z['embedding'].shape,(90,688))
                np.testing.assert_array_equal(z['margin'],np.arange(90));np.testing.assert_array_equal(z['group_keys'],keys)

    def test_all_U_fit_registry_freeze_before_Q_and_current_DA_closed(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.target_config(Path(td))
            with self.target_patches(m,c,events):
                done=m.predict(c,'cpu');checked=m._predictions(c['output_root'])
            self.assertEqual(events[0],('smoke',));self.assertEqual(events[1],('open','U'));self.assertIn(('fit',16),events)
            self.assertEqual(events[-1],('open','Q'));self.assertEqual(done['Q_fusion_count'],4)
            p=checked[1];self.assertEqual(p['closed_prediction'].shape,(2,8));self.assertEqual(p['prediction'].shape,(4,8))
            self.assertFalse(np.array_equal(p['closed_prediction'][0],p['closed_prediction'][1]))
            for name in m.STATES:np.testing.assert_array_equal(p['knownness'][m.STATES.index(name)],p['joint_margins_'+name].max(1))
            with self.assertRaises(FileExistsError):m.predict(c,'cpu')

    def test_mutating_frozen_registry_stops_and_never_completes(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.target_config(Path(td))
            with self.target_patches(m,c,events,mutation=True),self.assertRaises(ValueError):m.predict(c,'cpu')
            self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_bad_source_parent_rejected_before_model_or_L(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.source_config(Path(td));root=Path(c['class_risk_prediction_root']);root.mkdir()
            (root/'complete.json').write_text(json.dumps({'status':'wrong','code_commit':'a'*40}))
            with patch.object(m,'_source_backend',side_effect=AssertionError('bad prior before F0')),self.assertRaises(ValueError):m.source(c,'cpu')


if __name__=='__main__':unittest.main()
