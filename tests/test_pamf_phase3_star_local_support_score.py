"""Synthetic fixed-bank evidence, truth isolation and primary-only decisions."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class LocalSupportScoreTests(unittest.TestCase):
    def fixture(self,root,main_pass=True):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_local_support_score')
        identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
            source_output='original/source',source_contract='contract',base_release='release')
        dura=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940)
        bank={}
        for role,axis in (('L',0),('OE',1)):
            n=40;features=np.zeros((n,160),np.float32);features[:,axis]=1
            bank.update({role+'_ids':np.asarray([role+str(i) for i in range(n)]),role+'_unit_features':features,
                role+'_rx_i':np.repeat(np.asarray(m.RECEIVERS,np.int64),8)})
        rows=[(c,r,s) for c in range(6) for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        known=dict(ids=np.asarray(['V'+str(i) for i in range(n)]),y=np.asarray([v[0] for v in rows],np.int64),
            rx_i=np.asarray([v[1] for v in rows],np.int64),day_i=(np.arange(n)%3+1).astype(np.int64),scenario=np.asarray([v[2] for v in rows]))
        known['closed']=known['y'].copy();known['baseline_closed']=known['y'].copy();known['baseline_closed'][0]=1
        known['default_logits']=np.eye(6,dtype=np.float32)[known['closed']]*5
        known['unit_features']=np.zeros((n,160),np.float32);known['unit_features'][:,0]=1
        rows=[(tx,r,s) for tx in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];nu=len(rows)
        novel=dict(ids=np.asarray(['novel'+str(i) for i in range(nu)]),rx_i=np.asarray([v[1] for v in rows],np.int64),
            day_i=(np.arange(nu)%3+1).astype(np.int64),closed=np.zeros(nu,np.int64),
            default_logits=np.tile(np.asarray([5,0,0,0,0,0],np.float32),(nu,1)))
        novel['unit_features']=np.zeros((nu,160),np.float32);novel['unit_features'][:,1]=1
        truth=dict(ids=novel['ids'][::-1],tx=np.asarray([v[0] for v in rows])[::-1],rx_i=novel['rx_i'][::-1],
            day_i=novel['day_i'][::-1],scenario=np.asarray([v[2] for v in rows])[::-1])
        counts=dict(L=40,OE=40,V=n,novel=nu);cache=root/'cache';handles={}
        for role,data in (('L',None),('OE',None),('V',known),('novel',novel)):
            folder=cache/role;folder.mkdir(parents=True)
            ids=bank[role+'_ids'] if data is None else data['ids'];rx=bank[role+'_rx_i'] if data is None else data['rx_i']
            day=(np.arange(len(ids))%3+1).astype(np.int64) if data is None else data['day_i']
            meta=dict(rx_i=rx,day_i=day,domain=np.searchsorted(m.RECEIVERS,rx)*3+day-1)
            if role=='L':meta['y']=np.arange(len(ids),dtype=np.int64)%6
            if role=='V':meta.update(y=known['y'],scenario=known['scenario'])
            manifest=dict(schema=m.CACHE_SCHEMA,status=m.CACHE_STATUS,run_id=m.CACHE_RUN,role=role,count=len(ids),
                source_only=True,all_new_IQ_satellite=True,received_input_path='P0',restored_input_path='P1',
                parent_F0_identity=identity,dura_identity=dura)
            handles[role]=manifest;(folder/'manifest.json').write_text(json.dumps(manifest));np.save(folder/'ids.npy',ids)
            np.savez(folder/'metadata.npz',**meta)
            if role=='V':np.save(folder/'baseline_closed.npy',np.column_stack((known['y'],known['baseline_closed'])))
        teacher=root/'teacher';teacher.mkdir()
        for role in ('L','OE'):np.savez(teacher/(role+'.npz'),ids=bank[role+'_ids'],unit_features=bank[role+'_unit_features'])
        tm=dict(schema=m.TEACHER_SCHEMA,status=m.TEACHER_STATUS,run_id=m.TEACHER_RUN,code_commit=m.TEACHER_CODE,
            source_only=True,target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,OE_truth_reads=0,
            parent_F0_identity=identity,dura_identity=dura,cache_handles={k:handles[k] for k in ('L','OE')},
            source_L_count=40,source_OE_count=40,input_path='cached_P1_only',feature='unit_fused160',
            model_parameters=232993,model_buffers_unchanged=True,encoder_updates=0,identity_head_updates=0,G_updates=0,G_forward_rows=0)
        (teacher/'complete.json').write_text(json.dumps(tm))
        pred=root/'pred';(pred/'calibration').mkdir(parents=True);np.savez(pred/'source_bank.npz',**bank)
        tau=np.asarray([[-1.]*6,[0.]*6,[0. if main_pass else 100.]*6],np.float64)
        state={role:bank[role+'_unit_features'].astype(np.float64) for role in ('L','OE')}
        state.update(L_rx=bank['L_rx_i'],OE_rx=bank['OE_rx_i'])
        for role,data in (('known',known),('novel',novel)):
            data['raw_scores']=m.core.score(data['unit_features'],state)
            data['scores']=data['raw_scores']-tau[:,data['closed']].T
            np.savez(pred/(role+'_predictions.npz'),**data)
        np.savez(pred/'thresholds.npz',conditional=tau)
        cal=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            method_names=list(m.METHODS),conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,
            source_only=True,target_reads=0,group_count=42,methods={name:dict(constraints_passed=True,
                conditional_thresholds=tau[j].tolist(),groups={key:{} for key in m.GROUP_NAMES}) for j,name in enumerate(m.METHODS)})
        (pred/'calibration/complete.json').write_text(json.dumps(cal))
        marker=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            model_identity=identity,teacher_run_id=m.TEACHER_RUN,teacher_code_commit=m.TEACHER_CODE,teacher_root=str(teacher),
            cache_root=str(cache),cache_identities=handles,dura_identity=dura,known_count=n,novel_count=nu,bank_L_count=40,bank_OE_count=40,
            method_names=list(m.METHODS),primary_method=m.PRIMARY,classification_path='P1',source_only=True,
            score_threshold=0.,parameters=232993,default_head_replaced=False,source_U_reads=0,received_P0_values_read=0,
            source_bank_uses_OE_membership=True,source_bank_uses_TX_labels=False,
            all_thresholds_frozen_before_novel=True,source_bank_frozen_before_V=True,source_bank_unchanged=True,model_parameters_buffers_unchanged=True,
            target_reads=0,source_novel_truth_reads=0,neural_updates=0,G_updates=0,G_forward_rows=0,F_forward_rows=n+nu+2,
            cpu_readout_seconds=1.25,cpu_readout_rows=n+nu,
            new_V_forward_rows=n,new_novel_forward_rows=nu,synthetic_P0_forward_rows=2,source_L_forward_rows=0,source_OE_forward_rows=0)
        (pred/'complete.json').write_text(json.dumps(marker));truth_path=root/'truth.npz';np.savez(truth_path,**truth)
        return m,pred,truth_path,counts

    def test_fixed_bank_scores_report_true_class_macro_and_cache_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t,counts=self.fixture(Path(td))
            with patch.object(m,'COUNTS',counts),patch.object(m.core,'fit',side_effect=AssertionError('No new bank or query fitting')):
                result=m.score(p,t,Path(td)/'score')
            self.assertTrue(result['source_gate_passed']);self.assertTrue(result['eligible_primary'])
            self.assertEqual(result['status'],'STAR_LOCAL_SUPPORT_SCORES_COMPLETE');self.assertEqual(result['primary_method'],'rx_consensus_relative')
            row=result['primary_summary'];self.assertEqual(row['known_macro_open_accuracy'],1.);self.assertEqual(row['unknown_macro_rejection'],1.)
            self.assertLess(row['original_F0_macro_closed_accuracy'],row['fresh_F0_macro_closed_accuracy'])
            self.assertEqual(len(result['methods'][m.PRIMARY]['groups']),42);self.assertEqual(result['A_N'],None)
            self.assertEqual(result['full_goal_status'],'ACTIVE');self.assertEqual(result['target_reads'],0)
            self.assertEqual(result['CPU_readout']['production_seconds'],1.25);self.assertEqual(result['CPU_readout']['production_rows'],270)
            self.assertGreaterEqual(result['CPU_readout']['public_evidence_verification_seconds'],0)
            with self.assertRaises(FileExistsError):m.score(p,t,Path(td)/'score')

    def test_higher_control_cannot_replace_failed_primary(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t,counts=self.fixture(Path(td),main_pass=False)
            with patch.object(m,'COUNTS',counts):result=m.score(p,t,Path(td)/'score')
            self.assertTrue(result['methods']['pooled_relative']['goals']['passed']);self.assertFalse(result['source_gate_passed'])
            self.assertFalse(result['eligible_primary']);self.assertEqual(result['next_step'],'NO_PROMOTION')

    def test_non_axis_units_reconstruct_using_exact_fixed_core_transform(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);m,p,t,counts=self.fixture(root);rng=np.random.default_rng(321)
            with np.load(p/'source_bank.npz') as z:bank={k:z[k].copy() for k in z.files}
            for role in ('L','OE'):
                x=rng.normal(size=(40,160)).astype(np.float32);x/=np.linalg.norm(x,axis=1,keepdims=True)
                bank[role+'_unit_features']=x
                np.savez(root/'teacher'/(role+'.npz'),ids=bank[role+'_ids'],unit_features=x)
            np.savez(p/'source_bank.npz',**bank)
            state=m.core.fit(bank['L_unit_features'],bank['L_rx_i'],bank['OE_unit_features'],bank['OE_rx_i'])
            with np.load(p/'thresholds.npz') as z:tau=z['conditional'].copy()
            for role in ('known','novel'):
                with np.load(p/(role+'_predictions.npz')) as z:data={k:z[k].copy() for k in z.files}
                n=len(data['ids']);data['unit_features']=np.tile(bank['L_unit_features'],((n+39)//40,1))[:n]
                data['raw_scores']=m.core.score(data['unit_features'],state);data['scores']=data['raw_scores']-tau[:,data['closed']].T
                np.savez(p/(role+'_predictions.npz'),**data)
            with patch.object(m,'COUNTS',counts),patch.object(m.core,'fit',side_effect=AssertionError('No fit call in scorer')):
                m._prediction(p)

    def test_all_public_evidence_validation_precedes_private_truth(self):
        for defect in ('marker','model','head','input_cost','teacher_code','teacher_bank','teacher_role','private','argmax','raw','margin','tau','bank_dtype','bank_rx','overlap','baseline'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                m,p,t,counts=self.fixture(Path(td))
                if defect in ('marker','model','head','input_cost','teacher_code'):
                    path=p/'complete.json';d=json.loads(path.read_text())
                    if defect=='marker':d['source_novel_truth_reads']=1
                    elif defect=='model':d['model_identity']['epoch']=51
                    elif defect=='head':d['default_head_replaced']=True
                    elif defect=='input_cost':d['received_P0_values_read']=1
                    else:d['teacher_code_commit']='b'*40
                    path.write_text(json.dumps(d))
                elif defect=='teacher_role':
                    path=Path(td)/'teacher/complete.json';d=json.loads(path.read_text());d['cache_handles']['L']['role']='V';path.write_text(json.dumps(d))
                elif defect in ('bank_dtype','bank_rx','overlap','teacher_bank'):
                    path=p/'source_bank.npz'
                    if defect=='teacher_bank':path=Path(td)/'teacher/L.npz'
                    with np.load(path) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='bank_dtype':d['L_unit_features']=d['L_unit_features'].astype(np.float64)
                    elif defect=='bank_rx':d['L_rx_i'][0]=99
                    elif defect=='overlap':d['L_ids'][0]='V0'
                    else:d['unit_features'][0]=0;d['unit_features'][0,2]=1
                    np.savez(path,**d)
                elif defect=='tau':
                    with np.load(p/'thresholds.npz') as z:tau=z['conditional'].copy()
                    tau[0,0]+=1;np.savez(p/'thresholds.npz',conditional=tau)
                else:
                    path=p/('known_predictions.npz' if defect=='baseline' else 'novel_predictions.npz')
                    with np.load(path) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='private':d['tx']=np.asarray(['19-15']*len(d['ids']))
                    elif defect=='argmax':d['closed'][0]=1
                    elif defect=='raw':d['raw_scores'][0,0]+=.5
                    elif defect=='baseline':d['baseline_closed']=d['closed'].copy()
                    else:d['scores'][0,0]+=.5
                    np.savez(path,**d)
                with patch.object(m,'COUNTS',counts),patch.object(m,'_truth',side_effect=AssertionError('truth opened early')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(p,t,Path(td)/'score')


if __name__=='__main__':unittest.main()
