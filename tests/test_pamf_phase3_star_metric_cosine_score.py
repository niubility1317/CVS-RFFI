"""Synthetic cosine deployment, source provenance and one truth-read contract."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

torch.set_num_threads(1)


class MetricCosineScoreTests(unittest.TestCase):
    def fixture(self,root,main_pass=True):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_metric_cosine_score')
        parent=root/'parent/predict';(parent/'deployment').mkdir(parents=True);(parent/'calibration').mkdir()
        out=root/'pred';(out/'deployment').mkdir(parents=True);(out/'calibration').mkdir()
        identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701)
        dura=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940)
        rows=[(c,r,s) for c in range(6) for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        known=dict(ids=np.asarray(['V'+str(i) for i in range(n)]),y=np.asarray([a[0] for a in rows],np.int64),rx_i=np.asarray([a[1] for a in rows],np.int64),
            day_i=np.ones(n,np.int64),scenario=np.asarray([a[2] for a in rows]))
        known.update(closed=known['y'].copy(),baseline_closed=known['y'].copy(),unit_features=np.eye(160,dtype=np.float32)[known['y']],default_logits=np.eye(6,dtype=np.float32)[known['y']]*5)
        rows=[(tx,r,s) for tx in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];nu=len(rows)
        novel=dict(ids=np.asarray(['novel'+str(i) for i in range(nu)]),rx_i=np.asarray([a[1] for a in rows],np.int64),day_i=np.ones(nu,np.int64),closed=np.zeros(nu,np.int64),
            default_logits=np.tile(np.asarray([5,0,0,0,0,0],np.float32),(nu,1)),unit_features=np.tile(np.eye(160,dtype=np.float32)[20],(nu,1)))
        truth=dict(ids=novel['ids'][::-1],tx=np.asarray([a[0] for a in rows])[::-1],rx_i=novel['rx_i'][::-1],day_i=novel['day_i'][::-1],scenario=np.asarray([a[2] for a in rows])[::-1])
        tp=root/'truth.npz';np.savez(tp,**truth);counts=dict(V=n,novel=nu)
        torch.manual_seed(2026092701);net=m.embedding.core.MetricNet().eval();state=net.state_dict()
        for folder in (parent,out):torch.save(state,folder/'deployment/model.pt')
        metric=dict(class_means=.5*np.eye(160,dtype=np.float64)[:6],background_mean=np.zeros(160),within_whitener=np.eye(160),background_whitener=np.eye(160))
        np.savez(parent/'deployment/metric.npz',**metric)
        proto=m.evaluation.prototypes(metric['class_means']);np.savez(out/'deployment/prototypes.npz',prototypes=proto)
        ptau=np.asarray([[0.]*6,[-.75]*6],np.float64);tau=np.asarray([[.5]*6,[.5 if main_pass else 2.]*6],np.float64)
        np.savez(parent/'deployment/thresholds.npz',conditional=ptau);np.savez(out/'deployment/thresholds.npz',conditional=tau)
        report=lambda t:dict(conditional_thresholds=t.tolist(),constraints_passed=True,groups={k:{} for k in m.GROUP_NAMES})
        pc=dict(schema=m.parent.SCHEMA,status=m.parent.CALIBRATION_STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_CODE,arm=m.PARENT_ARM,
            method_names=list(m.parent.METHODS),conditional_thresholds=ptau.tolist(),group_count=42,known_only=True,unknown_inputs_used=False,source_V_neural_fit=False,
            source_only=True,target_reads=0,methods={name:report(ptau[j]) for j,name in enumerate(m.parent.METHODS)})
        (parent/'calibration/complete.json').write_text(json.dumps(pc))
        cal=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,method_names=list(m.METHODS),primary_method=m.PRIMARY,
            conditional_thresholds=tau.tolist(),group_count=42,known_only=True,unknown_inputs_used=False,source_V_neural_fit=False,source_only=True,target_reads=0,
            methods={name:report(tau[j]) for j,name in enumerate(m.METHODS)})
        (out/'calibration/complete.json').write_text(json.dumps(cal))
        handles={r:dict(role=r,count=c,parent_F0_identity=identity,dura_identity=dura) for r,c in dict(L=6300,OE=6134,V=n,novel=nu).items()}
        pm=dict(schema=m.parent.SCHEMA,status=m.parent.STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_CODE,arm=m.PARENT_ARM,model_identity=identity,dura_identity=dura,cache_identities=handles,
            cache_root='never-open-cache',teacher_root='never-open-teacher',teacher_run_id=m.TEACHER_RUN,teacher_code_commit=m.TEACHER_CODE,known_count=n,novel_count=nu,
            source_L_count=6300,source_OE_count=6134,g_parameters=20704,g_training_steps=1000,g_training_L_exposures=120000,g_training_OE_exposures=128000,
            neural_F_updates=0,G_updates=0,source_U_reads=0,target_reads=0,source_novel_truth_reads=0,source_only=True,classification_path='P1',default_head_replaced=False,
            model_parameters_buffers_unchanged=True,metric_parameters_buffers_frozen_for_V_novel=True,aggregate_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
            source_free_deployment_state=True,per_record_source_rows_retained=0,fresh_closed_vs_cached_baseline_changed_count=0)
        (parent/'complete.json').write_text(json.dumps(pm))
        pd=dict(schema=m.embedding.DEPLOYMENT_SCHEMA,status=m.embedding.DEPLOYMENT_STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_CODE,arm=m.PARENT_ARM,parameters=20704,source_free=True,per_record_source_rows_retained=0)
        (parent/'deployment/complete.json').write_text(json.dumps(pd))
        nd=dict(schema=m.evaluation.DEPLOYMENT_SCHEMA,status=m.evaluation.DEPLOYMENT_STATUS,run_id=m.RUN_ID,code_commit='a'*40,parameters=20704,source_free=True,per_record_source_rows_retained=0)
        (out/'deployment/complete.json').write_text(json.dumps(nd))
        for role,data in (('known',known),('novel',novel)):
            data['metric_unit_features']=data['unit_features'].copy();praw=m.embedding.core.metric_scores(data['metric_unit_features'],data['closed'],metric)
            np.savez(parent/(role+'_predictions.npz'),**dict(data,raw_scores=praw,scores=praw-ptau[:,data['closed']].T))
            raw=m.evaluation.cosine_scores(data['metric_unit_features'],data['closed'],proto)
            np.savez(out/(role+'_predictions.npz'),**dict(data,raw_scores=raw,scores=raw-tau[:,data['closed']].T))
        marker=dict(run_id=m.RUN_ID,code_commit='a'*40,schema=m.evaluation.SCHEMA,status=m.evaluation.STATUS,parent_run_id=m.PARENT_RUN,parent_code_commit=m.PARENT_CODE,parent_arm=m.PARENT_ARM,
            parent_prediction_root=str(parent),model_identity=identity,dura_identity=dura,cache_identities=handles,cache_root=pm['cache_root'],teacher_root=pm['teacher_root'],
            teacher_run_id=m.TEACHER_RUN,teacher_code_commit=m.TEACHER_CODE,method_names=list(m.METHODS),primary_method=m.PRIMARY,known_count=n,novel_count=nu,g_parameters=20704,
            inherited_g_training_steps=1000,new_neural_updates=0,F_forward_rows=0,G_forward_rows=0,g_forward_rows=2,synthetic_unit_smoke_rows=2,real_IQ_values_read=0,
            source_L_forward_rows=0,source_OE_forward_rows=0,source_L_metadata_labels_read=0,target_reads=0,source_U_reads=0,source_novel_truth_reads=0,source_only=True,classification_path='P1',
            default_head_replaced=False,prototypes_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_free_deployment_state=True,parent_public_features_reused=True,
            deployed_g_unchanged=True,per_record_source_rows_retained=0,cpu_readout_seconds=1.,cpu_readout_rows=n+nu,fresh_closed_is_parent_reuse=True,A_N=None,H_OW=None,final_phase3_complete=False)
        (out/'complete.json').write_text(json.dumps(marker))
        m.embedding._reference_smoke(net.double(),pm,out/'smoke.json')
        return m,out,tp,counts,parent

    def test_one_truth_read_new_main_and_exact_external_MD_reference(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t,counts,parent=self.fixture(Path(td));original=m._truth;read=m._read;forbidden=parent.parent/'score';forbidden.mkdir()
            (forbidden/'complete.json').write_text('{"forbidden_truth_derived_parent_score":true}')
            def public_read(path):
                if Path(path).is_relative_to(forbidden):raise AssertionError('Parent private scores are not public inputs')
                return read(path)
            with patch.object(m,'COUNTS',counts),patch.object(m,'_truth',wraps=original) as truth,\
                 patch.object(m.embedding,'_prediction',side_effect=AssertionError('No parent full validation')),patch.object(m.embedding,'_training',side_effect=AssertionError('No parent full trace')),\
                 patch.object(m,'_read',side_effect=public_read),patch.object(m.embedding,'_read',side_effect=public_read):
                result=m.score(p,t,Path(td)/'score')
            self.assertEqual(truth.call_count,1);self.assertTrue(result['source_gate_passed']);self.assertTrue(result['eligible_primary'])
            self.assertEqual(result['primary_method'],'maximum_cosine');self.assertEqual(result['status'],'STAR_METRIC_COSINE_SCORES_COMPLETE')
            self.assertEqual(result['primary_summary']['H_OS'],1.);self.assertEqual(result['method_names'],['predicted_cosine','maximum_cosine'])
            ref=result['reference_original_mahalanobis'];self.assertEqual(ref['conditional_thresholds'],[-.75]*6)
            self.assertEqual(ref['overall']['H_OS'],1.);self.assertFalse(ref['deployable_from_new_four_files']);self.assertTrue(ref['saved_parent_margin_reused'])
            self.assertEqual(result['target_reads'],0);self.assertEqual(result['full_goal_status'],'ACTIVE');self.assertIsNone(result['A_N'])
            with self.assertRaises(FileExistsError):m.score(p,t,Path(td)/'score')

    def test_control_and_original_MD_cannot_replace_failed_maximum_main(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t,counts,parent=self.fixture(Path(td),main_pass=False)
            with patch.object(m,'COUNTS',counts):result=m.score(p,t,Path(td)/'score')
            self.assertTrue(result['methods']['predicted_cosine']['goals']['passed']);self.assertTrue(result['reference_original_mahalanobis']['goals']['passed'])
            self.assertFalse(result['eligible_primary']);self.assertEqual(result['next_step'],'NO_PROMOTION')

    def test_maximum_cosine_only_routes_rejection_and_preserves_original_identity(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t,counts,parent=self.fixture(Path(td));index=150
            with np.load(parent/'deployment/metric.npz') as z:metric={k:z[k].copy() for k in z.files}
            with np.load(parent/'deployment/thresholds.npz') as z:ptau=z['conditional'].copy()
            with np.load(p/'deployment/prototypes.npz') as z:proto=z['prototypes'].copy()
            with np.load(p/'deployment/thresholds.npz') as z:tau=z['conditional'].copy()
            for folder in (parent,p):
                path=folder/'known_predictions.npz'
                with np.load(path) as z:d={k:z[k].copy() for k in z.files}
                self.assertEqual(d['closed'][index],5)
                d['unit_features'][index]=np.eye(160,dtype=np.float32)[2];d['metric_unit_features'][index]=d['unit_features'][index]
                raw=m.embedding.core.metric_scores(d['metric_unit_features'],d['closed'],metric) if folder==parent else m.evaluation.cosine_scores(d['metric_unit_features'],d['closed'],proto)
                d.update(raw_scores=raw,scores=raw-(ptau if folder==parent else tau)[:,d['closed']].T);np.savez(path,**d)
            with patch.object(m,'COUNTS',counts):_,_,_,known,_,_=m._prediction(p)
            self.assertEqual(known['closed'][index],5);self.assertEqual(known['raw_scores'][index,0],0.);self.assertEqual(known['raw_scores'][index,1],1.)

    def test_source_model_prototype_reference_and_tamper_checks_precede_truth(self):
        for defect in ('parent_arm','parent_code','default_head','freeze','smoke','model','prototype','private','identity','unit','raw','margin','tau','old_tau','old_margin','extra_deployment'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);m,p,t,counts,parent=self.fixture(root)
                if defect in ('parent_arm','parent_code','default_head','freeze','smoke'):
                    path=p/('smoke.json' if defect=='smoke' else 'complete.json');d=json.loads(path.read_text())
                    if defect=='parent_arm':d['parent_arm']='compact_oe'
                    elif defect=='parent_code':d['parent_code_commit']='b'*40
                    elif defect=='default_head':d['default_head_replaced']=True
                    elif defect=='smoke':d['arm']='compact_oe'
                    else:d['all_thresholds_frozen_before_novel']=False
                    path.write_text(json.dumps(d))
                elif defect=='model':
                    path=p/'deployment/model.pt';d=torch.load(path,map_location='cpu',weights_only=True);d['fc1.bias'][0]+=.01;torch.save(d,path)
                elif defect=='extra_deployment':(p/'deployment/source_rows.txt').write_text('forbidden')
                else:
                    path=p/'novel_predictions.npz'
                    if defect=='prototype':path=p/'deployment/prototypes.npz'
                    elif defect=='tau':path=p/'deployment/thresholds.npz'
                    elif defect=='old_tau':path=parent/'deployment/thresholds.npz'
                    elif defect=='old_margin':path=parent/'novel_predictions.npz'
                    with np.load(path) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='prototype':d['prototypes'][0]=np.eye(160)[1]
                    elif defect=='private':d['tx']=np.asarray(['19-15']*len(d['ids']))
                    elif defect=='identity':d['closed'][0]=1
                    elif defect=='unit':d['metric_unit_features'][0]=np.eye(160,dtype=np.float32)[21]
                    elif defect=='raw':d['raw_scores'][0,1]+=.1
                    elif defect in ('tau','old_tau'):d['conditional'][1,0]+=1.
                    else:d['scores'][0,1]+=.1
                    np.savez(path,**d)
                with patch.object(m,'COUNTS',counts),patch.object(m,'_truth',side_effect=AssertionError('Truth opened early')),self.assertRaises((ValueError,RuntimeError,FileNotFoundError)):
                    m.score(p,t,root/'score')


if __name__=='__main__':unittest.main()
