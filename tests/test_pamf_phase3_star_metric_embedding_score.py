"""Synthetic deployment-only reconstruction and full training-budget evidence."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

torch.set_num_threads(1)  # Match the launcher's independent CPU-score environment.


class MetricEmbeddingScoreTests(unittest.TestCase):
    def fixture(self,root,arm='compact_oe_structure',main_pass=True):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_metric_embedding_score')
        parent=root/'parent';parent.mkdir();out=root/'pred';(out/'deployment').mkdir(parents=True);(out/'calibration').mkdir()
        identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701)
        dura=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940)
        ly=np.repeat(np.arange(6,dtype=np.int64),40);lr=np.tile(np.repeat(np.asarray(m.RECEIVERS,np.int64),8),6);nl=len(ly);no=128
        lids=np.asarray(['L'+str(i) for i in range(nl)]);oeids=np.asarray(['OE'+str(i) for i in range(no)])
        np.savez(parent/'source_bank.npz',L_ids=lids,L_rx_i=lr,L_unit_features=np.eye(160,dtype=np.float32)[ly],OE_ids=oeids,
            OE_rx_i=np.asarray(m.RECEIVERS,np.int64)[np.arange(no)%5],OE_unit_features=np.tile(np.eye(160,dtype=np.float32)[20],(no,1)))
        cache=root/'cache';(cache/'L').mkdir(parents=True);np.save(cache/'L/ids.npy',lids)
        np.savez(cache/'L/metadata.npz',y=ly,rx_i=lr,day_i=np.ones(nl,np.int64),domain=np.searchsorted(m.RECEIVERS,lr)*3)
        handles={r:dict(role=r,count=n,parent_F0_identity=identity,dura_identity=dura) for r,n in dict(L=nl,OE=no,V=180,novel=90).items()}
        (cache/'L/manifest.json').write_text(json.dumps(handles['L']))
        rows=[(c,r,s) for c in range(6) for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        known=dict(ids=np.asarray(['V'+str(i) for i in range(n)]),y=np.asarray([v[0] for v in rows],np.int64),rx_i=np.asarray([v[1] for v in rows],np.int64),
            day_i=np.ones(n,np.int64),scenario=np.asarray([v[2] for v in rows]))
        known.update(closed=known['y'].copy(),baseline_closed=known['y'].copy(),unit_features=np.eye(160,dtype=np.float32)[known['y']],default_logits=np.eye(6,dtype=np.float32)[known['y']]*5)
        rows=[(tx,r,s) for tx in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];nu=len(rows)
        novel=dict(ids=np.asarray(['novel'+str(i) for i in range(nu)]),rx_i=np.asarray([v[1] for v in rows],np.int64),day_i=np.ones(nu,np.int64),
            closed=np.zeros(nu,np.int64),default_logits=np.tile(np.asarray([5,0,0,0,0,0],np.float32),(nu,1)),unit_features=np.tile(np.eye(160,dtype=np.float32)[20],(nu,1)))
        truth=dict(ids=novel['ids'][::-1],tx=np.asarray([v[0] for v in rows])[::-1],rx_i=novel['rx_i'][::-1],day_i=novel['day_i'][::-1],scenario=np.asarray([v[2] for v in rows])[::-1])
        tp=root/'truth.npz';np.savez(tp,**truth);counts=dict(L=nl,OE=no,V=n,novel=nu)
        torch.manual_seed(2026092701);net=m.core.MetricNet().eval();torch.save(net.state_dict(),out/'deployment/model.pt')
        metric=dict(class_means=np.eye(160,dtype=np.float64)[:6],background_mean=np.zeros(160,np.float64),within_whitener=np.eye(160),background_whitener=np.eye(160))
        np.savez(out/'deployment/metric.npz',**metric);tau=np.asarray([[0.]*6,[0. if main_pass else 100.]*6],np.float64)
        np.savez(out/'deployment/thresholds.npz',conditional=tau)
        for role,data in (('known',known),('novel',novel)):
            np.savez(parent/(role+'_predictions.npz'),**dict(data,raw_scores=np.zeros((len(data['ids']),3),np.float64),scores=np.zeros((len(data['ids']),3),np.float64)))
            with torch.no_grad():u=np.asarray(net(torch.tensor(data['unit_features'].tolist(),dtype=torch.float32)).tolist(),np.float32)
            raw=m.core.metric_scores(u,data['closed'],metric)
            np.savez(out/(role+'_predictions.npz'),**dict(data,metric_unit_features=u,raw_scores=raw,scores=raw-tau[:,data['closed']].T))
        pm=dict(run_id=m.PARENT_RUN,code_commit=m.PARENT_CODE,schema=m.parent.SCHEMA,status=m.parent.STATUS,model_identity=identity,dura_identity=dura,
            cache_identities=handles,cache_root=str(cache),teacher_root='original_teacher',teacher_run_id=m.TEACHER_RUN,teacher_code_commit=m.TEACHER_CODE,
            classification_path='P1',source_only=True,target_reads=0,source_novel_truth_reads=0,source_U_reads=0,neural_updates=0,G_updates=0,G_forward_rows=0,
            default_head_replaced=False,fresh_closed_vs_cached_baseline_changed_count=0,F_forward_rows=n+nu+2,
            known_count=n,novel_count=nu,bank_L_count=nl,bank_OE_count=no,
            model_parameters_buffers_unchanged=True,all_thresholds_frozen_before_novel=True,source_bank_frozen_before_V=True,source_bank_unchanged=True)
        (parent/'complete.json').write_text(json.dumps(pm))
        steps=0 if arm=='identity' else 1000;loss=dict(total=2.,known_contrast=1.,OE_repulsion=.1,OE_structure=0.)
        indices=np.asarray([np.flatnonzero((ly==c)&(lr==r))[:4] for c in range(6) for r in m.RECEIVERS]).reshape(-1).tolist()
        with (out/'training.jsonl').open('w') as f:
            for step in range(1,steps+1):f.write(json.dumps(dict(step=step,**loss,g_grad_norm=1.,L_indices=indices,OE_indices=list(range(no)),
                L_exposures=step*120,OE_exposures=step*128,L_unique=120,OE_unique=no))+'\n')
        tc=dict(schema=m.TRAINING_SCHEMA,status=m.TRAINING_STATUS,run_id=m.RUN_ID,code_commit='a'*40,arm=arm,g_parameters=20704,g_training_steps=steps,
            g_training_L_exposures=steps*120,g_training_OE_exposures=steps*128,g_training_L_unique=120 if steps else 0,g_training_OE_unique=no if steps else 0,
            g_training_forward_rows=steps*248,model_seed=2026092701,sampling_seed=2026103002,loss_initial=loss if steps else None,loss_final=loss if steps else None,
            optimizer=dict(name='Adam',lr=.001,weight_decay=.0001) if steps else None,initialization_cpu_seeded=True,F_updates=0,G_updates=0,source_only=True,
            target_reads=0,source_OE_TX_or_K_used=False,source_L_TX_RX_labels_used=True,trace_rows=steps,train_seconds=1.,g_parameters_finite=True)
        (out/'training_complete.json').write_text(json.dumps(tc))
        deploy=dict(schema=m.DEPLOYMENT_SCHEMA,status=m.DEPLOYMENT_STATUS,run_id=m.RUN_ID,code_commit='a'*40,arm=arm,parameters=20704,source_free=True,per_record_source_rows_retained=0)
        (out/'deployment/complete.json').write_text(json.dumps(deploy))
        cal=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,arm=arm,method_names=list(m.METHODS),
            conditional_thresholds=tau.tolist(),group_count=42,known_only=True,unknown_inputs_used=False,source_only=True,target_reads=0,
            source_V_neural_fit=False,
            methods={name:dict(constraints_passed=True,conditional_thresholds=tau[j].tolist(),groups={k:{} for k in m.GROUP_NAMES}) for j,name in enumerate(m.METHODS)})
        (out/'calibration/complete.json').write_text(json.dumps(cal))
        marker=dict(pm,run_id=m.RUN_ID,code_commit='a'*40,schema=m.evaluation.SCHEMA,status=m.evaluation.STATUS,arm=arm,parent_run_id=m.PARENT_RUN,parent_code_commit=m.PARENT_CODE,
            parent_prediction_root=str(parent),method_names=list(m.METHODS),primary_arm=m.PRIMARY_ARM,primary_method=m.PRIMARY,
            source_L_count=nl,source_OE_count=no,g_parameters=20704,g_training_steps=steps,g_training_L_exposures=steps*120,g_training_OE_exposures=steps*128,
            g_forward_rows=steps*248+nl+n+nu,F_forward_rows=2,real_IQ_values_read=0,source_L_metadata_labels_read=nl,neural_F_updates=0,
            synthetic_P0_forward_rows=2,parent_public_features_reused=True,source_L_TX_RX_labels_used=True,source_OE_TX_or_K_used=False,
            metric_parameters_buffers_frozen_for_V_novel=True,aggregate_frozen_before_V=True,source_free_deployment_state=True,per_record_source_rows_retained=0,
            cpu_readout_seconds=1.25,g_evaluation_seconds=.5)
        (out/'complete.json').write_text(json.dumps(marker));return m,out,tp,counts

    def test_fixed_main_eligibility_controls_and_deployment_only_rebuild(self):
        for arm in ('identity','compact_oe','compact_oe_structure'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as td:
                m,p,t,counts=self.fixture(Path(td),arm)
                with patch.object(m,'COUNTS',counts),patch.object(m.core,'fit_metric',side_effect=AssertionError('No refit')),\
                     patch.object(m.core,'losses',side_effect=AssertionError('No training')),patch.object(m.previous,'_prediction',side_effect=AssertionError('No old three-score validation')):
                    result=m.score(p,t,Path(td)/'score')
                self.assertTrue(result['source_gate_passed']);self.assertEqual(result['eligible_primary'],arm==m.PRIMARY_ARM)
                self.assertEqual(result['primary_method'],'predicted_identity_mahalanobis')
                self.assertEqual(result['next_step'],'SOURCE_METRIC_EMBEDDING_READY_FOR_TARGET_FALSIFICATION' if arm==m.PRIMARY_ARM else 'CONTROL_ONLY_COMPLETE')
                self.assertEqual(result['primary_summary']['H_OS'],1.);self.assertEqual(result['target_reads'],0);self.assertEqual(result['full_goal_status'],'ACTIVE')
                self.assertIsNone(result['A_N']);self.assertTrue(result['source_free_deployment_state'])
                self.assertLess(result['novel_cross_RX_retrieval']['metric']['macro_hit_rate'],.9)
                self.assertFalse(result['novel_structure_is_promotion_gate'])
                self.assertEqual(result['input_costs']['g_evaluation_rows'],270)
                self.assertEqual(result['input_costs']['g_source_L_aggregate_rows'],counts['L'])
                self.assertEqual(result['input_costs']['g_training_forward_rows'],0 if arm=='identity' else 248000)
                with self.assertRaises(FileExistsError):m.score(p,t,Path(td)/'score')

    def test_higher_minimum_relative_cannot_replace_conditioned_primary(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t,counts=self.fixture(Path(td),main_pass=False)
            with patch.object(m,'COUNTS',counts):result=m.score(p,t,Path(td)/'score')
            self.assertTrue(result['methods']['minimum_relative']['goals']['passed']);self.assertFalse(result['eligible_primary']);self.assertEqual(result['next_step'],'NO_PROMOTION')

    def test_public_source_deployment_training_and_score_errors_precede_truth(self):
        for defect in ('freeze','parent','private','identity','g_unit','raw','margin','tau','aggregate','deployment_extra','model_extra','model_tensor','budget','trace','source_L'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);m,p,t,counts=self.fixture(root)
                if defect in ('freeze','parent'):
                    path=p/'complete.json';d=json.loads(path.read_text());d['all_thresholds_frozen_before_novel' if defect=='freeze' else 'parent_code_commit']=False if defect=='freeze' else 'b'*40;path.write_text(json.dumps(d))
                elif defect=='deployment_extra':(p/'deployment/source_rows.txt').write_text('forbidden')
                elif defect in ('model_extra','model_tensor'):
                    d=torch.load(p/'deployment/model.pt',map_location='cpu',weights_only=True)
                    if defect=='model_extra':d['source_ids']=torch.ones(1)
                    else:d['fc1.weight']=torch.ones((1,160))
                    torch.save(d,p/'deployment/model.pt')
                elif defect=='budget':
                    path=p/'training_complete.json';d=json.loads(path.read_text());d['g_training_steps']=999;path.write_text(json.dumps(d))
                elif defect=='trace':
                    path=p/'training.jsonl';lines=path.read_text().splitlines();d=json.loads(lines[-1]);d['L_indices'][0]=d['L_indices'][1];lines[-1]=json.dumps(d);path.write_text('\n'.join(lines)+'\n')
                else:
                    path=p/'novel_predictions.npz'
                    if defect=='tau':path=p/'deployment/thresholds.npz'
                    elif defect=='aggregate':path=p/'deployment/metric.npz'
                    elif defect=='source_L':path=root/'cache/L/metadata.npz'
                    with np.load(path) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='private':d['tx']=np.asarray(['19-15']*len(d['ids']))
                    elif defect=='identity':d['closed'][0]=1
                    elif defect=='g_unit':d['metric_unit_features'][0]=np.eye(160,dtype=np.float32)[21]
                    elif defect=='raw':d['raw_scores'][0,1]+=.1
                    elif defect=='margin':d['scores'][0,1]+=.1
                    elif defect=='tau':d['conditional'][0,0]=1.
                    elif defect=='aggregate':d['source_rows']=np.ones((2,160))
                    else:d['y'][0]=1
                    np.savez(path,**d)
                with patch.object(m,'COUNTS',counts),patch.object(m,'_truth',side_effect=AssertionError('Truth opened before public validation')),self.assertRaises((ValueError,RuntimeError,FileNotFoundError)):
                    m.score(p,t,root/'score')


if __name__=='__main__':unittest.main()
