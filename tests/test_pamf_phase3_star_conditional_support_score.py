"""Synthetic conditional-bank provenance, parent control and truth-last checks."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class ConditionalSupportScoreTests(unittest.TestCase):
    def fixture(self,root,main_pass=True):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_conditional_support_score')
        parent=root/'parent';(parent/'calibration').mkdir(parents=True);out=root/'pred';(out/'calibration').mkdir(parents=True)
        identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,source_output='original/source')
        dura=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940)
        Lrx=np.tile(np.repeat(np.asarray(m.RECEIVERS,np.int64),8),6);Ly=np.repeat(np.arange(6,dtype=np.int64),40);nl=len(Ly);no=40
        bank=dict(L_ids=np.asarray(['L'+str(i) for i in range(nl)]),L_rx_i=Lrx,L_unit_features=np.tile(np.eye(160,dtype=np.float32)[0],(nl,1)),
            OE_ids=np.asarray(['OE'+str(i) for i in range(no)]),OE_rx_i=np.repeat(np.asarray(m.RECEIVERS,np.int64),8),
            OE_unit_features=np.tile(np.eye(160,dtype=np.float32)[1],(no,1)))
        cache=root/'cache';(cache/'L').mkdir(parents=True);np.save(cache/'L/ids.npy',bank['L_ids'])
        np.savez(cache/'L/metadata.npz',y=Ly,rx_i=Lrx,day_i=np.ones(nl,np.int64),domain=np.searchsorted(m.RECEIVERS,Lrx)*3)
        manifest=dict(role='L',count=nl,parent_F0_identity=identity,dura_identity=dura)
        (cache/'L/manifest.json').write_text(json.dumps(manifest));np.savez(parent/'source_bank.npz',**bank)
        handles={'L':manifest,**{k:dict(role=k) for k in ('OE','V','novel')}}
        rows=[(y,r,s) for y in range(6) for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        known=dict(ids=np.asarray(['V'+str(i) for i in range(n)]),y=np.asarray([a[0] for a in rows],np.int64),
            rx_i=np.asarray([a[1] for a in rows],np.int64),day_i=np.ones(n,np.int64),scenario=np.asarray([a[2] for a in rows]))
        known['closed']=known['y'].copy();known['baseline_closed']=known['y'].copy()
        known['default_logits']=np.eye(6,dtype=np.float32)[known['closed']]*5
        known['unit_features']=np.tile(np.eye(160,dtype=np.float32)[0],(n,1))
        rows=[(tx,r,s) for tx in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];nu=len(rows)
        novel=dict(ids=np.asarray(['novel'+str(i) for i in range(nu)]),rx_i=np.asarray([a[1] for a in rows],np.int64),day_i=np.ones(nu,np.int64),
            closed=np.zeros(nu,np.int64),default_logits=np.tile(np.asarray([5,0,0,0,0,0],np.float32),(nu,1)),
            unit_features=np.tile(np.eye(160,dtype=np.float32)[1],(nu,1)))
        truth=dict(ids=novel['ids'][::-1],tx=np.asarray([a[0] for a in rows])[::-1],rx_i=novel['rx_i'][::-1],day_i=novel['day_i'][::-1],
            scenario=np.asarray([a[2] for a in rows])[::-1]);truth_path=root/'truth.npz';np.savez(truth_path,**truth)
        counts=dict(L=nl,OE=no,V=n,novel=nu)
        ptau=np.asarray([[-1.]*6,[0.]*6,[0.]*6],np.float64);tau=np.asarray([[0.]*6,[0. if main_pass else 100.]*6],np.float64)
        report=lambda thresholds:dict(constraints_passed=True,conditional_thresholds=thresholds.tolist(),groups={key:{} for key in m.GROUP_NAMES})
        pcal=dict(schema=m.parent.SCHEMA,status=m.parent.CALIBRATION_STATUS,run_id=m.PARENT_RUN,code_commit=m.PARENT_CODE,
            group_count=42,source_only=True,target_reads=0,
            method_names=list(m.parent.METHODS),methods={name:report(ptau[j]) for j,name in enumerate(m.parent.METHODS)},conditional_thresholds=ptau.tolist())
        (parent/'calibration/complete.json').write_text(json.dumps(pcal));np.savez(parent/'thresholds.npz',conditional=ptau)
        state=dict(L=bank['L_unit_features'].astype(np.float64),L_y=Ly,OE=bank['OE_unit_features'].astype(np.float64))
        from experiments.pamf_phase3_open_world import star_local_support_core as oldcore
        pstate=dict(L=state['L'],OE=state['OE'],L_rx=bank['L_rx_i'],OE_rx=bank['OE_rx_i'])
        for role,data in (('known',known),('novel',novel)):
            pdata=dict(data,raw_scores=oldcore.score(data['unit_features'],pstate));pdata['scores']=pdata['raw_scores']-ptau[:,data['closed']].T
            np.savez(parent/(role+'_predictions.npz'),**pdata)
            raw=m.core.score(data['unit_features'],data['closed'],state);np.savez(out/(role+'_predictions.npz'),**dict(data,raw_scores=raw,scores=raw-tau[:,data['closed']].T))
        pmarker=dict(run_id=m.PARENT_RUN,code_commit=m.PARENT_CODE,schema=m.parent.SCHEMA,status=m.parent.STATUS,model_identity=identity,dura_identity=dura,
            teacher_run_id=m.TEACHER_RUN,teacher_code_commit=m.TEACHER_CODE,teacher_root='teacher',cache_root=str(cache),cache_identities=handles,
            known_count=n,novel_count=nu,bank_L_count=nl,bank_OE_count=no,method_names=list(m.parent.METHODS),primary_method=m.parent.PRIMARY_METHOD,
            classification_path='P1',source_only=True,target_reads=0,source_novel_truth_reads=0,neural_updates=0,G_updates=0,G_forward_rows=0,
            default_head_replaced=False,parameters=232993,score_threshold=0.,
            F_forward_rows=n+nu+2,all_thresholds_frozen_before_novel=True,source_bank_frozen_before_V=True,source_bank_unchanged=True,model_parameters_buffers_unchanged=True)
        (parent/'complete.json').write_text(json.dumps(pmarker));np.savez(out/'source_bank.npz',**bank,L_y=Ly);np.savez(out/'thresholds.npz',conditional=tau)
        cal=dict(schema=m.evaluation.SCHEMA,status=m.evaluation.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,method_names=list(m.METHODS),
            conditional_thresholds=tau.tolist(),methods={m.METHODS[0]:pcal['methods']['pooled_relative'],m.PRIMARY:report(tau[1])},group_count=42,
            primary_threshold_fit_known_only=True,control_thresholds_reused=True,source_only=True,target_reads=0,source_L_TX_labels_used=True,source_OE_TX_or_K_used=False)
        (out/'calibration/complete.json').write_text(json.dumps(cal))
        marker=dict(pmarker,run_id=m.RUN_ID,code_commit='a'*40,schema=m.evaluation.SCHEMA,status=m.evaluation.STATUS,method_names=list(m.METHODS),primary_method=m.PRIMARY,
            parent_run_id=m.PARENT_RUN,parent_code_commit=m.PARENT_CODE,parent_prediction_root=str(parent),source_U_reads=0,F_forward_rows=2,
            synthetic_P0_forward_rows=2,new_V_forward_rows=0,new_novel_forward_rows=0,source_L_forward_rows=0,source_OE_forward_rows=0,
            real_IQ_values_read=0,source_L_metadata_labels_read=nl,parent_public_features_reused=True,control_scores_thresholds_acceptance_reproduced=True,
            cpu_readout_seconds=1.25,cpu_readout_rows=n+nu,fresh_closed_vs_cached_baseline_changed_count=0)
        (out/'complete.json').write_text(json.dumps(marker));return m,out,truth_path,counts

    def test_control_reproduction_macro_metrics_and_terminal_contract(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t,counts=self.fixture(Path(td))
            with patch.object(m,'COUNTS',counts),patch.object(m.core,'fit',side_effect=AssertionError('No refit')),\
                 patch.object(m.previous,'_prediction',side_effect=AssertionError('No repeated full parent scoring')):
                result=m.score(p,t,Path(td)/'score')
            self.assertEqual(result['status'],'STAR_CONDITIONAL_SUPPORT_SCORES_COMPLETE');self.assertFalse(result['eligible_primary'])
            self.assertTrue(result['source_gate_passed']);self.assertTrue(result['arm_source_gate_passed'])
            self.assertTrue(result['source_free_export_required']);self.assertFalse(result['source_bank_is_deployment_state'])
            self.assertFalse(result['target_eligible'])
            self.assertEqual(result['next_step'],'SOURCE_CONDITIONAL_SUPPORT_READY_FOR_SOURCE_FREE_EXPORT')
            self.assertEqual(result['primary_method'],'predicted_identity_relative');self.assertEqual(result['primary_summary']['H_OS'],1.)
            self.assertEqual(len(result['methods'][m.PRIMARY]['groups']),42);self.assertEqual(result['target_reads'],0)
            self.assertEqual(result['A_N'],None);self.assertEqual(result['full_goal_status'],'ACTIVE')
            with self.assertRaises(FileExistsError):m.score(p,t,Path(td)/'score')

    def test_control_cannot_replace_failed_primary(self):
        with tempfile.TemporaryDirectory() as td:
            m,p,t,counts=self.fixture(Path(td),main_pass=False)
            with patch.object(m,'COUNTS',counts):result=m.score(p,t,Path(td)/'score')
            self.assertTrue(result['methods']['pooled_relative']['goals']['passed']);self.assertFalse(result['eligible_primary'])
            self.assertEqual(result['next_step'],'NO_PROMOTION')

    def test_all_public_identity_bank_and_control_checks_precede_truth(self):
        for defect in ('freeze','parent_code','parent_identity','parent_head','parent_control','source_L_y','bank_y','bank_features','private','closed','argmax','meta','baseline','raw','control','tau','control_report'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);m,p,t,counts=self.fixture(root)
                if defect in ('freeze','parent_code','parent_identity','parent_head'):
                    path=p/'complete.json' if defect not in ('parent_identity','parent_head') else root/'parent/complete.json';d=json.loads(path.read_text())
                    if defect=='freeze':d['all_thresholds_frozen_before_novel']=False
                    elif defect=='parent_code':d['parent_code_commit']='b'*40
                    elif defect=='parent_head':d['default_head_replaced']=True
                    else:d['model_identity']['epoch']=51
                    path.write_text(json.dumps(d))
                elif defect=='control_report':
                    path=p/'calibration/complete.json';d=json.loads(path.read_text());d['methods']['pooled_relative']['constraints_passed']=False;path.write_text(json.dumps(d))
                else:
                    path=p/'novel_predictions.npz'
                    if defect=='source_L_y':path=root/'cache/L/metadata.npz'
                    elif defect in ('bank_y','bank_features'):path=p/'source_bank.npz'
                    elif defect=='tau':path=p/'thresholds.npz'
                    elif defect=='parent_control':path=root/'parent/novel_predictions.npz'
                    elif defect=='baseline':path=p/'known_predictions.npz'
                    with np.load(path) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='source_L_y':d['y'][0]=1
                    elif defect=='bank_y':d['L_y'][0]=1
                    elif defect=='bank_features':d['L_unit_features'][0]=np.eye(160,dtype=np.float32)[2]
                    elif defect=='private':d['tx']=np.asarray(['19-15']*len(d['ids']))
                    elif defect in ('closed','argmax'):
                        d['closed'][0]=1
                        if defect=='argmax':
                            with np.load(root/'parent/novel_predictions.npz') as z:old={k:z[k].copy() for k in z.files}
                            old['closed'][0]=1;np.savez(root/'parent/novel_predictions.npz',**old)
                    elif defect=='meta':d['day_i'][0]=2
                    elif defect=='baseline':d['baseline_closed'][0]=1
                    elif defect=='tau':d['conditional'][0,0]=1
                    elif defect=='raw':d['raw_scores'][0,1]+=.1
                    else:d['scores'][0,1 if defect=='parent_control' else 0]+=.1
                    np.savez(path,**d)
                with patch.object(m,'COUNTS',counts),patch.object(m,'_truth',side_effect=AssertionError('Private truth opened early')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(p,t,root/'score')


if __name__=='__main__':unittest.main()
