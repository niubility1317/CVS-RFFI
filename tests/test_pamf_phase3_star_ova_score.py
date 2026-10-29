"""Synthetic source-only OVA evidence gates and four-scalar paired metrics."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarOvaScoreTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_ova_score')

    def state(self,m):
        return dict(schema='pamf_phase3_star_ova_readouts_v1',descriptor_schema='class_conditional_paired_branch_evidence_v1',
            descriptor_dim=106,class_count=6,model_names=['membership','ova_hard'],mean=np.zeros(106,np.float64),scale=np.ones(106,np.float64),
            models={name:{'weight1':np.zeros((32,106),np.float32),'bias1':np.zeros(32,np.float32),
                'weight2':np.zeros((1,32),np.float32),'bias2':np.zeros(1,np.float32)} for name in ('membership','ova_hard')},
            training=dict(steps_per_arm=1000,total_optimizer_updates=2000,initial_parameters_equal=True,
                parameter_counts={'membership':3457,'ova_hard':3457},model_seed=2026092701,
                losses={name:[1.]*1000 for name in ('membership','ova_hard')}))

    def save(self,folder,known,novel):
        with np.load(folder/'thresholds.npz') as z:conditional=z['conditional']
        for role,data in (('known',known),('novel',novel)):
            n=len(data['ids']);c=data['closed'];raw=data['scores']+conditional[:,c].T
            evidence=np.full((n,2,6),-10.,np.float64)
            for j,(routed,maximum) in enumerate(((0,1),(2,3))):
                if np.any(raw[:,maximum]<raw[:,routed]):raise AssertionError('Synthetic max must exceed routed')
                evidence[np.arange(n),j,c]=raw[:,routed]
                evidence[np.arange(n),j,(c+1)%6]=raw[:,maximum]
            data['class_scores']=evidence;np.savez(folder/'predict'/(role+'_predictions.npz'),**data)

    def fixture(self,root,main_pass=True):
        m=self.module();rows=[(c,r,s) for c in range(6) for r in m.RECEIVERS for s in m.SCENES];n=len(rows)
        known=dict(ids=np.asarray(['k'+str(i) for i in range(n)]),y=np.asarray([a[0] for a in rows],np.int64),
            rx_i=np.asarray([a[1] for a in rows],np.int64),day_i=(np.arange(n)%3+1).astype(np.int64),
            scenario=np.asarray([a[2] for a in rows]),scores=np.ones((n,4),np.float64),
            closed=np.asarray([a[0] for a in rows],np.int64),baseline_closed=np.asarray([a[0] for a in rows],np.int64))
        rows=[(tx,r,s) for tx in m.UNKNOWN_TX for r in m.RECEIVERS for s in m.SCENES];nu=len(rows)
        novel=dict(ids=np.asarray(['u'+str(i) for i in range(nu)]),rx_i=np.asarray([a[1] for a in rows],np.int64),
            day_i=(np.arange(nu)%3+1).astype(np.int64),scores=np.full((nu,4),-1.,np.float64),closed=np.zeros(nu,np.int64))
        if not main_pass:novel['scores'][:,-1]=1.
        truth=dict(ids=novel['ids'][::-1],tx=np.asarray([a[0] for a in rows])[::-1],rx_i=novel['rx_i'][::-1],
            day_i=novel['day_i'][::-1],scenario=np.asarray([a[2] for a in rows])[::-1])
        folder=root/'prediction';(folder/'predict').mkdir(parents=True);(folder/'calibration').mkdir()
        conditional=np.asarray([[0.]*6,[2.]*6,[0.]*6,[2.]*6],np.float64)
        np.savez(folder/'thresholds.npz',**{'global':np.zeros(4,np.float64),'conditional':conditional})
        self.save(folder,known,novel)
        np.savez(folder/'readouts.npz',schema=np.asarray('synthetic codec stub'))
        (folder/'readouts.json').write_text('{}',encoding='utf-8')
        marker=dict(schema=m.PREDICTION_SCHEMA,status=m.PREDICTION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            method_names=list(m.METHODS),primary_method=m.PRIMARY,classification_path='P1',score_threshold=0.,
            source_only=True,target_reads=0,source_novel_truth_reads=0,new_encoder_updates=0,new_identity_head_updates=0,
            source_V_readout_fit=False,source_novel_fit_reads=0,model_parameters_buffers_unchanged=True,
            L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,
            known_count=n,novel_count=nu,model_names=['membership','ova_hard'],readout_model_count=2,readout_steps_per_arm=1000,
            readout_optimizer_updates_total=2000,readout_parameter_counts={'membership':3457,'ova_hard':3457},
            known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz')
        (folder/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
        calibration=dict(schema=m.PREDICTION_SCHEMA,status=m.CALIBRATION_STATUS,run_id=m.RUN_ID,code_commit='a'*40,
            method_names=list(m.METHODS),group_count=42,known_only=True,unknown_inputs_used=False,
            classification_path='P1',score_threshold=0.,global_thresholds=[0.]*4,conditional_thresholds=conditional.tolist(),
            methods={name:dict(constraints_passed=True,conditional_thresholds=conditional[i].tolist(),
                groups={key:dict(constraints_passed=True) for key in m.GROUP_NAMES}) for i,name in enumerate(m.METHODS)})
        (folder/'calibration/complete.json').write_text(json.dumps(calibration),encoding='utf-8')
        tp=root/'truth.npz';np.savez(tp,**truth)
        return folder,tp,known,novel,truth,self.state(m)

    def evaluate(self,m,root,fixture,name='score'):
        folder,tp,k,u,t,state=fixture
        with patch.object(m,'_readout_gate',return_value=state):return m.score(folder,tp,root/name)

    def test_four_scalars_two_models_complete_trueTX_metrics_and_CI(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);done=self.evaluate(m,root,v)
            self.assertTrue(done['source_gate_passed']);self.assertEqual(done['next_step'],'SOURCE_OVA_READY_FOR_TARGET_FALSIFICATION')
            self.assertEqual(done['primary_method'],'ova_max');self.assertEqual(tuple(done['methods']),m.METHODS)
            self.assertEqual(done['readout_model_count'],2);self.assertEqual(done['readout_parameters_total'],6914)
            self.assertEqual(len(done['paired_comparisons']),4)
            for row in done['methods'].values():
                self.assertEqual(len(row['groups']),42);self.assertEqual(len(row['by_known_TX']),6);self.assertEqual(len(row['by_unknown_TX']),3)
                self.assertEqual(len(row['confidence_intervals']['A_K_per_class']),6);self.assertEqual(len(row['confidence_intervals']['R_U_per_class']),3)
                self.assertEqual(row['overall']['H_OS'],1.)
            self.assertIsNone(done['A_N']);self.assertIsNone(done['H_OW']);self.assertFalse(done['final_phase3_complete'])
            with self.assertRaises(FileExistsError):self.evaluate(m,root,v)

    def test_better_membership_and_routed_controls_cannot_rescue_primary(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);done=self.evaluate(m,root,self.fixture(root,False))
            self.assertTrue(done['methods']['membership_max']['goals']['passed']);self.assertTrue(done['methods']['ova_routed']['goals']['passed'])
            self.assertFalse(done['source_gate_passed']);self.assertEqual(done['next_step'],'NO_PROMOTION')

    def test_all_fits_thresholds_and_four_columns_precede_private_truth(self):
        m=self.module()
        for defect in ('marker','calibration','calibration_overall_only','missing_readout','bad_model','short_trace','unequal_initial','primary','path','thresholdshape',
                       'thresholdmismatch','scores2','class_scores_nan','max_routed_swapped','nan','private_novel','duplicate','baseline','Vfit'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                root=Path(td);v=self.fixture(root);folder,tp,k,u,t,state=v
                if defect=='marker':(folder/'complete.json').unlink()
                elif defect=='calibration':(folder/'calibration/complete.json').unlink()
                elif defect=='calibration_overall_only':
                    p=folder/'calibration/complete.json';d=json.loads(p.read_text())
                    d['methods'][m.PRIMARY]['groups']={'overall':dict(constraints_passed=True)};p.write_text(json.dumps(d))
                elif defect=='missing_readout':(folder/'readouts.npz').unlink()
                elif defect=='bad_model':state['models']['ova_hard']['weight1'][0,0]=np.nan
                elif defect=='short_trace':state['training']['losses']['membership']=[1.]*999
                elif defect=='unequal_initial':state['training']['initial_parameters_equal']=False
                elif defect in ('primary','path','Vfit'):
                    p=folder/'complete.json';d=json.loads(p.read_text());key,value={'primary':('primary_method','membership_max'),
                        'path':('classification_path','P0'),'Vfit':('source_V_readout_fit',True)}[defect];d[key]=value;p.write_text(json.dumps(d))
                elif defect in ('thresholdshape','thresholdmismatch'):
                    np.savez(folder/'thresholds.npz',**{'global':np.zeros(4,np.float64),
                        'conditional':np.ones((4,5 if defect=='thresholdshape' else 6),np.float64)})
                elif defect=='baseline':k['baseline_closed'][0]=1;np.savez(folder/'predict/known_predictions.npz',**k)
                else:
                    if defect=='scores2':u['scores']=u['scores'][:,:2]
                    elif defect=='class_scores_nan':u['class_scores'][0,0,0]=np.nan
                    elif defect=='max_routed_swapped':u['scores'][0,1]+=1.
                    elif defect=='nan':u['scores'][0,0]=np.nan
                    elif defect=='private_novel':u['tx']=np.asarray(['19-15']*len(u['ids']))
                    else:u['ids'][0]=u['ids'][1]
                    np.savez(folder/'predict/novel_predictions.npz',**u)
                with patch.object(m,'_readout_gate',return_value=state),patch.object(m,'_truth',side_effect=AssertionError('truth touched')),self.assertRaises((ValueError,FileNotFoundError)):
                    m.score(folder,tp,root/'score')

    def test_paired_objective_and_inference_effects_count_correct_not_only_accept(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);folder,tp,k,u,t,state=v
            k['closed'][0]=k['baseline_closed'][0]=1;k['scores'][0,0]=-1.;k['scores'][1:3,1]=-1.;k['scores'][:3,3]=-1.
            u['scores'][:]=1.
            for j,n in enumerate((10,20,30,40)):u['scores'][:n,j]=-1.
            self.save(folder,k,u)
            done=self.evaluate(m,root,v);pairs=done['paired_comparisons']
            expected={'ova_minus_membership_routed':(1,0,20),'ova_minus_membership_max':(-1,0,20),
                'membership_max_minus_routed':(-1,-2,10),'ova_max_minus_routed':(-3,-2,10)}
            for key,(accept,correct,rejected) in expected.items():
                with self.subTest(comparison=key):
                    pair=pairs[key];c=pair['counts'];self.assertEqual(c['known_accept_net'],accept)
                    self.assertEqual(c['known_correct_accept_net'],correct);self.assertEqual(c['unknown_rejected_net'],rejected)
                    self.assertEqual(len(pair['groups']),42);self.assertEqual(len(pair['by_known_TX']),6);self.assertEqual(len(pair['by_unknown_TX']),3)
                    self.assertAlmostEqual(pair['overall']['known_macro_open_accuracy'],correct/180)

    def test_zero_margin_is_accepted_wrong_identity_not_AK_and_v3_boundaries(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);folder,tp,k,u,t,state=v
            k['closed'][:]=k['baseline_closed'][:]=0;k['scores'][:]=u['scores'][:]=0.
            self.save(folder,k,u)
            row=self.evaluate(m,root,v)['primary_summary'];self.assertEqual(row['known_macro_acceptance'],1.)
            self.assertAlmostEqual(row['known_macro_open_accuracy'],1/6);self.assertEqual(row['unknown_macro_rejection'],0.)
        base=dict(coverage=dict(known_complete=True,unknown_complete=True),known_macro_acceptance=.9,
            original_closed_minus_new_open=.02,unknown_macro_rejection=.9)
        groups={key:copy.deepcopy(base) for key in m.GROUP_NAMES};self.assertTrue(m._goals(groups,True)['passed'])
        for field,value in [('known_macro_acceptance',.89999),('original_closed_minus_new_open',.02001),('unknown_macro_rejection',.89999)]:
            with self.subTest(field=field):
                g=copy.deepcopy(groups);g['overall'][field]=value;self.assertFalse(m._goals(g,True)['passed'])

    def test_missing_required_class_is_NA_unknown_RXscene_remains_diagnostic(self):
        m=self.module()
        for kind in ('unknown_cell','unknown_full','known_cell'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                root=Path(td);v=self.fixture(root);folder,tp,k,u,t,state=v
                if kind=='unknown_full':t['tx'][t['tx']=='19-15']='19-16'
                elif kind=='unknown_cell':t['tx'][(t['tx']=='19-15')&(t['rx_i']==1)&(t['scenario']==m.SCENES[0])]='19-16'
                else:k['y'][(k['y']==0)&(k['rx_i']==1)&(k['scenario']==m.SCENES[0])]=1
                np.savez(tp,**t);np.savez(folder/'predict/known_predictions.npz',**k)
                done=self.evaluate(m,root,v);self.assertEqual(done['source_gate_passed'],kind=='unknown_cell')
                key='overall' if kind=='unknown_full' else 'rxscene/1/'+m.SCENES[0]
                self.assertIsNone(done['methods'][m.PRIMARY]['groups'][key]['H_OS'])

    def test_scoring_never_changes_margin_threshold_or_aggregate_files(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);folder=v[0]
            files=[folder/'readouts.npz',folder/'thresholds.npz',folder/'predict/known_predictions.npz',folder/'predict/novel_predictions.npz']
            before=[p.read_bytes() for p in files];self.evaluate(m,root,v);self.assertEqual(before,[p.read_bytes() for p in files])


if __name__=='__main__':unittest.main()
