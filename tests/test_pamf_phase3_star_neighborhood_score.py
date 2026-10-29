"""Synthetic four-state U-only matching and current-DA curve tests."""
from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


STATES=('DA0_REG0','DA1_REG0','DA0_REG1','DA1_REG1')
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


class StarNeighborhoodScoreTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_neighborhood_score')

    def targets(self):return Path(__file__).resolve().parents[1]/'configs/pamf_phase3_targets_20261029.json'

    def fixture(self,root,K=2):
        uy=np.tile(np.repeat(np.arange(8,dtype=np.int64),8),6)
        qy=np.tile(np.repeat(np.arange(9,dtype=np.int64),16),6)
        ui=np.asarray(['U'+str(i) for i in range(384)]);qi=np.asarray(['Q'+str(i) for i in range(864)])
        us=np.repeat(SCENES,64);qs=np.repeat(SCENES,144);closed=np.broadcast_to(np.where(qy<6,qy,0),(2,864)).copy()
        margins={}
        for i,name in enumerate(STATES):
            reg=i>=2;values=np.full((864,6+(K if reg else 0)),-2.,np.float64)
            values[np.arange(864),closed[i%2]]=np.where(qy<6,1.,-1.)
            if reg and K>=2:values[qy==6,6]=1.;values[qy==7,7]=1.
            margins[name]=values
        # One old record per scene rejected in DA0, recovered by the actual DA1 neighborhood fusion.
        for s in range(6):
            margins['DA0_REG0'][s*144,:]=-1.;margins['DA0_REG1'][s*144,:]=-1.
        p=dict(ids=qi,U_ids=ui,state_names=np.asarray(STATES),closed_prediction=closed,
            baseline_closed_prediction=closed[0].copy(),knownness=np.stack([margins[n].max(1) for n in STATES]),
            prediction=np.stack([np.where(margins[n].max(1)>=0,margins[n].argmax(1),-1) for n in STATES]).astype(np.int64))
        p.update({'joint_margins_'+n:v for n,v in margins.items()})
        assignment=np.where(uy>=6,uy-6,-1).astype(np.int64) if K else np.full(384,-1,np.int64)
        diag=dict(ids=ui,cluster_labels=uy.copy(),assignments=assignment,candidate_mask=assignment>=0,
            cluster_sizes=np.bincount(uy),anonymous_ids=np.asarray(['anon'+str(i) for i in range(K)],dtype='U16'))
        state=dict(schema='pamf_phase3_star_neighborhood_state_v1',cluster_embedding=np.zeros((8,688),np.float64),
            cluster_logits=np.zeros((8,6),np.float64),cluster_margin=np.zeros(8,np.float64),cluster_radius=np.ones(8,np.float64),
            anonymous_indices=np.arange(6,6+K,dtype=np.int64),anonymous_penalty=np.zeros(K,np.float64))
        done=dict(status='STAR_NEIGHBORHOOD_PREDICTIONS_COMPLETE',schema='pamf_phase3_star_neighborhood_target_v1',
            run_id='20261029-phase3-star-neighborhood-s2026092701-r01',condition='P1',view_family='practical_cross_receiver',
            classification_path='P1',rejection_input_paths=['P1','P0'],count=864,U_count=384,states=list(STATES),
            prototype_counts={'DA0':K,'DA1':K},all_new_IQ_satellite=True,truth_read=False,query_fit=False,
            registry_frozen_before_Q=True,U_day='2021-03-08',Q_day='2021-03-15',DA1_fusion_applied=True)
        ut=root/'U.npz';qt=root/'Q.npz'
        np.savez(ut,ids=ui[::-1],y=uy[::-1],receiver=np.full(384,26,np.int64),scenario=us[::-1])
        np.savez(qt,ids=qi[::-1],y=qy[::-1],receiver=np.full(864,26,np.int64),scenario=qs[::-1])
        return done,p,diag,state,ut,qt,uy,qy,qs

    def evaluate(self,m,root,values,name='score'):
        done,p,diag,state,ut,qt,*_=values
        with patch.object(m,'_prediction_gate',return_value=(done,p,diag,state)):
            return m.score(root,ut,qt,self.targets(),root/name)

    def test_full_four_states_main_H_rosters_CI_and_registered_competition(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);values=self.fixture(root);before=deepcopy(values[:4]);result=self.evaluate(m,root,values)
            self.assertTrue(result['primary_pass']);self.assertEqual(result['primary_state'],'DA1_REG1')
            self.assertEqual(result['cluster_diagnostics']['mapping'],{'0':6,'1':7})
            for name,row in result['states'].items():
                self.assertEqual(set(row['by_scenario']),set(SCENES));self.assertEqual(len(row['by_transmitter']),9)
                self.assertEqual(set(row['by_receiver']),{'26'});self.assertEqual(set(row['by_day']),{'2021-03-15'})
                self.assertTrue(row['overall']['coverage']['complete']);self.assertEqual(len(row['overall']['confidence_intervals']['A_K_per_class']),6)
                self.assertEqual(row['overall']['R_U_labels'],[8] if name.endswith('REG1') else [6,7,8])
                self.assertAlmostEqual(row['overall']['H_score'],row['overall']['H_OS'])
                if name.endswith('REG0'):
                    self.assertIsNone(row['overall']['A_N']);self.assertIsNone(row['overall']['H_old_new']);self.assertIsNone(row['overall']['H_OW'])
            main=result['states']['DA1_REG1']['overall'];self.assertEqual(main['A_N'],1.);self.assertEqual(main['H_OW'],1.)
            self.assertEqual(main['intended_registered_macro_acceptance'],1.)
            self.assertIsNone(result['deltas']['DA1_REG1_minus_DA1_REG0']['unknown_rejected_net'])
            self.assertEqual(result['deltas']['DA1_REG0_minus_DA0_REG0']['counts']['old_correct_accept_net'],6)
            self.assertTrue(result['states']['DA1_REG1']['goals']['passed'])
            for current,original in zip(values[:4],before):
                if isinstance(current,dict):
                    for k,v in current.items():
                        if isinstance(v,np.ndarray):self.assertTrue(np.array_equal(v,original[k]))
            with self.assertRaises(FileExistsError):self.evaluate(m,root,values)

    def test_current_DA_identity_and_joint_winner_drive_OSCR_not_baseline(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);done,p,diag,state,ut,qt,uy,qy,qs=v
            wrong=np.where(qy<6,(qy+1)%6,0);p['closed_prediction'][1]=wrong
            for index,name in ((1,'DA1_REG0'),(3,'DA1_REG1')):
                a=p['joint_margins_'+name];a[:,:6]=-2.;a[np.arange(864),wrong]=np.where(qy<6,1.,-1.)
                p['knownness'][index]=a.max(1);p['prediction'][index]=np.where(a.max(1)>=0,a.argmax(1),-1)
            r=self.evaluate(m,root,v)
            for name in ('DA1_REG0','DA1_REG1'):
                row=r['states'][name]['overall'];self.assertEqual(row['original_F0_macro_closed_accuracy'],1.)
                self.assertEqual(row['old_macro_open_accuracy'],0.);self.assertEqual(row['oscr_auc_macro'],0.)
                self.assertAlmostEqual(row['auroc_known_macro'],1.)
            self.assertFalse(r['primary_pass'])

    def test_U_only_mapping_does_not_change_when_Q_labels_swap_or_arrival_missing(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);a=self.evaluate(m,root,v,'first');done,p,diag,state,ut,qt,uy,qy,qs=v
            changed=qy.copy();changed[qy==6]=7;changed[qy==7]=6
            np.savez(qt,ids=p['ids'],y=changed,receiver=np.full(864,26,np.int64),scenario=qs)
            b=self.evaluate(m,root,v,'swapped');self.assertEqual(a['cluster_diagnostics']['mapping'],b['cluster_diagnostics']['mapping'])
            self.assertEqual(b['states']['DA1_REG1']['overall']['A_N'],0.)
            changed=qy.copy();changed[qy==7]=8
            np.savez(qt,ids=p['ids'],y=changed,receiver=np.full(864,26,np.int64),scenario=qs)
            c=self.evaluate(m,root,v,'missing');row=c['states']['DA1_REG1']['overall']
            self.assertEqual(row['new_macro_denominator'],2);self.assertIsNone(row['A_N']);self.assertFalse(row['coverage']['complete'])
            self.assertIsNone(c['states']['DA1_REG0']['overall']['R_U']);self.assertFalse(c['primary_pass'])

    def test_zeroK_and_unmatched_anonymous_accepts_are_wrong_not_rejected(self):
        m=self.module()
        for mode in ('zero','no_partition','pure_old','one_missed'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);v=self.fixture(root,K=0 if mode in ('zero','no_partition') else 2);done,p,diag,state,ut,qt,uy,qy,qs=v
                if mode=='no_partition':
                    diag['cluster_labels']=np.full(384,-1,np.int64);diag['cluster_sizes']=np.zeros(0,np.int64)
                    state['cluster_embedding']=np.zeros((0,688),np.float64);state['cluster_logits']=np.zeros((0,6),np.float64)
                    state['cluster_margin']=state['cluster_radius']=np.zeros(0,np.float64)
                    done['DA1_fusion_applied']=False;p['closed_prediction'][1]=p['closed_prediction'][0]
                    for index,name,other in ((1,'DA1_REG0','DA0_REG0'),(3,'DA1_REG1','DA0_REG1')):
                        p['joint_margins_'+name]=p['joint_margins_'+other].copy()
                        p['knownness'][index]=p['knownness'][index-1];p['prediction'][index]=p['prediction'][index-1]
                elif mode=='pure_old':
                    diag['assignments']=np.where(uy==0,0,np.where(uy==1,1,-1)).astype(np.int64)
                elif mode=='one_missed':diag['assignments']=np.where(uy==6,0,-1).astype(np.int64)
                diag['candidate_mask']=diag['assignments']>=0
                if mode not in ('zero','no_partition'):
                    for index,name in ((2,'DA0_REG1'),(3,'DA1_REG1')):
                        a=p['joint_margins_'+name];a[qy>=6,6:]=-2.
                        if mode=='pure_old':a[qy>=6,6]=1.
                        else:a[qy==6,6]=1.
                        p['knownness'][index]=a.max(1);p['prediction'][index]=np.where(a.max(1)>=0,a.argmax(1),-1)
                result=self.evaluate(m,root,v);row=result['states']['DA1_REG1']['overall']
                self.assertEqual(row['new_macro_denominator'],2);self.assertEqual(row['A_N'],.5 if mode=='one_missed' else 0.)
                self.assertFalse(result['primary_pass'])
                if mode=='pure_old':
                    self.assertEqual(result['cluster_diagnostics']['mapping'],{});self.assertEqual(row['unmatched_anonymous_accepted'],288)
                    self.assertEqual(row['R_U'],0.);self.assertEqual(row['planned_later_macro_rejection'],0.)
                if mode=='zero':self.assertEqual(row['intended_registered_macro_acceptance'],.75)
                if mode=='no_partition':
                    self.assertEqual(result['all_U_partition_diagnostics']['partition_cluster_count'],0)
                    self.assertFalse(result['adaptation_goals']['passed'])

    def test_missing_scene_or_required_class_is_NA_not_zero_fill(self):
        m=self.module()
        for mode in ('empty_scene','old','arrival','later'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);v=self.fixture(root);done,p,diag,state,ut,qt,uy,qy,qs=v;qs=qs.copy();select=qs==SCENES[0]
                if mode!='empty_scene':select&=qy=={'old':5,'arrival':7,'later':8}[mode]
                qs[select]=SCENES[1];np.savez(qt,ids=p['ids'],y=qy,receiver=np.full(864,26,np.int64),scenario=qs)
                result=self.evaluate(m,root,v);self.assertFalse(result['primary_pass'])
                row=result['states']['DA1_REG1']['by_scenario'][SCENES[0]];self.assertFalse(row['coverage']['complete'])
                self.assertIsNone(row['H_score']);self.assertIsNone(row['H_OW'])
                if mode in ('empty_scene','old'):self.assertIsNone(row['old_macro_open_accuracy'])
                if mode in ('empty_scene','arrival'):self.assertIsNone(row['A_N'])
                if mode in ('empty_scene','later'):self.assertIsNone(row['R_U'])

    def test_v3_90_and_same_DA_old_retention_and_later_scene_constraints(self):
        m=self.module();base=dict(coverage=dict(complete=True),old_macro_acceptance=.9,original_F0_minus_old_open=.02,
            R_U=.9,A_N=.8,intended_registered_macro_acceptance=.9,old_loss_from_same_DA_REG0=.02)
        self.assertTrue(all(m._registration_checks(base).values()))
        for key,value in [('old_macro_acceptance',.8999),('original_F0_minus_old_open',.0201),('R_U',.8999),
                          ('intended_registered_macro_acceptance',.8999),('old_loss_from_same_DA_REG0',.0201)]:
            with self.subTest(key=key):
                row=dict(base,**{key:value});self.assertFalse(all(m._registration_checks(row).values()))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);v=self.fixture(root);done,p,diag,state,ut,qt,uy,qy,qs=v
            a=p['joint_margins_DA1_REG1'];bad=np.flatnonzero((qy==8)&(qs==SCENES[0]))[:6];a[bad,6]=1.
            p['knownness'][3]=a.max(1);p['prediction'][3]=np.where(a.max(1)>=0,a.argmax(1),-1)
            result=self.evaluate(m,root,v);self.assertAlmostEqual(result['states']['DA1_REG1']['overall']['R_U'],.9375)
            self.assertFalse(result['registration_goals']['DA1']['by_scenario'][SCENES[0]]['planned_later_unknown_protection'])
            self.assertFalse(result['primary_pass'])

    def test_all_state_gate_and_U_Q_disjointness_precede_truth(self):
        m=self.module()
        for mode in ('incomplete','truthflag','overlap','ids','wrong_path','knownness','state_missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);v=self.fixture(root);done,p,diag,state,*_=v
                if mode=='truthflag':done['truth_read']=True
                elif mode=='overlap':p['U_ids'][0]=p['ids'][0]
                elif mode=='ids':diag['ids']=diag['ids'][::-1]
                elif mode=='wrong_path':done['classification_path']='P0'
                elif mode=='knownness':p['knownness'][0,0]+=1
                elif mode=='state_missing':del p['joint_margins_DA1_REG1']
                gate=patch.object(m,'_prediction_gate',side_effect=ValueError('incomplete')) if mode=='incomplete' else patch.object(m,'_prediction_gate',return_value=(done,p,diag,state))
                with gate,patch.object(m,'_truth',side_effect=AssertionError('truth opened')) as truth,self.assertRaises((ValueError,KeyError)):
                    m.score(root,root/'U',root/'Q',self.targets(),root/'score')
                self.assertEqual(truth.call_count,0)


if __name__=='__main__':unittest.main()
