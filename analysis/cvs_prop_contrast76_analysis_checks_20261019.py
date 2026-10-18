"""P76 fake saved-output fixtures; no Torch, IQ, checkpoint or network."""
import copy
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_prop_contrast76_analysis_20261019.py')
    spec=importlib.util.spec_from_file_location('_p76_analysis_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture(a,arm):
    share=arm[1]=='1';on=arm[2]=='1';namespace='FcrossRX/train/e21/step4441/L_S';labels=[0,1,0,1]
    groups=[(1,1,1),(1,1,1),(3,2,1),(3,2,1)];scene=a.c.SCENES[0];obs=[]
    for i,(rx,day,eq) in enumerate(groups):
        group=(rx,day,eq);pid=[eq,rx,day,900+i,800+i];session=f'rx{rx}:day{day}:eq{eq}'
        personal=a.q.stable_seed(20261016421,namespace,scene,pid);env=a.q.stable_seed(20261016421,namespace,scene,'environment',group)
        obs.append(dict(sample_id=pid,namespace=namespace,scene=scene,state_end='LOS',receiver_lock_status='locked',
            channel_equalization_applied=False,group=list(group),environment_group=list(group),group_id=session,session_id=session,
            TX_used_in_group=False,sig_i_is_TX=False,seed=personal,scene_seed=a.q.stable_seed(20261016421,namespace,'scene',group),
            personal_seed=personal,packet_seed=personal,environment_seed=env,share_propagation=share,
            propagation_source='environment' if share else 'personal',propagation_fields=list(a.PROPAGATION),
            private_streams_preserved=True,propagation_arrays_not_aliased=True,same_raw_input=True,
            native_noise_chain_retained=True,native_agc_recomputed=True,environment_initialized_not_processed=True,
            private_stream_seeds={name:a.q.stable_seed(personal,name) for name in a.PRIVATE},propagation_matches_environment=share,
            receiver=dict(oscillator_cfo_hz=1.),environment_index=i//2,geometry=dict(elevation_deg=float(rx)),
            orbital_doppler_hz=100.,residual_kernel_energy=.1,propagation_kernel_length=5,residual_kernel_length=5,boundary='synthetic_reflect',
            environment_initializations_this_step=2,personal_initializations_this_step=4,native_waveforms_this_step=4,environment_processes_this_step=0))
    second=copy.deepcopy(obs)
    for r in second:r.update(view_kind='same_IQ_reuse',channel_generated=False,agc_recomputed=False)
    group_stats=[dict(RX=rx,day=day,eq=eq,TX_counts={'0':1,'1':1},count=2,scenes=[scene]) for rx,day,eq in ((1,1,1),(3,2,1))]
    eligible=a.eligibility(labels,groups)
    details=[dict(phase=p,level=l,loss=.7,total_anchors=4,temperature=.2,same_RX_positive_pairs=0,cross_group_negative_pairs=0,**eligible)
             for p in (0,1) for l in a.LEVELS] if on else []
    record=dict(step=1,cumulative_step=4441,method=arm,share_propagation=share,As_contrast=on,
        generated_As0=4,generated_As1=0,As_generated_packets=4,actual_residual_snapshots=4,As_forward_packets=8,
        logical_second_slots=4,As_same_IQ=True,L_NN=768,cross_channel_KL=False,two_separate_CE_mean=True,
        all_labeled_CE_retained=True,bypassed_elements=0,As_IQ_pair_MSE=0.,group_count=2,As_details=details,
        As_loss_raw=.7 if on else 0.,As_loss_weighted=.035 if on else 0.,ordinary_SupCon_coefficient=.1,
        As_SupCon_coefficient=.05 if on else 0.,As_loss_evaluated=on,
        **{'As_'+k:eligible[k] if on else 0. for k in ('valid_anchors','zero_anchors','positive_pairs','negative_pairs')})
    physics=dict(step=1,cumulative_step=4441,arm=arm,share_propagation=share,As_contrast=on,view0=obs,view1=second,
        labels_L_As=labels,group_stats=group_stats)
    return record,physics


def summary_fixture(proof):
    arm=proof['arm'];on=arm[2]=='1';n=proof['steps'];total=proof['native_waveforms']
    return dict(arm=arm,steps=n,generated_As0=total,generated_As1=0,native_waveforms=total,As_forward_NN=2*total,L_NN=768*n,
        logical_second_slots=total,share_propagation=arm[1]=='1',As_contrast=on,ordinary_SupCon_coefficient=.1,
        As_SupCon_coefficient=.05 if on else 0.,total_L_CE_scale=1.25,As_CE_coefficient=.25,per_pass_As_CE_coefficient=.125,
        As_KL=False,original_dropout=True,original_ordinary_raw_objectives=True,new_deployment_parameters=0,extra_NN=0,extra_U_NN=0,
        U_truth_read=False,target_access=False,physics_records=n,session_count=proof['session_count'],
        means={k+'_mean':v for k,v in proof['scalar_means'].items()})


def checks(names=None):
    a=load_analysis();c=a.c
    class Checks(unittest.TestCase):
        def test_six_rows_and_four_shared_legal_parents(self):
            rows=a.row_specs('PROJECT');self.assertEqual(tuple(r['key'] for r in rows),a.ARMS)
            self.assertEqual(sum(not r['reused'] for r in rows),4)
            self.assertEqual(len({str(r['parent']) for r in rows if r['key'] in a.CELLS}),1)
            self.assertNotEqual(rows[1]['parent'],rows[2]['parent'])
        def test_interaction_pp_not_single_combined_gain(self):
            rows={arm:dict(accuracy=value,macro_f1=value,worst_RX=value,mean_margin_over30=value,satellite_retention=value)
                for arm,value in zip(a.CELLS,(.8,.81,.82,.835))}
            value=a.interaction(rows);self.assertAlmostEqual(value['accuracy_interaction_pp'],.5)
            self.assertAlmostEqual(value['margin_over30_interaction'],.005)
            rows['C11']['accuracy']=None;self.assertIsNone(a.interaction(rows)['accuracy_interaction_pp'])
        def test_three_absolute_source_criteria(self):
            vals={arm:dict(fullV_clean=.8,midlow=value) for arm,value in zip(a.ARMS,(.0,.7,.7,.705,.706,.716))}
            vals['C11']['fullV_clean']=.799
            self.assertTrue(a.source_criteria(vals)['source_qualified'])
            bad=copy.deepcopy(vals);bad['C11']['midlow']=.7159
            self.assertFalse(a.source_criteria(bad)['checks']['midlow_interaction_at_least_0p5pp'])
            bad=copy.deepcopy(vals);bad['C11']['fullV_clean']=.79899
            self.assertFalse(a.source_criteria(bad)['checks']['clean_within_NoAs_0p1pp'])
        def test_constructor_counts_not_waveforms_and_copy_not_generation(self):
            r,p=fixture(a,'C11');proof,steps=a.prop_recount([r],[p],'C11',1,4)
            self.assertEqual(proof['native_waveforms'],4);self.assertEqual(proof['As_forward_slots'],8)
            self.assertEqual(proof['environment_initializations'],2);self.assertEqual(proof['personal_initializations'],4)
            self.assertEqual(proof['environment_processes'],0)
            self.assertTrue(proof['TX_from_labels_L_As_only'])
            self.assertEqual({r['TX'] for r in proof['label_coverage']},{0,1})
            self.assertNotIn(900,{r['TX'] for r in proof['label_coverage']})
            result=dict(TX_RX_scene_state_lock=proof['label_coverage'],L_As_total=4,L_As_forward_NN=8)
            a.validate_summary(summary_fixture(proof),proof,result)
            bad=copy.deepcopy(result);bad['TX_RX_scene_state_lock'][0]['TX']=900
            with self.assertRaises(ValueError):a.validate_summary(summary_fixture(proof),proof,bad)
        def test_OFF_stats_zero_are_not_physical_pair_absence(self):
            r,p=fixture(a,'C00');proof,steps=a.prop_recount([r],[p],'C00',1,4)
            self.assertFalse(r['As_loss_evaluated']);self.assertEqual(r['As_valid_anchors'],0)
            self.assertEqual(steps[0]['potential_valid_anchors'],4)
            self.assertGreater(proof['potential_pair_counts']['negative_pairs'],0)
        def test_shared_H_private_streams_and_true_L_negatives(self):
            r,p=fixture(a,'C11')
            for key,value in [('private_streams_preserved',False),('share_propagation',False),('TX_used_in_group',True),('sig_i_is_TX',True),('residual_kernel_energy',.2)]:
                bad=copy.deepcopy(p);bad['view0'][1][key]=value;bad['view1'][1][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):a.prop_recount([r],[bad],'C11',1,4)
            bad=copy.deepcopy(p);bad.pop('labels_L_As')
            with self.assertRaises(ValueError):a.prop_recount([r],[bad],'C11',1,4)
            bad=copy.deepcopy(r);bad['As_details'][0]['negative_pairs']+=1
            with self.assertRaises(ValueError):a.prop_recount([bad],[p],'C11',1,4)
            bad=copy.deepcopy(p);bad['view1'][0]['channel_generated']=True
            with self.assertRaises(ValueError):a.prop_recount([r],[bad],'C11',1,4)
        def test_environment_negative_mask_uses_day_and_true_label(self):
            v=a.eligibility([0,1,0,1],[(1,1,1),(1,1,1),(3,2,1),(3,2,1)])
            self.assertEqual(v,dict(valid_anchors=4,zero_anchors=0,positive_pairs=4,negative_pairs=4,valid_positive_pairs=4,valid_negative_pairs=4))
            v=a.eligibility([0,1],[(1,1,1),(3,1,1)]);self.assertEqual(v['negative_pairs'],0)
            with self.assertRaises(ValueError):a.eligibility([0,1],[(99,1,1),(3,1,1)])
        def test_extra_loss_parse_adapter_restores_and_keeps_actual_curve(self):
            steps=[dict(step=i,cumulative_step=i+4440,method='C01',As_loss_raw=.7,As_loss_weighted=.035,
                ordinary_SupCon_coefficient=.1,As_SupCon_coefficient=.05,As_loss_evaluated=True) for i in range(1,2221)]
            folder=Path('TOY')
            def lines(path):
                if Path(path).name=='prop_contrast_steps.jsonl.gz':return iter(steps)
                if Path(path).name=='reference_steps.jsonl.gz':return iter([dict(step=1,total_L=2.035,As_loss_weighted=.035)])
                return iter([])
            def parsed(path,is_as):
                record=next(c.lines(folder/'reference_steps.jsonl.gz'));self.assertAlmostEqual(record['total_L'],2.)
                return dict(curve=[dict(local_epoch=e,total_L=2.) for e in range(1,11)]),[]
            with patch.object(c,'lines',lines),patch.object(a.h,'parse_tail',parsed):
                result,_=a.parse_tail(folder,True,'C01');self.assertIs(c.lines,lines)
            self.assertAlmostEqual(result['curve'][0]['total_L'],2.035)
            self.assertEqual(result['curve'][0]['base_total_L'],2.)
            def fail(*args):raise RuntimeError('synthetic parser failure')
            with patch.object(c,'lines',lines),patch.object(a.h,'parse_tail',fail),self.assertRaises(RuntimeError):
                try:a.parse_tail(folder,True,'C01')
                finally:self.assertIs(c.lines,lines)
        def test_formal_enrichment_all6_and_interaction(self):
            rows={arm:{view:dict(accuracy=.8,macro_f1=.8,worst_RX=.7,mean_margin_over30=.2,satellite_retention=.9) for view in c.VIEWS} for arm in a.ARMS}
            for view in c.VIEWS:rows['C11'][view]['accuracy']+=.01
            obj=dict(rows=rows,stage={arm:{} for arm in a.CELLS},all4_rows_all7_verified_before_truth=True);a.enrich_formal(obj)
            self.assertTrue(obj['all6_rows_all7_verified_before_truth']);self.assertNotIn('all4_rows_all7_verified_before_truth',obj)
            self.assertAlmostEqual(obj['interaction']['urban_mean']['accuracy_interaction_pp'],1.)
            self.assertFalse(obj['stage']['C11']['reproducible_dual_gain'])
        def test_pooled_source_confusions_and_strata_interactions(self):
            records=[]
            for arm in a.ARMS:
                for scene in c.SCENES[1:]:
                    for dim,label,n in [('overall','ALL',60),*(('RX',str(rx),12) for rx in c.RX)]:
                        records.append(dict(row_id=arm,scene=scene,draw=-1,dimension=dim,stratum=label,count=n,correct=n,
                            confusion=(np.eye(6,dtype=int)*(n//6)).tolist(),accuracy=1.,macro_f1=1.,mean_margin_over30=.2,
                            satellite_retention=1.,fixed_NOAS30_clean_correct_count=n,fixed_NOAS30_clean_correct_retained=n))
            value=a.pooled_source(records);self.assertEqual(value['rows']['C11']['count'],120)
            self.assertEqual(value['rows']['C11']['macro_f1'],1.)
            self.assertEqual(value['interaction']['accuracy_interaction_pp'],0.)
            self.assertEqual(len(a.interaction_rows(records,('scene','draw','dimension','stratum'))),12)
        def test_smoke_constructor_counts_are_metadata_only(self):
            _,physics=fixture(a,'C00');proof=a.smoke_constructors([physics],4)
            self.assertEqual(proof,dict(environment_initializations=2,personal_initializations=4,native_waveforms=4,environment_processes=0))
            bad=copy.deepcopy(physics);bad['view0'][0]['environment_initializations_this_step']=8
            with self.assertRaises(ValueError):a.smoke_constructors([bad],4)
        def test_budget_arithmetic_and_unchanged_references(self):
            self.assertEqual(4*(3162962+12600+1176002)+3076,17409332)
            self.assertEqual(4*2220,8880);self.assertEqual(4*142080,568320)
            self.assertEqual(3*20+6*10,120);self.assertEqual(120*222,26640)
        def test_saved_three_check_verdict_before_formal(self):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json';path.parent.mkdir(parents=True)
                value=dict(status='SOURCE_READOUT_DECIDED',source_qualified=False,checks=dict(clean=True,urban=False,interaction=False),
                    formal_scores_read=False,target_feedback=False,NN=0,model_seed_count=1)
                path.write_text(json.dumps(value),encoding='utf-8');self.assertEqual(a.read_source_decision(temp),value)
                value['formal_scores_read']=True;path.write_text(json.dumps(value),encoding='utf-8')
                with self.assertRaises(ValueError):a.read_source_decision(temp)
        def test_all6_all7_predictions_checked_before_truth(self):
            with tempfile.TemporaryDirectory() as temp:
                specs=a.row_specs(temp);ids=np.asarray(['toy'+str(i) for i in range(6)])
                def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj),encoding='utf-8')
                for s in specs:
                    root=s['root'];cfg=dict(row_id=s['row_id']);p=root/'prediction'
                    dump(p/'complete.json',dict(status='PREDICTIONS_COMPLETE',row_id=root.name,count=6,views=list(c.VIEWS),config=cfg,
                        truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=44))
                    dump(p/'resolved_config.json',dict(config=cfg,views=list(c.VIEWS),model_frozen=True,separate_head_fit=False,
                        truth_read=False,query_fit=False,checkpoint=str(root/'source/final_ssdg.pth')))
                    dump(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',row_id=root.name,views=list(c.VIEWS),truth_last=True))
                    dump(root/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=root.name))
                    dump(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,
                        results=[dict(view=v,dimension='overall',stratum='ALL',query_count=6) for v in c.VIEWS]))
                    np.savez(p/'predictions.npz',ids=ids,**{view:np.arange(6) for view in c.VIEWS})
                    for view in c.VIEWS:np.savez(p/(view+'_scores.npz'),ids=ids,logits=np.eye(6)*30)
                truth=Path(temp)/'truth.json';dump(truth,{value:dict(label=i,receiver='toyRX',day='toyday') for i,value in enumerate(ids)})
                observed=[];read=c.read
                def spy(path):observed.append('truth' if Path(path)==truth else str(path));return read(path)
                checker=lambda s,p:dict(config=dict(row_id=s['row_id']))
                with patch.object(c,'read',side_effect=spy):c.truth_last_arrays(specs,ids,truth,temp,6,checker)
                self.assertEqual(observed.count('truth'),1);self.assertEqual(observed[-1],'truth')
                (specs[-1]['root']/'prediction/practical_low_urban_scores.npz').unlink();observed.clear()
                with patch.object(c,'read',side_effect=spy),self.assertRaises(FileNotFoundError):c.truth_last_arrays(specs,ids,truth,temp,6,checker)
                self.assertNotIn('truth',observed)
    suite=unittest.TestSuite(Checks(name) for name in names) if names else unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    return dict(status='PASS',tests=result.testsRun,NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,target_access=False)


if __name__=='__main__':print(json.dumps(checks()))
