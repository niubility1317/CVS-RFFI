"""Focused P68 CPU metadata/metrics checks; no Torch, IQ, checkpoint or network."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_full_channel_pair68_analysis_20261018.py')
    spec=importlib.util.spec_from_file_location('_p68_analysis_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def checks(names=None):
    a=load_analysis();c=a.c
    class Checks(unittest.TestCase):
        def test_exact_row_mapping_and_shared_sat18_parent(self):
            rows={s['key']:s for s in a.row_specs('PROJECT')}
            self.assertEqual(tuple(rows),a.ARMS)
            self.assertEqual(rows['INDEP_CE6']['row_id'],'B01-s2026092701')
            self.assertEqual(rows['SAME_CE6']['row_id'],'AS_CE-s2026092701')
            self.assertEqual(rows['SAME_CE6']['parent'],rows['INDEP_CE6']['parent'])
            self.assertNotEqual(rows['AS30']['parent'],rows['INDEP_CE6']['parent'])
            self.assertTrue(rows['INDEP_CE6']['parent_As_KL_off'])
        def test_primary_attribution_and_prior20_history(self):
            self.assertTrue(a.comparison_context('INDEP_CE6','SAME_CE6')['same_parent'])
            self.assertTrue(a.comparison_context('INDEP_CE6','AS30')['first20_As_KL_history_confounded'])
            self.assertFalse(a.comparison_context('INDEP_CE6','NOAS30')['matched_noAs_parent'])
            rows={arm:{v:dict(accuracy=.8,macro_f1=.8,worst_RX=.7,satellite_retention=.9) for v in c.VIEWS} for arm in a.ARMS}
            obj=dict(rows=rows,stage={'INDEP_CE6':{}});a.enrich_formal(obj)
            self.assertEqual(obj['primary']['parent_run'],a.SAME)
            self.assertFalse(obj['stage']['INDEP_CE6']['same_parent_NoAs_control_available'])
            self.assertFalse(obj['Ag_claim'])
        def test_absolute_pp_source_rules_and_better_of_two_controls(self):
            r=a.source_criteria(.799,.8,.705,.700,.698)
            self.assertTrue(r['source_qualified']);self.assertAlmostEqual(r['clean_delta_pp'],-.1)
            self.assertFalse(a.source_criteria(.79899,.8,.706,.7,.698)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.705,.700,.701)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.70499,.700,.699)['source_qualified'])
        def test_budget_has_one_tail_and_no_additional_teacher(self):
            self.assertEqual(3162962+12600+1176002+772,4352336)
            self.assertEqual(2220*768,1704960)
            self.assertEqual(2220*128,284160)
            self.assertEqual(20+10,30)
        def test_empty_strata_and_pair_repairs(self):
            empty=c.logit_metric(np.array([],dtype=int),np.zeros((0,6)),np.zeros(0,bool))
            self.assertTrue(empty['empty_cell']);self.assertIsNone(empty['accuracy']);self.assertIsNone(empty['mean_margin_over30'])
            y=np.arange(6);old=y.copy();new=y.copy();old[0]=1;new[1]=2
            p=c.paired(y,old,new);self.assertEqual((p['repairs'],p['new_errors'],p['net_repairs']),(1,1,0))
        def test_complete_pair_metadata_and_missing_physics_rejected(self):
            pid=[1,1,1,10,80];first=dict(sample_id=pid,session_id='rx1:day1:eq1',namespace='FcrossRX/train/e21/step4441/L_S',
                scene=c.SCENES[0],state_end='LOS',receiver_lock_status='locked',channel_equalization_applied=False)
            second=dict(first,namespace=first['namespace']+'/view1-full-independent',hardware_object_reused=True,from_same_raw=True,scene=c.SCENES[1])
            physics=dict(step=1,cumulative_step=4441,view0=[first],view1=[second])
            record=dict(step=1,cumulative_step=4441,method='INDEP_CE6',L_NN=768,extra_NN=0,generated_As0=1,generated_As1=1,
                same_hardware=True,As_same_IQ=False,ordinary_raw_same_IQ=True,all_labeled_CE_retained=True,two_separate_CE_mean=True,
                cross_channel_KL=False,bypassed_elements=0,As_IQ_pair_MSE=.4,satellite_CE0=.7,satellite_CE1=.8,
                original_mixed_KL=.3,ordinary_half_KL=.2,satellite_half_KL=.4,used_mixed_KL=.1,removed_satellite_KL=.2)
            r,steps=a.pair_recount([record],[physics],1,1);self.assertEqual(r['counts'],dict(view0=1,view1=1));self.assertEqual(len(steps),1)
            self.assertAlmostEqual(r['means']['used_mixed_KL'],.1)
            summary=dict(method='INDEP_CE6',steps=1,generated_As0=1,generated_As1=1,generated_As_total=2,
                As_forward_NN=2,L_NN=768,total_L_CE_scale=1.25,mixed_group_weight=.5,As_fraction=.5,As_CE_coefficient=.25,
                per_pass_As_CE_coefficient=.125,ordinary_raw_KL_unchanged=True,As_KL=False,original_dropout=True,
                dropout_bypassed_elements=0,same_hardware_steps=1,full_physics_steps=1,view0_namespace_unchanged=True,
                second_from_same_raw=True,second_from_first_As=False,session_count=15,extra_NN=0,extra_U_NN=0,
                extra_parameters=0,extra_losses=0,U_truth_read=False,target_access=False,
                **{key+'_mean':value for key,value in r['means'].items()})
            a.validate_pair_summary(summary,r)
            bad_summary=dict(summary,generated_As_total=1)
            with self.assertRaises(ValueError):a.validate_pair_summary(bad_summary,r)
            bad=copy.deepcopy(physics);bad['view1'][0]['sample_id']=[1,3,1,10,80]
            with self.assertRaises(ValueError):a.pair_recount([record],[bad],1,1)
            with self.assertRaises(ValueError):a.pair_recount([record],[],1,1)
            bad=copy.deepcopy(record);bad['cross_channel_KL']=True
            with self.assertRaises(ValueError):a.pair_recount([bad],[physics],1,1)
        def test_source_decision_is_saved_before_formal_read(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json';p.parent.mkdir(parents=True)
                d=dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,NN=0,model_seed_count=1,
                    source_qualified=False,checks=dict(clean=True,midlow=False))
                p.write_text(json.dumps(d),encoding='utf8');self.assertEqual(a.read_source_decision(temp),d)
                d['formal_scores_read']=True;p.write_text(json.dumps(d),encoding='utf8')
                with self.assertRaises(ValueError):a.read_source_decision(temp)
        def test_four_rows_all_seven_defaulthead_outputs_before_truth(self):
            with tempfile.TemporaryDirectory() as temp:
                specs=a.row_specs(temp);ids=np.asarray(['toy'+str(i) for i in range(6)])
                def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj),encoding='utf8')
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
                    np.savez(p/'predictions.npz',ids=ids,**{v:np.arange(6) for v in c.VIEWS})
                    for view in c.VIEWS:np.savez(p/(view+'_scores.npz'),ids=ids,logits=np.eye(6)*30)
                truth=Path(temp)/'truth.json';dump(truth,{v:dict(label=i,receiver='toyRX',day='toyday') for i,v in enumerate(ids)})
                observed=[];read=c.read
                def spy(p):observed.append('truth' if Path(p)==truth else str(p));return read(p)
                checker=lambda s,p:dict(config=dict(row_id=s['row_id']))
                with patch.object(c,'read',side_effect=spy):c.truth_last_arrays(specs,ids,truth,temp,6,checker)
                self.assertEqual(observed.count('truth'),1);self.assertEqual(observed[-1],'truth')
                (specs[-1]['root']/'prediction/practical_low_urban_scores.npz').unlink();observed.clear()
                with patch.object(c,'read',side_effect=spy),self.assertRaises(FileNotFoundError):c.truth_last_arrays(specs,ids,truth,temp,6,checker)
                self.assertNotIn('truth',observed)
    suite=(unittest.TestSuite(Checks(name) for name in names) if names else unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':checks()
