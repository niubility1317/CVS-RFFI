"""P69 focused CPU metadata/metrics checks; no Torch, CP, IQ or network."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_optimizer_continuity69_analysis_20261018.py')
    spec=importlib.util.spec_from_file_location('_p69_analysis_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def checks(names=None):
    a=load_analysis();c=a.c
    class Checks(unittest.TestCase):
        def test_four_rows_share_two_exact_P43_parents(self):
            specs=a.row_specs('PROJECT');self.assertEqual(tuple(s['key'] for s in specs),a.ARMS)
            self.assertEqual(specs[0]['parent'],specs[2]['parent']);self.assertEqual(specs[1]['parent'],specs[3]['parent'])
            self.assertNotEqual(specs[0]['parent'],specs[1]['parent'])
            self.assertEqual([s['row_id'] for s in specs],['B00-s2026092701','B01-s2026092701']*2)
            self.assertTrue(all(s['parent'].parent.parent.name==a.REF for s in specs))
        def test_only_two_absolute_pp_source_criteria(self):
            result=a.source_criteria(.799,.8,.705,.7)
            self.assertEqual(len(result['checks']),2);self.assertTrue(result['source_qualified'])
            self.assertFalse(a.source_criteria(.79899,.8,.706,.7)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.70499,.7)['source_qualified'])
        def test_factorial_and_matched_satellite_effect(self):
            rows={arm:{view:dict(accuracy=v,macro_f1=v,worst_RX=v,satellite_retention=v) for view in c.VIEWS}
                  for arm,v in zip(a.ARMS,(.7,.8,.85,.8495))}
            result=dict(rows=rows,stage={});a.enrich_formal(result)
            self.assertAlmostEqual(result['stage']['AS_KEEP']['clean_delta_pp'],-.05)
            self.assertTrue(result['stage']['AS_KEEP']['clean_within_0p1pp'])
            self.assertAlmostEqual(result['mechanism']['AS_KEEP']['effects']['clean']['accuracy_delta_pp'],4.95)
            self.assertAlmostEqual(result['interactions']['clean']['accuracy_interaction_pp'],-10.05)
            self.assertFalse(result['stage']['AS_KEEP']['reproducible_dual_gain'])
        def test_shared_parent_and_new_forward_cost_not_double_counted(self):
            self.assertEqual(2*3162962+2*12600+2*1176002+8,8703136)
            self.assertEqual(2*20+4*10,80);self.assertEqual(2*4440+4*2220,17760)
            self.assertEqual(2*10,20)
        def test_source_verdict_precedes_formal_access(self):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json';path.parent.mkdir(parents=True)
                value=dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,NN=0,
                    model_seed_count=1,source_qualified=False,checks=dict(clean=False,midlow=True))
                path.write_text(json.dumps(value),encoding='utf8');self.assertEqual(a.read_source_decision(temp),value)
                bad=dict(value,checks=dict(clean=True,midlow=True,extra=True));path.write_text(json.dumps(bad),encoding='utf8')
                with self.assertRaises(ValueError):a.read_source_decision(temp)
                bad=dict(value,formal_scores_read=True);path.write_text(json.dumps(bad),encoding='utf8')
                with self.assertRaises(ValueError):a.read_source_decision(temp)
        def test_parent_cohort_rejects_EMA_ASCE20_and_mismatched_order(self):
            meta=dict(status='SOURCE_READOUT_COMPLETE',run_id=a.PARENT_COHORT,model_seed=a.SEED,
                checkpoint_run=a.REF,checkpoint_row='B01-s2026092701',checkpoint_stage='stage20',checkpoint_key='model',
                checkpoint_epoch=20,checkpoint_optimizer_steps=4440,checkpoint_selection='final_only',source_role='V',
                source_unique_V=1800,channel_draws=2,scenes=list(c.SCENES),condition='full',condition_index=0,NN=12602,
                training_updates=0,head_fit_updates=0,U_truth_read=False,target_access=False,formal_scores_read=False,
                new_channel_calls=0,model_state_unchanged=True,all_parameters_frozen=True,model_parameters=900722,
                model_state_keys=206,default_head_shape=[6,160],default_head_scale=30.)
            a.parent_cohort_metadata(meta)
            for patch_value in (dict(checkpoint_key='ema_model'),dict(checkpoint_row='AS_CE-s2026092701'),dict(checkpoint_epoch=30)):
                with self.assertRaises(ValueError):a.parent_cohort_metadata(dict(meta,**patch_value))
            y=np.arange(6);z=np.eye(6)*30
            parent=dict(ids=np.asarray(['p'+str(i) for i in y]),TX=y,RX=np.ones(6,int),day=np.ones(6,int),clean=z,satellite=np.tile(z,(2,3,1,1)))
            current={k:v.copy() for k,v in parent.items()};current['satellite'][0,1,0]=np.eye(6)[1]*30
            result=a.parent_cohort_arrays(parent,current)
            row=next(r for r in result['records'] if (r['scene'],r['draw'],r['dimension'])==(c.SCENES[1],0,'overall'))
            self.assertEqual(row['parent_clean_correct_count'],6);self.assertEqual(row['new_errors_within_parent_cohort'],1)
            bad={k:v.copy() for k,v in current.items()};bad['ids']=bad['ids'][::-1]
            with self.assertRaises(ValueError):a.parent_cohort_arrays(parent,bad)
            parent['clean']=np.roll(z,1,axis=1)
            self.assertTrue(all(r['empty_cohort'] for r in a.parent_cohort_arrays(parent,current)['records']))
        def test_optimizer_must_inherit_not_reset_and_have_all_epochs(self):
            def stats(step):
                return dict(state_entries=2,parameter_count=3,state_missing_parameters=1,
                    state_step_min=1,state_step_max=step,state_step_histogram={'1':1,str(step):1},
                    exp_avg_numel=5,exp_avg_sq_numel=5,exp_avg_l2=.1,exp_avg_sq_l2=.2,
                    groups=[dict(state_entries=2,lr=.0002,fresh_lr=.0002,restored_lr=.0002,parent_lr=.000001)])
            restored=stats(4440)
            restored.update({key:True for key in ('all_state_tensors_equal','per_parameter_steps_equal',
                'fresh_hyperparameters_equal','current_parameter_pointers_retained','custom_attributes_retained',
                'named_layout_checked','parent_named_layout_checked')})
            initialization=dict(reset=False,parent_state_loaded=True,schedule='cosine10',lr=.0002,weight_decay=.0001,
                parent_optimizer_steps=4440,fresh_sinc_unit_steps=0,fresh_sinc_physical_clip_calls=0,
                extra_NN=0,optimizer_updates_during_load=0,state_entries=2,restored=restored)
            records=[dict(epoch=i,cumulative_epoch=i+20,successful_updates=i*222,parent_successful_updates=4440,
                optimizer_state=stats(4440+i*222),inherited_state_entries=2,new_state_entries=0,
                inherited_step_delta_histogram={'0':1,str(i*222):1},sinc_unit_steps=i*222,sinc_physical_clip_calls=i*222,
                last_unit_step=dict(step=i*222,physical_clip_calls=i*222),no_extra_optimizer_updates=True,extra_NN=0) for i in range(1,11)]
            value=a.validate_optimizer_telemetry(initialization,records,records[-1])
            self.assertTrue(value['optimizer_readout_is_not_accuracy'])
            self.assertTrue(value['unused_parameters_not_forced_to_step4440'])
            with self.assertRaises(ValueError):a.validate_optimizer_telemetry(dict(initialization,reset=True),records,records[-1])
            with self.assertRaises(ValueError):a.validate_optimizer_telemetry(initialization,records[:-1],records[-1])
            bad=copy.deepcopy(initialization);bad['restored']['all_state_tensors_equal']=False
            with self.assertRaises(ValueError):a.validate_optimizer_telemetry(bad,records,records[-1])
            bad=copy.deepcopy(records);bad[0]['inherited_step_delta_histogram']={'223':2}
            with self.assertRaises(ValueError):a.validate_optimizer_telemetry(initialization,bad,bad[-1])
        def test_all_four_rows_all_seven_views_checked_before_truth(self):
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


if __name__=='__main__':checks()
