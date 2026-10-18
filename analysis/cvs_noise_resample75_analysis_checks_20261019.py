"""P75 CPU fixtures: saved metadata, cost arithmetic and truth-last reads only."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_noise_resample75_analysis_20261019.py')
    spec=importlib.util.spec_from_file_location('_p75_analysis_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixtures(a,arm):
    independent=arm==a.PRIMARY[0];namespace='FcrossRX/train/e21/step4441/L_S';pid=[1,1,1,10,80];scene=a.c.SCENES[0]
    seed=a.stable_seed(20261016421,namespace,scene,pid)
    first=dict(sample_id=pid,session_id='rx1:day1:eq1',namespace=namespace,scene=scene,seed=seed,
        state_end='LOS',receiver_lock_status='locked',channel_equalization_applied=False,receiver=dict(foo=.1),geometry=dict(elevation_deg=20.),
        quality_snr_db=20.,residual_scale=1.,estimated_frequency_hz=100.,output_frequency_hz=3.,phase_tracking_applied=True,
        iq_compensation_applied=True,residual_kernel_energy=.2,boundary='synthetic_reflect',agc_gain_db=2.,
        measured_added_noise_power_pre_agc=.001,signal_to_added_noise_db=18.,agc_limited=False,output_metrics=dict(rms=1.))
    second=copy.deepcopy(first)
    if independent:
        ns=namespace+a.SECOND_SUFFIX
        second.update(namespace=ns,physics_namespace=namespace,physics_seed=seed,noise1_namespace=ns,
            noise1_seed=a.stable_seed(20261016421,ns,scene,pid),noise_stream_only_resampled=True,non_noise_metadata_equal=True,
            hardware_object_reused=True,from_same_raw=True,native_noise_chain_retained=True,native_agc_recomputed=True,
            output_white_noise_injection=False,exact_noise_energy_calibration=False,first_agc_reused=False,
            non_noise_metadata_fields_checked=sorted(set(first)-a.NOISE_FIELDS-{'sample_id','session_id','namespace','scene'}),
            channel_generated=True,view_kind='independent_noise',agc_recomputed=True,agc_gain_db=2.1,measured_added_noise_power_pre_agc=.0011)
    else:second.update(channel_generated=False,view_kind='same_IQ_reuse',agc_recomputed=False)
    physics=dict(step=1,cumulative_step=4441,noise_arm=a.ARM_IDS[arm],actual_residual_snapshots=2 if independent else 1,
        physical_latent_draws=1,secondary_noise_redraws=int(independent),view0=[first],view1=[second])
    record=dict(step=1,cumulative_step=4441,method=a.RECORD_METHODS[arm][0],noise_arm=a.ARM_IDS[arm],L_NN=768,extra_NN=0,
        generated_As0=1,generated_As1=int(independent),As_generated_packets=2 if independent else 1,
        actual_residual_snapshots=2 if independent else 1,actual_secondary_snapshots=int(independent),secondary_noise_redraws=int(independent),
        physical_latent_draws=1,As_forward_packets=2,As_same_IQ=not independent,same_hardware=True,ordinary_raw_same_IQ=True,
        all_labeled_CE_retained=True,two_separate_CE_mean=True,cross_channel_KL=False,bypassed_elements=0,
        As_IQ_pair_MSE=.03 if independent else 0.,satellite_CE0=.7,satellite_CE1=.8,
        original_mixed_KL=.3,ordinary_half_KL=.2,satellite_half_KL=.4,used_mixed_KL=.1,removed_satellite_KL=.2)
    return record,physics


def summary_fixture(a,pair):
    n=pair['steps'];first=pair['counts']['view0'];second=pair['actual_secondary_snapshots']
    return dict(method=a.METHODS[pair['arm']],steps=n,generated_As0=first,generated_As1=second,generated_As_total=first+second,
        As_forward_NN=first*2,L_NN=n*768,physical_latent_draws=first,logical_second_slots=first,total_L_CE_scale=1.25,
        As_CE_coefficient=.25,per_pass_As_CE_coefficient=.125,ordinary_raw_KL_unchanged=True,As_KL=False,original_dropout=True,
        dropout_bypassed_elements=0,extra_NN=0,extra_U_NN=0,extra_parameters=0,extra_losses=0,U_truth_read=False,target_access=False,
        arm=a.ARM_IDS[pair['arm']],first_noisestream_unchanged=True,nonnoise_replayed=True,AGC_per_view_native=True,
        **{k+'_mean':v for k,v in pair['means'].items()})


def checks(names=None):
    a=load_analysis();c=a.c
    class Checks(unittest.TestCase):
        def test_five_rows_common_new_parent_and_reference_history(self):
            rows={s['key']:s for s in a.row_specs('PROJECT')}
            self.assertEqual(tuple(rows),a.ARMS)
            self.assertEqual(rows[a.PRIMARY[0]]['row_id'],'N01-s2026092701')
            self.assertEqual(rows[a.PRIMARY[1]]['row_id'],'C01-s2026092701')
            self.assertEqual(rows[a.PRIMARY[0]]['parent'],rows[a.PRIMARY[1]]['parent'])
            self.assertEqual(rows[a.PRIMARY[0]]['parent'],rows['INDEP_FULL_CE6']['parent'])
            self.assertNotEqual(rows['AS30']['parent'],rows[a.PRIMARY[0]]['parent'])
            self.assertEqual(sum(not r['reused'] for r in rows.values()),2)
        def test_source_absolute_pp_and_best_control(self):
            self.assertTrue(a.source_criteria(.799,.8,.705,.700,.698)['source_qualified'])
            self.assertFalse(a.source_criteria(.79899,.8,.706,.700,.698)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.705,.700,.701)['source_qualified'])
            self.assertFalse(a.source_criteria(.8,.8,.70499,.700,.699)['source_qualified'])
        def test_formal_primary_context_and_single_seed_boundary(self):
            rows={arm:{v:dict(accuracy=.8,macro_f1=.8,worst_RX=.7) for v in c.VIEWS} for arm in a.ARMS}
            rows[a.PRIMARY[0]]['clean']['accuracy']=.799
            obj=dict(rows=rows,stage={arm:{} for arm in a.NEW_ARMS},all4_rows_all7_verified_before_truth=True)
            a.enrich_formal(obj)
            self.assertTrue(obj['all5_rows_all7_verified_before_truth'])
            self.assertNotIn('all4_rows_all7_verified_before_truth',obj)
            self.assertAlmostEqual(obj['primary']['by_view']['clean']['accuracy_delta_pp'],-.1)
            self.assertTrue(a.comparison_context(*a.PRIMARY)['same_parent'])
            self.assertTrue(a.comparison_context(a.PRIMARY[0],'AS30')['first20_As_KL_history_confounded'])
            self.assertFalse(obj['stage'][a.PRIMARY[0]]['reproducible_dual_gain'])
        def test_C_copied_slot_not_new_physical_generation(self):
            r,p=fixtures(a,a.NEW_ARMS[0]);result,steps=a.pair_recount([r],[p],a.NEW_ARMS[0],1,1)
            self.assertEqual(result['counts'],dict(view0=1,view1=1))
            self.assertEqual(result['actual_residual_snapshots'],1)
            self.assertEqual(result['actual_secondary_snapshots'],0)
            self.assertEqual(result['logical_As_forward_slots'],2)
            self.assertEqual(result['AGC_changed_pairs'],0)
            a.validate_pair_summary(summary_fixture(a,result),result)
            bad=copy.deepcopy(r);bad['generated_As1']=1
            with self.assertRaises(ValueError):a.pair_recount([bad],[p],a.NEW_ARMS[0],1,1)
            bad=copy.deepcopy(p);bad['view1'][0]['agc_gain_db']=2.1
            with self.assertRaises(ValueError):a.pair_recount([r],[bad],a.NEW_ARMS[0],1,1)
        def test_N_only_noise_state_fixed_AGC_allowed(self):
            r,p=fixtures(a,a.PRIMARY[0]);result,steps=a.pair_recount([r],[p],a.PRIMARY[0],1,1)
            self.assertEqual(result['actual_residual_snapshots'],2)
            self.assertEqual(result['physical_latent_draws'],1)
            self.assertEqual(result['AGC_changed_pairs'],1)
            self.assertAlmostEqual(steps[0]['mean_AGC_delta_db'],.1)
            a.validate_pair_summary(summary_fixture(a,result),result)
            for key,value in [('quality_snr_db',21.),('residual_scale',2.),('phase_tracking_applied',False),
                              ('native_noise_chain_retained',False),('exact_noise_energy_calibration',True),('first_agc_reused',True),
                              ('noise1_seed',p['view1'][0]['seed'])]:
                bad=copy.deepcopy(p);bad['view1'][0][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):a.pair_recount([r],[bad],a.PRIMARY[0],1,1)
            bad=summary_fixture(a,result);bad['generated_As_total']=1
            with self.assertRaises(ValueError):a.validate_pair_summary(bad,result)
        def test_missing_physics_KL_and_unknown_arm_rejected(self):
            r,p=fixtures(a,a.PRIMARY[0])
            with self.assertRaises(ValueError):a.pair_recount([r],[],a.PRIMARY[0],1,1)
            bad=copy.deepcopy(r);bad['cross_channel_KL']=True
            with self.assertRaises(ValueError):a.pair_recount([bad],[p],a.PRIMARY[0],1,1)
            with self.assertRaises(ValueError):a.pair_recount([r],[p],'UNKNOWN',1,1)
        def test_cost_arithmetic_and_slot_counts(self):
            self.assertEqual(2*(3162962+12600+1176002)+1540,8704668)
            self.assertEqual(2220*768,1704960)
            self.assertEqual(2220*64,142080)
            self.assertEqual(142080+284160,426240)
            self.assertEqual(3*20+5*10,110)
        def test_saved_source_verdict_precedes_formal(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json';p.parent.mkdir(parents=True)
                value=dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,NN=0,model_seed_count=1,
                    source_qualified=False,checks=dict(clean=True,midlow=False))
                p.write_text(json.dumps(value),encoding='utf-8');self.assertEqual(a.read_source_decision(temp),value)
                value['formal_scores_read']=True;p.write_text(json.dumps(value),encoding='utf-8')
                with self.assertRaises(ValueError):a.read_source_decision(temp)
        def test_all5_rows_all7_before_only_truth_read(self):
            with tempfile.TemporaryDirectory() as temp:
                specs=a.row_specs(temp);ids=np.asarray(['toy'+str(i) for i in range(6)])
                def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj),encoding='utf-8')
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
        def test_empty_cells_and_error_repairs(self):
            empty=c.logit_metric(np.array([],dtype=int),np.zeros((0,6)),np.zeros(0,bool))
            self.assertTrue(empty['empty_cell']);self.assertIsNone(empty['accuracy'])
            y=np.arange(6);old=y.copy();new=y.copy();old[0]=1;new[1]=2
            result=c.paired(y,old,new);self.assertEqual((result['repairs'],result['new_errors'],result['net_repairs']),(1,1,0))
    suite=unittest.TestSuite(Checks(name) for name in names) if names else unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    return dict(status='PASS',tests=result.testsRun,NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,target_access=False)


if __name__=='__main__':print(json.dumps(checks()))
