"""P77 deterministic saved-output fixtures; zero model, IQ, checkpoint or network."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('cvs_quality_entropy77_analysis_20261019.py')
    spec=importlib.util.spec_from_file_location('_p77_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def entropy_fixture(a,arm='Q',packets=4):
    namespace='FcrossRX/train/e21/step4441/L_S';db=[-10.,0.,10.,30.][:packets]
    original=np.asarray([a.epsilon(x) for x in db]);seed=None if arm=='Q' else a.q.stable_seed(20261016421,namespace,'quality_entropy77/permutation')
    permutation=np.arange(packets) if arm=='Q' else np.random.default_rng(seed).permutation(packets)
    assigned=original[permutation];change=.125*(.6-.8+.7-.9)
    record=dict(step=1,cumulative_step=4441,method=arm,entropy_weighted_delta=change,entropy=dict(mode=arm,
        epsilon_min=float(original.min()),epsilon_max=float(original.max()),epsilon_mean=float(original.mean()),
        epsilon_std=float(original.std()),assigned_epsilon_min=float(assigned.min()),assigned_epsilon_max=float(assigned.max()),
        assigned_epsilon_mean=float(assigned.mean()),epsilon_multiset_preserved=True,
        As_CE_passes=2,epsilon_same_both_passes=True,nominal_CE0=.8,nominal_CE1=.9,new_CE0=.6,new_CE1=.7,
        weighted_delta=change,base_total_L=2.,total_L=2.+change))
    observations=[dict(sample_id=[1,1,1,600+i,800+i],namespace=namespace,scene=a.c.SCENES[0],state_end='LOS',
        receiver_lock_status='locked',signal_to_added_noise_db=value) for i,value in enumerate(db)]
    physics=dict(step=1,cumulative_step=4441,arm=arm,view0=observations,labels_L_As=list(range(packets)),entropy=dict(
        signal_to_added_noise_db=db,epsilon_original=original.tolist(),epsilon_assigned=assigned.tolist(),
        signal_to_added_noise_db_native=db,snr_boundary_kinds=['finite']*packets,
        permutation=permutation.tolist(),permutation_seed=seed,namespace=namespace,mode=arm))
    counts=dict(snr_finite_count=packets,snr_explicit_infinite_count=0,snr_zero_signal_count=0,snr_zero_added_noise_count=0)
    record['entropy'].update(counts);physics['entropy'].update(counts)
    return record,physics


def formal_fixture(a,base):
    specs=a.row_specs(base);ids=np.asarray(['toy'+str(i) for i in range(6)])
    def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8')
    for spec in specs:
        root=spec['root'];cfg=dict(row_id=spec['row_id']);pred=root/'prediction'
        dump(pred/'complete.json',dict(status='PREDICTIONS_COMPLETE',row_id=root.name,count=6,views=list(a.c.VIEWS),config=cfg,
            truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=44))
        dump(pred/'resolved_config.json',dict(config=cfg,views=list(a.c.VIEWS),model_frozen=True,separate_head_fit=False,
            truth_read=False,query_fit=False,checkpoint=str(root/'source/final_ssdg.pth')))
        dump(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',row_id=root.name,views=list(a.c.VIEWS),truth_last=True))
        dump(root/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=root.name))
        dump(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,
            results=[dict(view=view,dimension='overall',stratum='ALL',query_count=6) for view in a.c.VIEWS]))
        np.savez(pred/'predictions.npz',ids=ids,**{view:np.arange(6) for view in a.c.VIEWS})
        for view in a.c.VIEWS:np.savez(pred/(view+'_scores.npz'),ids=ids,logits=np.eye(6)*30)
    truth=Path(base)/'truth.json';dump(truth,{value:dict(label=i,receiver='toyRX',day='toyday') for i,value in enumerate(ids)})
    return specs,ids,truth,lambda spec,project:dict(config=dict(row_id=spec['row_id']))


def checks(names=None):
    a=load_analysis();c=a.c
    class Checks(unittest.TestCase):
        def test_five_rows_three_reused_and_shared_ASCE20(self):
            rows=a.row_specs('PROJECT');self.assertEqual(tuple(r['key'] for r in rows),a.ARMS)
            self.assertEqual(sum(not r['reused'] for r in rows),2)
            self.assertEqual(len({str(r['parent']) for r in rows[2:]}),1)
            self.assertEqual(rows[2]['run'],a.REFERENCE);self.assertEqual(rows[3]['run'],a.RUN)
            self.assertNotEqual(rows[1]['parent'],rows[2]['parent'])
        def test_fixed_epsilon_formula_limits_and_invalid(self):
            self.assertAlmostEqual(a.epsilon(0.),.105);self.assertAlmostEqual(a.epsilon(10.),.01+.19/11)
            self.assertEqual(a.epsilon('+inf'),.01);self.assertEqual(a.epsilon('-inf'),.20)
            self.assertAlmostEqual(a.epsilon(10000.),.01);self.assertAlmostEqual(a.epsilon(-10000.),.20)
            for value in (None,float('nan'),float('inf'),True,'10'):
                with self.subTest(value=value),self.assertRaises(ValueError):a.epsilon(value)
        def test_Q_identity_R_named_permutation_same_multiset(self):
            for arm in a.NEW_ARMS:
                r,p=entropy_fixture(a,arm);value=a.entropy_record(r,p,arm,1,4)
                self.assertAlmostEqual(value['loss_delta'],-.05)
                self.assertAlmostEqual(value['epsilon_mean'],value['assigned_epsilon_mean'])
                self.assertTrue(value['multiset_exact']);self.assertEqual(len(value['packet_rows']),4)
                if arm=='Q':self.assertEqual(value['permutation_moved'],0)
        def test_assignment_negative_fields_and_physical_ratio(self):
            r,p=entropy_fixture(a,'R')
            changes=[('epsilon_original',[.01]*4),('epsilon_assigned',[.01]*4),('permutation',[0]*4),
                ('permutation_seed',1),('mode','Q'),('signal_to_added_noise_db',[30.]*4),('namespace','unknown')]
            for key,value in changes:
                bad=copy.deepcopy(p);bad['entropy'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):a.entropy_record(r,bad,'R',1,4)
            bad=copy.deepcopy(p);bad['view0'][0]['signal_to_added_noise_db']=None
            with self.assertRaises(ValueError):a.entropy_record(r,bad,'R',1,4)
        def test_native_zero_power_limits_keep_all_epsilon_and_CE_slots(self):
            r,p=entropy_fixture(a,'Q')
            original=[None,None,'-inf','+inf'];resolved=['-inf','+inf','-inf','+inf']
            kinds=['zero_signal','zero_added_noise','explicit_negative_infinity','explicit_positive_infinity']
            eps=[a.epsilon(v) for v in resolved]
            for row,db,power in zip(p['view0'],original,(1.,0.,1.,0.)):
                row['signal_to_added_noise_db']=db;row['measured_added_noise_power_pre_agc']=power
            p['entropy'].update(signal_to_added_noise_db=resolved,signal_to_added_noise_db_native=original,
                snr_boundary_kinds=kinds,epsilon_original=eps,epsilon_assigned=eps)
            counts=dict(snr_finite_count=0,snr_explicit_infinite_count=2,snr_zero_signal_count=1,snr_zero_added_noise_count=1)
            p['entropy'].update(counts);r['entropy'].update(counts)
            for prefix in ('epsilon','assigned_epsilon'):
                r['entropy'].update({prefix+'_min':min(eps),prefix+'_max':max(eps),prefix+'_mean':float(np.mean(eps))})
            r['entropy']['epsilon_std']=float(np.std(eps))
            value=a.entropy_record(r,p,'Q',1,4);self.assertEqual(len(value['packet_rows']),4)
            self.assertEqual(value['snr_zero_signal_count'],1);self.assertEqual(value['snr_zero_added_noise_count'],1)
            for power in (None,-1.,float('nan'),True):
                with self.subTest(power=power),self.assertRaises(ValueError):a.resolved_native_snr(dict(signal_to_added_noise_db=None,measured_added_noise_power_pre_agc=power))
            with self.assertRaises(ValueError):a.resolved_native_snr({})
        def test_ce_correction_total_and_both_passes_negatives(self):
            r,p=entropy_fixture(a)
            for key,value in [('weighted_delta',.1),('total_L',3.),('new_CE0',-1.),('epsilon_multiset_preserved',False),
                ('epsilon_same_both_passes',False),('As_CE_passes',1),('epsilon_mean',.5)]:
                bad=copy.deepcopy(r);bad['entropy'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):a.entropy_record(bad,p,'Q',1,4)
        def test_exact_assignment_distribution_excludes_only_global_mean(self):
            r,p=entropy_fixture(a,'R');x=np.asarray(p['entropy']['epsilon_original']);y=np.asarray(p['entropy']['epsilon_assigned'])
            self.assertTrue(np.array_equal(np.sort(x),np.sort(y)));self.assertEqual(x.sum(),y.sum())
            # Correlated per-packet target changes need not cancel merely because
            # the mean epsilon is identical; no parameter-gradient equality claim.
            hard=np.asarray([.2,.7,1.,1.4]);uniform=np.asarray([2.,1.7,1.5,1.8])
            qloss=((1-x)*hard+x*uniform).mean();rloss=((1-y)*hard+y*uniform).mean()
            self.assertAlmostEqual(qloss-rloss,((x-y)*(uniform-hard)).mean())
        def test_epsilon_coverage_counts_original_assigned(self):
            r,p=entropy_fixture(a,'Q');value=a.entropy_record(r,p,'Q',1,4);rows=a.epsilon_strata([value])
            overall=next(x for x in rows if x['dimension']=='overall');self.assertEqual(overall['count'],4)
            self.assertAlmostEqual(overall['epsilon_mean'],overall['assigned_epsilon_mean'])
            bins=[x for x in rows if x['dimension']=='signal_to_added_noise_bin'];self.assertEqual(sum(x['count'] for x in bins),4)
        def test_cross_row_original_epsilon_and_native_physics_exact(self):
            _,q=entropy_fixture(a,'Q');_,r=entropy_fixture(a,'R')
            result=a.matched_assignments([q],[r],1);self.assertTrue(result['exact_original_epsilon'])
            bad=copy.deepcopy(r);bad['entropy']['epsilon_original'][0]+=.001
            with self.assertRaises(ValueError):a.matched_assignments([q],[bad],1)
            bad=copy.deepcopy(r);bad['view0'][0]['sample_id'][-1]+=1
            with self.assertRaises(ValueError):a.matched_assignments([q],[bad],1)
        def test_two_source_checks_absolute_percentage_points(self):
            values={key:dict(fullV_clean=.9,midlow=.7) for key in a.ARMS};values['Q']=dict(fullV_clean=.899,midlow=.705)
            self.assertTrue(a.source_criteria(values)['source_qualified'])
            bad=copy.deepcopy(values);bad['Q']['fullV_clean']=.89899;self.assertFalse(a.source_criteria(bad)['source_qualified'])
            bad=copy.deepcopy(values);bad['R']['midlow']=.7001;self.assertFalse(a.source_criteria(bad)['source_qualified'])
        def test_stored_source_verdict_not_recomputed(self):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'runs'/a.RUN/'source_matrix_decision.json';path.parent.mkdir(parents=True)
                value=dict(status='SOURCE_READOUT_DECIDED',checks=dict(clean=True,midlow=False),source_qualified=False,
                    formal_scores_read=False,target_feedback=False,NN=0,model_seed_count=1)
                path.write_text(json.dumps(value),encoding='utf8');self.assertEqual(a.read_source_decision(temp),value)
                for change in (dict(formal_scores_read=True),dict(source_qualified=True),dict(checks=dict(clean=1))):
                    bad=dict(value,**change);path.write_text(json.dumps(bad),encoding='utf8')
                    with self.subTest(change=change),self.assertRaises(ValueError):a.read_source_decision(temp)
        def test_total_L_adapter_subtracts_then_restores_actual_smoothed_value(self):
            folder=Path('TOY');records=[dict(step=i) for i in range(1,2221)]
            verified=[dict(step=i,base_total_L=2.,total_L=2.05,loss_delta=.05,old_As_CE=.8,new_As_CE=1.,
                epsilon_mean=.1,assigned_epsilon_mean=.1,packet_rows=[]) for i in range(1,2221)]
            def lines(path):
                if Path(path).name in ('prop_contrast_steps.jsonl.gz','prop_contrast_physics.jsonl.gz'):return iter(records)
                if Path(path).name=='reference_steps.jsonl.gz':return iter([dict(step=1,total_L=2.05)])
                return iter([])
            def parse(path,is_as):
                self.assertAlmostEqual(next(c.lines(folder/'reference_steps.jsonl.gz'))['total_L'],2.)
                return dict(curve=[dict(local_epoch=e,total_L=2.) for e in range(1,11)]),[]
            with patch.object(c,'lines',lines),patch.object(a.h,'parse_tail',parse),patch.object(a,'entropy_record',side_effect=verified),patch.object(a,'epsilon_strata',return_value=[dict(dimension='overall',count=142080)]):
                value,_=a.parse_tail(folder,True,'Q');self.assertIs(c.lines,lines)
            self.assertAlmostEqual(value['curve'][0]['base_total_L'],2.);self.assertAlmostEqual(value['curve'][0]['total_L'],2.05)
        def test_formal_five_row_enrichment_primary_and_whole_recipe(self):
            rows={arm:{view:dict(accuracy=.8,macro_f1=.8,worst_RX=.7,mean_margin_over30=.2,satellite_retention=.9) for view in c.VIEWS} for arm in a.ARMS}
            for view in c.VIEWS:rows['Q'][view]['accuracy']+=.01
            obj=dict(rows=rows,stage={arm:{} for arm in a.NEW_ARMS},all4_rows_all7_verified_before_truth=True)
            a.enrich_formal(obj);self.assertTrue(obj['all5_rows_all7_verified_before_truth'])
            self.assertEqual(obj['primary']['control'],'R');self.assertAlmostEqual(obj['primary']['effects']['clean']['accuracy_delta_pp'],1.)
            self.assertFalse(obj['stage']['Q']['same_parent_NoAs_control_available']);self.assertFalse(obj['reproducible_dual_gain'])
        def test_actual_parent_cohort_is_distinct_and_counts_only_fixed_set(self):
            y=np.arange(4);parent=np.eye(6)[y]*30;current=parent.copy();cohort=np.asarray([True,True,False,False])
            parent[1]=np.eye(6)[5]*30;current[0]=np.eye(6)[5]*30
            value=a.cohort_metric(y,parent,current,cohort)
            self.assertEqual(value['parent_clean_correct_count'],2);self.assertEqual(value['repairs_within_parent_cohort'],1)
            self.assertEqual(value['new_errors_within_parent_cohort'],1);self.assertEqual(value['retention_delta_pp'],0.)
            empty=a.cohort_metric(y,parent,current,np.zeros(4,bool));self.assertIsNone(empty['current_retention']);self.assertTrue(empty['empty_cohort'])
        def test_actual_parent_cohort_provenance_rejects_wrong_parent_or_EMA(self):
            project=Path('PROJECT');panel=project/'runs'/c.PANEL_RUN/'diagnostic'
            meta=dict(status='SOURCE_READOUT_COMPLETE',run_id=a.PARENT_COHORT_RUN,model_seed=a.SEED,
                checkpoint=str(project/'runs'/a.PARENT_RUN/'AS_CE-s2026092701/stage20/final_ssdg.pth'),
                checkpoint_run=a.PARENT_RUN,checkpoint_row='AS_CE-s2026092701',checkpoint_stage='stage20',checkpoint_key='model',
                checkpoint_epoch=20,checkpoint_optimizer_steps=4440,checkpoint_selection='final_only',
                source_contract=str(project/'runs'/c.CONTRACT_RUN/'source_contract.json'),source_role='V',source_unique_V=1800,
                source_role_counts=dict(L_s=6300,U_s=56700,V=27000),source_panel=str(panel),channel_draws=2,
                scenes=list(c.SCENES),condition='full',condition_index=0,source_NN=12600,smoke_NN=2,NN=12602,
                training_updates=0,head_fit_updates=0,teacher_forwards=0,U_truth_read=False,U_records_used=0,
                target_access=False,formal_scores_read=False,query_access=False,model_state_unchanged=True,
                all_parameters_frozen=True,default_head_shape=[6,160],default_head_scale=30.)
            a.validate_parent_cohort_metadata(meta,project)
            for key,value in [('checkpoint_key','ema_model'),('checkpoint_row','B01-s2026092701'),('default_head_scale',15.),('U_truth_read',True),('source_role','L_s')]:
                with self.subTest(key=key),self.assertRaises(ValueError):a.validate_parent_cohort_metadata(dict(meta,**{key:value}),project)
        def test_all_five_all_seven_views_before_truth_and_late_failure(self):
            with tempfile.TemporaryDirectory() as temp:
                specs,ids,truth,checker=formal_fixture(a,temp);read=c.read;seen=[]
                def spy(path):seen.append('truth' if Path(path)==truth else str(path));return read(path)
                with patch.object(c,'read',side_effect=spy):data,meta=c.truth_last_arrays(specs,ids,truth,temp,6,checker)
                self.assertEqual(len(data),5);self.assertEqual(seen.count('truth'),1);self.assertEqual(seen[-1],'truth')
                (specs[-1]['root']/'prediction/practical_low_urban_scores.npz').unlink();seen.clear()
                with patch.object(c,'read',side_effect=spy),self.assertRaises(FileNotFoundError):c.truth_last_arrays(specs,ids,truth,temp,6,checker)
                self.assertNotIn('truth',seen)
        def test_cost_new_history_and_curve_counts_do_not_double_count_parent(self):
            self.assertEqual(2*(3162962+12600+1176002)+1540,8704668)
            self.assertEqual(3*20+3*10+2*10,110);self.assertEqual(110*222,24420)
            self.assertEqual(5*30,150);self.assertNotEqual(150,110)
        def test_cost_function_two_actual_tails_three_parents_and_reused_readouts(self):
            specs=a.row_specs('PROJECT')
            def native(path):
                n=100 if Path(path).name=='stage20' else 3162962
                return dict(native_NN=n,training_total_NN=n,extra_student_U_NN=0,extra_frozen_teacher_U_NN=0,wall_seconds=1.)
            def read(path):
                name=Path(path).name
                if name=='real_checkpoint_smoke.json':return dict(status='PASS',NN=1540)
                if Path(path).parent.name=='source_qualification':return dict(NN=12600,wall_seconds=1.)
                if Path(path).parent.name=='prediction':return dict(actual_NN=1176002,wall_seconds=2.)
                raise AssertionError('Unexpected source '+str(path))
            with patch.object(c,'native_cost',side_effect=native),patch.object(c,'read',side_effect=read):cost=a.costs('PROJECT',specs)
            self.assertEqual(cost['new_total_remote_NN'],8704668);self.assertEqual(len(cost['unique_reused_parents']),3)
            self.assertEqual(cost['new_training_wall_seconds_sum'],2.);self.assertEqual(cost['new_formal_wall_seconds_sum'],4.)
            self.assertEqual(cost['reference_reprediction'],0);self.assertEqual(cost['reference_rescoring'],0)
    suite=unittest.TestSuite(Checks(name) for name in names) if names else unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    started=__import__('time').perf_counter();result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    return dict(status='PASS',tests=result.testsRun,wall_seconds=__import__('time').perf_counter()-started,
        NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,target_access=False)


if __name__=='__main__':print(json.dumps(checks()))
