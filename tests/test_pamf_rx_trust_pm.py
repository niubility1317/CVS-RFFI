from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from experiments.pamf_u_history.rx_trust_pm_source import RUN,PREDECESSOR,SEEDS,RECIPE,P_RECIPE,mode_coefficient,config,validate_train,prepare,calibrate


def steps():
    return [dict(step=k+1,base='raw' if k%2==0 else 'H',gradient=dict(norm=(k+1)*(.05 if k%2==0 else .25),base_norm=k+1)) for k in range(12)]


def context():
    from test_pamf_rx_mean import RXMeanTests
    c=RXMeanTests().context();c['predecessor_complete']=c['project']+'/runs/'+PREDECESSOR+'/complete.json';return c


class ModeCalibrationTest(unittest.TestCase):
    def test_each_mode_median_is_bounded_and_scale_invariant(self):
        alpha,m=mode_coefficient(steps());self.assertAlmostEqual(alpha,.4)
        self.assertEqual(m,dict(raw=.05,H=.25))
        self.assertTrue(all(alpha*v<=.1+1e-12 for v in m.values()))
        scaled=steps()
        for r in scaled:r['gradient']['norm']*=100;r['gradient']['base_norm']*=100
        self.assertAlmostEqual(mode_coefficient(scaled)[0],alpha)

    def test_missing_or_invalid_source_batches_are_not_dropped(self):
        with self.assertRaises(ValueError):mode_coefficient(steps()[:-1])
        for field,value in [('norm',float('nan')),('base_norm',0),('norm',-1)]:
            x=steps();x[0]['gradient'][field]=value
            with self.assertRaises(ValueError):mode_coefficient(x)

    def test_only_calibration_formula_and_source_config_identity_change(self):
        differences={k for k in RECIPE if RECIPE[k]!=P_RECIPE[k]};self.assertEqual(differences,{'coefficient_rule'})
        c=config(context(),'commit',SEEDS[0],.2);validate_train(c)
        self.assertEqual(c['plain_PL_threshold'],.95);self.assertEqual(c['sig_time_views'],1)
        self.assertNotEqual(c['output_root'],c['project']+'/runs/'+PREDECESSOR+'/F/F0-RXTrust-P-s'+str(SEEDS[0])+'/source')
        for key,value in [('source_G','G'),('plain_PL_threshold',.9),('target_truth','target')]:
            with self.assertRaises(ValueError):validate_train(dict(c,**{key:value}))

    def test_queue_and_truth_isolation_for_all_three_seeds(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);c=p/'context.json';c.write_text(json.dumps(context()));e=p/'evaluation.json'
            e.write_text(json.dumps(dict(classes=list(range(6)),tests={t:dict(views_root='/'+t,truth='/truth') for t in ('cross_rx','cross_both')})))
            jobs=prepare(dict(run_id=RUN,context=str(c),evaluation_context=str(e),run_root=str(p/'run'),release_root='/release',python='/python',code_commit='commit'))
            self.assertEqual(len(jobs),12)
            for job in jobs:
                self.assertEqual(job['entrypoint'],'experiments.pamf_u_history.rx_trust_pm_source')
                if job['kind']=='train':continue
                value=json.loads(Path(job['config']).read_text())
                if job['kind']=='calibrate':self.assertEqual(job['dependencies'],[context()['predecessor_complete']]);self.assertNotIn('evaluation_context',value)
                if job['kind']=='predict':self.assertNotIn('truth',json.dumps(value));self.assertNotIn('mean_bank',value)
                if job['kind']=='score':self.assertNotIn('source_output',value)

    def fixture(self,p):
        c=context();old=c['project'];project=p.as_posix()
        def replace(v):
            if isinstance(v,str):return v.replace(old,project)
            if isinstance(v,list):return [replace(x) for x in v]
            if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
            return v
        c=replace(c);contract=p/'contract.json';contract.write_text('{}')
        c['source_contract']=contract.as_posix()
        for row in c['cases']:row['baseline_config']['source_contract']=row['baseline_config']['expected_source_contract']=contract.as_posix()
        own=config(c,'commit',SEEDS[0],1.)
        from experiments.pamf_u_history import rx_trust_p_source as parent
        pc=deepcopy(c);pc['predecessor_complete']=pc['project']+'/runs/'+parent.PREDECESSOR+'/complete.json';refcfg=parent.config(pc,'P-commit',SEEDS[0],.3)
        ref=p/'runs'/PREDECESSOR;cal=ref/'calibration'/('s'+str(SEEDS[0]));cal.mkdir(parents=True);(ref/'configs').mkdir()
        (ref/'configs'/('train__'+str(SEEDS[0])+'.json')).write_text(json.dumps(refcfg))
        reference=dict(status='RX_TRUST_P_GRADIENT_CALIBRATED',formula=P_RECIPE['coefficient_rule'],model_seed=SEEDS[0],optimizer_updates=0,V_reads=0,target_reads=0,parameters_unchanged=True,
            L_seen=1536,U_seen=3072,raw_U=1536,H_U=1536,steps=steps())
        (cal/'calibration.json').write_text(json.dumps(reference));(cal/'smoke.json').write_text(json.dumps(dict(status='PASS',actual_parent=own['shared_SIG1_E20'],actual_model_parameters=863730,optimizer_updates=0,U_truth_read=False,target_reads=0,L_scope_isolated=True)))
        ctx=p/'context.json';ctx.write_text(json.dumps(c));return dict(context=str(ctx),model_seed=SEEDS[0],output_root=str(p/'newcal'),run_root=str(p/'newrun'),code_commit='commit'),cal

    def test_reusing_source_statistics_reads_no_weights_and_runs_no_F(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);rt,_=self.fixture(p)
            with mock.patch('torch.load',side_effect=AssertionError('No weight read')),mock.patch('torch.nn.Module.__call__',side_effect=AssertionError('No new F forward')):calibrate(rt)
            c=json.loads((p/'newcal/calibration.json').read_text());self.assertAlmostEqual(c['coefficient'],.4)
            self.assertEqual(c['new_F_forwards'],0);self.assertFalse(c['model_state_loaded']);self.assertFalse((p/'newcal/smoke.json').exists())
            resolved=json.loads((p/'newrun/configs'/('train__'+str(SEEDS[0])+'.json')).read_text());self.assertEqual(resolved['run_id'],RUN);self.assertEqual(resolved['plain_PL_weight'],1.)

    def test_target_or_different_objective_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);rt,cal=self.fixture(p);ref=json.loads((cal/'calibration.json').read_text());ref['target_reads']=1;(cal/'calibration.json').write_text(json.dumps(ref))
            with self.assertRaises(ValueError):calibrate(rt)
            self.assertFalse((p/'newrun/configs'/('train__'+str(SEEDS[0])+'.json')).exists())


if __name__=='__main__':unittest.main()
