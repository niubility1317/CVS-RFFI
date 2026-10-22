import copy,tempfile,unittest
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.f_binding_recovery import prepare,ABLATION,CONTROLS
ROOT=Path(__file__).resolve().parents[1]/'local_artifacts/F_ablation_repair_20261022/tests'

class RecoveryScope(unittest.TestCase):
    def test_mechanism_only_timing_does_not_import_unused_external_baselines(self):
        from unittest.mock import patch
        from experiments.cric_dura_comparison.timing import external_models_api
        with patch('builtins.__import__',side_effect=AssertionError('unused external import')):
            self.assertIsNone(external_models_api(dict(external_models=[])))
    def test_excluded_history_does_not_prevent_effective_completion(self):
        from experiments.cric_dura_comparison.multiprocess_control import completion_status
        state=dict(jobs={'old':dict(status='SKIPPED_INPUT_BINDING_REPLACED'),'new':dict(status='COMPLETE')},excluded_job_ids=['old'])
        self.assertEqual(completion_status(state),'COMPLETE')
        state['jobs']['new']['status']='FAILED'
        self.assertEqual(completion_status(state),'PARTIAL_TECHNICAL_FAILURE')
    def test_exact69_replacements_preserve_healthy_PIDs_contracts_and_G(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);jobs=[];state=dict(jobs={});active={};sources=[]
            def add(key,kind,c,deps=()):
                path=root/(key.replace('/','__')+'.json');write(path,c)
                j=dict(id=key,kind=kind,config=str(path),completion=c['output_root']+'/complete.json',dependencies=list(deps),log_name=key.replace('/','__'),successful_statuses=['SOURCE_FROZEN'])
                jobs.append(j);state['jobs'][key]=dict(status='QUEUED');return j
            for seed in (1,2,3):
                for arm in [*('F%02d'%i for i in range(1,22)),'sup','pl']:
                    key='old/'+arm+'-s'+str(seed);source=str(root/'original'/key/'source');sources.append((arm,source))
                    cfg=dict(stage='F',run_id='old',row_id=arm+'-s'+str(seed),protocol_id='F0',ablation=arm if arm.startswith('F') else None,
                        output_root=source,model_seed=seed,source_contract='contract unchanged',regime='native_cric')
                    f=add(key,'F',cfg)
                    p=add(key+'/predict','predict',dict(stage='predict',run_id='old',row_id=cfg['row_id'],source_output=source,output_root=source+'/pred',tests='unchanged truth-blind',readouts={'G':'same'}),[f['completion'],'G-unchanged'])
                    add(key+'/score','score',dict(stage='score',run_id='old',prediction_output=source+'/pred',output_root=source+'/score'),[p['completion']])
            mechanism=add('latency/mechanisms','timing',dict(stage='timing',output_root='old/timing',F_mechanisms=[dict(name=a,source_output=s) for a,s in sources if a in ('F06','F07','F08','F09','F10')][:5],G_mechanisms='unchanged'),[s+'/complete.json' for a,s in sources if a in ('F06','F07','F08','F09','F10')][:5])
            g=add('healthy/G','G',dict(stage='G',output_root='G-stays',execution_variant='existing cache'))
            state['jobs']['old/F01-s1']=dict(status='RUNNING',pid=123)
            active['old/F01-s1']=dict(pid=123,gpu=0)
            before=copy.deepcopy(g);rt=dict(project=str(root),control_root=str(root/'control'),release_root='new-release',code_commit='fix',frozen_sources={})
            out=prepare(rt,jobs,state,active)
            self.assertEqual(len(out),208+209)
            self.assertEqual(state['jobs']['old/F01-s1'],dict(status='RUNNING',pid=123))
            self.assertEqual(g,before);self.assertEqual(len(state['excluded_job_ids']),208)
            self.assertEqual(sum(j['kind']=='F' and j.get('execution_commit')=='fix' for j in out),69)
            new=next(j for j in out if j['id']==ABLATION+'/F01-s1');cfg=read(new['config'])
            self.assertEqual(cfg['model_seed'],1);self.assertEqual(cfg['source_contract'],'contract unchanged')
            pred=read(next(j for j in out if j['id']==new['id']+'/predict')['config'])
            self.assertEqual(pred['source_output'],cfg['output_root']);self.assertEqual(pred['readouts'],{'G':'same'})
            new_timing=out[-1]
            self.assertTrue(all(ABLATION in dep for dep in new_timing['dependencies']))
            self.assertEqual(state['jobs']['old/sup-s1']['status'],'SKIPPED_INPUT_BINDING_REPLACED')
    def test_repair_runtime_keeps_default_and_existing_resource_limits(self):
        from experiments.cric_dura_comparison.publish_F_binding_repair import runtime_for
        old=dict(run_id='20261022-phase1-DURA-teacher-cache-control-r01',project='/p',jobs='/resolved',control_root='/old',runtime_file='/old-runtime',release_root='/old-release',worker_release='/3e',load_balance={'unchanged':1})
        r=runtime_for(old,'new')
        self.assertEqual(r['worker_release'],'/3e');self.assertEqual(r['load_balance'],old['load_balance'])
        self.assertFalse(r['G_teacher_acceleration']);self.assertTrue(r['F_source_binding_recovery'])
        self.assertEqual(r['source_jobs'],'/resolved')

if __name__=='__main__':unittest.main()
