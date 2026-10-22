import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.cric_dura_comparison.physics_device_recovery import (
    ORIGINAL, ORIGINAL_TIMING, RUN, TIMING, ERROR, prepare, recovery_needed)


class DeviceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.rt=dict(project=str(self.root/'project'),control_root=str(self.root/'controller'),
                     release_root=str(self.root/'release'),code_commit='newcommit',frozen_sources={})
        self.jobs=[];self.state=dict(jobs={},excluded_job_ids=[],failures={ERROR:1})
        self.bad=ORIGINAL+'/WRONG_REFERENCE-s2026092701'
        for seed in (2026092701,2026092702,2026092703):
            row='WRONG_REFERENCE-s'+str(seed);base=self.root/'old'/row
            source=dict(run_id=ORIGINAL,row_id=row,model_seed=seed,supervised_architecture='WRONG_REFERENCE',
                        source_contract='same-contract',optimizer_updates=11100,output_root=str(base/'source'))
            f=self.add(ORIGINAL+'/'+row,'F',source,[], 'FAILED' if seed==2026092701 else 'QUEUED')
            pc=dict(run_id=ORIGINAL,row_id=row,source_run_id=ORIGINAL,source_output=source['output_root'],
                    readouts={'raw':None,'DURA-L':{'checkpoint':'existing-G-L'}},output_root=str(base/'predictions'))
            pj=self.add(f['id']+'/predict','predict',pc,[f['completion']],
                        'BLOCKED_TECHNICAL_DEPENDENCY' if seed==2026092701 else 'QUEUED')
            self.add(f['id']+'/score','score',dict(run_id=ORIGINAL,row_id=row,
                     prediction_output=pc['output_root'],output_root=str(base/'scores')), [pj['completion']],
                     'BLOCKED_TECHNICAL_DEPENDENCY' if seed==2026092701 else 'QUEUED')
        tc=dict(run_id=ORIGINAL_TIMING,output_root=str(self.root/'old-timing'),
                F_mechanisms=[dict(name='WRONG_REFERENCE',source_output=str(self.root/'old/WRONG_REFERENCE-s2026092701/source'))],
                batch_sizes=[1,8,16,32,64],rounds=5)
        self.add(ORIGINAL_TIMING,'timing',tc,[self.jobs[0]['completion']],'BLOCKED_TECHNICAL_DEPENDENCY')
        self.healthy=self.add('healthy/F','F',dict(output_root=str(self.root/'healthy'),method='fixed'),[], 'RUNNING')
        self.active={self.healthy['id']:dict(pid=4242,gpu=0)}

    def tearDown(self):
        self.tmp.cleanup()

    def add(self,key,kind,config,deps,status):
        path=self.root/'configs'/(key.replace('/','__')+'.json');path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(config),encoding='utf-8')
        j=dict(id=key,kind=kind,config=str(path),dependencies=deps,
               completion=config['output_root']+'/complete.json',log_name=key.replace('/','__'),
               worker_release='old-release',execution_commit='oldcommit',successful_statuses=['ORIGINAL_STATUS'],
               exclusive=kind=='timing')
        self.jobs.append(j);self.state['jobs'][key]=dict(status=status)
        if status=='FAILED':self.state['jobs'][key]['failure']=ERROR
        return j

    def test_only_failed_row_and_unmeasured_timing_replaced(self):
        before=copy.deepcopy(self.jobs);old_states=copy.deepcopy(self.state['jobs'])
        result=prepare(self.rt,self.jobs,self.state,self.active)
        self.assertEqual(len(result)-len(before),4)
        self.assertEqual(self.state['jobs'][self.bad]['status'],'FAILED')
        self.assertEqual(set(self.state['excluded_job_ids']),
                         {self.bad,self.bad+'/predict',self.bad+'/score',ORIGINAL_TIMING})
        self.assertEqual(self.state['jobs'][self.healthy['id']],old_states[self.healthy['id']])
        self.assertEqual(result[-5],before[-1])
        self.assertNotIn(ERROR,self.state['failures'])
        record=self.state['physics_device_recovery']
        self.assertEqual((record['failed_F_replacements'],record['new_G_training'],record['new_hypotheses']),(1,0,0))
        # Predict configs intentionally do not carry supervised_architecture;
        # these consumers still need the corrected builder when loading weights.
        expected={ORIGINAL+'/WRONG_REFERENCE-s'+str(seed)+suffix
                  for seed in (2026092702,2026092703) for suffix in ('','/predict')}
        self.assertEqual(set(record['migrated_unstarted_execution_only']),expected)
        for old,new in zip(before,result):
            if old['id'] in expected:
                self.assertEqual({k:v for k,v in old.items() if k not in ('worker_release','execution_commit')},
                                 {k:v for k,v in new.items() if k not in ('worker_release','execution_commit')})
            else:self.assertEqual(old,new)
        f,p,s,t=result[-4:]
        fc=json.loads(Path(f['config']).read_text());pc=json.loads(Path(p['config']).read_text())
        self.assertEqual((fc['model_seed'],fc['optimizer_updates'],fc['source_contract']),
                         (2026092701,11100,'same-contract'))
        self.assertEqual(pc['source_output'],fc['output_root'])
        self.assertEqual(p['dependencies'],[f['completion']])
        self.assertEqual(s['dependencies'],[p['completion']])
        self.assertEqual(t['dependencies'],[f['completion']])
        self.assertEqual(t['id'],TIMING);self.assertTrue(t['exclusive'])
        self.assertEqual(json.loads(Path(t['config']).read_text())['F_mechanisms'][0]['source_output'],fc['output_root'])

    def test_rejects_other_failures_and_started_timing(self):
        state=copy.deepcopy(self.state);state['jobs'][self.bad]['failure']='other CUDA problem'
        with self.assertRaises(ValueError):recovery_needed(self.jobs,state)
        state=copy.deepcopy(self.state);state['jobs'][ORIGINAL_TIMING]['status']='COMPLETE'
        with self.assertRaises(ValueError):recovery_needed(self.jobs,state)
        with self.assertRaises(ValueError):recovery_needed(self.jobs,self.state,{self.bad:dict(pid=1)})

    def test_existing_recovery_output_refused_without_queue_mutation(self):
        path=Path(self.rt['project'])/'runs'/RUN/'WRONG_REFERENCE-s2026092701';path.mkdir(parents=True)
        before=copy.deepcopy((self.jobs,self.state))
        with self.assertRaises(FileExistsError):prepare(self.rt,self.jobs,self.state,self.active)
        self.assertEqual((self.jobs,self.state),before)


if __name__=='__main__':unittest.main()
