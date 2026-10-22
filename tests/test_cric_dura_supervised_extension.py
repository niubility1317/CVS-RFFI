import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.supervised_extension import prepare,repair_pending_concat,RUN,READOUT,SEEDS


class SupervisedExtensionTests(unittest.TestCase):
    def test_concat_dependency_only_moves_unstarted_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=[];state={'jobs':{}};snapshots={}
            for i in range(9):
                key='binding/'+str(i);path=root/(str(i)+'.json')
                c=dict(satellite_concat=i<6,ablation=None if i<6 else 'F15',output_root=str(root/'outputs'/str(i)))
                write(path,c);snapshots[str(path)]=c
                for suffix,kind in (('','F'),('/predict','predict')):
                    cfg=path
                    if kind=='predict':
                        cfg=root/(str(i)+'-predict.json');pc=dict(c,output_root=str(root/'predictions'/str(i)))
                        write(cfg,pc);snapshots[str(cfg)]=pc
                    jobs.append(dict(id=key+suffix,kind=kind,config=str(cfg),worker_release='old'))
                    state['jobs'][key+suffix]={'status':'QUEUED'}
            rt=dict(previous_controller_cwd='old',release_root='new',code_commit='new-code')
            original=copy.deepcopy(jobs)
            promoted=repair_pending_concat(rt,jobs,state,{'unrelated-healthy':{'pid':1}})
            self.assertEqual(len(promoted),18)
            self.assertTrue(all(j['worker_release']=='new' for j in jobs))
            self.assertEqual({p:read(p) for p in snapshots},snapshots)
            state['jobs']['binding/0']['status']='RUNNING'
            with self.assertRaises(ValueError):repair_pending_concat(rt,original,state,{'binding/0':{'pid':2}})
            state['jobs']['binding/0']['status']='QUEUED'
            (root/'predictions/5').mkdir(parents=True)
            with self.assertRaises(ValueError):repair_pending_concat(rt,original,state,{})
            self.assertTrue(all(j['worker_release']=='old' for j in original))

    def test_supervised_owner_preserves_existing_accelerated_G_metadata(self):
        from experiments.cric_dura_comparison import multiprocess_control as control
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=root/'old_jobs.json';write(jobs,[])
            for name in ('ManySig','ManyRx'):
                write(root/('observations-'+name+'-submit.json'),{'pid':0})
                write(root/'g_data'/name/'complete.json',{'status':'COMPLETE'})
            previous=dict(pid=1,jobs={},accelerated_G_jobs=['already-running-G'],failures={})
            rt=dict(control_root=str(root/'new'),source_jobs=str(jobs),jobs=str(root/'new/jobs.json'),
                run_root=str(root),worker_commit='unchanged',worker_release='old-release',
                code_commit='new',launch_owner='new',previous_control_root=str(root/'old'),
                runtime_file=str(root/'runtime.json'),allowed_gpus=[],supervised_extension=True)
            from experiments.cric_dura_comparison import supervised_extension as extension
            with patch.object(control,'handover',return_value=(previous,{})),patch.object(control,'external_pending',return_value=False),\
                    patch.object(extension,'smoke'),patch.object(extension,'prepare',side_effect=lambda rt,jobs,prev,active:jobs),\
                    patch('experiments.cric_dura_comparison.bootstrap.activate'),patch('experiments.cric_dura_comparison.bootstrap.numerical_policy'):
                control.run(rt)
            self.assertEqual(read(root/'new/pipeline_state.json')['accelerated_G_jobs'],['already-running-G'])

    def test_missing_only_matrix_and_healthy_old_jobs_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=[];state={'jobs':{},'excluded_job_ids':[]}
            def old(key,kind,c):
                path=root/'old_configs'/(key.replace('/','__')+'.json');write(path,c)
                j=dict(id=key,kind=kind,config=str(path),completion=str(root/'old_outputs'/key/'complete.json'))
                jobs.append(j);state['jobs'][key]={'status':'QUEUED'}
                return j
            common=dict(method_id='pamf_t1_e50',optimizer_updates=11100,num_classes=6,
                source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_contract='contract',
                source_cache='cache',satellite_concat=True,frozen_sources={})
            pids=['F0']+['p'+str(i) for i in range(16)]
            for pid in pids:
                for seed in SEEDS:
                    key='main/'+pid+'-s'+str(seed)
                    old(key,'F',dict(common,run_id='main',row_id=key.split('/')[1],model_seed=seed,
                        protocol_id=pid,regime='matched_sat_supervised',ablation=None,output_root=str(root/'old_outputs'/key/'source')))
            first=jobs[0]
            pred=old(first['id']+'/predict','predict',dict(run_id='main',row_id='F0-s'+str(SEEDS[0]),
                protocol_id='F0',source_contract='contract',source_cache='cache',tests={'main':{'views_root':'views'}},
                readouts={'raw':None,'DURA-L':dict(kind='contract_matched',roles='L',checkpoint=str(root/'G/source/r69/final.pt'))}))
            old(first['id']+'/score','score',dict(classes=[0,1,2,3,4,5],tests={'main':{'truth':'closed'}},source_days=[1,2,3]))
            for arm in ('F15','F20'):
                for seed in SEEDS:
                    key='binding/'+arm+'-s'+str(seed);folder=root/'old_outputs'/key/'source'
                    old(key,'F',dict(common,run_id='binding',row_id=key.split('/')[1],model_seed=seed,
                        protocol_id='F0',regime='native_cric',ablation=arm,output_root=str(folder)))
                    old(key+'/predict','predict',dict(output_root=str(folder.parent/'predictions')))
            old('F-binding-mechanism/timing','timing',dict(run_id='timing',output_root='old',G={},external_models=[],G_mechanisms=[]))
            before=copy.deepcopy(jobs);old_configs={j['config']:read(j['config']) for j in jobs}
            rt=dict(project=str(root),control_root=str(root/'control'),code_commit='new-commit',release_root='new-release',frozen_sources={})
            prepare(rt,jobs,state,{'healthy':{'pid':123}})
            self.assertEqual(jobs[:len(before)],before)
            self.assertEqual({p:read(p) for p in old_configs},old_configs)
            new=jobs[len(before):]
            self.assertEqual(len(new),85)
            self.assertEqual(sum(j['kind']=='F' for j in new),24)
            self.assertEqual(sum(j['kind']=='predict' for j in new),30)
            self.assertEqual(sum(j['kind']=='score' for j in new),30)
            self.assertEqual(len({j['id'] for j in jobs}),len(jobs))
            for j in new:
                self.assertEqual(j['worker_release'],'new-release')
                c=read(j['config'])
                if j['kind']=='predict':
                    self.assertTrue(c['strict_supervised_system'])
                    self.assertEqual(set(c['readouts']),{'raw','DURA-L'})
                    self.assertEqual('raw_predictions_from' in c,j['id'].startswith(READOUT+'/'))
                if j['kind']=='F':
                    self.assertEqual(c['optimizer_updates'],11100)
                    self.assertFalse(c['U_use']);self.assertEqual(c['ablation'],'F15')
            self.assertEqual(state['supervised_extension']['new_G_training'],0)
            self.assertEqual(state['supervised_extension']['healthy_PIDs_preserved'],['healthy'])


if __name__=='__main__':unittest.main()
