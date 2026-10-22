import copy
from pathlib import Path
import tempfile
import unittest
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.core_extension import prepare
from experiments.cric_dura_comparison.core_groups import MAIN,REPAIR,L_READOUT,SEEDS


class CoreExtensionTests(unittest.TestCase):
    def test_append_only_55_jobs_actual_repair_reuse_and_CPU_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs=[];previous={'jobs':{},'excluded_job_ids':[]}
            def old(key,kind,c):
                path=root/'configs'/(key.replace('/','__')+'.json');write(path,c)
                j=dict(id=key,kind=kind,config=str(path),completion=str(root/'old'/key/'complete.json'),
                    worker_release='unchanged-science',log_name=key.replace('/','__'))
                jobs.append(j);previous['jobs'][key]={'status':'QUEUED'};return j
            base=dict(method_id='pamf_t1_e50',protocol_id='F0',num_classes=6,optimizer_updates=11100,
                source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_contract='source-contract',
                source_cache='cache',regime='native_cric',satellite_concat=False)
            pc=dict(protocol_id='F0',source_contract='source-contract',source_cache='cache',
                tests={'main':{'views_root':'fixed-views'}},readouts={'raw':None,
                    'DURA-LU':dict(kind='original_F0_LU',roles='LU',checkpoint='original.pt')})
            sc=dict(protocol_id='F0',classes=list(range(6)),source_days=[1,2,3],tests={'main':{'truth':'scorer-only'}})
            for seed in SEEDS:
                row='F0-native_cric-s'+str(seed);key=MAIN+'/'+row
                old(key+'/predict','predict',dict(pc,run_id=MAIN,row_id=row,model_seed=seed,original_arm='T1',output_root=str(root/'original'/row/'pred')))
                old(key+'/score','score',dict(sc,run_id=MAIN,row_id=row,model_seed=seed,output_root=str(root/'original'/row/'scores')))
                for arm in ('F04','F16','F20'):
                    row=arm+'-s'+str(seed);key=REPAIR+'/'+row
                    old(key,'F',dict(base,run_id=REPAIR,row_id=row,model_seed=seed,ablation=arm,output_root=str(root/'repaired'/row/'source')))
                    old(key+'/predict','predict',dict(pc,run_id=REPAIR,row_id=row,model_seed=seed,output_root=str(root/'repaired'/row/'pred')))
                    old(key+'/score','score',dict(sc,run_id=REPAIR,row_id=row,model_seed=seed,output_root=str(root/'repaired'/row/'scores')))
                row='T1_SUP-s'+str(seed)
                old(L_READOUT+'/'+row+'/score','score',dict(sc,run_id=L_READOUT,row_id=row,model_seed=seed,output_root=str(root/'L-only'/row)))
            row='F0-matched_sat_supervised-s'+str(SEEDS[0])
            old('20261022-phase1-T1-DURA-F0-controls-binding-m6-r01/'+row+'/predict','predict',
                dict(pc,readouts={'raw':None,'DURA-L':dict(kind='contract_matched',roles='L',checkpoint=str(root/'G/source/r69/final.pt'))}))
            original=copy.deepcopy(jobs);before={j['config']:read(j['config']) for j in jobs}
            active={REPAIR+'/F04-s'+str(SEEDS[0]):dict(pid=123,gpu=0)}
            previous['jobs'][next(iter(active))]={'status':'RUNNING',**next(iter(active.values()))}
            statuses=copy.deepcopy(previous['jobs'])
            rt=dict(project=str(root),control_root=str(root/'new-control'),code_commit='new',release_root='new-release',frozen_sources={})
            extended=prepare(rt,jobs,previous,active)
            oldmap={j['id']:j for j in extended if j['id'] in {x['id'] for x in original}}
            self.assertEqual(oldmap,{j['id']:j for j in original})
            self.assertEqual({p:read(p) for p in before},before)
            for k,v in statuses.items():self.assertEqual(previous['jobs'][k],v)
            new=[j for j in extended if j['id'] not in oldmap]
            self.assertEqual(len(new),55);self.assertEqual(sum(j['kind']=='F' for j in new),12)
            self.assertEqual(sum(j['kind']=='predict' for j in new),21)
            analysis=next(j for j in new if read(j['config']).get('analysis_kind')=='core_factorial')
            self.assertEqual(analysis['kind'],'score')
            ac=read(analysis['config']);self.assertEqual(len(ac['cells']),24)
            self.assertFalse(Path(ac['output_root']).exists())
            self.assertEqual(previous['core_extension']['healthy_PIDs_preserved'],list(active))
            for j in new:
                if j['kind']=='predict':
                    c=read(j['config']);self.assertNotIn('truth',c['tests']['main'])
                    if 'raw_predictions_from' in c:self.assertEqual(set(c['readouts']),{'raw','DURA-L'})
                    else:self.assertEqual(set(c['readouts']),{'raw','DURA-L','DURA-LU'})


if __name__=='__main__':unittest.main()
