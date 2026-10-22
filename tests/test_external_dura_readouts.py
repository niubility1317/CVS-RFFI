import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.cric_dura_comparison import external_dura as e
from experiments.cric_dura_comparison.external_dura_extension import recovery_needed,BAD_CORE,ERROR
from experiments.cric_dura_comparison.io import write


def rows():
    cases=[(p,r,f,v) for p in ('F0','F1','F2','chrono','ManyRx')
        for r in (('supervised','pl') if p=='F0' else ('supervised',))
        for f in e.FAMILIES for v in ('ce','full')]
    cases += [(p,'supervised','ibrff',v) for p in ('label01','label05') for v in ('ce','full')]
    cases += [('m%d-%s'%(m,b),r,f,v) for m in (1,2,3,4,6) for b in ('natural','fixed')
        for r in (('supervised','pl') if b=='natural' else ('supervised',)) for f in e.FAMILIES for v in ('ce','full')]
    return [dict(run_id='source',row_id='%s-%s-%s-%s-%s'%(p,r,f,v,s),protocol_id=p,
        training_regime=r,family=f,variant=v,model_seed=s,U_s_use=r=='pl')
        for p,r,f,v in cases for s in e.SEEDS]


class FrozenExternalTests(unittest.TestCase):
    def test_external_namespace_and_existing_vendor_paths(self):
        import sys,types
        experiments=types.SimpleNamespace(__path__=['existing']);baselines=types.SimpleNamespace(__path__=['old-base'])
        original=list(sys.path)
        try:
            with patch.dict(sys.modules,{'experiments':experiments,'baselines':baselines}):
                e.activate_external({'external_releases':['author-release','count-release']})
            self.assertEqual(baselines.__path__[0],str(Path('author-release')/'baselines'))
            self.assertIn(str(Path(e.__file__).resolve().parent/'vendor'),sys.path)
        finally:sys.path[:]=original

    def test_fp32_cache_bytes_identical_without_numpy_bridge(self):
        x=torch.tensor([0.,-0.,1./3,-1e-20,1e20],dtype=torch.float32)
        copied=e.fp32_copy(x)
        self.assertEqual(copied.dtype,np.float32)
        self.assertEqual(copied.tobytes(),bytes(x.untyped_storage()))
        with self.assertRaises(ValueError):e.fp32_copy(x.double())

    def test_complete516_and_missing_or_duplicate_seed_rejected(self):
        self.assertEqual(len(e.validate_rows(rows())),516)
        for damage in ('missing','seed','m5'):
            r=rows()
            if damage=='missing':r.pop()
            elif damage=='seed':r[1]['model_seed']=r[0]['model_seed']
            else:r[-1]['protocol_id']='m5-natural'
            with self.assertRaises(ValueError):e.validate_rows(r)

    def test_source_permission_flag_differs_but_physical_contract_must_match(self):
        c=dict(role_ids={'L_s':['a'],'U_s':['b'],'V':['c']},counts={'L_s':1,'U_s':1,'V':1},source_rxs=[1],source_days=[1])
        g=dict(c,classes=['TX'],U_s_use=True)
        e.compatible_sources(dict(c,U_s_use=False),g,['TX'])
        for key,bad in [('role_ids',{'L_s':['other'],'U_s':['b'],'V':['c']}),('source_rxs',[2])]:
            with self.assertRaisesRegex(ValueError,'CONTRACT_MISMATCH'):e.compatible_sources(dict(c,**{key:bad}),g,['TX'])

    def test_native_prediction_template_gets_source_days_from_its_contract(self):
        item=dict(protocol_id='chrono',model_seed=e.SEEDS[0],suite='native219',run_id='old',row_id='old-row',source_output='source',
            source_contract='old-contract',classes=['a'],family='riei',variant='full',training_regime='supervised',prediction_output='raw')
        with patch.object(e,'read',return_value={'source_days':[0,1,2]}):
            c=e.model_config(item,{'source_contract':'G-contract'},{'external_releases':['old'],'frozen_sources':{},'code_commit':'fixed'})
        self.assertEqual(c['source_days'],[0,1,2])

    def test_raw_reuse_requires_all_views_and_same_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ids=np.array(['a','b']);(root/'main').mkdir()
            write(root/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',row_id='old',truth_read=False,tests={'main':{}}))
            values={v:np.array([0,1]) for v in e.VIEWS};np.savez(root/'main/predictions.npz',ids=ids,**values)
            c=dict(protocol_id='F0',raw_predictions_from=str(root),source_row_id='old',classes=['a','b'])
            self.assertEqual(set(e.load_raw(c,'main',ids)),set(e.VIEWS))
            with self.assertRaises(ValueError):e.load_raw(c,'main',ids[::-1])
            values.pop(e.VIEWS[-1]);np.savez(root/'main/predictions.npz',ids=ids,**values)
            with self.assertRaises(ValueError):e.load_raw(c,'main',ids)

    def test_cache_cannot_change_role_or_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'cache';(cache/'main').mkdir(parents=True);ids=np.array(['a','b'])
            cc=dict(protocol_id='F0',row_id='F0-DURA-L',readout='DURA-L',G={'roles':'L'},
                source_contract='frozen',output_root=str(cache),tests={'main':{'views_root':'fixed'}})
            cfg=root/'config.json';write(cfg,cc)
            write(cache/'complete.json',dict(status='DURA_IQ_CACHE_COMPLETE',run_id=e.CACHE,row_id=cc['row_id'],
                truth_read=False,F_reads=0,tests=['main'],views=list(e.VIEWS)))
            np.savez(cache/'main/index.npz',ids=ids)
            c=dict(protocol_id='F0',source_contract='frozen',cache_configs={'DURA-L':str(cfg)})
            self.assertEqual(e.cache_preflight(c,'DURA-L',ids,'main'),cache/'main')
            cc['G']['roles']='LU';cfg.write_text(__import__('json').dumps(cc))
            with self.assertRaises(ValueError):e.cache_preflight(c,'DURA-L',ids,'main')

    def test_recovery_only_accepts_exact_known_failures(self):
        state={'jobs':{BAD_CORE+'/'+a+'-s2026092701/predict':dict(status='FAILED',failure=ERROR) for a in ('ABC101','ABC111')}}
        self.assertEqual(len(recovery_needed(state)),2)
        state['jobs']['unrelated']={'status':'FAILED','failure':'OOM'}
        with self.assertRaises(ValueError):recovery_needed(state)

    def test_append516_readouts_and_only_failed_source_recovery(self):
        from experiments.cric_dura_comparison.external_dura_extension import prepare,OLD_ANALYSIS
        from experiments.cric_dura_comparison.io import read
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=[];state={'jobs':{},'failures':{ERROR:2},'excluded_job_ids':[]}
            def job(key,kind,c,status):
                p=root/'old-configs'/(key.replace('/','__')+'.json');write(p,c)
                j=dict(id=key,kind=kind,config=str(p),completion=c['output_root']+'/complete.json',
                    dependencies=[],log_name='old',worker_release='old-science')
                old.append(j);state['jobs'][key]=dict(status=status)
                if status=='FAILED':state['jobs'][key]['failure']=ERROR
                return j
            cells=[]
            for bits in ('ABC101','ABC111'):
                row=bits+'-s2026092701';key=BAD_CORE+'/'+row
                pc=dict(run_id=BAD_CORE,row_id=row,source_run_id='source',source_row_id=row,
                    output_root=str(root/'old-output'/row/'predict'),source_physics_diagnostics=True)
                job(key+'/predict','predict',pc,'FAILED')
                sj=job(key+'/score','score',dict(run_id=BAD_CORE,row_id=row,output_root=str(root/'old-output'/row/'scores')),'BLOCKED_TECHNICAL_DEPENDENCY')
                cells.append(dict(row_id=row,run_id=BAD_CORE,scores=str(Path(sj['completion']).parent/'scores.json'),completion=sj['completion']))
            aj=job(OLD_ANALYSIS,'score',dict(output_root=str(root/'old-analysis'),cells=cells),'BLOCKED_TECHNICAL_DEPENDENCY')
            aj['dependencies']=[v['completion'] for v in cells]
            healthy=job('healthy-F','F',dict(output_root='kept'),'RUNNING');active={'healthy-F':dict(pid=77,gpu=1)}
            old_before=copy.deepcopy(old);status_before=copy.deepcopy(state['jobs']['healthy-F'])
            contract=dict(role_ids={'L_s':['x'],'U_s':['u'],'V':['v']},counts={},classes=['TX'],source_rxs=[1],source_days=[1])
            cp=root/'contract.json';write(cp,contract)
            rs=rows()
            for r in rs:r.update(suite='native219',source_output='no-training',source_contract=str(cp),classes=['TX'],
                source_cache='cache',prediction_output='old-raw',tests={'main':{'views_root':'fixed','truth':'CPU-only'}})
            ts={p:dict(source_contract=str(cp),source_cache='cache',source_days=[1],tests={'main':{'views_root':'fixed'}},
                readouts={'DURA-L':dict(kind='contract_matched',roles='L',checkpoint='old/G/source/r69/final.pt'),
                    'DURA-LU':dict(kind='contract_matched',roles='LU',checkpoint='old/GU/source/r69/final.pt')}) for p in {r['protocol_id'] for r in rs}}
            rt=dict(project=str(root),control_root=str(root/'control'),release_root='new-execution',code_commit='commit',
                frozen_sources={},external_releases=['immutable'],external_manifest={'rows':rs,'G_templates':ts})
            extended=prepare(rt,old,state,active)
            self.assertEqual(len(extended)-len(old),1072);self.assertEqual(old,old_before)
            self.assertEqual(state['jobs']['healthy-F'],status_before);self.assertEqual(healthy['worker_release'],'old-science')
            new=extended[len(old):];self.assertEqual(sum(j['kind']=='G' for j in new),34)
            self.assertEqual(sum(j['kind']=='F' for j in new),0)
            self.assertNotIn(ERROR,state['failures'])
            for j in new:
                c=read(j['config'])
                if c.get('external_action'):
                    e.reject_truth(c)
                    if c['external_action']=='predict':self.assertEqual(len(j['dependencies']),3)
            recovered=next(j for j in new if j['id']==e.CORE_RECOVERY)
            ac=read(recovered['config'])
            self.assertTrue(all(x['run_id']==e.RECOVERY for x in ac['cells']))
            self.assertTrue(all(e.RECOVERY in x['scores'] for x in ac['cells']))

    def test_source_physics_avoids_numpy_index_and_preserves_selection(self):
        from experiments.cric_dura_comparison.source_physics import run
        class SourceTensor:
            def __init__(self,x):self.x=x
            def __getitem__(self,index):raise AssertionError('NumPy indexing must never reach Torch2.1')
            def index_select(self,dim,index):
                self.observed=index.clone();return self.x.index_select(dim,index)
        x=SourceTensor(torch.arange(12*512,dtype=torch.float32).reshape(12,2,256)/6000)
        class Dataset:pass
        data=Dataset();data.x=x;data.iq_binding={'status':'aligned'}
        data.metadata={k:np.asarray(v) for k,v in dict(y=[0]*6+[1]*6,rx_i=[0,1]*6,
            day_i=[0]*12,ids=[str(i) for i in range(12)]).items()}
        class Reader:
            def features(self,x):return x.flatten(1)
            def classify_features(self,z):return z[:,:2]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'contract.json',{})
            with patch('experiments.cric_dura_comparison.f_data.SourceDataset',return_value=data):
                run(Reader(),dict(source_contract=str(root/'contract.json'),source_cache='unused'),root/'result.json','cpu')
            r=__import__('json').loads((root/'result.json').read_text())
            self.assertEqual(r['target_reads'],0);self.assertEqual(r['source_U_reads'],0)
            self.assertEqual(set(r['physical_ids']),set(map(str,range(12))))
            self.assertEqual(x.observed.dtype,torch.int64);self.assertEqual(len(r['interventions']),11)


if __name__=='__main__':unittest.main()
