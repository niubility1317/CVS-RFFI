import copy,csv,importlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.cric_dura_comparison import pamf_gfactor as p
from experiments.cric_dura_comparison.pamf_gfactor_extension import prepare
from experiments.cric_dura_comparison.analyze_gfactor import run as analyze
from experiments.cric_dura_comparison.plan import make_plan


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
def cells(root):
    rows=[]
    for pid in make_plan()['protocols']:
        for seed in p.SEEDS:
            source=p.main_source(pid,'matched_sat_plain_pl',seed)
            tests={t:dict(views_root=str(root/'views'),truth=str(root/'truth.json')) for t in (['main','source_unseen_date'] if pid=='chrono' else ['main'])}
            rows.append(dict(protocol_id=pid,model_seed=seed,row_id=pid+'-PL-G-L-s'+str(seed),
                source=dict(source_run_id=source['run_id'],source_row_id=source['row_id'],source_output=str(root/'source')),
                source_contract=str(root/'contract.json'),source_cache=str(root/'L'),source_days=[1,2,3],classes=[0,1],
                raw_predictions_from=str(root/'raw'),source_completion=str(root/'source/complete.json'),
                raw_completion=str(root/'raw/complete.json'),cache_config=str(root/'cache.json'),cache_completion=str(root/'cache/complete.json'),
                tests=tests,existing_scores={r:dict(run_id='old',row_id=pid+r+str(seed),scores=str(root/(pid+r+str(seed)+'scores.json')),
                    completion=str(root/(pid+r+str(seed)+'done.json'))) for r in ('matched_sat_supervised','matched_sat_plain_pl')}))
    return rows


class GfactorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'local_artifacts');self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_local_publisher_needs_no_torch(self):
        script="import sys; import experiments.cric_dura_comparison.publish_gfactor; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',script],check=True,cwd=Path(__file__).resolve().parents[1],capture_output=True)
    def test_exact_scope_and_binding_replacement(self):
        c=cells(self.root);self.assertEqual(len(p.validate_cells(c)),51)
        bad=copy.deepcopy(c);f=next(v for v in bad if v['protocol_id']=='F0');f['source']['source_run_id']=p.main_source('F1','matched_sat_plain_pl',p.SEEDS[0])['run_id']
        with self.assertRaises(ValueError):p.validate_cells(bad)
        with self.assertRaises(ValueError):p.validate_cells(c[:-1])
    def test_append_preserves_existing_and_truth_split(self):
        c=cells(self.root);old=[dict(id='oldjob',config='oldcfg',kind='F',unchanged=True)];before=copy.deepcopy(old)
        rt=dict(project=str(self.root/'project'),control_root=str(self.root/'control'),pamf_gfactor_manifest=dict(cells=c),frozen_sources={},code_commit='test',release_root='newrelease')
        state=dict(jobs={'oldjob':dict(status='RUNNING',pid=123)});out=prepare(rt,old,state,{'oldjob':{'pid':123}})
        self.assertEqual(old,before);self.assertEqual(out[0],before[0]);self.assertEqual(state['jobs']['oldjob']['pid'],123)
        self.assertEqual(len(out),104);self.assertEqual(sum(j['kind']=='predict' for j in out),51)
        self.assertFalse(any(j['kind'] in ('G','G_student') for j in out[1:]))
        for j in out[1:]:
            cfg=json.loads(Path(j['config']).read_text())
            if j['kind']=='predict':p.reject_truth(cfg);self.assertEqual(cfg['readouts'],['raw','DURA-L']);self.assertEqual(len(j['dependencies']),3)
        self.assertEqual(len(out[-1]['dependencies']),153)
    def raw_fixture(self,c):
        raw=Path(c['raw_predictions_from']);raw.mkdir();(raw/'main').mkdir();ids=np.array([7,8],dtype=np.int64)
        np.savez(raw/'main/predictions.npz',ids=ids,**{'raw__'+v:np.array([0,1]) for v in p.VIEWS})
        write(raw/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,
            run_id=c['source_run_id'],row_id=c['source_row_id'],tests={'main':{}},readouts=['raw','DURA-LU']))
        return ids
    def test_raw_reuse_rejects_wrong_ID_or_source(self):
        c=p.config(next(x for x in cells(self.root) if x['protocol_id']=='F0'));ids=self.raw_fixture(c)
        self.assertEqual(p.raw(c,'main',ids)['clean'].tolist(),[0,1])
        with self.assertRaises(ValueError):p.raw(c,'main',ids[::-1])
        c['source_row_id']='wrong'
        with self.assertRaises(ValueError):p.raw(c,'main',ids)
    def test_predict_and_independent_score(self):
        import torch
        from experiments.cric_dura_comparison.evaluation import score
        cell=next(x for x in cells(self.root) if x['protocol_id']=='F0');c=p.config(cell);c['output_root']=str(self.root/'pred');ids=self.raw_fixture(c)
        cache=self.root/'cache';cache.mkdir()
        for v in p.VIEWS:np.save(cache/(v+'.npy'),np.ones((2,2,256),np.float32))
        class Reader:
            def features(self,x):return x
            def classify_features(self,x):return torch.tensor([[2.,1.],[1.,2.]],device=x.device)[:len(x)]
        with patch.object(p,'load_reader',return_value=Reader()),patch.object(p,'handle',return_value=({},ids,None)),patch.object(p,'cache_preflight',return_value=cache):p.predict(c,'cpu')
        manifest=dict(status='VALIDATED_ONCE',classes=[0,1],scenes=list(p.VIEWS[1:]),source_target_disjoint=True,count=2)
        write(self.root/'views/manifest.json',manifest);np.savez(self.root/'views/index.npz',ids=ids)
        write(self.root/'truth.json',{str(i):dict(label=k,receiver='1-1',day='2021_03_01') for k,i in enumerate(ids)})
        sc=dict(c,prediction_output=c['output_root'],output_root=str(self.root/'scores'),tests=cell['tests'])
        score(sc);self.assertEqual(json.loads((self.root/'scores/complete.json').read_text())['status'],'SCORED_COMPLETE')
        bad=dict(c,output_root=str(self.root/'bad'));bad['tests']['main']['truth']='forbidden'
        with self.assertRaises(ValueError):p.predict(bad,'cpu')
    def synthetic_analysis(self):
        rows=[]
        def cm(n):return [[n//2,(20-n)//2],[(20-n)//2,n//2]]
        for cell in cells(self.root):
            pid=cell['protocol_id'];seed=cell['model_seed'];records={}
            for suffix,readouts in [('SUP',[('DURA-L',10),('DURA-LU',12)]),('OLDPL',[('DURA-LU',18)]),('NEWPL',[('DURA-L',14)])]:
                row=pid+suffix+str(seed);folder=self.root/row;results=[];paired=[]
                for task in cell['tests']:
                    for view in p.VIEWS:
                        rawcm=cm(8)
                        for name,n in [('raw',8)]+readouts:
                            confusion=cm(n);results.append(dict(test=task,readout=name,view=view,dimension='overall',stratum='ALL',count=20,
                                confusion=confusion,accuracy=n/20,macro_f1=n/20))
                            if name!='raw':paired.append(dict(test=task,readout=name,view=view,dimension='overall',stratum='ALL',count=20,repairs=n-8,damage=0,net=n-8))
                write(folder/'scores.json',dict(status='SCORED',run_id='test',row_id=row,model_seed=seed,results=results,paired_raw=paired,truth_last=True,target_feedback_forbidden=True))
                write(folder/'complete.json',dict(status='SCORED_COMPLETE'))
                for name,_ in readouts:records[('Sup' if suffix=='SUP' else 'PL')+'+'+name.replace('DURA','G')]=dict(run_id='test',row_id=row,
                    scores=str(folder/'scores.json'),completion=str(folder/'complete.json'),readout=name)
            write(self.root/'source/complete.json',dict(seconds=100))
            write(self.root/'cache.json',dict(readout='DURA-L',output_root=str(self.root/'cache'),G={'checkpoint':'same-G.pt'}))
            write(self.root/'cache/complete.json',dict(status='DURA_IQ_CACHE_COMPLETE',seconds=1))
            rows.append(dict(protocol_id=pid,model_seed=seed,classes=[0,1],records=records,source_completion=str(self.root/'source/complete.json'),cache_config=str(self.root/'cache.json')))
        return dict(run_id='test-analysis',cells=rows,output_root=str(self.root/'analysis'))
    def test_complete_factorial_interaction_and_macroF1(self):
        c=self.synthetic_analysis();analyze(c)
        with (Path(c['output_root'])/'gfactor_paired_effects.csv').open() as f:r=list(csv.DictReader(f))
        selected=[x for x in r if x['effect']=='F_PL_x_G_U_interaction']
        self.assertTrue(selected);self.assertTrue(all(abs(float(x['mean'])-10)<1e-10 for x in selected))
        self.assertTrue(all(float(x['sd'])==0 for x in selected))
        self.assertEqual(json.loads((Path(c['output_root'])/'complete.json').read_text())['cells'],204)
    def test_analysis_rejects_reused_raw_change_and_partial_matrix(self):
        c=self.synthetic_analysis();r=c['cells'][0]['records']['PL+G-L'];path=Path(r['scores']);s=json.loads(path.read_text());s['results'][0]['confusion']=[[5,5],[5,5]];write(path,s)
        with self.assertRaises(ValueError):analyze(c)
        c['cells']=c['cells'][:-1]
        with self.assertRaises(ValueError):analyze(c)

if __name__=='__main__':unittest.main()
