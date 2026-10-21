import json
from pathlib import Path
import tempfile
import unittest
import sys,importlib
import numpy as np
from experiments.cric_dura_comparison.evaluation import VIEWS,preflight,score
from experiments.cric_dura_comparison.g_cache import CacheView
from experiments.cric_dura_comparison.pipeline import block_failed_dependencies


class ContinuationContracts(unittest.TestCase):
    def test_frozen_G_root_is_available_to_its_tools_imports(self):
        from experiments.cric_dura_comparison.bootstrap import activate
        import experiments
        old_paths=list(sys.path);old_namespace=list(experiments.__path__)
        with tempfile.TemporaryDirectory() as folder:
            roots={k:str(Path(folder)/k) for k in ('F','AF','BASE','FOLD','G')}
            for path in roots.values():Path(path).mkdir()
            (Path(roots['G'])/'_cric_frozen_tool_probe.py').write_text('VALUE=35\n',encoding='utf-8')
            try:
                activate(dict(frozen_sources=roots))
                self.assertEqual(importlib.import_module('_cric_frozen_tool_probe').VALUE,35)
                self.assertLess(sys.path.index(roots['G']),sys.path.index(roots['BASE']))
            finally:
                sys.path[:]=old_paths;experiments.__path__=old_namespace
                sys.modules.pop('_cric_frozen_tool_probe',None)

    def test_cache_failure_blocks_only_dependent_closure(self):
        jobs=[dict(id='G',completion='/G/complete',dependencies=['/cache/complete']),
              dict(id='predict',completion='/P/complete',dependencies=['/G/complete']),
              dict(id='score',completion='/S/complete',dependencies=['/P/complete']),
              dict(id='healthy',completion='/F/complete',dependencies=[])]
        state=dict(jobs={j['id']:dict(status='QUEUED') for j in jobs})
        block_failed_dependencies(jobs,state,['/cache/complete'])
        self.assertEqual(state['jobs']['healthy']['status'],'QUEUED')
        for key in ('G','predict','score'):
            self.assertEqual(state['jobs'][key]['status'],'BLOCKED_TECHNICAL_DEPENDENCY')

    def test_incomplete_prediction_cannot_open_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'complete.json').write_text(json.dumps(dict(status='INCOMPLETE')),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Truth closed'):
                score(dict(prediction_output=folder,run_id='r',row_id='x',tests={'main':{'truth':'DO_NOT_OPEN'}},protocol_id='F0'))

    def test_all_readouts_before_truth_and_paired_correction(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);views=p/'views';views.mkdir();pred=p/'prediction';pred.mkdir();(pred/'main').mkdir()
            classes=['A','B'];ids=np.asarray(['a','b'])
            (views/'manifest.json').write_text(json.dumps(dict(status='VALIDATED_ONCE',classes=classes,scenes=list(VIEWS[1:]),source_target_disjoint=True,count=2)),encoding='utf-8')
            np.savez(views/'index.npz',ids=ids)
            values={name+'__'+v:np.asarray([0,0] if name=='raw' else [0,1],np.int64) for name in ('raw','G') for v in VIEWS}
            np.savez(pred/'main/predictions.npz',ids=ids,**values)
            (pred/'complete.json').write_text(json.dumps(dict(status='ALL_PREDICTIONS_COMPLETE',run_id='r',row_id='x',tests={'main':{}},views=list(VIEWS),readouts=['raw','G'],truth_read=False,query_fit=False)),encoding='utf-8')
            truth=p/'truth.json';truth.write_text(json.dumps({'a':dict(label=0,receiver='R1',day='2021_03_08'),'b':dict(label=1,receiver='R2',day='2021_03_08')}),encoding='utf-8')
            c=dict(prediction_output=str(pred),output_root=str(p/'score'),run_id='r',row_id='x',protocol_id='F0',model_seed=1,classes=classes,readouts=['raw','G'],source_days=[1],tests={'main':dict(views_root=str(views),truth=str(truth))})
            self.assertEqual(set(preflight(c)),{'main'})
            score(c)
            result=json.loads((p/'score/scores.json').read_text())
            gains=[r for r in result['paired_raw'] if r['dimension']=='overall']
            self.assertEqual(len(gains),7);self.assertTrue(all(r['repairs']==1 and r['damage']==0 for r in gains))
            del values['G__practical_low_urban']
            np.savez(pred/'main/predictions.npz',ids=ids,**values)
            with self.assertRaisesRegex(ValueError,'Truth closed'):
                preflight(c)

    def test_G_L_selector_cannot_read_U_position(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);old=p/'old/train';old.mkdir(parents=True);new=p/'new';new.mkdir()
            np.save(old/'clean.npy',np.zeros((2,2,256),np.float32));np.save(old/'physical_ids.npy',np.asarray(['L','U']))
            np.save(new/'clean.npy',np.zeros((1,2,256),np.float32));np.save(new/'physical_ids.npy',np.asarray(['OTHER']))
            contract=p/'contract.json';contract.write_text(json.dumps(dict(role_ids={'L_s':['L'],'U_s':['U']})),encoding='utf-8')
            selector=p/'selector.npz'
            np.savez(selector,ids=np.asarray(['L']),kind=np.asarray([0]),position=np.asarray([0]),is_L=np.asarray([True]))
            c=dict(selector=str(selector),source_contract=str(contract),roles='L',old_action=str(old.parent),old_coarse='unused',new_cache=str(new))
            self.assertEqual(len(CacheView(c)),1)
            np.savez(selector,ids=np.asarray(['L']),kind=np.asarray([0]),position=np.asarray([1]),is_L=np.asarray([True]))
            with self.assertRaisesRegex(ValueError,'physical IQ'):
                CacheView(c)


if __name__=='__main__':unittest.main()
