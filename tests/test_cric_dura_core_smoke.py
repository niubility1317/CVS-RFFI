import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from tests.test_cric_dura_core_groups import config
from experiments.cric_dura_comparison.core_groups import REPAIR,SEEDS
from experiments.cric_dura_comparison.core_extension import smoke


class ActualGroupedSmokeTest(unittest.TestCase):
    def test_actual_native_call_captures_ordinary_input_for_all_missing_cells(self):
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        source=Path(__file__).resolve().parents[1]/'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
        if not all((source/k).is_dir() for k in ('F','AF','BASE','FOLD','G')):
            self.skipTest('Fixed native source releases not present')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=config('011');c['source_contract']=str(root/'contract.json')
            (root/'contract.json').write_text('{}',encoding='utf-8')
            (root/'config.json').write_text(json.dumps(c),encoding='utf-8')
            (root/'jobs.json').write_text(json.dumps([dict(id=REPAIR+'/F16-s'+str(SEEDS[0]),config=str(root/'config.json'))]),encoding='utf-8')
            rt=dict(project=str(root),source_jobs=str(root/'jobs.json'),code_commit='test',
                frozen_sources={k:str(source/k) for k in ('F','AF','BASE','FOLD','G')})
            activate(rt);numerical_policy();torch.manual_seed(7)
            sample=SimpleNamespace(x=torch.randn(128,2,256),metadata=dict(
                y=np.arange(128)%6,domain=np.arange(128)%15,
                rx_i=np.asarray([1,3,4,6,8])[np.arange(128)%5],day_i=np.ones(128,dtype=np.int64),
                eq_i=np.ones(128,dtype=np.int64),sig_i=np.arange(128),base_index=np.arange(128)))
            with patch('experiments.cric_dura_comparison.f_binding_recovery.smoke') as checkpoint,\
                    patch('experiments.cric_dura_comparison.f_data.SourceDataset',return_value=sample):
                smoke(rt,root/'smoke.json')
            checkpoint.assert_called_once()
            result=json.loads((root/'smoke.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['target_reads'],0)
            self.assertEqual(len(result['missing_cells']),4)
            self.assertTrue(all(r['L_forward_packets']==768 and r['optimizer_updates']==0 for r in result['missing_cells']))


if __name__=='__main__':unittest.main()
