import sys
import unittest
import json
import tempfile
from unittest.mock import patch
import numpy as np
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'code'), str(Path(__file__).resolve().parents[1]/'tools')]
from build_practical_phase2_data import assignments
from build_practical_phase2_data import main


class BuilderTests(unittest.TestCase):
    def test_capsule_has_no_truth_and_nested_support_uses_fixed_query(self):
        raw = dict(tx_list=['a','b'],rx_list=['target'],capture_date_list=['d0','d1','d2','d3'],equalized_list=[1],
            data=[[[[np.ones((50,256,2),dtype='float32')] for day in range(4)]] for tx in range(2)])
        def fake_channel(x,cfg,**kw):
            self.assertEqual(cfg.processing_route,'residual')
            self.assertFalse(cfg.equalization_enabled)
            return x,[{} for _ in x],None
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=dict(output_root=str(root/'data'),manytx_path='builder-only-raw.pkl',
                old_classes=['a'],new_classes=['b'],source_receivers=['source'],target_receivers=['target'],
                fs_hz=25000000,data_seed=100,augmentation_seed=101,receiver_seed=2027,evaluation_seed=102,
                scenarios=['practical_high','practical_mid','practical_low_urban'],
                shots=[1,5,20],support_seeds=[103],new_counts=[0,1])
            path=root/'config.json'
            path.write_text(json.dumps(config))
            with patch('dataset_wisig.load_wisig_compact_pkl',return_value=raw), \
                 patch('leo_practical.batch.apply_leo_practical_channel_batch',side_effect=fake_channel), \
                 patch.object(sys,'argv',['builder','--config',str(path)]):
                main()
            capsule=root/'data/capsule'
            with np.load(capsule/'received.npz',allow_pickle=False) as z:
                self.assertEqual(set(z.files),{'iq','ids'})
                self.assertEqual(len(z['ids']),396)
                self.assertEqual(len(set(z['ids'])),396)
            rows=[json.loads(p.read_text()) for p in (capsule/'splits').glob('*.json')]
            self.assertEqual(len(rows),18)
            group=sorted([r for r in rows if r['scenario']=='practical_high' and len(r['registered_classes'])==2],key=lambda r:r['k'])
            self.assertEqual(group[0]['query_indices'],group[-1]['query_indices'])
            self.assertTrue(set(group[0]['support_indices']) < set(group[-1]['support_indices']))
            self.assertTrue(all(not(set(r['support_indices'])&set(r['query_indices'])) for r in rows))
            self.assertTrue(all('query_labels' not in r and 'old' not in r and 'manytx_path' not in r for r in rows))
            self.assertFalse((capsule/'truth.json').exists())

    def test_one_physical_observation_per_scene_role_and_fixed_counts(self):
        records = [(day, i) for day in range(4) for i in range(50)]
        result = assignments(records, seed=2026092705, receiver='1-1', class_id='14-10')
        seen = set()
        for groups in result.values():
            self.assertEqual(len(groups['support_pool']), 36)
            self.assertEqual(len(groups['query']), 30)
            for ids in groups.values():
                self.assertFalse(seen & set(ids))
                seen.update(ids)
        self.assertEqual(len(seen),198)
        self.assertEqual(result, assignments(records,seed=2026092705,receiver='1-1',class_id='14-10'))

    def test_missing_or_duplicate_records_rejected(self):
        for records in [list(range(150)), [0]*200]:
            with self.assertRaises(ValueError):
                assignments(records, seed=1,receiver='rx',class_id='tx')


if __name__ == '__main__':
    unittest.main()
