import gzip
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.data import sample_id
from experiments.pamf_repl_step_guard.u_scoring import score


class UTruthScoringTests(unittest.TestCase):
    def test_frozen_U_id_join_and_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';cache=root/'cache';source.mkdir();cache.mkdir()
            ids=[sample_id('ManyRx',tx,2,1,0,7) for tx in (0,1)]
            np.savez(cache/'U_s_metadata.npz',ids=np.array(ids),rx_i=np.array([2,2]),day_i=np.array([1,1]),
                     eq_i=np.array([0,0]),sig_i=np.array([7,7]))
            contract=root/'contract.json';contract.write_text(json.dumps({'role_ids':{'U_s':ids}}))
            cfg=dict(source_contract=str(contract),source_cache=str(cache),protocol_id='ManyRx',row_id='x',num_classes=2)
            (source/'complete.json').write_text(json.dumps(dict(status='SOURCE_FROZEN',epoch=50,updates=11100,
                target_access=False,U_truth_read=False,config=cfg)))
            for epoch,step in ((30,4441),(50,6661)):
                segment=source/f'segment{epoch}';segment.mkdir()
                with gzip.open(segment/'U_pseudo_observations.jsonl.gz','wt') as out:
                    out.write(json.dumps(dict(step=step,physical_ids=ids,pseudo=[0,0],confidence=[.99,.96],accepted=[True,True]))+'\n')
            score(dict(source_output=str(source),protocol_id='ManyRx',row_id='x',output_root=str(root/'out')))
            result=json.loads((root/'out'/'scores.json').read_text())
            self.assertEqual(result['overall'],dict(seen=4,accepted=4,accepted_correct=2))
            self.assertEqual((result['unique_U_total'],result['unique_accepted']),(2,2))
            self.assertFalse(result['training_or_selection_use'])

    def test_rejects_target_input_before_reading_truth(self):
        with self.assertRaises(ValueError):score(dict(target_truth='forbidden'))


if __name__=='__main__':unittest.main()
