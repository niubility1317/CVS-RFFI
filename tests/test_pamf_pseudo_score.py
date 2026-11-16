import gzip
import json
import tempfile
import unittest
from pathlib import Path
from experiments.pamf_repl_weight.pseudo_score import score_observations


class PseudoScoreTests(unittest.TestCase):
    def test_counts_exposures_separately_from_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'obs.gz'
            rows=[dict(step=4441,physical_ids=['a','b'],pseudo=[0,0],confidence=[.99,.96],accepted=[True,True]),
                  dict(step=4442,physical_ids=['a','b'],pseudo=[1,1],confidence=[.92,.99],accepted=[False,True])]
            with gzip.open(path,'wt') as output:
                for row in rows:output.write(json.dumps(row)+'\n')
            result=score_observations(path,truth={'a':0,'b':1},allowed_ids=['a','b'],num_classes=2)
            self.assertEqual(result['overall'],dict(seen=4,accepted=3,accepted_correct=2))
            self.assertEqual((result['unique_seen'],result['unique_accepted']),(2,2))
            self.assertEqual(result['per_epoch']['21']['accepted_correct'],2)
            self.assertEqual(result['per_true_class']['1']['accepted_correct'],1)

    def test_rejects_mismatched_truth_role(self):
        with self.assertRaises(ValueError):
            score_observations('unused',truth={'a':0,'target':1},allowed_ids=['a','b'],num_classes=2)


if __name__=='__main__':unittest.main()
