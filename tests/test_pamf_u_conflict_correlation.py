import gzip
import json
import tempfile
import unittest
from pathlib import Path
from experiments.pamf_repl_step_guard.correlate_u_conflict import analyze


class UConflictTests(unittest.TestCase):
    def test_same_step_and_exposure_weighted_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            p,g=Path(folder)/'p.gz',Path(folder)/'g.gz'
            decisions=[dict(step=4441,physical_ids=['a','b'],pseudo=[0,0],accepted=[True,True]),
                       dict(step=4442,physical_ids=['a'],pseudo=[1],accepted=[False])]
            guards=[dict(step=i,groups={name:{'anchor_cosines_before':{'L_full':v}} for name in ('t2','cls_head','other')})
                    for i,v in ((1,-.1),(2,.2))]
            for path,rows in ((p,decisions),(g,guards)):
                with gzip.open(path,'wt') as output:
                    for row in rows:output.write(json.dumps(row)+'\n')
            done=dict(status='SOURCE_FROZEN',epoch=50,target_access=False,U_truth_read=False)
            result=analyze(p,g,truth={'a':0,'b':1},step_offset=4440,source_completion=done)
            self.assertEqual(result['groups']['t2']['conflict'],dict(steps=1,seen=2,accepted=2,accepted_correct=1))
            self.assertEqual(result['groups']['t2']['nonconflict']['steps'],1)
            with self.assertRaises(ValueError):analyze(p,g,truth={'a':0,'b':1},step_offset=0,source_completion=done)
            with self.assertRaises(ValueError):analyze(p,g,truth={'a':0,'b':1},step_offset=4440,source_completion={})


if __name__=='__main__':unittest.main()
