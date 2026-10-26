import unittest
import numpy as np
from experiments.pamf_u_history.fonly_results import metrics,index,TASKS,VIEWS,SEED


class FonlyResultTests(unittest.TestCase):
    def test_population_pooling_uses_counts(self):
        a=np.zeros((6,6),np.int64);b=a.copy();a[0,0]=90;a[0,1]=10;b[0,0]=10;b[0,1]=90
        self.assertEqual(metrics(a+b)['accuracy'],50.)
        self.assertEqual(metrics(3*a+b)['accuracy'],70.)
        with self.assertRaises(ValueError):metrics(np.zeros((6,6),np.int64))

    def test_saved_score_is_checked_against_confusion_and_coverage(self):
        cm=np.eye(6,dtype=np.int64)
        rows=[dict(test=t,view=v,readout='raw',dimension='overall',stratum='ALL',count=6,accuracy=1.,macro_f1=1.,confusion=cm.tolist()) for t in TASKS for v in VIEWS]
        d=dict(status='SCORED',truth_last=True,target_feedback_forbidden=True,model_seed=SEED,results=rows)
        self.assertEqual(len(index(d)),14)
        d['results'][0]['accuracy']=.9
        with self.assertRaises(ValueError):index(d)
        d['results']=rows[1:]
        with self.assertRaises(ValueError):index(d)
