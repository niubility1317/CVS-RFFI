from copy import deepcopy
import unittest
from experiments.pamf_u_history.compare import index_scores
from experiments.pamf_u_history.confirm import RUN,ROW
from experiments.practical_full_2x2.plan import TASKS,VIEWS


class CompareTests(unittest.TestCase):
    def scores(self):
        cm=[[5 if i==j else 0 for j in range(6)] for i in range(6)]
        return dict(status='SCORED',truth_last=True,model_seed=2026092701,run_id=RUN,row_id=ROW,
            results=[dict(test=t,readout=r,view=v,dimension='overall',stratum='ALL',count=30,confusion=deepcopy(cm),accuracy=1.,macro_f1=1.)
                     for t in TASKS for r in ('raw','DURA-LU') for v in VIEWS])

    def test_all_populations_and_views_required(self):
        data=self.scores();self.assertEqual(len(index_scores(data,candidate=True)),42)
        data['results'].pop()
        with self.assertRaises(ValueError):index_scores(data,candidate=True)

    def test_scores_must_match_integer_confusion(self):
        data=self.scores();data['results'][0]['accuracy']=.99
        with self.assertRaises(ValueError):index_scores(data,candidate=True)

    def test_duplicate_and_non_truth_last_rejected(self):
        data=self.scores();data['results'].append(deepcopy(data['results'][0]))
        with self.assertRaises(ValueError):index_scores(data,candidate=True)
        data=self.scores();data['truth_last']=False
        with self.assertRaises(ValueError):index_scores(data,candidate=True)


if __name__=='__main__':unittest.main()
