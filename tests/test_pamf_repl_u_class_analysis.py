import unittest
from experiments.pamf_repl_step_guard.analyze_u_class import summarize


def make_score():
    segments=[]
    for start,end in ((21,30),(31,50)):
        n=end-start+1
        segments.append(dict(role='U_s',truth_used_only_for_offline_diagnostic=True,training_or_selection_use=False,
            per_epoch={str(i):dict(seen=2,accepted=2,accepted_correct=1) for i in range(start,end+1)},
            per_true_class={'0':dict(seen=n,accepted=n,accepted_correct=n),'1':dict(seen=n,accepted=n,accepted_correct=0)},
            per_pseudo_class={'0':dict(seen=n,accepted=n,accepted_correct=n),'1':dict(seen=n,accepted=n,accepted_correct=0)}))
    return dict(status='SCORED_U_DIAGNOSTIC',training_or_selection_use=False,segments=segments,
        overall=dict(seen=60,accepted=60,accepted_correct=30),unique_U_total=2,unique_accepted=2)


class UClassAnalysisTests(unittest.TestCase):
    def test_counts_close_and_errors_are_exposure_weighted(self):
        result=summarize(make_score(),num_classes=2)
        self.assertEqual(len(result['dimensions']['per_epoch']),30)
        self.assertEqual(result['dimensions']['per_true_class']['1']['accepted_wrong'],30)
        self.assertEqual(result['dimensions']['per_true_class']['1']['wrong_exposure_share'],1.)
        self.assertEqual(result['dimensions']['per_true_class']['0']['accuracy'],1.)

    def test_rejects_selection_and_nonclosing_counts(self):
        value=make_score();value['training_or_selection_use']=True
        with self.assertRaisesRegex(ValueError,'diagnostic-only'):summarize(value,num_classes=2)
        value=make_score();value['segments'][0]['per_true_class']['0']['accepted_correct']-=1
        with self.assertRaisesRegex(ValueError,'close'):summarize(value,num_classes=2)


if __name__=='__main__':unittest.main()
