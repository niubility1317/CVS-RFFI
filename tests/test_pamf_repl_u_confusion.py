import unittest
from experiments.pamf_repl_step_guard.analyze_u_confusion import aggregate,confidence_bin,validate_marginals


class UConfusionTests(unittest.TestCase):
    def test_unique_ids_exposures_and_confusion_are_distinct(self):
        rows=[dict(physical_ids=['a','b','c'],pseudo=[1,0,0],confidence=[.95,.97,.3],accepted=[True,True,False]),
              dict(physical_ids=['a','b'],pseudo=[0,0],confidence=[.99,.999],accepted=[True,True]),
              dict(physical_ids=['a'],pseudo=[1],confidence=[1.],accepted=[True])]
        result=aggregate(rows,truth={'a':0,'b':1,'c':0},receivers={'a':0,'b':1,'c':0},num_classes=2)
        self.assertEqual(result['overall'],dict(seen=6,accepted=5,accepted_correct=1))
        self.assertEqual(result['confusion'],[[1,2],[2,0]])
        self.assertEqual(result['unique']['accepted'],2)
        self.assertEqual(result['unique']['ever_wrong'],2)
        self.assertEqual(result['unique']['always_wrong_when_accepted'],1)
        self.assertEqual(result['unique']['mixed_correct_and_wrong'],1)
        self.assertEqual(result['unique']['wrong_exposures'],4)
        self.assertEqual(result['confidence_bins']['0.999-1.0']['accepted'],2)
        self.assertEqual(result['receiver_true_class']['1:1']['accepted'],2)
        marginals={'0':{'accepted':3,'accepted_correct':1},'1':{'accepted':2,'accepted_correct':0}}
        score={'segments':[{'per_true_class':marginals,'per_pseudo_class':marginals}]}
        validate_marginals(result,score,2)
        score['segments'][0]['per_true_class']={'0':{'accepted':4,'accepted_correct':1}}
        with self.assertRaisesRegex(ValueError,'marginals'):validate_marginals(result,score,2)

    def test_rejects_bad_accepted_confidence(self):
        self.assertEqual(confidence_bin(.949999988079071),'0.95-0.97')
        with self.assertRaisesRegex(ValueError,'confidence'):confidence_bin(.9499)
        with self.assertRaisesRegex(ValueError,'confidence'):confidence_bin(float('nan'))


if __name__=='__main__':unittest.main()
