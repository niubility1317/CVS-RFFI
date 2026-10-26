import unittest
from experiments.pamf_u_history.confirm import prediction_config,scoring_config,validate_context,ROW
from experiments.pamf_u_history.source import RUN as SOURCE_RUN
from experiments.cric_dura_comparison.evaluation import reject_truth


class ConfirmationTests(unittest.TestCase):
    def context(self):
        return dict(project='PROJECT',python='PYTHON',frozen_sources={},source_output='PROJECT/runs/'+SOURCE_RUN+'/F/'+ROW+'/source',
            source_contract='contract',source_cache='source',classes=list(range(6)),restored_cache='G-cache',
            tests={t:dict(views_root=t+'/views',truth=t+'/truth.json') for t in ('cross_date','cross_rx','cross_both')},
            reference_predictions={m:m+'/predictions' for m in ('ordinary_PL','supervised','REPL')},
            reference_scores={m:m+'/scores' for m in ('ordinary_PL','supervised','REPL')})

    def test_truth_only_passed_to_independent_scorer(self):
        c=self.context();predict=prediction_config(c,'commit');score=scoring_config(c,'commit')
        reject_truth(predict)
        self.assertTrue(all('truth' in v for v in score['tests'].values()))
        self.assertFalse({'source_output','source_contract','source_cache','restored_caches','frozen_sources'}&set(score))
        predict['tests']['cross_rx']['truth']='forbidden'
        with self.assertRaises(ValueError):reject_truth(predict)

    def test_primary_checkpoint_cannot_be_reselected(self):
        c=self.context();c['source_output']=c['source_output'].replace('PL-History','PL-Negative')
        with self.assertRaises(ValueError):validate_context(c)


if __name__=='__main__':unittest.main()
