import unittest
from copy import deepcopy
from experiments.pamf_u_history.fonly_confirm import RUN,SOURCE_RUN,METHODS,TASKS,config,validate_config,validate_context


def context():
    return dict(project='/p',python='/python',frozen_sources={'F':'/f','AF':'/af','BASE':'/b','FOLD':'/fold'},source_run_root='/p/runs/'+SOURCE_RUN,
        source_contract='/contract',source_cache='/cache',classes=['a','b','c','d','e','f'],
        tests={t:dict(views_root='/views/'+t,truth='/truth/'+t,kind='unseen_receiver') for t in TASKS},
        reference_predictions={'SIG1-PL':'/p/pl','SIG1-Sup':'/p/sup'},reference_scores={'SIG1-PL':'/s/pl','SIG1-Sup':'/s/sup'})


class FonlyConfirmationTests(unittest.TestCase):
    def test_all_three_configs_are_raw_only_and_truth_is_isolated(self):
        for method in METHODS:
            p=config(context(),'test',method);s=config(context(),'test',method,True)
            self.assertEqual(p['readouts'],['raw']);self.assertEqual(p['run_id'],RUN)
            self.assertTrue(all(set(v)=={'views_root'} for v in p['tests'].values()))
            self.assertTrue(all('truth' in v for v in s['tests'].values()))
            self.assertTrue({'source_output','source_cache','source_contract','frozen_sources','source_pilot'}.isdisjoint(s))
            self.assertEqual(s['prediction_output'],p['output_root'])

    def test_G_handles_and_truth_in_prediction_are_rejected(self):
        p=config(context(),'test','CovSel-U')
        for key in ('restored_caches','G','same_G','source_G','teacher_checkpoint'):
            bad=deepcopy(p);bad[key]=None
            with self.assertRaises(ValueError):validate_config(bad)
        bad=deepcopy(p);bad['tests']['cross_rx']['truth']='/truth'
        with self.assertRaises(ValueError):validate_config(bad)
        s=config(context(),'test','CovSel-U',True);s['source_cache']='/cache'
        with self.assertRaises(ValueError):validate_config(s,True)

    def test_noneligible_method_or_G_context_rejected(self):
        with self.assertRaises(ValueError):config(context(),'test','CovFull-U')
        c=context();c['frozen_sources']['G']='/g'
        with self.assertRaises(ValueError):validate_context(c)
        p=config(context(),'test','CovSel-U');p['tests'].pop('cross_both')
        with self.assertRaises(ValueError):validate_config(p)
