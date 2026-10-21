import copy
import unittest
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.phase1_dg_sourcecount6.plan import make_plan,validate_plan,counts,TARGET
from experiments.phase1_dg_sourcecount6.data import selections


class SourceCountProtocolTests(unittest.TestCase):
    def test_fixed_target_and_every_count_without_standalone_resnet(self):
        p=make_plan()
        self.assertEqual(len(p['rows']),432)
        self.assertEqual(sum(bool(r['reuse_parent_row_id']) for r in p['rows']),48)
        self.assertEqual(Counter(r['source_count'] for r in p['rows']),{m:72 for m in range(1,7)})
        self.assertEqual(set(r['family'] for r in p['rows']),{'riei','drift','ibrff','sdg'})
        for key,value in [('source_rxs',[0]),('target_rxs',[2,5,7,9,10,11])]:
            bad=copy.deepcopy(p);next(iter(bad['protocols'].values()))[key]=value
            with self.assertRaises(ValueError):validate_plan(bad)
        for key,value in [('epochs',99),('lambda_sat_cls',0.),('teacher',True)]:
            bad=copy.deepcopy(p);bad['rows'][0][key]=value
            with self.assertRaises(ValueError):validate_plan(bad)

    def test_fixed_annotations_and_physical_disjointness_for_all_six_counts(self):
        # Metadata-only synthetic inventory: no target IQ or truth is opened.
        data=dict(tx_list=list(range(6)),equalized_list=[0,1],
            data=[[[[None,range(1000)] for day in range(4)] for rx in range(12)] for tx in range(6)])
        for m in range(1,7):
            p=make_plan()['protocols']['m%d-fixed-supervised'%m]
            roles=selections(data,p)
            self.assertEqual({k:len(v) for k,v in roles.items()},counts(m,'fixed'))
            ids={k:{r[-1] for r in v} for k,v in roles.items()}
            self.assertFalse(ids['L_s']&ids['U_s'] or ids['L_s']&ids['V'] or ids['U_s']&ids['V'])
            for role,total in [('L_s',1080),('V',5400)]:
                balanced=Counter((r[0],r[1],r[2]) for r in roles[role])
                self.assertEqual(set(balanced.values()),{total//(6*m*3)})
            self.assertTrue(all(r[1] not in TARGET for v in roles.values() for r in v))
            self.assertTrue(all(r[4]<700 for r in roles['U_s']))

    def test_source_and_predictor_reject_truth_before_reading_input(self):
        from experiments.phase1_dg_sourcecount6.runtime import bind
        source,evaluation=bind();row=make_plan()['rows'][0]
        with self.assertRaises(ValueError):source.check_training(dict(row,target_truth='forbidden'))
        with self.assertRaises(ValueError):evaluation.reject_truth(dict(tests=dict(main=dict(truth='forbidden'))))
        with self.assertRaises(ValueError):evaluation.score_preflight(dict(row_id=row['row_id'],model_seed=row['model_seed'],tests={}))


if __name__=='__main__':unittest.main()
