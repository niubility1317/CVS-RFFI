"""Protocol-negative checks for the fixed paper reproduction matrix."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from experiments.pamf_single_view_paper.plan import make_plan,configuration,validate_config,PROTOCOLS

def context():
    return dict(handles={v[0]:dict(source_cache='source/'+k,source_contract='contract/'+k,classes=list(range(10 if k=='ten_class' else 6))) for k,v in PROTOCOLS.items()},
        protocols={v[0]:dict(source_rxs=[1,3,4,6,8]) for k,v in PROTOCOLS.items()},
        templates={str(s):dict(base=dict(F21={})) for s in (2026092701,2026092702,2026092703)},runtime=dict(augmentation_cache='source_cache'))

class ProtocolTests(unittest.TestCase):
    def test_clean_only_score_preserves_seven_view_capsule(self):
        from experiments.pamf_single_view_paper.evaluation import score
        from experiments.pamf_single_view_paper.plan import VIEWS
        fake=SimpleNamespace(VIEWS=VIEWS,tasks=lambda _:['old'])
        def handle(*args):
            self.assertEqual(fake.VIEWS,VIEWS)
            return 'validated full capsule'
        fake.handle=handle
        def scoring(c):
            self.assertEqual(fake.VIEWS,('clean',))
            self.assertEqual(fake.handle('existing capsule'),'validated full capsule')
            self.assertEqual(fake.VIEWS,('clean',))
        fake.score=scoring
        with patch.dict('sys.modules',{'experiments.cric_dura_comparison.evaluation':fake}):score({'views':['clean']})
        self.assertEqual(fake.VIEWS,VIEWS)
    def test_ce_pl_loader_does_not_inherit_F15_U_exclusion(self):
        from experiments.pamf_single_view_paper.model import loader_configuration
        row=next(r for r in make_plan()['rows'] if r['family']=='branch' and r['learning_regime']=='pl')
        c=configuration(context(),row,'/project','testcommit');loader=loader_configuration(c)
        self.assertEqual(c['ablation'],'F15')
        self.assertEqual(loader['ablation'],'F21')
        self.assertEqual(loader['regime'],'native_cric')
        self.assertTrue(loader['U_use'])
        self.assertEqual(loader['unlabeled_batch'],256)
    def test_exact_matrix_and_reuse(self):
        p=make_plan();self.assertEqual((p['logical_F'],p['new_F'],p['reused_F']),(120,108,12))
        self.assertEqual(len({r['row_id'] for r in p['rows']}),120)
        self.assertTrue(all(not r['row_id'].startswith('F0') for r in p['rows']))
    def test_all_source_configurations(self):
        for row in make_plan()['rows']:
            c=configuration(context(),row,'/project','testcommit');validate_config(c)
            self.assertEqual(c['U_use'],row['learning_regime']=='pl')
            self.assertEqual(c['ablation']=='F15',row['family']=='branch')
    def test_illegal_protocols_and_objectives_rejected(self):
        row=next(r for r in make_plan()['rows'] if r['family']=='branch' and r['learning_regime']=='pl')
        base=configuration(context(),row,'/project','testcommit')
        for key,value in [('truth','forbidden'),('tests',{'main':'target'}),('checkpoint','foreign'),('shared_prefix','foreign'),
            ('own_EMA',True),('native_U_gate',True),('posthoc_head_fit',True),('U_use',False),('plain_PL_threshold',.8),
            ('plain_PL_weight',6.),('optimizer_updates',4440),('satellite_CE_weight',1.),('ablation','F21'),('U_start_epoch',1)]:
            c=copy.deepcopy(base);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(c)
    def test_display_and_role_names_are_explicit(self):
        p=make_plan();names=[r['split_name'] for r in p['rows']]
        self.assertIn('Main split',names);self.assertIn('Receiver split 1',names)
        self.assertIn('1% labeled',names);self.assertIn('6 source receivers',names)
        for r in p['rows']:
            self.assertEqual(r['views']==['clean'],not r['channel_augmentation'])

if __name__=='__main__':unittest.main()
