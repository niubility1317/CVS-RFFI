"""Focused negative checks for the next authorized paper run."""
import copy,json,tempfile,unittest
from pathlib import Path
from experiments.pamf_paper_completion.plan import make_plan,configuration,validate_config,PROTOCOLS,RUN

def context():
    protocols={pid:dict(source_rxs=([1] if setting=='source_rx_1' else [1,3,4,6,8])) for setting,(pid,*_) in PROTOCOLS.items()}
    handles={pid:dict(source_cache='/source/'+pid,source_contract='/contract/'+pid,classes=list(range(10 if pid=='ManyRx' else 6))) for pid in protocols}
    base=dict(source_cache='/source/F0',source_contract='/contract/F0')
    repl=dict(hard_pl=dict(threshold=.95,weight=6.),hard_pl_no_uas=True)
    return dict(protocols=protocols,handles=handles,templates={str(s):dict(base={'F21':base},repl=repl) for s in (2026092701,2026092702,2026092703)},
                ce_template=dict(method='standard_ce_baselines',epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,
                drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,source_contract='/contract/F0',dataset='/data/ManySig.pkl'))

class MatrixAndPermissions(unittest.TestCase):
    def setUp(self):self.plan=make_plan();self.ctx=context()
    def config(self,**criteria):
        row=next(r for r in self.plan['rows'] if all(r[k]==v for k,v in criteria.items()))
        return configuration(self.ctx,row,'/project','commit')
    def test_dedup_and_two_layers(self):
        self.assertEqual((self.plan['logical_F'],self.plan['new_F'],self.plan['reused_F']),(200,119,81))
        self.assertEqual(sum(r['new_training'] and r['family']=='structure' for r in self.plan['rows']),12)
        self.assertEqual(sum(r['family']=='architecture' for r in self.plan['rows']),8)
        self.assertFalse(any(r['learning_regime']=='pl' and r['permuted'] for r in self.plan['rows']))
    def test_all_declared_configs_validate(self):
        for r in self.plan['rows']:validate_config(configuration(self.ctx,r,'/project','commit'))
    def test_no_target_truth_in_training(self):
        c=self.config(method='ssl')
        for key in ('truth','target_truth','tests','views_root','teacher_checkpoint','shared_prefix','fork_parent_source'):
            bad=dict(c,**{key:'forbidden'})
            with self.assertRaises(ValueError):validate_config(bad)
    def test_sup_has_no_U_and_no_EMA(self):
        c=self.config(method='sup_noaug')
        self.assertFalse(c['U_use']);self.assertEqual(c['regime'],'matched_sat_supervised');self.assertEqual(c['ablation'],'F17')
        for key in ('U_use','own_EMA','native_U_gate','native_unlabeled_loop'):
            with self.assertRaises(ValueError):validate_config(dict(c,**{key:True}))
    def test_plain_PL_cannot_inherit_full_SSL(self):
        c=self.config(method='pl_noaug');self.assertEqual((c['plain_PL_threshold'],c['plain_PL_weight']),(.95,1.))
        with self.assertRaises(ValueError):validate_config(dict(c,hard_pl={'weight':6.}))
        with self.assertRaises(ValueError):validate_config(dict(c,plain_PL_threshold=.8))
    def test_SSL_cannot_degrade_to_ordinary_PL(self):
        c=self.config(method='ssl');bad=copy.deepcopy(c);bad['hard_pl']['weight']=1.
        with self.assertRaises(ValueError):validate_config(bad)
    def test_source_method_not_relabelled(self):
        c=self.config(method='ssl');bad=copy.deepcopy(c);bad['paper_row']['permuted']=False
        with self.assertRaises(ValueError):validate_config(bad)
        with self.assertRaises(ValueError):validate_config(dict(c,model_seed=2026092708))
    def test_budget_and_same_row_segments(self):
        c=self.config(method='ssl')
        for extra in ({'optimizer_updates':4440},{'epochs':20},{'segment_epochs':20,'epoch_offset':20,'step_offset':4440}):
            with self.assertRaises(ValueError):validate_config(dict(c,**extra))
        validate_config(dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source='/same-row/segment20'))
    def test_CE_200_has_no_augmentation(self):
        c=self.config(family='architecture')
        for extra in ({'augmentation':True},{'epochs':50},{'extra_losses':['PL']},{'initial_checkpoint':'forbidden'}):
            with self.assertRaises(ValueError):validate_config(dict(c,**extra))
    def test_legacy_SSL_bookkeeping_is_not_a_mechanism(self):
        from experiments.pamf_paper_completion.prepare import validate_reused_SSL
        ref=dict(hard_pl={'weight':6.,'threshold':.95},hard_pl_parent_epoch=20,hard_pl_continuation_epochs=30)
        done=dict(config={'hard_pl':{'weight':6.,'threshold':.95}},actual_REPL={'calls':6660},
                  segments=[dict(epoch=n,updates=n*222,config={'step_offset':o*222}) for n,o in ((20,0),(10,20),(20,30))])
        validate_reused_SSL(done,ref)
        bad=copy.deepcopy(done);bad['config']['hard_pl']['weight']=1.
        with self.assertRaises(ValueError):validate_reused_SSL(bad,ref)
        bad=copy.deepcopy(done);bad['actual_REPL']['calls']=4440
        with self.assertRaises(ValueError):validate_reused_SSL(bad,ref)
    def test_source_constructor_arguments_have_physical_input_length(self):
        from types import SimpleNamespace
        from experiments.pamf_paper_completion.model import source_arguments
        native=SimpleNamespace(num_classes=6,use_unlabeled=True,use_ema_teacher=True)
        a=source_arguments(lambda c,device:native,{},'cpu')
        self.assertEqual(a.input_len,256)
        self.assertFalse(a.use_unlabeled);self.assertFalse(a.use_ema_teacher)
        self.assertEqual((a.lambda_u,a.lambda_ent),(0.,0.))

if __name__=='__main__':unittest.main()
