from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.pamf_no_signature_ablation.method import RUN,ARMS,SEEDS,READOUTS,TASKS,rows,configuration,validate_config


def template(seed):
    return dict(model_seed=seed,run_id='previous',row_id='previous',regime='matched_sat_supervised',ablation='F15',
        U_use=False,own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,posthoc_head_fit=False,
        epochs=50,optimizer_updates=11100,trajectory=[20,10,20],satellite_concat=True,satellite_CE_weight=.68,
        channel_augmentation=True,augmentation_seed=2026102301,source_rxs=[1,3,4,6,8],source_days=[1,2,3],num_classes=6,
        repl_row=dict(model_seed=seed,row_id='previous',structure='NO_SIGNATURE',group='strict_structure',U=False,
                      L_A=False,L_B=False,blocks=dict(style=False,guard=False,joint=False,extrapolation=False)))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.context=dict(baselines={str(s):dict(complete=dict(config=template(s))) for s in SEEDS})
        self.config=configuration(self.context,rows()[3],'/project','abc')

    def test_exact_matrix_and_scratch_budget(self):
        self.assertEqual(len(rows()),15)
        self.assertEqual(sum(not r['reuse'] for r in rows()),12)
        self.assertNotIn('SHUFFLE_TIME',ARMS)
        for row in rows():validate_config(configuration(self.context,row,'/project','abc'))

    def test_direct_smoke_supplies_native_input_length(self):
        from argparse import Namespace
        from experiments.pamf_no_signature_ablation.runner import smoke_arguments
        factory=lambda c,device:Namespace(num_classes=6)
        # Reproduce the failed adapter boundary without importing a remote model.
        with self.assertRaises(AttributeError):int(factory({},'cpu').input_len)
        self.assertEqual(int(smoke_arguments(factory,{},'cpu').input_len),256)

    def test_reject_target_U_ancestors_and_recipe_changes(self):
        for key,value in [('truth','private'),('tests',{'cross_both':'private'}),('parent_source','old'),
                          ('shared_prefix','old'),('U_use',True),('own_EMA',True),('epochs',20),('ablation','F20'),
                          ('channel_augmentation',False),('augmentation_seed',1),('source_rxs',[0,1,2])]:
            c=deepcopy(self.config);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(c)

    def test_reject_unregistered_architecture_and_objective(self):
        for key,value in [('structure','SHUFFLE_TIME'),('U',True),('L_A',True)]:
            c=deepcopy(self.config);c['repl_row'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(c)

    def test_pooled_CM_and_paired_summary(self):
        from experiments.pamf_no_signature_ablation.analyze import run
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sources={};views=['clean','p1','p2','p3','p4','p5','p6']
            for row in rows():
                d=root/row['row_id'];d.mkdir();results=[]
                for t,n in zip(TASKS,(2,3,5)):
                    cm=[[n if i==j else 0 for j in range(6)] for i in range(6)]
                    for rd in READOUTS:
                        for v in views:results.append(dict(test=t,readout=rd,view=v,dimension='overall',stratum='ALL',
                             count=6*n,confusion=cm,accuracy=1.,macro_f1=1.))
                (d/'scores.json').write_text(json.dumps(dict(truth_last=True,target_feedback_forbidden=True,
                         model_seed=row['model_seed'],results=results,paired_raw=[])),encoding='utf-8')
                (d/'complete.json').write_text(json.dumps(dict(epoch=50,updates=11100,segments=[dict(seconds=1.)])),encoding='utf-8')
                sources[row['row_id']]=dict(row=row,source_output=str(d),scores=str(d))
            (root/'sources.json').write_text(json.dumps(sources),encoding='utf-8')
            run(dict(run_root=str(root),views=views))
            import csv
            with (root/'analysis/summary.csv').open(encoding='utf-8') as f:summary=list(csv.DictReader(f))
            self.assertTrue(all(float(r['mean'])==100. and float(r['sd'])==0. for r in summary))
            self.assertTrue(any(r['test']=='pooled' and r['view']=='six_practical_mean' for r in summary))
            with (root/'analysis/paired_vs_baseline.csv').open(encoding='utf-8') as f:
                self.assertTrue(all(float(r['mean'])==0. for r in csv.DictReader(f)))


if __name__=='__main__':unittest.main()
