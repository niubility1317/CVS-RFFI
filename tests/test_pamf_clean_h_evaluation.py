import tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.pamf_channel_pl import clean_h_evaluation as e
from experiments.pamf_channel_pl.clean_h_source import REFERENCES


def config(scoring=False,reference=False):
    method='SIG1-Sup' if reference else e.METHODS[0];project='/project'
    c=dict(schema='pamf_clean_h_raw_eval_v1',run_id='confirm',row_id='row',protocol_id='F0',purpose='baseline_cross_date' if reference else 'selected_method',method=method,model_seed=2026092701,
        source_days=[1,2,3],classes=list('abcdef'),tests={t:dict(views_root=t) for t in (('cross_date',) if reference else e.TASK_COUNTS)},readouts=['raw'],output_root='output')
    if scoring:
        c['prediction_output']='prediction'
        for v in c['tests'].values():v.update(truth='truth_sentinel',kind='frozen')
    else:c.update(project=project,selection_source=project+'/runs/'+e.SOURCE_RUN+'/complete.json',source_run_id=e.BASELINE_RUN if reference else e.SOURCE_RUN,source_row_id='source_row',source_output='source',source_contract='contract',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')})
    return c


def selection():
    scores={m:dict(raw_correct=26574,H3_correct=60889) for m in REFERENCES}
    scores.update({m:dict(raw_correct=26550,H3_correct=61140) for m in e.METHODS})
    return dict(status='CLEAN_H_SOURCE_COMPLETE',run_id=e.SOURCE_RUN,model_seed=2026092701,U_truth_reads=0,target_reads=0,source_scores=scores,source_selected=e.METHODS[0])


def source(c):
    sup=c['purpose']=='baseline_cross_date' and c['method']=='SIG1-Sup'
    cfg=dict(run_id=c['source_run_id'],row_id=c['source_row_id'],model_seed=c['model_seed'],paper_method=c['method'],protocol_id='F0',sig_time_views=1,default_head_joint=True,U_use=not sup,posthoc_head_fit=False,source_contract='contract',frozen_sources=c['frozen_sources'])
    return dict(status='SOURCE_FROZEN',epoch=50,updates=11100,target_access=False,U_truth_read=False,head_fit_updates=0,config=cfg,segments=[dict(config=dict(U_use=not sup)) for _ in range(3)])


class RawEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_predict_and_score_have_disjoint_handles_and_no_recovery(self):
        for ref in (False,True):
            c=config(reference=ref);e.validate(c)
            for extra in (dict(truth='x'),dict(restored_caches={}),dict(G={})):
                with self.assertRaises(ValueError):e.validate(dict(c,**extra))
            bad=deepcopy(c);bad['tests'][next(iter(bad['tests']))]['truth']='x'
            with self.assertRaises(ValueError):e.validate(bad)
            s=config(scoring=True,reference=ref);e.validate(s,scoring=True)
            with self.assertRaises(ValueError):e.validate(dict(s,source_output='source'),scoring=True)
        bad=config();bad['frozen_sources']['G']='forbidden'
        with self.assertRaises(ValueError):e.validate(bad)

    def test_target_closed_until_complete_fixed_source_winner(self):
        c=config();good=selection()
        with patch.object(e,'read',return_value=good):self.assertEqual(e.selected_method(c),e.METHODS[0])
        bad=deepcopy(good)
        for m in e.METHODS:bad['source_scores'][m]['H3_correct']=61131
        for header in (dict(good,status='RUNNING'),dict(good,target_reads=1),dict(good,source_selected=e.METHODS[1]),bad):
            with patch.object(e,'read',return_value=header),self.assertRaises(ValueError):e.selected_method(c)
        with patch.object(e,'read',return_value=good),self.assertRaises(ValueError):e.selected_method(dict(c,method=e.METHODS[1]))

    def test_failed_selection_precedes_activation_and_target_IO(self):
        with patch.object(e,'selected_method',side_effect=ValueError('not selected')),patch.object(e,'activate') as act,patch('experiments.cric_dura_comparison.evaluation.handle') as handle:
            with self.assertRaises(ValueError):e.predict(config(),'cpu')
        act.assert_not_called();handle.assert_not_called()

    def test_selected_source_data_contract_anchor_precedes_target_IO(self):
        c=config()
        def reader(path):
            v=str(path).replace('\\','/')
            if v==c['selection_source']:return selection()
            if v.endswith('/configs/'+e.METHODS[0]+'.json'):return dict(source_contract='anchor')
            if v=='contract':return dict(classes=c['classes'],roles='wrong_F1')
            if v=='anchor':return dict(classes=c['classes'],roles='actual_F0')
            raise AssertionError(v)
        with patch.object(e,'read',reader),patch.object(e,'activate'),patch('experiments.cric_dura_comparison.evaluation.handle') as handle:
            with self.assertRaises(ValueError):e.predict(c,'cpu')
        handle.assert_not_called()

    def test_frozen_source_rejects_wrong_seed_head_target_or_U_visibility(self):
        for ref in (False,True):
            c=config(reference=ref);s=source(c);contract=dict(classes=c['classes'])
            with patch.object(e,'read',return_value=contract):e.source_metadata(s,c,contract)
            for top in (dict(epoch=30),dict(target_access=True),dict(U_truth_read=True),dict(head_fit_updates=1)):
                with patch.object(e,'read',return_value=contract),self.assertRaises(ValueError):e.source_metadata(dict(s,**top),c,contract)
            for key,value in (('model_seed',2026092702),('sig_time_views',3),('U_use',ref)):
                bad=deepcopy(s);bad['config'][key]=value
                with patch.object(e,'read',return_value=contract),self.assertRaises(ValueError):e.source_metadata(bad,c,contract)
            if ref:
                bad=deepcopy(s);bad['segments'][0]['config']['U_use']=True
                with patch.object(e,'read',return_value=contract),self.assertRaises(ValueError):e.source_metadata(bad,c,contract)

    def test_original_physical_positions_and_independent_six_class_argmax(self):
        class Reader:
            def features(self,x):return x[:,0,:6]
            def classify_features(self,h):return h
        a=np.zeros((4,2,256),np.float32);a[0,0,2]=1;a[3,0,4]=1;positions=np.array([3,0],np.int64)
        np.testing.assert_array_equal(e.array_predictions(Reader(),a,positions,2,6,'cpu'),[4,2])
        np.testing.assert_array_equal(e.array_predictions(Reader(),a,positions[::-1].copy(),2,6,'cpu'),[2,4])
        with self.assertRaises(ValueError):e.array_predictions(Reader(),a,None,2,6,'cpu')
        with self.assertRaises(ValueError):e.array_predictions(Reader(),a,np.array([4,0]),2,6,'cpu')
        a[0,0,0]=np.nan
        with self.assertRaises(FloatingPointError):e.array_predictions(Reader(),a,positions,2,6,'cpu')

    def test_truth_stays_closed_when_any_declared_prediction_is_incomplete(self):
        c=config(scoring=True);calls=[]
        def reader(path):
            calls.append(str(path));return dict(status='PREDICTING',parameters_and_buffers_unchanged=True,source_frozen_verified=True)
        with patch.object(e,'read',reader),patch('experiments.cric_dura_comparison.evaluation.read',reader):
            with self.assertRaises(ValueError):e.score(c)
        self.assertFalse(any('truth_sentinel' in path for path in calls))


if __name__=='__main__':unittest.main()
