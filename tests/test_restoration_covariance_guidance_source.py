import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def test_source_metrics_use_clean_reference_class_and_optimized_feature_scale():
    from experiments.pamf_phase3_restoration.covariance_guidance_source import source_metrics
    clean=torch.ones(2,2,256);restored=clean.clone();restored[0]=0
    target=torch.ones(2,160);features=target.clone();features[0]=2
    original=torch.tensor([[0.,3.,0.,0.,0.,0.],[5.,0.,0.,0.,0.,0.]])
    logits=torch.tensor([[2.,1.,0.,0.,0.,0.],[4.,1.,0.,0.,0.,0.]])
    m=source_metrics(restored,clean,features,logits,target,original,torch.ones(160))
    torch.testing.assert_close(m['iq_nmse'],torch.tensor([1.,0.]))
    torch.testing.assert_close(m['feature_standardized_error'],torch.tensor([1.,0.]))
    torch.testing.assert_close(m['optimized_feature_objective'],torch.tensor([.01,0.]),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(m['Fclean_reference_margin'],torch.tensor([-1.,3.]))
    torch.testing.assert_close(m['Fclean_margin_change'],torch.tensor([-4.,-2.]))
    torch.testing.assert_close(m['clean_decision_agreement'],torch.tensor([0.,1.]))


def test_both_source_parent_identities_and_permissions_are_required():
    from experiments.pamf_phase3_restoration.covariance_guidance_source import validate_parent_metadata
    root=Path(__file__).resolve().parents[1]/'automation_reports/CV-SincNet'
    cases=[('physical','20261102-phase1-Restore-CovarianceMAP-E30-s2026092701-r01','TF_CovarianceMAP'),
           ('guided','20261102-phase1-Restore-CovarianceFeature-E30-s2026092701-r01','TF_CovarianceFeature')]
    for method,run,arm in cases:
        p=root/run/('training_analysis.json' if method=='physical' else 'source_independent_check.json')
        data=json.loads(p.read_text(encoding='utf-8'))
        done=data['complete'] if method=='physical' else data['training'][arm]['complete']
        validate_parent_metadata(done,method)
        for key,value in [('code_commit','other'),('epoch',29),('TX_labels_read',1),('target_training_reads',1)]:
            bad=copy.deepcopy(done);bad[key]=value
            with pytest.raises(ValueError):validate_parent_metadata(bad,method)
        bad=copy.deepcopy(done);bad['F0_identity']['source_code_commit']='other'
        with pytest.raises(ValueError):validate_parent_metadata(bad,method)


def test_missing_predictions_prevent_truth_open(tmp_path):
    from experiments.pamf_phase3_restoration.covariance_guidance_source import score
    from types import SimpleNamespace
    # No input config exists either: prediction completion must fail first.
    with pytest.raises(FileNotFoundError,match='predictions_complete'):
        score(SimpleNamespace(root=tmp_path,inputs=tmp_path/'must-not-open.json',commit='test'))


def test_metric_recomputation_and_surrogate_damage_cells():
    from experiments.pamf_phase3_restoration.covariance_guidance_source import numpy_metrics,paired_cells
    clean=np.ones((3,2,256));target=np.ones((3,160));original=np.tile([0.,3.,0.,0.,0.,0.],(3,1))
    logits=np.array([[2.,1.,0,0,0,0],[0.,4.,0,0,0,0],[0.,4.,0,0,0,0]])
    a=numpy_metrics(clean,clean,target,logits,target,original,np.ones(160))
    np.testing.assert_array_equal(a['Fclean_reference_margin'],[-1.,4.,4.])
    physical={'prediction':np.array([1,0,0]),'optimized_feature_objective':np.array([2.,2.,2.])}
    guided={'prediction':np.array([0,1,0]),'optimized_feature_objective':np.array([1.,1.,2.])}
    cells=paired_cells(physical,guided,np.array([1,1,1]),np.ones(3,bool))
    assert cells['feature_better']['TX_damage']==1 and cells['feature_better']['TX_repair']==1
    assert sum(sum(v.values()) for v in cells.values())==3
