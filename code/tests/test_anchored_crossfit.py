import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_anchored_cache import toy_rows
from cvsrffi.anchored_fit import FitConfig
from cvsrffi.anchored_crossfit import build_rx_folds,run_expert_oof,run_nested_fusion_audit


def test_fold_physical_and_rx_disjointness():
    rows=toy_rows(receivers=(1,3,4,6,8))
    folds=build_rx_folds(rows)
    assert len(folds)==5
    for fold in folds:
        assert not set(fold.train_physical_ids)&set(fold.predict_physical_ids)
        assert fold.heldout_rx not in fold.train_rx
        assert len(fold.train_rx)==4


def test_nested_models_reuse_training_sets_and_do_not_train_outer_rx(tmp_path):
    torch.set_num_threads(2)
    rows=toy_rows(receivers=(1,3,4,6,8));calls=[]
    def trainer(cache,train_rx,config,seed,output,**kwargs):
        calls.append(tuple(train_rx))
        return SimpleNamespace(train_rx=tuple(train_rx),predict=lambda data: data.baseline_inference_logits.clone(),
                               state={'test_only':True})
    w0=torch.randn(3,8)
    oof=run_expert_oof(rows,FitConfig(epochs=1),4,tmp_path/'oof',w0=w0,tau0=10.,fit_fn=trainer)
    assert len(calls)==6
    assert torch.equal(oof.scores,rows.baseline_inference_logits)
    import pytest
    for changed_rows,cfg,seed in ((rows.take(list(reversed(range(len(rows))))),FitConfig(epochs=1),4),
                                  (rows,FitConfig(candidate='A3',epochs=1),4),(rows,FitConfig(epochs=1),5)):
        with pytest.raises(ValueError):run_nested_fusion_audit(changed_rows,oof,cfg,seed,tmp_path/'bad',w0=w0,tau0=10.,fit_fn=trainer)
    nested=run_nested_fusion_audit(rows,oof,FitConfig(epochs=1),4,tmp_path/'nested',w0=w0,tau0=10.,fit_fn=trainer)
    assert len(calls)==16 and len(set(calls))==16
    for fold in nested['folds']:
        assert fold['heldout_rx'] not in fold['gate_train_rx']
        assert fold['heldout_rx'] not in fold['expert_train_rx']
        assert not set(fold['gate_train_physical_ids'])&set(fold['predict_physical_ids'])
    assert (nested['selected_actions']==0).all()
