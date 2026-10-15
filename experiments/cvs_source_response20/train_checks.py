"""Real native graph, synthetic IQ; no checkpoint or source files."""
import copy
import json
from pathlib import Path
import numpy as np
from .train_runner import bootstrap


def run():
    roots=Path('E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/validation_dependencies')
    bootstrap([roots/'original',roots/'legacy',roots/'fold'])
    from experiments.cvs_source_response20 import train_model as api,train_design as d,response
    rng=np.random.default_rng(24);predicted=rng.normal(size=(18,5,8));predicted-=predicted.mean(1,keepdims=True)
    api._bank=response.bank(predicted)
    from experiments.cvs_mixed_iq20.train_checks import run as original_check
    result=original_check()
    from experiments.cvs_mixed_iq20.train_protocol_checks import run as protocol_check
    negatives=protocol_check()
    import torch
    c=d.config(d.rows()[0]);m=api.build(c,'cpu');ema=copy.deepcopy(m)
    assert c['baseline_reference'].startswith('/home/') and '\\' not in c['baseline_reference']
    assert ema._response_base_hook.__self__ is ema and ema._source_labeled_feature_hook.__self__ is ema
    x=torch.randn(5,2,256);meta=dict(rx_i=[1,3,4,6,8],day_i=[1]*5,eq_i=[1]*5)
    empirical,_=api.generalization(x,meta,1,kind='empirical')
    rotated,_=api.generalization(x,meta,1,kind='rotated')
    assert not torch.equal(empirical,x) and not torch.equal(rotated,x) and not torch.equal(empirical,rotated)
    assert all('TX' not in v for v in api._pending)
    api._pending=None
    from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS
    raw=d.previous.config(next(v for v in d.previous.rows() if v['arm']=='B00'))
    a=d.previous.make_args(raw,'cpu');b=d.make_args(c,'cpu')
    assert all(getattr(a,k)==getattr(b,k) for k in RECIPE_FIELDS if k not in ('run_id','candidate_id','output_dir'))
    assert c['training_mechanism']['slot_weights']==[.25,.25,.5] and not c['training_mechanism']['A_s']
    result.update(protocol_negatives=negatives['negative_cases'],paired_directions_changed_IQ=True,
        source_parameter_TX_hidden=True,response_EMA_binding=True,raw_control_recipe_match=True,
        additional_operator_packets=10,new_models_trained=0)
    print('SOURCE_RESPONSE_NATIVE_CHECKS '+json.dumps(result));return result


if __name__=='__main__':run()
