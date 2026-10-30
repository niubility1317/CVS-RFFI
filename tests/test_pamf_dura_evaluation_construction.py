import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


class ConstructorReached(Exception):
    pass


@pytest.mark.parametrize('package',['pamf_dura_prior_refine','pamf_dura_direct_train'])
@pytest.mark.parametrize('entry',['predict','profile'])
def test_model_construction_preserves_parameter_version_counters(monkeypatch,tmp_path,package,entry):
    module=importlib.import_module('experiments.'+package+'.evaluation')
    def constructor(*args,**kwargs):
        # FrozenCenteredSinc caches its actual parameter version. Constructing
        # under inference_mode makes this exact operation fail on real weights.
        parameter=torch.ones(1)
        assert parameter._version==0
        raise ConstructorReached
    monkeypatch.setattr(module,'read_json',lambda p:dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_F_access=False,target_reads=0))
    monkeypatch.setattr(module,'reference',lambda p,r:(tmp_path,np.array(['a','b']),None,None))
    monkeypatch.setattr(module,'frozen',constructor)
    monkeypatch.setattr(module,'parent_model',constructor)
    monkeypatch.setitem(sys.modules,'experiments.pamf_stage2_inference.engine',SimpleNamespace(InferenceEngine=object))
    monkeypatch.setitem(sys.modules,'experiments.pamf_fcrossrx_readout.frozen',SimpleNamespace(load_reader=None))
    args=SimpleNamespace(arm='continuation',commit='a'*40)
    with pytest.raises(ConstructorReached):
        if entry=='predict':module.predict(args,dict(output=tmp_path),'source')
        else:module.profile(args,dict(output=tmp_path))
