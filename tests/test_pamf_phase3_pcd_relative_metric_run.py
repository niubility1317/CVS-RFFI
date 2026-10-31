"""Cached L metric fits keep original F0 identity and all future roles frozen."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_relative_metric_run')


def test_no_private_config_or_implicit_new_role():
    m=api()
    with pytest.raises(ValueError):m.validate_config(dict(run_id=m.RUN_ID,code_commit='b'*40,reference_config={},source_geometry_root='source',output_root='output',batch_size=128,truth='forbidden'))


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_prototype_identity_run import fixture as inherited
    with inherited(root,mp) as (parent,c,pm,b,model,_,meta,geo):
        m=api();c=dict(deepcopy(c),run_id=m.RUN_ID,output_root=str(root/'relative'))
        mp.setattr(m,'COUNTS',dict(parent.COUNTS));mp.setattr(m,'Q_COUNT',parent.Q_COUNT);mp.setattr(m,'SOURCE_COUNT',parent.SOURCE_COUNT)
        mp.setattr(m,'_backend',lambda *a:b);mp.setattr(m,'_load_model',lambda *a:(model,pm['model_identities']['TARGET_PCD']))
        events=[];source=m._source;rows=m._parent_rows;smoke=m._smoke
        def noquery(*a):events.append('smoke');return smoke(*a)
        def labelled(*a):
            assert events==['smoke'];events.append('L');return source(*a)
        def cached(config,key):
            out=Path(c['output_root']);assert (out/'models/complete.json').exists()
            if key!='source_V':assert (out/'calibration/complete.json').exists()
            events.append(key);return rows(config,key)
        mp.setattr(m,'_smoke',noquery);mp.setattr(m,'_source',labelled);mp.setattr(m,'_parent_rows',cached)
        mp.setattr(parent,'fit_prototypes',lambda *a:pytest.fail('No prototype identity fit'))
        yield m,c,events
