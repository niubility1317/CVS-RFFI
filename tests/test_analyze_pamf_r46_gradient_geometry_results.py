import copy,json
import pytest
from experiments.pamf_r46_gradient_geometry import design as d
from tools.analyze_pamf_r46_gradient_geometry_results import audit_collection,METHOD


@pytest.fixture(scope='module')
def collection(tmp_path_factory):
    from _pytest.monkeypatch import MonkeyPatch
    from tests.test_pamf_r46_gradient_geometry import test_full_CPU_source_fixture_has_6930_inputs_and_324_summary_rows
    tmp=tmp_path_factory.mktemp('gradsource');patch=MonkeyPatch()
    try:test_full_CPU_source_fixture_has_6930_inputs_and_324_summary_rows(tmp,patch)
    finally:patch.undo()
    root=tmp/d.RUN;result=json.loads((root/'results.json').read_text(encoding='utf-8'));result['method_commit']=METHOD
    return dict(result=result,actual_owned_handle_live=False,failure=None,
        batch_records={a:{v:json.loads((root/a/(v+'_batches.json')).read_text(encoding='utf-8')) for v in d.VIEWS} for a in d.ARMS})


def test_independent_batch_arithmetic_matches_all324_aggregates(collection):
    assert len(audit_collection(collection)['summary_rows'])==324


@pytest.mark.parametrize('bad',['live','F','missingview','count','roles','dot','ratio','null','aggregate'])
def test_incomplete_or_inconsistent_source_gradient_evidence_rejected(collection,bad):
    c=copy.deepcopy(collection);row=c['batch_records'][d.ARMS[1]]['received01']['records'][0]
    if bad=='live':c['actual_owned_handle_live']=True
    elif bad=='F':c['result']['F_output_reads']=1
    elif bad=='missingview':c['batch_records'][d.ARMS[0]].pop('clean')
    elif bad=='count':c['result']['total_actual_G_IQ_inputs']-=1
    elif bad=='roles':row['L_s']=9
    elif bad=='dot':row['geometry']['whole']['pairs']['base_vs_moment_clean']['dot']+=.1
    elif bad=='ratio':row['geometry']['whole']['pairs']['base_vs_moment_clean']['norm_ratio']+=.1
    elif bad=='null':
        row=c['batch_records'][d.ARMS[0]]['clean']['records'][0]
        row['geometry']['whole']['pairs']['base_vs_moment_teacher'].update(valid=False,cosine=0.)
    else:c['result']['summary_rows'][0]['mean']+=1
    with pytest.raises(ValueError):audit_collection(c)
