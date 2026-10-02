"""Narrow source-contract repair tests; all metadata below is literal synthetic."""
import ast
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import score_d92_group_barrier_joint_benchmark_contract_r02 as score


def producer_contract():
    """Resolve the native branch declaration without importing any exporter."""
    text=(ROOT/'tools/export_d92_branch_support_features.py').read_text(encoding='utf-8')
    tree=ast.parse(text)
    assignments={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name)}
    branches=ast.literal_eval(assignments['BRANCHES'])
    call=assignments['FEATURE_CONTRACT']
    assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='dict'
    branch_expr=next(v.value for v in call.keywords if v.arg=='branch_keys')
    assert isinstance(branch_expr,ast.Call) and isinstance(branch_expr.func,ast.Name)
    assert branch_expr.func.id=='list' and len(branch_expr.args)==1 and not branch_expr.keywords
    assert isinstance(branch_expr.args[0],ast.Name) and branch_expr.args[0].id=='BRANCHES'
    return {v.arg:list(branches) if v.arg=='branch_keys' else ast.literal_eval(v.value) for v in call.keywords}


def test_exact_clone_has_only_the_native_branch_list_correction():
    original=(ROOT/'tools/score_d92_group_barrier_joint_benchmark.py').read_bytes()
    repaired=(ROOT/'tools/score_d92_group_barrier_joint_benchmark_contract_r02.py').read_bytes()
    old=b"branch_keys=['z_id','fft','t_emb','f_emb','pa_local']"
    new=b"branch_keys=['z_id','t_emb','f_emb','pa_local']"
    assert original.count(old)==1
    assert repaired==original.replace(old,new)
    ast.parse(repaired.decode('utf-8',errors='strict'))


def test_contract_is_exact_producer_literal_with_fft_separately_declared():
    contract=producer_contract()
    assert score.RAW_FEATURE_CONTRACT==contract
    assert contract['branch_keys']==['z_id','t_emb','f_emb','pa_local']
    assert contract['fft_dim']==96 and contract['fft']=='historical_spectral_logmag_sketch'
    assert contract['fft_norm_floor']==1e-8
    # Follow both actual import links as source, without native/model imports.
    links=[('tools/evaluate_d92_group_barrier_joint_benchmark.py','export_d92_branch_features'),
           ('tools/export_d92_branch_features.py','export_d92_branch_support_features')]
    for path,module in links:
        tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
        assert any(isinstance(n,ast.ImportFrom) and n.module==module and
            any(a.name=='FEATURE_CONTRACT' and a.asname is None for a in n.names) for n in tree.body)


def source_metadata(tmp_path):
    expected=dict(checkpoint_sha256='a'*64,model_seed=17)
    classes=['old_'+str(i) for i in range(6)]
    source=dict(expected,source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',source_role_comparison='EXACT_MATCH',
        checkpoint_epoch=200,checkpoint_inheritance=[],target_access_before_freeze=False,
        cache_schema='d92_branch_received_features_v1',feature_contract=producer_contract())
    ground=dict(expected,path=str(tmp_path/'not-read-ground-packet'),ordered_classes=classes,
        source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',scale=7.,norm_eps=1e-4,packet_total_file_bytes=123,
        feature_contract=dict(checkpoint_sha256=expected['checkpoint_sha256'],cache_key='z_id',
            source_tensor='feat_joint',representation='raw',dtype='float32',feature_dim=160))
    complete=dict(source_identity=source,ground_packet_identity=ground,ordered_ground_classes=classes,
        run_binding=dict(ground_packet=ground['path']))
    return deepcopy(complete),complete,expected


def forbid_artifact_read(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Source contract validation must not read truth, predictions, packets or features')
    monkeypatch.setattr(score,'read',forbidden)
    monkeypatch.setattr(score.np,'load',forbidden)
    monkeypatch.setattr(Path,'open',forbidden)


def test_valid_literal_source_contract_accepts_without_any_artifact_or_truth_read(tmp_path,monkeypatch):
    startup,complete,expected=source_metadata(tmp_path)
    forbid_artifact_read(monkeypatch)
    assert score._source_identity(startup,complete,expected) is None


@pytest.mark.parametrize('fault',['legacy_five_branches','branch_order','fft_dim','fft_name',
    'fft_norm_floor','dtype','missing_field','extra_field'])
def test_strict_source_contract_rejects_coherently_wrong_fields_before_truth(tmp_path,monkeypatch,fault):
    startup,complete,expected=source_metadata(tmp_path)
    contract=complete['source_identity']['feature_contract']
    if fault=='legacy_five_branches':contract['branch_keys'].insert(1,'fft')
    elif fault=='branch_order':contract['branch_keys'].reverse()
    elif fault=='fft_dim':contract['fft_dim']=160
    elif fault=='fft_name':contract['fft']='different_spectral_transform'
    elif fault=='fft_norm_floor':contract['fft_norm_floor']=1e-7
    elif fault=='dtype':contract['cache_dtype']='float64'
    elif fault=='missing_field':contract.pop('normalization')
    else:contract['alias']='raw'
    # Both metadata copies agree. Rejection must come from the exact producer
    # contract comparison, not the preceding startup/complete equality check.
    startup['source_identity']=deepcopy(complete['source_identity'])
    forbid_artifact_read(monkeypatch)
    with pytest.raises(ValueError,match='Wrong current five-branch raw feature source contract'):
        score._source_identity(startup,complete,expected)
