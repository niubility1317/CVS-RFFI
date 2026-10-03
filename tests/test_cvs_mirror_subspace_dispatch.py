"""Fixed source universe and deployment boundary; all inputs synthetic."""
import copy
import pytest
from experiments.cvs_mirror_subspace_identity import dispatch as d
from experiments.cvs_mirror_subspace_identity import publish as p


def records():
    return [dict(variant=v,seed=s,accuracy=.9,worst_rx=.8,parameters=247731,macs=9000000)
            for v in d.CANDIDATES for s in d.SEEDS]


def test_twenty_records_and_twelve_metadata_controls():
    rows=d.control_rows()
    assert len(rows)==12 and len(d.CANDIDATES)==5
    assert {r['method'] for r in rows}=={'cvs_neural_residual_identity','cvs_response_fusion_identity','cvs_spectral_relation_identity'}
    result=d.select_source_candidate(records())
    assert result['selected_variant']==d.CONTROL and not result['new_candidate_selected']
    assert result['target_access'] is result['target_score_used'] is False


@pytest.mark.parametrize('winner',d.CANDIDATES)
def test_source_score_wins_regardless_of_higher_cost(winner):
    r=records()
    for row in r:
        if row['variant']==winner:row.update(accuracy=.91,macs=99999999,parameters=9999999)
    result=d.select_source_candidate(r)
    assert result['selected_variant']==winner
    assert result['new_candidate_selected']==(winner in d.VARIANTS)


@pytest.mark.parametrize('kind',['missing','duplicate','nan','foreign','badcost'])
def test_incomplete_or_invalid_source_cannot_freeze(kind):
    r=records()
    if kind=='missing':r.pop()
    elif kind=='duplicate':r[-1]=copy.deepcopy(r[-2])
    elif kind=='nan':r[-1]['accuracy']=float('nan')
    elif kind=='foreign':r[-1]['variant']='not_registered'
    else:r[-1]['macs']=0
    with pytest.raises(ValueError):d.select_source_candidate(r)


def test_spectral_control_uses_its_original_source_reader(monkeypatch):
    from experiments.cvs_spectral_relation_identity import dispatch as old
    seen=[]
    monkeypatch.setattr(old,'read_source_record',lambda *args:seen.append(args) or {'ok':True})
    row=next(r for r in d.control_rows() if r['variant']==d.SPECTRAL_CONTROL)
    assert d.read_source_record(row,{'roles':'original'},row['method'])=={'ok':True}
    assert seen==[(row,{'roles':'original'},'cvs_spectral_relation_identity')]


def test_remote_script_compiles_without_running():
    cfg=dict(project=d.PROJECT,release=p.RELEASE,run=p.RUN,archive=p.RELEASE+'.tar.gz',sha256='x',commit='x')
    compile(p.REMOTE.replace('c=CONFIG','c='+repr(cfg),1),'remote','exec')
    compile(p.PREFLIGHT.replace('c=CONFIG','c='+repr(cfg),1),'preflight','exec')
    assert 'cvs_mirror_subspace_identity.cpu_smoke' in p.REMOTE
    assert 'cvs_spectral_relation_identity' in p.REMOTE
