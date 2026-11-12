import pytest

from experiments.pamf_phase2_practical import clean_orthogonal_source as source


def test_source_preparation_rejects_target_and_existing_output(tmp_path):
    release=tmp_path/'release';release.mkdir()
    data=tmp_path/'source';data.mkdir()
    contract=tmp_path/'contract.json';contract.write_text('{}')
    root=tmp_path/'run'
    config=dict(run_root=str(root),release_root=str(release),source_cache=str(data),
                source_contract=str(contract),augmentation_cache=str(root/'augmentation'))
    assert source.check(config,2026092701)['target_reads']==0
    with pytest.raises(ValueError,match='target/query'):
        source.check(dict(config,data_root='forbidden'),2026092701)
    with pytest.raises(ValueError,match='seed'):
        source.check(config,1)
    output=root/'source/orthogonal-s2026092701';output.mkdir(parents=True)
    with pytest.raises(ValueError,match='already exists'):
        source.check(config,2026092701)
