import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from experiments.cvs_all_frozen_clean_backfill import contracts as c
from experiments.cvs_all_frozen_clean_backfill import score as scoring
from experiments.cvs_all_frozen_clean_backfill import publish

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'experiments/cvs_all_frozen_clean_backfill/configs/launch_spec.json').read_text(encoding='utf-8'))

def local_read(path):
    rel=str(path).split('/releases/')[1].split('/',1)[1]
    return json.loads((ROOT/rel).read_text(encoding='utf-8'))

def test_complete_registered_matrix_has_every_four_seed_group():
    with patch.object(c,'read',side_effect=local_read):c.validate_spec(SPEC)
    assert len({r['source_run'] for r in SPEC['rows']})==35
    assert sum(r['reuse'] for r in SPEC['rows'])==88

@pytest.mark.parametrize('change',[
    {'views':['clean','leo_clear_weak']},{'frozen_epoch':199},{'p1_capsule':'other'},
    {'source_output':'/tmp/last'},{'output_root':'/tmp/out'},{'model_seed':42},
    {'authorization':'source_selected'},{'target_truth':'secret'},
])
def test_reject_changed_prediction_scope(change):
    row=next(r for r in SPEC['rows'] if not r['reuse']);cfg=local_read(row['config']);cfg.update(change)
    with pytest.raises(ValueError):c.validate_config(cfg)

@pytest.mark.parametrize('change',['missing','duplicate','seed','output','view'])
def test_reject_partial_or_mismatched_matrix(change):
    s=copy.deepcopy(SPEC)
    if change=='missing':s['rows'].pop()
    if change=='duplicate':s['rows'][1]=copy.deepcopy(s['rows'][0])
    if change=='seed':s['rows'][0]['model_seed']=42
    if change=='output':s['rows'][0]['output_root']='/tmp/wrong'
    if change=='view':s['views']=['satellite']
    with patch.object(c,'read',side_effect=local_read),pytest.raises(ValueError):c.validate_spec(s)

def test_incomplete_predictions_never_open_truth(tmp_path):
    spec=dict(SPEC,runtime_root=str(tmp_path),p1_truth='forbidden_truth')
    with patch.object(scoring,'preflight_predictions',side_effect=ValueError('missing prediction')),patch.object(scoring,'read') as read:
        with pytest.raises(ValueError):scoring.score(spec)
        read.assert_not_called()

def test_remote_release_scripts_compile():
    compile(publish.REMOTE,'remote','exec');compile(publish.INSPECT,'inspect','exec')

def test_all_missing_variants_build_finite_without_target():
    import torch
    from experiments.cvs_clean_eval.contracts import build_model
    torch.set_num_threads(2)
    for variant in sorted({r['variant'] for r in SPEC['rows'] if not r['reuse']}):
        model=build_model(variant).eval()
        with torch.no_grad():y=model(torch.zeros(2,2,256))
        assert y.shape==(2,6) and torch.isfinite(y).all(),variant
