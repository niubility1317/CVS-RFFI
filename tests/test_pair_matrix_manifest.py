import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from pair_matrix_manifest import build_manifest


def test_frozen_matrix_balanced_and_source_only():
    saved=json.loads((ROOT/'configs/phase1_adv3b02_pair24_manifest.json').read_text(encoding='utf-8'))
    generated=build_manifest(**{k:saved[k] for k in ('code_root','output_root','python','dataset','checkpoint')})
    assert generated == saved
    rows=saved['rows']
    assert len(rows)==len({r['row_id'] for r in rows})==24
    assert Counter(r['assigned_gpu'] for r in rows)==dict.fromkeys(range(8),3)
    assert set(Counter(r['candidate'] for r in rows).values())=={3}
    assert saved['max_active']==24 and saved['max_new_per_gpu']==3
    for row in rows:
        argv=row['argv']
        value=lambda flag:argv[argv.index('--'+flag)+1]
        assert value('epochs')=='200'
        assert value('muse_external_final_eval')=='true'
        assert value('phase1_source_val_selection_only')=='true'
        assert value('checkpoint_selection')=='final_only'
        assert '\\' not in argv[2]
        assert row['epochs']==200


def test_recovery_preserves_science_and_combined_gpu_balance():
    from pair_recovery_manifest import build_recovery, FAILED
    old=json.loads((ROOT/'configs/phase1_adv3b02_pair24_manifest.json').read_text(encoding='utf-8'))
    new=json.loads((ROOT/'configs/phase1_adv3b02_pair12_recovery_manifest.json').read_text(encoding='utf-8'))
    assert build_recovery(old)==new
    assert len(new['rows'])==12
    for row in new['rows']:
        original=next(r for r in old['rows'] if r['row_id']==row['row_id'])
        assert row['argv']==[v.replace(old['run_id'],new['run_id']) for v in original['argv']]
        assert row['assigned_gpu']==original['assigned_gpu']
    healthy=[r for r in old['rows'] if r['candidate'] not in FAILED]
    assert len(healthy)==12
    assert Counter(r['assigned_gpu'] for r in healthy+new['rows'])==dict.fromkeys(range(8),3)
    assert '--amp' in new['smoke_argv'] and '--training-manifest' in new['smoke_argv']


def test_safe_recovery_only_replaces_three_failed_rows():
    from pair_recovery_manifest import build_safe_recovery
    old=json.loads((ROOT/'configs/phase1_adv3b02_pair12_recovery_manifest.json').read_text(encoding='utf-8'))
    new=json.loads((ROOT/'configs/phase1_adv3b02_safe3_recovery_manifest.json').read_text(encoding='utf-8'))
    assert build_safe_recovery(old)==new
    assert len(new['rows'])==3 and {r['candidate'] for r in new['rows']}=={'B_SAFE'}
    for row in new['rows']:
        original=next(r for r in old['rows'] if r['row_id']==row['row_id'])
        assert row['argv']==[v.replace(old['run_id'],new['run_id']) for v in original['argv']]
        assert row['assigned_gpu']==original['assigned_gpu']
    assert 'NO_PROMOTION' in new['selection']['comparison_boundary']
