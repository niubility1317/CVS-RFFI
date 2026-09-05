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
