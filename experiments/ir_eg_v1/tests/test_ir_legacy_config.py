from pathlib import Path
from copy import deepcopy
import json
import subprocess
import sys
import pytest
from cvsrffi.xuc_fusion.response_config import validate_prepared_row,IR_DEFAULT_KEYS


def saved_row():
    path=Path(__file__).resolve().parents[1]/'configs/separate_controls/PURE_EG_s392005.json'
    return json.loads(path.read_text(encoding='utf-8'))['row']


def test_old_pure_EG_readonly_validation_preserves_baseline(tmp_path):
    saved=saved_row();before=deepcopy(saved)
    row=validate_prepared_row(saved)
    assert saved==before and row['joint']==saved['joint']
    assert row['response']['method']=='EG'
    assert set(row['response'])-set(saved['response'])==set(IR_DEFAULT_KEYS)
    assert list(tmp_path.iterdir())==[]


def test_old_config_actual_launcher_only_validates(tmp_path):
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/'must_not_be_created'
    process=subprocess.run([sys.executable,str(root/'code/scripts/train_response_games.py'),
        '--config',str(root/'configs/separate_controls/PURE_EG_s392005.json'),
        '--output',str(output),'--device','cpu'],capture_output=True,text=True,check=True)
    record=json.loads(process.stdout)
    assert record['status']=='VALIDATED_NOT_LAUNCHED'
    assert record['resolved']['ir_import_audit'] and not output.exists()


@pytest.mark.parametrize('kind',['native','unknown'])
def test_old_config_normalization_still_rejects_unapproved_edits(kind):
    row=saved_row()
    if kind=='native':row['joint']['lr']=.123
    else:row['response']['typo']=True
    with pytest.raises(ValueError):validate_prepared_row(row)
