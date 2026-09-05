import importlib.util
import math
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from SSDG.train_ssdg import build_arg_parser, _validate_daot_config

SPEC = importlib.util.spec_from_file_location('pair_dry', Path(__file__).resolve().parents[1]/'tools/pair_reform_dry_run.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def commands(rows=None):
    return MODULE.build_commands(root='E:/example', python='python.exe', dataset='E:/data.pkl',
        checkpoint='E:/baseline.pth', output_root='E:/pending', rows=rows)['commands']


def test_all_generated_rows_parse_and_validate_without_launch():
    rows = commands()
    assert len(rows) == 7
    for row in rows:
        args = build_arg_parser().parse_args(row['argv'][3:])
        _validate_daot_config(args)
        assert args.use_muse_ssdg and args.muse_level == 'M1'
        assert args.representation_mode == 'dual'
        assert args.epochs == 200 and args.seed == 392005
        assert args.sat_training_mode == 'concat_masked'
        assert args.sat_cons_start_epoch <= args.pair_start_epoch
        assert args.pair_gradient_projection and not args.fasttrust_rc4
        assert args.phase1_source_val_selection_only
        assert row['status'] == 'PENDING_NOT_RUN'


def test_matrix_is_single_factor_and_alias_does_not_restore_a0():
    parsed = {r['row']: vars(build_arg_parser().parse_args(r['argv'][3:])) for r in commands()}
    base = parsed['M0']
    ignore = {'output_dir', 'candidate_id', 'phase2_export_path'}
    expected = {'M1': {'pair_tangent_weight'}, 'M2': {'pair_route_weight'},
        'M3': {'pair_tangent_weight','pair_route_weight'}, 'B_SAFE': {'pair_reform'},
        'POINT_MEMORY': {'pair_memory'}, 'ASYMMETRIC': {'pair_reform'}}
    for row, keys in expected.items():
        differences = {k for k in base if base[k] != parsed[row][k]
            and not (isinstance(base[k], float) and isinstance(parsed[row][k], float)
                     and math.isnan(base[k]) and math.isnan(parsed[row][k]))}
        assert differences - ignore == keys
    assert commands(['A_POINT'])[0]['row'] == 'M0'
    with pytest.raises(ValueError):
        commands(['A0'])


def test_cli_only_prints_commands_and_never_creates_run_directory(tmp_path):
    output_root = tmp_path/'not_created'
    result = subprocess.run([sys.executable, '-X', 'utf8', str(MODULE.ROOT/'tools/pair_reform_dry_run.py'),
        '--root', str(MODULE.ROOT), '--python', sys.executable, '--dataset', 'missing-data.pkl',
        '--checkpoint', 'missing-checkpoint.pth', '--output-root', str(output_root), '--rows', 'B_SAFE'],
        capture_output=True, text=True, encoding='utf-8', check=True)
    payload = json.loads(result.stdout)
    assert payload['status'] == 'PENDING_NOT_RUN'
    assert payload['commands'][0]['row'] == 'B_SAFE'
    assert not output_root.exists()


@pytest.mark.parametrize('row', ['M0', 'M1', 'M2', 'M3', 'B_SAFE', 'POINT_MEMORY', 'ASYMMETRIC'])
def test_every_row_reaches_real_training_entrypoint_dry_run(row, tmp_path):
    output_root = tmp_path/'never_constructed'
    command = MODULE.build_commands(root=str(MODULE.ROOT), python=sys.executable,
        dataset=str(tmp_path/'missing.pkl'), checkpoint=str(tmp_path/'missing.pth'),
        output_root=str(output_root), rows=[row])['commands'][0]
    result = subprocess.run([*command['argv'], '--dry_run'], shell=False,
        cwd=command['cwd'], env={**os.environ, **command['environment'], 'PYTHONUTF8': '1'},
        capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '[DRY-RUN] Parsed arguments and skipped data/model construction.' in result.stdout
    assert not output_root.exists()
