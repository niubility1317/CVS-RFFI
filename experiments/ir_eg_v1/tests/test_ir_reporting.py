import json
from cvsrffi.game_tracking.solvers import StepResult
from cvsrffi.xuc_fusion.ir_telemetry import step_record, TelemetryWriter


def test_missing_metrics_are_null(tmp_path):
    row = step_record(StepResult(1., 2., 3., 'extragradient', True, 2, 'IR_EG'), accepted_index=1)
    assert row['hp_norm'] is None and row['missing_reasons']['hp_norm']
    path = tmp_path/'steps.jsonl'; TelemetryWriter(path).write(row)
    assert json.loads(path.read_text(encoding='utf-8')) == row


def test_executed_not_equal_configured():
    result = StepResult(1., 2., 3., 'extragradient', True, 2, 'IR_EG', telemetry={'configured':True,'eligible':False,'executed':False,'applied':False,'nonzero':False})
    row = step_record(result)
    assert row['configured'] and not row['executed'] and row['accepted']


def test_actual_clip_displacement_logged():
    result = StepResult(1., 2., 3., 'extragradient', True, 2, 'IR_EG')
    row = step_record(result, origin_clip_coefficient=.2, formal_clip_coefficient=.7, parameter_displacement_norm=.125)
    assert row['parameter_displacement_norm'] == .125 and row['formal_clip_coefficient'] == .7
