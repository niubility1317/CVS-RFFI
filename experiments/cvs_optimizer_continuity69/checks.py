"""Focused contract/source-rule cases and tensor-only AdamW continuity checks."""
import copy
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch
from . import design as d
from .protocol import final_metadata


def run(native_path=None):
    negative = 0
    def bad(fn):
        nonlocal negative
        try:
            fn()
        except (ValueError, TypeError, KeyError):
            negative += 1
        else:
            raise AssertionError('Invalid fixed optimizer KEEP contract accepted')
    assert tuple(r['arm'] for r in d.rows()) == ('B00', 'B01')
    parents = []
    for row in d.rows():
        c = d.config(row)
        d.validate(c)
        parents.append(c['parent_source'])
        assert c['As'] == (row['arm'] == 'B01')
        assert c['method'] == ('AS_KEEP' if c['As'] else 'NOAS_KEEP')
        expected, actual = d.old.make_args(d.old_config(row['arm']), 'cpu'), d.make_args(c, 'cpu')
        def equal(a, b):
            return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        differences = {k for k in set(vars(expected)) | set(vars(actual))
            if not equal(getattr(expected, k, None), getattr(actual, k, None))}
        assert differences <= {'run_id', 'candidate_id', 'output_dir'}
        assert actual.pseudo_temporal_mode == 'batch_neighbor' and actual.pseudo_temporal_gate
        for key, value in [('target_access', True), ('As', not c['As']), ('new_Ag', True),
            ('epochs', 20), ('updates', 4440), ('model_seed', 2026092702),
            ('parent_source', 'unknown'), ('label_epochs', 13), ('method', 'RESET')]:
            changed = copy.deepcopy(c)
            changed[key] = value
            bad(lambda changed=changed: d.validate(changed))
        changed = copy.deepcopy(c)
        changed['optimizer_keep']['old_LR_loaded'] = True
        bad(lambda: d.validate(changed))
        bad(lambda row=row: d.config(row, 20))
    assert len(set(parents)) == 2
    c = d.config(d.rows()[0])
    roles = {'L': ['toyL'], 'U': ['toyU'], 'V': ['toyV']}
    done = dict(config=c, epoch=10, optimizer_steps=2220, target_access=False, head_fit_updates=0)
    frozen = dict(config=c, selection='final_only', target_access=False)
    initial = dict(scratch_only=False, target_contact=False, checkpoint_sources=c['checkpoint_sources'], source_roles='EXACT_MATCH')
    final_metadata(c, done, frozen, initial, dict(role_ids=roles), roles)
    changed = dict(done, target_access=True)
    bad(lambda: final_metadata(c, changed, frozen, initial, dict(role_ids=roles), roles))
    from .optimizer_checks import run as optimizer_checks
    state_checks = optimizer_checks()
    from . import report
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        base, ref = root/'new', root/'runs'/d.REFERENCE_RUN
        fixed = {r['row_id']: d.config(r) for r in d.rows()}
        values = [('B00', False, .9, .4), ('B01', False, .899, .5),
                  ('B00', True, .91, .41), ('B01', True, .909, .505)]
        for arm, new, clean, rate in values:
            row = next(r for r in d.rows() if r['arm'] == arm)
            config = fixed[row['row_id']] if new else d.reference_config(arm)
            out = (base if new else ref)/row['row_id']
            d.write(out/'source/completion.json', dict(config=config, target_access=False,
                final_source_metrics=dict(source_val_count=27000, source_val_accuracy=clean)))
            d.write(out/'source_qualification/complete.json', dict(config=config, target_access=False,
                formal_scores_read=False, training_updates=0, NN=12600, source_accuracies=[[.8, rate, rate]]*2))
        with patch.object(d, 'BASE', base), patch.object(d, 'PROJECT', root), patch.object(d, 'config',
                side_effect=lambda row: fixed[row['row_id']]):
            result = report.source_decision()
            assert result['source_qualified'] is True and len(result['checks']) == 2
            assert result['candidate'] == 'AS_KEEP'
            json.dumps(result, allow_nan=False)
            (base/d.rows()[1]['row_id']/'source_qualification/complete.json').unlink()
            result = report.source_decision()
            assert result['status'] == 'SOURCE_READOUT_PARTIAL' and not result['source_qualified']
            assert result['formal_evaluation_always_required']
    return dict(status='PASS', protocol_negatives=negative, optimizer_checks=state_checks,
        source_decision_conditions=2, native_args_unchanged_except_identity=True,
        source_FAIL_still_all7=True, CVS_NN=0, IQ=0, CP=0, optimizer_updates=0,
        target_access=False, local_nn=d.LOCAL_NN, smoke_nn=d.SMOKE_NN)


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
