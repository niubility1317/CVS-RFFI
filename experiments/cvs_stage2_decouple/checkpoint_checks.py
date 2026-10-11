"""Cold-process regression for native objects embedded in historical checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def child(mode, folder):
    # Deliberately do not call make_args, installed, or native_modules before the
    # common entry point. This matches a fresh preparation worker on N607.
    import torch
    from . import common, design as d
    torch.set_num_threads(2)
    native_path = str(d.ROOT / 'experiments/adv3b02_xuc/code')
    assert native_path not in sys.path
    assert 'baseline_origin_sat_view' not in sys.modules
    path = folder / 'parent.pth'
    if mode == 'raw':
        try:
            torch.load(path, map_location='cpu', weights_only=False)
        except ModuleNotFoundError as error:
            assert error.name == 'baseline_origin_sat_view'
            print(json.dumps(dict(status='EXPECTED_FAILURE', missing_module=error.name, initially_cold=True)))
            return
        raise AssertionError('Negative control unexpectedly loaded the native object without registration')
    config = d.read(folder / 'config.json')
    if mode == 'verify':
        # Only fixture locations are substituted. All verify_parent provenance
        # comparisons, checkpoint deserialization, and args comparisons run intact.
        d.config = lambda row: config
        d.SOURCE = str(folder / 'source_contract.json')
        result = common.verify_parent(config)
        assert result['status'] == 'VERIFIED'
        print(json.dumps(dict(status='PASS', mode='cold_verify_parent', initially_cold=True,
                              parent_epoch=result['epoch'], native_object_deserialized=True)))
        return
    registered = d.config(d.rows()[0])
    model = common.build_model(registered, torch.device('cpu'), checkpoint=path)
    assert 'baseline_origin_sat_view' in sys.modules
    assert type(model._stage2_baseline_args['_cold_checkpoint_fixture']).__name__ == 'SatViewStage'
    model.eval()
    with torch.no_grad():
        actual = common.identity_logits(model, torch.zeros(2, 2, 256))
    expected = torch.load(folder / 'expected.pt', map_location='cpu', weights_only=True)
    assert torch.equal(actual, expected['logits'])
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in expected['state'].items())
    print(json.dumps(dict(status='PASS', mode='cold_build_model', initially_cold=True,
        strict_state_identical=True, no_query_logits_identical=True, native_object_deserialized=True)))


def run():
    import torch
    from . import design as d
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_feature_disentangle.source import clean
    torch.set_num_threads(2)
    c = d.config(d.rows()[0])
    a = d.make_args(c, 'cpu')
    a.num_domains, a.input_len = 5, 256
    a.from_scratch = a.a1_scratch_only = True
    a.baseline_ckpt = a.teacher_ckpt = ''
    a.epochs = a.label_epochs = 200
    a.pseudo_epochs = 0
    from baseline_origin_sat_view import SatViewStage
    a._cold_checkpoint_fixture = SatViewStage(1, ('unused_serialization_fixture',), 0.)
    with installed(c) as native:
        model = native.build_baseline_model(a, torch.device('cpu')).eval()
        from scripts.train_daot_rc4_baseline import resolved_config
        actual = clean(resolved_config(a))
    with torch.no_grad():
        logits = model(torch.zeros(2, 2, 256))
    with tempfile.TemporaryDirectory(prefix='stage2-checkpoint-cold-') as temp:
        folder = Path(temp)
        fixture_config = dict(c, parent_checkpoint=str(folder / 'parent.pth'),
                              source_contract=str(folder / 'source_contract.json'))
        old = dict(run_id=d.PARENT_ORIGIN, row_id='native-s' + str(c['model_seed']), arm='native',
            model_seed=c['model_seed'], checkpoint_sources=[], model_initialization='scratch', target_access=False)
        torch.save(dict(model=model.state_dict(), baseline_args=vars(a), args=vars(a), epoch=200,
            run_id=d.PARENT_ORIGIN, candidate_id=old['row_id'], checkpoint_selection='final_only',
            a1_ema_successful_updates=44400), folder / 'parent.pth')
        torch.save(dict(logits=logits, state=model.state_dict()), folder / 'expected.pt')
        d.write(folder / 'config.json', fixture_config)
        d.write(folder / 'source_contract.json', dict(native_role_comparison='EXACT_MATCH', fixture_only=True))
        d.write(folder / 'initialization.json', dict(scratch_only=True, checkpoint_sources=[], ancestors=[],
            target_contact=False, seed=c['model_seed'], source_roles='EXACT_MATCH'))
        d.write(folder / 'completion.json', dict(status='SOURCE_TRAINED', epoch=200, logged_steps=44400,
            optimizer_steps=44400, target_access=False,
            checkpoint_origin=dict(config=old, commit=d.PARENT_COMMIT, mode='post_training_diagnostic_recovery')))
        d.write(folder / 'resolved_config.json', dict(old, commit=d.PARENT_COMMIT))
        d.write(folder / 'resolved_native_args.json', actual)
        modes = []
        for mode in ('raw', 'build', 'verify'):
            command = [sys.executable, '-X', 'utf8', '-m', 'experiments.cvs_stage2_decouple.checkpoint_checks',
                       '--child', mode, '--fixture', str(folder)]
            # The remote immutable release does not include code's duplicate
            # baseline_origin_sat_view.py. Exclude that local-only fallback so
            # this regression cannot pass via an unrelated search path.
            env = dict(os.environ, PYTHONPATH=str(d.ROOT))
            result = subprocess.run(command, cwd=d.ROOT, env=env, capture_output=True, text=True,
                                    encoding='utf-8', check=False)
            if result.returncode:
                raise RuntimeError(f'Cold {mode} failed:\n{result.stdout}\n{result.stderr}')
            modes.append(json.loads(result.stdout.strip().splitlines()[-1]))
    return dict(status='PASS', checks=modes, source_data_read=False, target_access=False,
                scope='Synthetic real architecture checkpoint and strict provenance fixtures; independent fresh processes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=['raw', 'build', 'verify'])
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.fixture)
    else:
        result = run()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result))
