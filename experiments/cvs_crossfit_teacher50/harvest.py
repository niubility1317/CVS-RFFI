"""Retrieve completed source-only teacher evidence without model or IQ access."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261015-phase1-crossfit-RX1-teachers-e50-manysig-s2026092701-m2-r01'
ARMS = ('SAMPLE', 'EXCLUDE_RX1')


def program(spec):
    if spec['run_id'] != RUN:
        raise ValueError('Expected the registered fixed RX1 source-only pilot')
    rows = spec['rows']
    if [r['config']['arm'] for r in rows] != list(ARMS):
        raise ValueError('Both registered teacher rows required')
    base = spec['execution']['remote_run_root']
    files = ['experiment.json', 'source_smoke.json', 'launch.json', 'complete.json',
             'resources.json', 'training_curves.json', 'analysis.md',
             'comparison/results.json', 'comparison/complete.json']
    for row in rows:
        prefix = row['row_id']
        files += [prefix+'/complete.json', prefix+'/source_evaluation/complete.json']
        files += [prefix+'/source/'+name for name in (
            'completion.json', 'initialization.json', 'actual_native_args.json',
            'training_mechanism_complete.json', 'teacher_coverage.json',
            'gradient_history.json', 'parameter_update_history.json',
            'inner_source_contract.json')]
    return '''import json
from pathlib import Path
base=Path(%r)
done=base/'complete.json'
if not done.is_file():
    print(json.dumps(dict(status='WAITING_FOR_MATRIX',no_partial_curve_claim=True)))
else:
    files={}
    for name in %r:
        p=base/name
        files[name]=p.read_text(encoding='utf-8') if p.suffix=='.md' else json.loads(p.read_text(encoding='utf-8'))
    print(json.dumps(dict(status='COMPLETE_SOURCE_EVIDENCE',files=files)))
''' % (base, files)


def validate(payload):
    if payload['status'] != 'COMPLETE_SOURCE_EVIDENCE':
        return False
    files = payload['files']
    done, result, resources = (files[n] for n in ('complete.json', 'comparison/results.json', 'resources.json'))
    if (done['status'] != 'ANALYZED' or not done['source_only'] or done['query_read']
            or done['actual_model_count'] != 2 or done['seed_count'] != 1
            or done['student_models_trained'] or done['automatic_bank_expansion'] or done['target_predictions']
            or result['status'] != 'ANALYZED' or not result['source_only'] or result['query_read']
            or result['U_truth_read'] or result['student_models_trained'] or result['head_fit_updates']
            or result['comparison_forward_packets'] != 1440 or result['teacher_evaluation_forward_packets'] != 10084
            or resources['teacher_training_updates'] != 22200 or len(resources['rows']) != 2):
        raise ValueError('Complete legal two-teacher source-only evidence required')
    for arm in ARMS:
        prefix = f'{arm}-s2026092701'
        trained = files[prefix+'/source/completion.json']
        mechanism = files[prefix+'/source/training_mechanism_complete.json']
        evaluated = files[prefix+'/source_evaluation/complete.json']
        cost = next(r for r in resources['rows'] if r['arm'] == arm)
        if (trained['epoch'] != 50 or trained['optimizer_steps'] != 11100 or trained['logged_steps'] != 11100
                or trained['target_access'] or trained['head_fit_updates']
                or mechanism['status'] != 'VERIFIED' or mechanism['unique_train_samples'] != 4464
                or mechanism['forward_packets']['train'] != 1420800 or mechanism['U_forward_packets']
                or mechanism['EMA_updates'] or mechanism['U_truth_read'] or mechanism['head_fit_updates']
                or cost['full_step_log_read'] != 11100
                or evaluated['status'] != 'SOURCE_PREDICTIONS_COMPLETE' or evaluated['forward_packets'] != 5042
                or not evaluated['model_state_unchanged'] or evaluated['query_read'] or evaluated['U_truth_read']):
            raise ValueError('Actual fixed E50 source training/evaluation differs')
        if arm == 'EXCLUDE_RX1' and 1 in mechanism['train_receivers']:
            raise ValueError('Held RX1 entered excluded teacher training')
    primary = next(r for r in result['records'] if r['view'] == 'clean' and r['region'] == 'RX1')
    if primary['count'] != 144 or primary['net'] != primary['repairs']-primary['new_errors']:
        raise ValueError('Matched RX1 source comparison incomplete')
    return True


def store(output, files):
    # Finished run artifacts are immutable; repeat retrieval may only reuse them.
    for name, value in files.items():
        path = output/name
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)+'\n'
        if path.exists() and path.read_text(encoding='utf-8-sig') != text:
            raise FileExistsError('Existing local evidence differs: '+str(path))
    for name, value in files.items():
        path = output/name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def run(output):
    spec = json.loads((ROOT/'experiments/cvs_crossfit_teacher50/registration.json').read_text(encoding='utf-8-sig'))
    module_spec = importlib.util.spec_from_file_location('teacher_harvest_operations', ROOT/'experiments/cvs_urban_sigtime/operations.py')
    ops = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(ops)
    payload = ops.remote(program(spec))
    if not validate(payload):
        print(json.dumps(payload))
        return payload
    store(output, payload['files'])
    result = payload['files']['comparison/results.json']
    print(json.dumps(dict(status=payload['status'],run_id=RUN,output=str(output),
        source_rule_passed=result['source_decision']['passed'],full_steps=22200,
        source_only=True,target_predictions=0)))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
