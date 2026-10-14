import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
import torch
from . import design as d
from experiments.cvs_source_variance_angle import audit as shared
from experiments.cvs_backbone_fishr_seeds import predict as model_api


def source_ids(root, row):
    raw = torch.load(root / row['row_id'] / 'actual_source_gradients.pt',
                     map_location='cpu', weights_only=False)
    ids = [i for cohort in raw for i in cohort['source_physical_ids']]
    if len(ids) != 900 or len(set(ids)) != 900:
        raise ValueError('Fixed source physical coverage differs')
    return ids


def combined(base):
    first = d.first.read(d.first.BASE / 'analysis.json')
    d.first.validate(first['config'])
    if first['status'] != 'ANALYZED' or first['qualification']['status'] != 'PASS':
        raise ValueError('Frozen first source qualification differs')
    first_ids = source_ids(d.first.BASE, d.first.rows()[0])
    reports = [first]
    for row in d.rows():
        report = d.read(base / row['row_id'] / 'analysis.json')
        d.validate(report['config'])
        if report['config'] != d.config(row):
            raise ValueError('Remaining source model attribution differs')
        if source_ids(base, row) != first_ids:
            raise ValueError('Deterministic paired source cohorts differ across model seeds')
        reports.append(report)
    return dict(status='ANALYZED', source_only=True,
        source_decision_commit=d.SOURCE_DECISION_COMMIT,
        model_seeds=[2026092701, *d.SEEDS], model_source_observations=2700,
        unique_source_physical_ids=900, identical_ordered_cohorts_across_models=True,
        first_seed_remeasured=False,
        rows=[dict(model_seed=r['config']['model_seed'], source_packets=r['source_packets'],
                   gradient_summary=r['gradient_summary'], qualification=r['qualification'],
                   model_parameters_buffers_unchanged=r['model_parameters_buffers_unchanged'])
              for r in reports],
        qualification=dict(status='PASS' if all(r['qualification']['status'] == 'PASS'
                                              for r in reports) else 'FAIL',
                           no_automatic_training_release=True,
                           target_metrics_used=False, fixed_per_model_rules=d.QUALIFICATION),
        training_epochs=0, head_updates=0, optimizer_steps=0, query_read=False,
        limitation='Three frozen terminal models on the same fixed900-L subset; not training or generalization evidence')


def main():
    from experiments.cvs_nocfo_foundation.source import load_source
    from experiments.cvs_nocfo_foundation import design as nd
    torch.set_num_threads(2)
    spec = d.read(d.ROOT / 'registration/experiment.json')
    if spec['run_id'] != d.RUN or [r['config'] for r in spec['rows']] != [d.config(r) for r in d.rows()]:
        raise ValueError('Registered remaining source diagnostic differs')
    if d.BASE.exists():
        raise FileExistsError('Immutable output exists; reconcile instead of retry')
    d.BASE.mkdir(parents=True)
    try:
        L, U, V, contract = load_source(nd.config(nd.rows()[0]))
        if U is not None:
            raise ValueError('Unlabelled tensor materialized')
        del V
        reports = []
        for row in d.rows():
            report = shared.measure(d.config(row), L, design=d, model_api=model_api)
            subprocess.run([sys.executable, '-u', '-m', 'experiments.cvs_source_variance_angle_seeds.audit',
                            '--recount', str(d.BASE), '--row', row['row_id']], check=True)
            d.write(d.BASE / row['row_id'] / 'completion.json',
                    dict(status='ANALYZED', source_packets=900, source_only=True))
            reports.append(report)
        evidence = combined(d.BASE)
        recounts = [d.read(d.BASE / r['row_id'] / 'independent_artifact_recount.json') for r in d.rows()]
        d.write(d.BASE / 'independent_artifact_recount.json', dict(status='VERIFIED',
            models=2, source_packets=sum(r['source_packets'] for r in recounts),
            actual_matrices=sum(r['actual_matrices'] for r in recounts),
            gradient_batches=10, layer_records=20, saved_vectors=60, model_forward=0,
            query_read=False, head_updates=0, optimizer_steps=0, rows=recounts))
        d.write(d.BASE / 'combined_three_seed_source_analysis.json', evidence)
        d.write(d.BASE / 'source_qualification.json', evidence['qualification'])
        d.write(d.BASE / 'analysis.json', dict(status='ANALYZED', rows=reports, combined=evidence))
        done = dict(status='ANALYZED', models=2, source_packets=1800, forward_packets=1800,
                    gradient_batches=10, layer_records=20, saved_vectors=60,
                    source_only=True, query_read=False, U_truth_read=False, head_updates=0, optimizer_steps=0)
        d.write(d.BASE / 'completion.json', done)
        print(json.dumps(done), flush=True)
    except Exception as exc:
        d.write(d.BASE / 'failure.json', dict(status='FAILED', error=repr(exc),
            traceback=traceback.format_exc(), no_automatic_retry=True, outputs_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recount', type=Path)
    parser.add_argument('--row')
    args = parser.parse_args()
    if args.recount:
        row = next((r for r in d.rows() if r['row_id'] == args.row), None)
        if row is None:
            raise ValueError('Exact registered source row required for recount')
        torch.set_num_threads(2)
        result = shared.recount(args.recount, design=d, row=row)
        d.write(args.recount / row['row_id'] / 'independent_artifact_recount.json', result)
        print(json.dumps(result), flush=True)
    elif args.row:
        parser.error('--row requires --recount')
    else:
        main()
