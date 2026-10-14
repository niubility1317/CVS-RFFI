import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from . import design as d
from .objective import angle, stats
from experiments.cvs_backbone_fishr50 import predict as api
from experiments.cvs_backbone_fishr50.objective import calculate, variance_terms
from experiments.cvs_source_layer_order.probe import cohorts
from experiments.cvs_equivariant_identity.precision import numerical_context


def summarize(raw, *, design=d):
    result = []
    for layer, name in enumerate(design.PARAMETERS):
        main = torch.stack([r['vectors']['CE'][layer].double() for r in raw]).mean(0)
        row = dict(parameter=name, cohorts=len(raw))
        for key in ('RAW', 'ANGLE'):
            other = torch.stack([r['vectors'][key][layer].double() for r in raw]).mean(0)
            row[key] = dict(global_gradient=stats(main, other),
                           nonzero_cohorts=sum(r['vectors'][key][layer].norm() > 1e-12 for r in raw))
            row[key]['nonzero_cohorts'] = int(row[key]['nonzero_cohorts'])
        result.append(row)
    return result


def qualification(summary, *, design=d):
    passed = len(summary) == 2 and all(
        r['cohorts'] == 5 and r['ANGLE']['nonzero_cohorts'] == 5
        and r['ANGLE']['global_gradient']['CE_norm'] > 1e-12
        and r['ANGLE']['global_gradient']['auxiliary_norm'] > 1e-12
        and r['ANGLE']['global_gradient']['cosine'] >= 0 for r in summary)
    return dict(status='PASS' if passed else 'FAIL', fixed_rules=design.QUALIFICATION,
                source_only=True, no_automatic_training_release=True,
                target_metrics_used=False, sample_scope='Fixed first five L180 cohorts only')


def recount(root, *, design=d, row=None):
    row = row or design.rows()[0]
    if row not in design.rows():
        raise ValueError('Unregistered source recount row')
    expected_config = design.config(row)
    row = root / row['row_id']
    report = design.read(row / 'analysis.json')
    design.validate(report['config'])
    if report['config'] != expected_config:
        raise ValueError('Saved source model differs from requested row')
    raw = torch.load(row / 'actual_source_gradients.pt', map_location='cpu', weights_only=False)
    if len(raw) != 5 or [r['cohort'] for r in raw] != list(range(5)):
        raise ValueError('Incomplete fixed source diagnostic')
    ids = []
    for r in raw:
        ids.extend(r['source_physical_ids'])
        losses = {'RAW': [], 'ANGLE': []}
        for k, G in enumerate(r['parameter_gradient_matrices']):
            if G.shape != (180, (3456, 6912)[k]):
                raise ValueError('Actual derivative object differs')
            losses['RAW'].append(float(variance_terms(G, r['y'], r['rx'], r['day'])[0]['RAW']))
            losses['ANGLE'].append(float(angle(G, r['y'], r['rx'], r['day'], r['source_physical_ids'])))
            if not torch.equal(G.mean(0), r['vectors']['CE'][k]):
                raise ValueError('Same-packet CE gradient differs')
        for key, values in losses.items():
            if abs(sum(values) / 2 - r['losses'][key]) > 1e-8:
                raise ValueError('Saved source loss differs')
    if len(ids) != 900 or len(set(ids)) != 900 or summarize(raw, design=design) != report['gradient_summary']:
        raise ValueError('Saved source vectors or physical coverage differ')
    if qualification(summarize(raw, design=design), design=design) != report['qualification']:
        raise ValueError('Saved source decision differs')
    return dict(status='VERIFIED', source_packets=900, actual_matrices=10,
                gradient_batches=5, layer_records=10, saved_vectors=30,
                model_forward=0, query_read=False, head_updates=0, optimizer_steps=0)


def measure(c, L, *, design=d, model_api=api):
    """One immutable row; numerical objective and observation rules are unchanged."""
    d = design
    d.validate(c)
    out = Path(c['output_root']); out.mkdir()
    started = time.time()
    try:
        model_api.source_proof(c['source_config'])
        if len(L['y']) != 6300:
            raise ValueError('Source role differs')
        old = cohorts(L)
        indices = [torch.cat((old[2 * j], old[2 * j + 1])) for j in range(5)]
        with numerical_context(d.FULL_FP32_POLICY), torch.random.fork_rng(devices=[]):
            model = model_api.own_model(c['source_config'], 'cpu').eval().requires_grad_(False)
            named = dict(model.named_parameters()); params = [named[n] for n in d.PARAMETERS]
            for p in params:
                p.requires_grad_(True)
            head = model.id_backbone.encoder.core.id_backbone.cls_head.weight
            head.requires_grad_(True)  # Only to test absence from the auxiliary graph.
            before = {k: v.clone() for k, v in model.state_dict().items()}
            raw = []
            d.write(out / 'resolved_config.json', dict(config=c,
                execution_commit=(d.ROOT / 'release_commit.txt').read_text().strip(),
                actual_checkpoint=(Path(c['source_config']['output_root']) / 'final_ssdg.pth').as_posix(),
                source_roles='EXACT_MATCH', query_read=False, optimizer_steps=0))
            for rank, ix in enumerate(indices):
                y, rx, day = (L[k][ix] for k in ('y', 'rx', 'day'))
                ids = [L['ids'][i] for i in ix.tolist()]
                old_losses, detail, matrices = calculate(model, L['x'][ix], y, rx, day, ids)
                angular = torch.stack([angle(G, y, rx, day, ids) for G in matrices]).mean()
                targets = dict(RAW=old_losses['RAW'], ANGLE=angular)
                vectors = dict(CE=[G.detach().mean(0).cpu() for G in matrices])
                for key, loss in targets.items():
                    if torch.autograd.grad(loss, head, allow_unused=True, retain_graph=True)[0] is not None:
                        raise ValueError('Auxiliary depends on class W')
                    gg = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
                    vectors[key] = [g.detach().flatten().cpu() for g in gg]
                    if any(not torch.isfinite(g).all() for g in vectors[key]):
                        raise ValueError('Nonfinite actual source derivative')
                raw.append(dict(cohort=rank, y=y.cpu(), rx=rx.cpu(), day=day.cpu(),
                    source_physical_ids=ids, parameter_gradient_matrices=[g.detach().cpu() for g in matrices],
                    vectors=vectors, losses={k: float(v.detach()) for k, v in targets.items()},
                    source_CE_mean=detail['source_CE_mean']))
                del matrices, targets, old_losses, angular, loss, gg
                print(json.dumps(dict(source_cohorts=rank + 1, source_packets=(rank + 1) * 180)), flush=True)
            if any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()) or any(p.grad is not None for p in model.parameters()):
                raise ValueError('Read-only source diagnostic mutated model state or accumulated gradients')
        summary = summarize(raw, design=d)
        report = dict(status='ANALYZED', config=c, source_packets=900,
            gradient_summary=summary, qualification=qualification(summary, design=d),
            cohort_losses=[dict(cohort=r['cohort'], **r['losses'], source_CE_mean=r['source_CE_mean']) for r in raw],
            model_parameters_buffers_unchanged=True, new_model_parameters=0,
            head_updates=0, optimizer_steps=0, query_read=False, U_truth_read=False,
            V_forward=False, seconds=time.time() - started,
            limitation='One frozen terminal model and fixed900-L subset; no training, U, target, generalization or new-TX benefit claim')
        torch.save(raw, out / 'actual_source_gradients.pt')
        d.write(out / 'analysis.json', report)
        return report
    except Exception as exc:
        d.write(d.BASE / 'failure.json', dict(status='FAILED', error=repr(exc),
                 traceback=traceback.format_exc(), no_automatic_retry=True, outputs_preserved=True))
        raise


def main():
    from experiments.cvs_nocfo_foundation.source import load_source
    from experiments.cvs_nocfo_foundation import design as nd
    torch.set_num_threads(2)
    c = d.config(d.rows()[0]); d.validate(c)
    spec = d.read(d.ROOT / 'registration/experiment.json')
    if spec['run_id'] != d.RUN or [r['config'] for r in spec['rows']] != [c]:
        raise ValueError('Registered source diagnostic differs')
    if d.BASE.exists():
        raise FileExistsError('Immutable output exists; reconcile instead of retry')
    d.BASE.mkdir(parents=True)
    try:
        L, U, V, contract = load_source(nd.config(nd.rows()[0]))
        if U is not None:
            raise ValueError('Unlabelled tensor materialized')
        del V
        report = measure(c, L)
        subprocess.run([sys.executable, '-u', '-m', 'experiments.cvs_source_variance_angle.audit',
                        '--recount', str(d.BASE)], check=True)
        d.write(d.BASE / 'source_qualification.json', report['qualification'])
        d.write(Path(c['output_root']) / 'completion.json', dict(status='ANALYZED', source_packets=900, source_only=True))
        d.write(d.BASE / 'analysis.json', report)
        done = dict(status='ANALYZED', models=1, source_packets=900, forward_packets=900,
                    gradient_batches=5, layer_records=10, saved_vectors=30,
                    source_only=True, query_read=False, U_truth_read=False, head_updates=0, optimizer_steps=0)
        d.write(d.BASE / 'completion.json', done)
        print(json.dumps(done), flush=True)
    except Exception as exc:
        d.write(d.BASE / 'failure.json', dict(status='FAILED', error=repr(exc),
                 traceback=traceback.format_exc(), no_automatic_retry=True, outputs_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--recount', type=Path)
    args = parser.parse_args()
    if args.recount:
        torch.set_num_threads(2)
        result = recount(args.recount)
        d.write(args.recount / 'independent_artifact_recount.json', result)
        print(json.dumps(result))
    else:
        main()
