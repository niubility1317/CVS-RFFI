"""Fixed-source counterfactual: set inclusion versus within-set logit margin."""
import argparse
import inspect
import itertools
import json
from pathlib import Path
import statistics
import numpy as np
from .source_confidence_metrics import probabilities
from .rx_mean_source import RUN, METHODS, SEEDS
from .rx_mean_reference_diagnostics import SCENES

METRICS = ('raw_precision_pct', 'set_inclusion_pct', 'raw_correct_H_wrong_pct',
           'wrong_conditional_target_mean', 'wrong_margin_logit_velocity_mean',
           'wrong_margin_increasing_pct', 'hard_raw_wrong_margin_logit_velocity_mean')


def set_statistics(raw, H, y, *, role):
    if role != 'V':
        raise ValueError('Visible frozen source V only; never U TX truth')
    p = probabilities(raw)
    q = probabilities(H)
    y = np.asarray(y)
    n = len(p)
    if q.shape != p.shape or y.shape != (n,) or y.dtype.kind not in 'iu' or y.min() < 0 or y.max() >= 6:
        raise ValueError('Aligned visible six-class source data required')
    index = np.arange(n)
    anchor, other = p.argmax(1), q.argmax(1)
    accepted = p.max(1) >= .95
    mask = np.eye(6, dtype=bool)[anchor] | np.eye(6, dtype=bool)[other]
    mass = (q * mask).sum(1)
    conditional = q * mask / mass[:, None]
    gradient = q - conditional
    hard_gradient = q - np.eye(6)[anchor]
    chosen = accepted & (anchor == y) & (other != y)
    # For a direct Euclidean step on logits, d(z_wrong-z_true)/d(learning_rate).
    velocity = gradient[index, y] - gradient[index, other]
    hard_velocity = hard_gradient[index, y] - hard_gradient[index, other]
    na, nc = int(accepted.sum()), int(chosen.sum())
    return dict(count=n, accepted=na, selected_raw_correct_H_wrong=nc,
                set_contains_truth=int((accepted & mask[index, y]).sum()),
                raw_precision_pct=100 * (anchor[accepted] == y[accepted]).mean() if na else None,
                set_inclusion_pct=100 * mask[index, y][accepted].mean() if na else None,
                raw_correct_H_wrong_pct=100 * nc / na if na else None,
                wrong_conditional_target_mean=float(conditional[index, other][chosen].mean()) if nc else None,
                wrong_margin_logit_velocity_mean=float(velocity[chosen].mean()) if nc else None,
                wrong_margin_increasing_pct=100 * (velocity[chosen] > 1e-12).mean() if nc else None,
                hard_raw_wrong_margin_logit_velocity_mean=float(hard_velocity[chosen].mean()) if nc else None)


def collection_script(rt):
    if rt['run_id'] != RUN:
        raise ValueError('Completed RXMean source artifacts only')
    script = 'import numpy as np\n' + inspect.getsource(probabilities) + '\n' + inspect.getsource(set_statistics) + r'''
import json
from pathlib import Path
rt=RUNTIME;root=Path(rt['run_root']);methods=METHODS;seeds=SEEDS;scenes=SCENES
d=json.loads((root/'complete.json').read_text())
if d['status']!='RX_MEAN_THREE_SEED_COMPLETE' or d['models']!=6 or d['methods']!=list(methods) or d['model_seeds']!=list(seeds):raise ValueError('Complete original matrix required')
rows=[];metadata=None;checks=[]
for method in methods:
 for seed in seeds:
  folder=root/'F'/('F0-'+method+'-s'+str(seed));source=json.loads((folder/'source/complete.json').read_text());v=json.loads((folder/'source_validation/complete.json').read_text())
  if (source['status']!='SOURCE_FROZEN' or source['epoch']!=50 or source['updates']!=11100 or source['config']['paper_method']!=method or source['config']['model_seed']!=seed
   or source['target_access'] or source['U_truth_read'] or source['head_fit_updates'] or v['status']!='MULTISEED_SOURCE_EVALUATED' or v['method']!=method or v['model_seed']!=seed or v['V_seen']!=27000 or v['G_reads'] or v['target_reads']):raise ValueError('Matched fixed source-only E50 outputs required')
  with np.load(folder/'source_validation/source_V_predictions.npz',allow_pickle=False) as a:
   if set(a.files)!={'logits','y','ids','rx','day','views'} or a['logits'].shape!=(4,27000,6) or not np.isfinite(a['logits']).all() or tuple(a['views'])!=('raw',*scenes):raise ValueError('Original finite same-packet source views required')
   current={k:a[k].copy() for k in ('ids','y','rx','day')}
   if any(x.shape!=(27000,) for x in current.values()) or len(np.unique(current['ids']))!=27000:raise ValueError('Unique complete V population required')
   if metadata is None:metadata=current
   elif any(not np.array_equal(x,metadata[k]) for k,x in current.items()):raise ValueError('Physical pairing differs')
   for j,scene in enumerate(scenes,1):
    for rx in [None,*np.unique(a['rx']).tolist()]:
     chosen=np.ones(27000,bool) if rx is None else a['rx']==rx
     rows.append(dict(method=method,model_seed=seed,scene=scene,RX='ALL' if rx is None else str(rx),**set_statistics(a['logits'][0,chosen],a['logits'][j,chosen],a['y'][chosen],role='V')))
  checks.append(dict(method=method,model_seed=seed,source_V=27000,exact_ID_label_RX_day_order=True))
print(json.dumps(dict(status='SOURCE_PARTIAL_SET_MARGIN_COLLECTED',rows=rows,metadata_checks=checks,models=6,source_V_per_model=27000,
 existing_logit_packets_analyzed=648000,new_F_forwards=0,new_training_updates=0,new_weight_reads=0,new_IQ_reads=0,U_TX_truth_reads=0,target_reads=0,G_forwards=0)))
'''
    for key, value in [('RUNTIME', rt), ('METHODS', METHODS), ('SEEDS', SEEDS), ('SCENES', SCENES)]:
        script = script.replace(key, repr(value))
    return script


def summarize(bundle):
    if bundle['status'] != 'SOURCE_PARTIAL_SET_MARGIN_COLLECTED':
        raise ValueError('Frozen source-only counterfactual required')
    for key in ('new_F_forwards', 'new_training_updates', 'new_weight_reads', 'new_IQ_reads', 'U_TX_truth_reads', 'target_reads', 'G_forwards'):
        if bundle[key] != 0:
            raise ValueError('Existing source logits only')
    expected = set(itertools.product(METHODS, SEEDS))
    checks = bundle['metadata_checks']
    if len(checks) != 6 or {(c['method'], c['model_seed']) for c in checks} != expected or any(c['source_V'] != 27000 or not c['exact_ID_label_RX_day_order'] for c in checks):
        raise ValueError('All six existing paired source models required')
    rows = bundle['rows']
    required = set(itertools.product(METHODS, SEEDS, SCENES, ('ALL', '1', '3', '4', '6', '8')))
    if len(rows) != 108 or {(r['method'], r['model_seed'], r['scene'], r['RX']) for r in rows} != required:
        raise ValueError('All three seeds/scenes/source RX strata required')
    summary = []
    for method, scene, metric in itertools.product(METHODS, SCENES, METRICS):
        selected = [r for r in rows if r['method'] == method and r['scene'] == scene and r['RX'] == 'ALL']
        values = [r[metric] for r in selected]
        defined = all(v is not None and np.isfinite(v) for v in values)
        summary.append(dict(method=method, scene=scene, metric=metric, models=3,
                            mean=statistics.mean(values) if defined else None,
                            sample_SD=statistics.stdev(values) if defined else None))
    return dict(status='ANALYZED_EXISTING_SOURCE_SET_MARGIN', model_seeds=list(SEEDS), models=6,
                summary=summary, rows=rows, metadata_checks=checks,
                **{k: bundle[k] for k in ('existing_logit_packets_analyzed', 'new_F_forwards', 'new_training_updates', 'new_weight_reads', 'new_IQ_reads', 'U_TX_truth_reads', 'target_reads', 'G_forwards')},
                fixed_E50_base_H_counterfactual=True, actual_selected_RX_style_H=False,
                actual_P_or_PM_model=False, actual_parameter_or_AdamW_update=False,
                causal_generalization_claim=False, goal_complete=False)


def collect(runtime, output):
    from experiments.cric_dura_comparison.publish import ssh
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserve prior diagnostic')
    script = collection_script(json.loads(Path(runtime).read_text(encoding='utf-8')))
    compile(script, 'source_set_margin', 'exec')
    bundle = ssh(script)
    result = summarize(bundle)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], models=6, rows=len(result['rows']), summary=result['summary'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    collect(a.runtime, a.output)
