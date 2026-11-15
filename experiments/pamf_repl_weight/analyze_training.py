"""Parse all complete training records, with no model, IQ or target access."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

FIELDS = ('train/loss_tx_labeled', 'HPL_loss', 'HPL_base_loss', 'HPL_confidence',
          'HPL_ordinary_KL', 'HPL_raw_KL', 'HPL_contrast', 'HPL_joint_loss',
          'HPL_RXE_ordinary_valid_CE', 'HPL_RXE_extrapolated_valid_CE')
FIELDS += tuple(f'HPL_GP_{g}_{x}' for g in ('t2', 'cls_head', 'other')
                for x in ('L_norm', 'U_norm', 'applied_U_norm', 'dot_before', 'dot_after', 'conflict', 'capped'))
FIELDS += tuple(f'HPL_GP_{g}_raw_{x}' for g in ('t2', 'cls_head', 'other')
                for x in ('U_norm', 'dot_before', 'conflict'))
FIELDS += tuple(f'HPL_component_guard_{part}_{g}_{x}'
                for part in ('shallow', 'extrap_delta') for g in ('t2', 'cls_head', 'other')
                for x in ('conflict', 'dot_before', 'correction_norm'))


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def analyze(root):
    root = Path(root)
    sources = read_json(root/'sources.json')
    results = []
    for rid, entry in sources.items():
        source = Path(entry['source_output'])
        done = read_json(source/'complete.json')
        if done['status'] != 'SOURCE_FROZEN' or done['target_access'] or done['U_truth_read']:
            raise ValueError('Source not frozen or unauthorized: '+rid)
        per_epoch, components, states, proxies = [], defaultdict(list), [], []
        step_count = 0
        for end, offset, expected in ((30, 20, 2220), (50, 30, 4440)):
            recovered = end == 30 and bool(done.get('recovered_e30_run'))
            folder = Path(done['segments'][1]['folder']) if recovered else source/f'segment{end}'
            state = read_json(source/'recovered_e30_state.json' if recovered else folder/'repl_state.json')
            if state['U_truth_read'] or state['calls'] != expected:
                raise ValueError('Missing or invalid REPL state: '+str(folder))
            states.append(state)
            proxy = read_json(source/'recovered_e30_v'/'source_v_pseudo_proxy.json' if recovered
                              else folder/'source_v_pseudo_proxy.json')
            if proxy['role'] != 'V' or proxy['U_truth_read'] or proxy['used_for_training']:
                raise ValueError('Source V proxy identity or permission differs')
            proxies.append(dict(epoch=end, **proxy))
            epochs = [json.loads(line) for line in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            if len(epochs) != end-offset:
                raise ValueError('Incomplete epoch records: '+str(folder))
            with (folder/'metrics_epoch.csv').open(encoding='utf-8-sig', newline='') as f:
                import csv
                if sum(1 for _ in csv.DictReader(f)) != len(epochs):
                    raise ValueError('Epoch CSV is incomplete: '+str(folder))
            sums, counts = defaultdict(Counter), defaultdict(Counter)
            n = 0
            with gzip.open(folder/'step_metrics.jsonl.gz', 'rt', encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line); n += 1
                    if record['step'] != n:
                        raise ValueError('Step discontinuity: '+str(folder))
                    m = record['metrics']; epoch = (n-1)//222+1+offset
                    for key in FIELDS:
                        v = m.get(key)
                        if v is not None:
                            if not math.isfinite(float(v)):
                                raise ValueError('Nonfinite '+key+' in '+str(folder))
                            sums[epoch][key] += float(v); counts[epoch][key] += 1
                    for group in ('t2', 'cls_head', 'other'):
                        prefix = f'HPL_GP_{group}_'
                        denominator = m[prefix+'L_norm'] * m[prefix+'U_norm']
                        if denominator > 0:
                            cosine = m[prefix+'dot_before'] / denominator
                            if not math.isfinite(cosine):
                                raise ValueError('Nonfinite gradient cosine in '+str(folder))
                            key = prefix+'cosine_before'
                            sums[epoch][key] += cosine; counts[epoch][key] += 1
                    if m.get('HPL_component_sample') == 1:
                        component = {k: v for k, v in m.items() if k.startswith('HPL_component_') and k != 'HPL_component_sample'}
                        if not component or any(not math.isfinite(float(v)) for v in component.values()):
                            raise ValueError('Missing/nonfinite sampled component gradients')
                        for key, value in component.items():
                            components[key].append(float(value))
            if n != expected:
                raise ValueError('Incomplete step records: '+str(folder))
            step_count += n
            for i, e in enumerate(epochs, 1):
                epoch = offset+i
                if e['epoch'] != i or counts[epoch]['train/loss_tx_labeled'] != 222:
                    raise ValueError('Epoch or step budget differs')
                per_epoch.append(dict(epoch=epoch, source_val_accuracy=e['val_tx_acc'],
                    means={key: sums[epoch][key]/count for key,count in counts[epoch].items()}))
        if step_count != 6660:
            raise ValueError('Continuation budget incomplete')
        log = root/'logs'/f'train__{rid}.log'
        text = log.read_text(encoding='utf-8', errors='replace')
        lower = text.lower()
        markers = {word: lower.count(word) for word in ('traceback (most recent call last)', 'out of memory', 'killed', 'warning', 'nonfinite')}
        seen = sum(x['seen'] for x in states); accepted = sum(x['accepted'] for x in states)
        hist, confidence = Counter(), Counter()
        for state in states:
            hist.update(state['U_pseudo_observation']['accepted_by_pseudo_class'])
            confidence.update(state['U_pseudo_observation']['confidence_bins'])
        if sum(hist.values()) != accepted:
            raise ValueError('Pseudo histogram does not match acceptance')
        if sum(confidence.values()) != seen:
            raise ValueError('Confidence histogram does not match exposures')
        results.append(dict(row_id=rid, steps=step_count, U_seen=seen, U_accepted=accepted,
            U_acceptance=accepted/seen, pseudo_class_counts=dict(hist), confidence_bins=dict(confidence), source_v_proxy=proxies,
            epochs=per_epoch, component_means={k:sum(v)/len(v) for k,v in components.items()},
            component_samples={k:len(v) for k,v in components.items()},
            component_negative_samples={k:sum(value < 0 for value in v) for k,v in components.items() if k.endswith('_cosine')},
            stdout_bytes=log.stat().st_size,
            stdout_marker_counts=markers))
    return dict(status='ANALYZED_COMPLETE_TRAINING_LOGS', rows=results, U_truth_read=False,
                target_inputs_read=False, target_scores_read=False, note='V is a labeled validation proxy, not U accuracy')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-root', required=True); a = p.parse_args()
    print(json.dumps(analyze(a.run_root), ensure_ascii=False, allow_nan=False))
