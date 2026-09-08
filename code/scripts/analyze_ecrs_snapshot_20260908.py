"""Parse a complete downloaded ECRS snapshot; independently recount source predictions."""
import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint-summary', type=Path)
    args = parser.parse_args()
    root, out = args.snapshot, args.output
    out.mkdir(parents=True, exist_ok=True)
    audit, cache = [], {}
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        text = p.read_text(encoding='utf-8')
        records = None
        if p.suffix == '.json':
            records = json.loads(text)
        elif p.suffix == '.jsonl':
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif p.suffix == '.csv':
            records = list(csv.DictReader(text.splitlines()))
        cache[p] = records if records is not None else text
        audit.append({'file': str(p.relative_to(root)), 'bytes': p.stat().st_size,
                      'lines': len(text.splitlines()), 'parsed': True})
    rows, epochs, sources, telemetry, loss_lines, source_finals = [], [], [], [], [], {}
    for run in sorted((root/'runs').iterdir()):
        for rowdir in sorted(run.iterdir()):
            row = rowdir.name
            logs = root/'logs'/run.name/row
            metrics = cache[logs/'logs.jsonl']
            csvmetrics = cache[logs/'metrics.csv']
            assert len(metrics) == len(csvmetrics), (row, 'CSV/JSONL row count')
            assert [int(x['epoch']) for x in metrics] == [int(x['epoch']) for x in csvmetrics]
            for m, c in zip(metrics, csvmetrics):
                for k in ('train_loss', 'train_tx_acc', 'epoch_time_s'):
                    assert abs(float(m[k])-float(c[k])) < 1e-6, (row, k)
                epochs.append({'row': row, **m})
            vals = cache[rowdir/'ecrs_source_validation.jsonl']
            for v in vals:
                for scene, sc in v['scenarios'].items():
                    for path, score in sc['paths'].items():
                        sources.append({'row': row, 'epoch': v['epoch'], 'scene': scene,
                                        'path': path, **{k: score[k] for k in ('tx_acc','tx_correct','tx_total','rx_floor','day_floor')}})
            best = max(vals, key=lambda x: x['score'])
            finalpath = rowdir/'ecrs_final_source_evaluation.json'
            if finalpath in cache:
                final = cache[finalpath]
                source_finals[row] = final
                for scene, sc in final['scenarios'].items():
                    p = next(rowdir.glob('source_predictions_*/'+scene+'.jsonl'))
                    preds = cache[p]
                    assert len(preds) == sc['count'] == 27000
                    assert len({x['physical_sample_id'] for x in preds}) == len(preds)
                    for path, score in sc['paths'].items():
                        correct = sum(x['predictions'][path] == x['source_truth'] for x in preds)
                        assert correct == score['tx_correct'], (row, scene, path)
                        for group, key in [('per_rx','receiver_id'),('per_day','day_id')]:
                            counts = Counter(str(x[key]) for x in preds)
                            hits = Counter(str(x[key]) for x in preds if x['predictions'][path] == x['source_truth'])
                            for name, metric in score[group].items():
                                assert counts[name] == metric['tx_total'] and hits[name] == metric['tx_correct']
            stdout = cache[root/'logs'/run.name/(row+'.stdout.log')]
            epoch = None
            for line in stdout.splitlines():
                match = re.search(r'\[EPOCH-BEGIN\].*?E(\d+)/', line)
                if match:
                    epoch = int(match[1])
                if line.startswith('[LOSS-') or line.startswith('[GRAD]'):
                    fields = dict(re.findall(r'([\w-]+)=([-+\d.eEnNaAfFiI]+)(?=\s|$)',line))
                    for key, value in fields.items():
                        try:
                            value = float(value)
                        except ValueError:
                            continue
                        loss_lines.append({'row': row, 'epoch': epoch, 'family': line.split(']')[0][1:], 'metric': key, 'value': value})
            path = rowdir/'ecrs_training.jsonl'
            activation = {}
            if path in cache:
                records = cache[path]
                for name in ('resp_ce','cross_rx','u_pair','fused_ce'):
                    active = [x for x in records if x['losses'][name]['executed'] and x['losses'][name]['valid_count'] > 0]
                    activation[name] = {'executed_valid_batches': len(active),
                        'successful_valid_batches': sum(x['successful_step'] for x in active),
                        'valid_sample_views': sum(x['losses'][name]['valid_count'] for x in active),
                        'first_epoch': active[0]['epoch'] if active else None,
                        'nonzero_weighted_batches': sum(x['losses'][name]['weighted_loss'] != 0 for x in active)}
                activation['total_batches'] = len(records)
                activation['successful_steps'] = sum(x['successful_step'] for x in records)
                activation['ema_updates'] = sum(x['step']['ema_updated'] for x in records)
                per_epoch = defaultdict(list)
                for x in records:
                    per_epoch[x['epoch']].append(x)
                for e, values in per_epoch.items():
                    t = {'row': row, 'epoch': e, 'batches': len(values), 'successful_steps': sum(x['successful_step'] for x in values)}
                    for name in ('resp_ce','cross_rx','u_pair','fused_ce'):
                        t[name+'_weighted_mean'] = statistics.mean(x['losses'][name]['weighted_loss'] for x in values)
                    for name in ('base','response','physical','fusion'):
                        norms = [g['grad_norm'] for x in values for g in x['groups'] if g['name']==name]
                        if norms:
                            t[name+'_grad_mean'] = statistics.mean(norms)
                    telemetry.append(t)
            recent = [x['epoch_time_s'] for x in metrics[-10:]]
            rows.append({'row': row, 'run': run.name, 'status': 'SOURCE_SCREEN_COMPLETE' if finalpath in cache else 'RUNNING',
                         'epoch': metrics[-1]['epoch'], 'best_epoch': best['epoch'], 'best_source_leo_mean': best['score'],
                         'latest_source_leo_mean': vals[-1]['score'], 'latest_source_epoch': vals[-1]['epoch'],
                         'latest_train_loss': metrics[-1]['train_loss'], 'latest_train_tx_acc': metrics[-1]['train_tx_acc'],
                         'train_hours': sum(x['train_time_s'] for x in metrics)/3600,
                         'epoch_hours': sum(x['epoch_time_s'] for x in metrics)/3600,
                         'recent_epoch_seconds_median': statistics.median(recent),
                         'recent_epoch_seconds_mean': statistics.mean(recent),
                         'eta_hours_at_recent_speed': max(0,200-metrics[-1]['epoch'])*statistics.mean(recent)/3600,
                         'skipped_batches': metrics[-1]['skipped_backward_batches_so_far'],
                         'unsafe_warning_count': stdout.count('unsafe backward/step skipped'),
                         'tracebacks': stdout.count('Traceback (most recent call last)'),
                         'activation': activation})
    def write_json(name, data):
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    def write_csv(name, data):
        keys = list(dict.fromkeys(k for x in data for k in x))
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(data)
    write_json('analysis.json', {'rows': rows, 'files': audit, 'source_prediction_rows_recounted': 324000})
    write_json('source_final_scores.json', source_finals)
    if args.checkpoint_summary:
        checkpoint_data = json.loads(args.checkpoint_summary.read_text(encoding='utf-8'))
        checkpoint_audit = {}
        for row, models in checkpoint_data.items():
            checkpoint_audit[row] = {}
            for name, checkpoint in models.items():
                config = checkpoint['args']
                split = checkpoint['split_info']['meta_ssl_source_split']
                assert split['overlap_count'] == 0 and split['labeled_size'] == 6300 and split['source_val_size'] == 27000
                item = {k: checkpoint.get(k) for k in ['epoch','path','size_bytes','feature_schema','optimizer_step_histogram','scaler','split_info']}
                item['configuration'] = {k:v for k,v in config.items() if k.startswith('ecrs') or k in ['use_ecrs','seed','epochs','stage_schedule','wisig_train_ratio','ssl_labeled_ratio','ssl_unlabeled_ratio','ssl_val_ratio','sat_view_schedule']}
                checkpoint_audit[row][name] = item
        write_json('checkpoint_audit.json', checkpoint_audit)
    paired = []
    for scene in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']:
        predictions = {}
        for row in source_finals:
            file = next(p for p in cache if p.name == scene+'.jsonl' and p.parent.parent.name == row)
            predictions[row] = {x['physical_sample_id']:x for x in cache[file]}
        for row in ('B3c','B4'):
            ref, cand = predictions['B0'], predictions[row]
            assert set(ref) == set(cand)
            rescue = harm = disagreed = 0
            for pid, record in ref.items():
                other = cand[pid]
                assert (record['source_truth'],record['receiver_id'],record['day_id']) == (other['source_truth'],other['receiver_id'],other['day_id'])
                rh = record['predictions']['raw'] == record['source_truth']
                ch = other['predictions']['raw'] == other['source_truth']
                rescue += ch and not rh; harm += rh and not ch
                disagreed += record['predictions']['raw'] != other['predictions']['raw']
            paired.append({'row':row,'reference':'B0','scene':scene,'count':len(ref),'rescue':rescue,'harm':harm,'disagreed':disagreed,'net_pp':100*(rescue-harm)/len(ref)})
    write_json('source_paired_vs_B0.json', paired)
    write_csv('epochs.csv', epochs); write_csv('source_validation.csv', sources)
    write_csv('telemetry_epochs.csv', telemetry); write_csv('stdout_losses.csv', loss_lines)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2,figsize=(13,8))
    for row in rows:
        name = row['row']; ms=[x for x in epochs if x['row']==name]
        vs=[x for x in sources if x['row']==name and x['path']=='raw']
        es=sorted({x['epoch'] for x in vs})
        ax[0,0].plot(es,[statistics.mean(x['tx_acc'] for x in vs if x['epoch']==e and x['scene']!='clean') for e in es],label=name)
        ax[0,1].plot([x['epoch'] for x in ms],[x['train_loss'] for x in ms],label=name)
        ax[1,0].plot([x['epoch'] for x in ms],[x['epoch_time_s'] for x in ms],label=name)
        ts=[x for x in telemetry if x['row']==name]
        if ts: ax[1,1].plot([x['epoch'] for x in ts],[x['resp_ce_weighted_mean'] for x in ts],label=name)
    for a,title in zip(ax.flat,['Source raw LEO mean (%)','Training total loss','Epoch time (seconds)','Response CE weighted loss']):
        a.set_title(title); a.set_xlabel('Epoch'); a.grid(alpha=.2); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=170); plt.close(fig)
    fig, axes = plt.subplots(4,2,figsize=(14,15))
    for a, row in zip(axes.flat, rows):
        for family in ('LOSS-CORE-W','LOSS-AUX-W','LOSS-SAT-W'):
            metrics = sorted({x['metric'] for x in loss_lines if x['row']==row['row'] and x['family']==family})
            for metric in metrics:
                points=[x for x in loss_lines if x['row']==row['row'] and x['family']==family and x['metric']==metric]
                if any(abs(x['value']) > 0 for x in points):
                    a.plot([x['epoch'] for x in points],[x['value'] for x in points],label=family[5:]+':'+metric,linewidth=.8)
        a.set_title(row['row']+' weighted loss components'); a.set_xlabel('Epoch'); a.grid(alpha=.2); a.legend(fontsize=6,ncol=2)
    axes.flat[-1].axis('off'); fig.tight_layout(); fig.savefig(out/'loss_components.png',dpi=150); plt.close(fig)
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
