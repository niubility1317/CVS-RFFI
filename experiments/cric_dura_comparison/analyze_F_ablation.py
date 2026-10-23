"""Pure-stdlib reproduction of the corrected21-arm F diagnostic panel."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .analyze_main import SEEDS, VIEWS, metrics, moments, read, write_csv

RUN = '20261022-phase1-T1-DURA-F-ablation-binding-m63-r01'
ARMS = tuple('F%02d' % i for i in range(1, 22))
READERS = ('raw', 'DURA-LU')


def key(r):
    return r['readout'], r['test'], r['view'], r['dimension'], str(r['stratum'])


def aggregate(folder):
    folder = Path(folder)
    manifest = read(folder / 'manifest.json')
    if (manifest['run_id'] != RUN or len(manifest['rows']) != 63
            or len(manifest['reference_rows']) != 3 or manifest['old_invalid_binding_rows_included']):
        raise ValueError('The63 corrected rows and3 fixed T1 references are required')
    expected = {(arm, seed) for arm in ARMS + ('T1',) for seed in SEEDS}
    seen = set()
    values, counts, nets, cost, stages = {}, {}, [], [], []
    gspec = None
    cm_count, reference_cm_count = 0, 0
    for item in manifest['rows'] + manifest['reference_rows']:
        rec = read(folder / item['member'])
        score, done = rec['score'], rec['complete']
        arm, seed = item['ablation'], item['seed']
        identity = arm, seed
        if identity not in expected or identity in seen:
            raise ValueError('Unexpected or duplicate model seed')
        seen.add(identity)
        pc = rec['prediction_config']
        if (score['row_id'] != item['row_id'] or score['model_seed'] != seed
                or score['status'] != 'SCORED' or score.get('truth_last') is not True
                or score.get('target_feedback_forbidden') is not True
                or done['status'] != 'SCORED_COMPLETE' or done['views'] != list(VIEWS)
                or set(done['readouts']) != set(READERS)):
            raise ValueError('Wrong-row or incomplete independent seven-view score')
        current_g = pc['readouts']['DURA-LU']
        if current_g['roles'] != 'LU':
            raise ValueError('This panel uses the fixed original F0 G-LU')
        if gspec is None:
            gspec = current_g
        elif current_g != gspec:
            raise ValueError('G changed across F mechanisms or default references')
        if arm != 'T1':
            cfg = rec['source_config']
            if (score['run_id'] != RUN or cfg['run_id'] != RUN or cfg['ablation'] != arm
                    or cfg['model_seed'] != seed or cfg['num_classes'] != 6 or cfg['protocol_id'] != 'F0'
                    or cfg.get('source_IQ_binding_fix') != 'physical_ID_index_v1'
                    or not cfg['default_head_joint'] or cfg['posthoc_head_fit']
                    or pc['source_output'] != cfg['output_root']
                    or cfg['source_contract'] != gspec['source_contract']):
                raise ValueError('Only corrected same-contract F0 sources may enter this panel')
            pred = rec['prediction_complete']
            if pred['status'] != 'ALL_PREDICTIONS_COMPLETE' or pred['truth_read'] is not False or pred['query_fit'] is not False:
                raise ValueError('Prediction freezing differs')
        elif not score['row_id'].startswith('F0-native_cric-s'):
            raise ValueError('Historical BASE/ALL_TX is not a final T1 reference')
        local = {}
        for r in score['results']:
            k = key(r)
            if k in local or k[0] not in READERS or k[1] != 'main' or k[2] not in VIEWS:
                raise ValueError('Duplicate or undeclared score cell')
            local[k] = r
            values[(*identity, *k)] = metrics(r, 6)
            counts[(*identity, *k)] = r['count']
            if arm == 'T1':
                reference_cm_count += 1
            else:
                cm_count += 1
        raw = {k[1:]: r for k, r in local.items() if k[0] == 'raw'}
        if ({k[1:] for k in local if k[0] == 'DURA-LU'} != set(raw)
                or set(raw) != {(t, v, d, s) for t, _, d, s in raw for v in VIEWS}
                or raw['main', 'clean', 'overall', 'ALL']['count'] != 168000):
            raise ValueError('The same seven-target168000 seven-view strata matrix is required')
        pairs = {key(r): r for r in score['paired_raw']}
        if len(pairs) != len(score['paired_raw']) or set(pairs) != {k for k in local if k[0] == 'DURA-LU'}:
            raise ValueError('Paired repair/damage strata missing')
        for k, p in pairs.items():
            rr, gr = raw[k[1:]], local[k]
            n = rr['count']
            nc = sum(rr['confusion'][i][i] for i in range(6))
            net = sum(gr['confusion'][i][i] - rr['confusion'][i][i] for i in range(6))
            if (gr['count'] != n or p['count'] != n or p['net'] != net or p['repairs'] - p['damage'] != net
                    or min(p['repairs'], p['damage']) < 0 or p['repairs'] > n - nc or p['damage'] > nc):
                raise ValueError('Correction nets or joint-count bounds differ from CMs')
            nets.append(dict(arm=arm, seed=seed, **dict(zip(('readout', 'test', 'view', 'dimension', 'stratum'), k)),
                             count=n, repairs=p['repairs'], damage=p['damage'], net=net))
        meta = rec['source_metadata']
        source = meta.get('complete.json')
        if arm == 'T1':
            ancestry = list(meta.values())
            if len(ancestry) != 3 or sum(s['optimizer_steps'] for s in ancestry) != 11100 or any(s['target_access'] for s in ancestry):
                raise ValueError('Fixed final T1 own-state ancestry differs')
            seconds = sum(s['elapsed_seconds'] for s in ancestry)
            cost.append(dict(arm=arm, seed=seed, newly_trained_for_fine_panel=False, model_epochs=50, updates=11100,
                shared_worker_seconds=seconds, L_packet_forward_evaluations=None, U_IQ_exposures=None,
                ordinary_PL_seen=None, ordinary_PL_accepted=None, ordinary_PL_acceptance=None,
                native_SSL_acceptance=None, cost_scope='Existing final T1 source ancestry; reused comparator, not incremental fine-panel training'))
            continue
        segments = source['segments']
        if (source['status'] != 'SOURCE_FROZEN' or source['epoch'] != 50 or source['updates'] != 11100
                or source['target_access'] is not False or source['head_fit_updates'] or source['U_truth_read']
                or len(segments) != 3 or [s['epoch'] for s in segments] != [20, 10, 20]
                or [s['updates'] for s in segments] != [4440, 2220, 4440]):
            raise ValueError('F trajectory budget/permissions differ')
        progress = [meta['segment%d/progress.json' % n] for n in (20, 30, 50)]
        for e, s, p in zip((20, 30, 50), segments, progress):
            if s['target_access'] or s['head_fit_updates'] or s['U_truth_read']:
                raise ValueError('Unauthorized source stage')
            stages.append(dict(arm=arm, seed=seed, cumulative_epoch=e, stage_epochs=s['epoch'], updates=s['updates'],
                shared_worker_seconds=s['seconds'], L_IQ_exposures=p['L_actual'],
                L_packet_forward_evaluations=p['L_actual'] + p['extra_L'], U_IQ_exposures=p['U_actual'],
                ordinary_PL_seen=p['PL_seen'] if arm == 'F21' else None,
                ordinary_PL_accepted=p['PL_accepted'] if arm == 'F21' else None))
        seconds = sum(s['seconds'] for s in segments)
        if abs(seconds - source['seconds']) > 1e-8:
            raise ValueError('F stage elapsed sum differs')
        exposure = sum(p['L_actual'] + p['extra_L'] for p in progress)
        expected_forward = 384 if arm in ('F05', 'F15') else 512 if arm == 'F19' else 768
        if exposure != 11100 * expected_forward:
            raise ValueError('Actual labeled forward exposure differs from registered mechanism')
        u = sum(p['U_actual'] for p in progress)
        if (arm in ('F15', 'F20') and u != 0) or (arm not in ('F15', 'F20') and u != 2097900):
            raise ValueError('Actual U exposure differs from F0 13/37 or L-only exception')
        pl_seen = sum(p['PL_seen'] for p in progress) if arm == 'F21' else None
        pl_accepted = sum(p['PL_accepted'] for p in progress) if arm == 'F21' else None
        if arm == 'F21' and (pl_seen != u or not 0 <= pl_accepted <= pl_seen):
            raise ValueError('F21 ordinary PL counters inconsistent')
        cost.append(dict(arm=arm, seed=seed, newly_trained_for_fine_panel=True, model_epochs=50, updates=11100,
            shared_worker_seconds=seconds, L_packet_forward_evaluations=exposure, U_IQ_exposures=u,
            ordinary_PL_seen=pl_seen, ordinary_PL_accepted=pl_accepted,
            ordinary_PL_acceptance=pl_accepted / pl_seen if pl_seen else None,
            native_SSL_acceptance=None, cost_scope='Corrected F full three-stage elapsed under shared load, not exclusive GPU time'))
    if seen != expected:
        raise ValueError('Missing a three-seed arm')
    base = {k[3:]: n for k, n in counts.items() if k[:3] == ('T1', SEEDS[0], 'raw')}
    for arm, seed in expected:
        for reader in READERS:
            if {k[3:]: n for k, n in counts.items() if k[:3] == (arm, seed, reader)} != base:
                raise ValueError('Target strata/counts differ across mechanism/seed/readout')
    for k in list(values):
        if k[4] == 'clean':
            six = [values[(*k[:4], v, *k[5:])] for v in VIEWS[1:]]
            values[(*k[:4], 'six_practical_mean', *k[5:])] = {m: statistics.mean(r[m] for r in six) for m in ('accuracy', 'macro_f1')}
    per, groups = [], defaultdict(dict)
    for k, row in sorted(values.items()):
        arm, seed, reader, test, view, dim, stratum = k
        for metric, v in row.items():
            per.append(dict(arm=arm, seed=seed, readout=reader, test=test, view=view, dimension=dim,
                            stratum=stratum, metric=metric, percent=100 * v))
            groups[arm, reader, test, view, dim, stratum, metric][seed] = 100 * v
    fields = ('arm', 'readout', 'test', 'view', 'dimension', 'stratum', 'metric')
    summary = [dict(zip(fields, k), **moments([v[s] for s in SEEDS])) for k, v in sorted(groups.items())]
    conditions = sorted({k[3:] for k in values if k[:3] == ('T1', SEEDS[0], 'raw')})
    paired, factor_seed, factor_summary = [], [], []
    def point(arm, seed, reader, condition, metric):
        return 100 * values[(arm, seed, reader, *condition)][metric]
    for condition in conditions:
        c = dict(zip(('test', 'view', 'dimension', 'stratum'), condition))
        for metric in ('accuracy', 'macro_f1'):
            for arm in ARMS:
                for reader in READERS:
                    v = [point('T1', s, reader, condition, metric) - point(arm, s, reader, condition, metric) for s in SEEDS]
                    paired.append(dict(contrast='T1_minus_variant', arm=arm, readout=reader, **c, metric=metric, **moments(v)))
            for arm in ARMS + ('T1',):
                v = [point(arm, s, 'DURA-LU', condition, metric) - point(arm, s, 'raw', condition, metric) for s in SEEDS]
                paired.append(dict(contrast='G_minus_raw', arm=arm, readout='paired', **c, metric=metric, **moments(v)))
            effects = defaultdict(list)
            for s in SEEDS:
                full_raw = point('T1', s, 'raw', condition, metric)
                full_g = point('T1', s, 'DURA-LU', condition, metric)
                none_raw = point('F16', s, 'raw', condition, metric)
                none_g = point('F16', s, 'DURA-LU', condition, metric)
                e = dict(CRIC_at_raw=full_raw - none_raw, CRIC_at_G=full_g - none_g,
                    G_at_T1=full_g - full_raw, G_at_F16=none_g - none_raw,
                    CRIC_x_G=(full_g - full_raw) - (none_g - none_raw))
                for name, v in e.items():
                    effects[name].append(v)
                    factor_seed.append(dict(seed=s, effect=name, **c, metric=metric, pp=v))
            for name, v in effects.items():
                factor_summary.append(dict(effect=name, **c, metric=metric, **moments(v)))
    data = dict(per_seed=per, summary=summary, paired=paired, net_corrections=nets,
                F_cost=cost, F_stage_cost=stages, factorial_per_seed=factor_seed, factorial_summary=factor_summary)
    verification = dict(status='ANALYZED_CORRECTED_FINE63',run_id=RUN,corrected_F_models=63,
        fixed_T1_references=3,actual_F_seeds=list(SEEDS),arms=list(ARMS),serialized_corrected_CM=cm_count,
        reused_reference_CM_checked=reference_cm_count,net_strata=len(nets),new_F_stages=len(stages),
        new_model_epochs=3150,new_optimizer_updates=699300,readout_view_cells=924,
        old_invalid_F_included=False,historical_factorial_used=False,conditional_on_one_fixed_G=True,
        target_truth_reads_by_analysis=0,target_feedback_used=False,
        corrected_F_shared_worker_seconds=sum(r['shared_worker_seconds'] for r in cost if r['newly_trained_for_fine_panel']),
        reused_T1_shared_worker_seconds=sum(r['shared_worker_seconds'] for r in cost if not r['newly_trained_for_fine_panel']),
        native_SSL_acceptance='Not present in fetched compact stage telemetry; do not interpret ordinary-PL-only zero counters as zero native acceptance')
    return data, verification


def run(folder, output):
    data, verification = aggregate(folder)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in data.items():
        write_csv(out / ('F_ablation_' + name + '.csv'), rows)
    for name in ('summary', 'paired', 'factorial_summary'):
        write_csv(out / ('F_ablation_' + name + '_overall.csv'), [r for r in data[name] if r['dimension'] == 'overall'])
    (out / 'F_ablation_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.input, a.output)
