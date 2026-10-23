"""Reproduce the completed F15/F20 three-seed, strictly L-only readouts."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .analyze_main import SEEDS, VIEWS, metrics, moments, read, write_csv

RUN = '20261022-phase1-PAMF-supervised-Lonly-readouts-m6-r01'
ARMS = ('X3_CE', 'T1_SUP')
READOUTS = ('raw', 'DURA-L')


def row_key(r):
    return r['readout'], r['test'], r['view'], r['dimension'], str(r['stratum'])


def aggregate(folder):
    folder = Path(folder)
    manifest = read(folder / 'manifest.json')
    if (manifest['run_id'] != RUN or len(manifest['rows']) != 6
            or manifest['new_F_training'] or manifest['new_G_training']):
        raise ValueError('Only the six completed, reused-F readouts belong here')
    expected = {(a, s) for a in ARMS for s in SEEDS}
    found = set()
    values, counts, nets, cost = {}, {}, [], []
    physics, sensitivity = defaultdict(dict), defaultdict(dict)
    physical_ids = None
    cm_count = 0
    for item in manifest['rows']:
        rec = read(folder / item['member'])
        score, cfg = rec['score'], rec['source_config']
        arm = item['row_id'].split('-s')[0]
        seed = item['seed']
        identity = arm, seed
        if identity not in expected or identity in found:
            raise ValueError('Unexpected/duplicate F seed')
        found.add(identity)
        source_arm = {'X3_CE': 'F15', 'T1_SUP': 'F20'}[arm]
        pc = rec['prediction_config']
        if (cfg['ablation'] != source_arm or cfg['protocol_id'] != 'F0'
                or cfg['source_IQ_binding_fix'] != 'physical_ID_index_v1'
                or cfg['model_seed'] != seed or not cfg['default_head_joint']
                or cfg['posthoc_head_fit'] or pc.get('strict_supervised_system') is not True
                or pc['source_run_id'] != cfg['run_id'] or pc['source_row_id'] != cfg['row_id']
                or pc['readouts']['DURA-L']['roles'] != 'L'):
            raise ValueError('Corrected same-contract L-only source/default head required')
        done, pred = rec['complete'], rec['prediction_complete']
        if ((score['run_id'], score['row_id'], score['model_seed']) != (RUN, item['row_id'], seed)
                or score['status'] != 'SCORED' or score.get('truth_last') is not True
                or score.get('target_feedback_forbidden') is not True
                or done['status'] != 'SCORED_COMPLETE' or done['views'] != list(VIEWS)
                or set(done['readouts']) != set(READOUTS)
                or pred['status'] != 'ALL_PREDICTIONS_COMPLETE' or pred['truth_read'] is not False
                or pred['query_fit'] is not False):
            raise ValueError('Full frozen seven-view predictions and independent scoring required')
        local = {}
        for r in score['results']:
            k = row_key(r)
            if k in local or k[0] not in READOUTS or k[1] != 'main' or k[2] not in VIEWS:
                raise ValueError('Duplicate/undeclared score cell')
            local[k] = r
            values[(*identity, *k)] = metrics(r, 6)
            counts[(*identity, *k)] = r['count']
            cm_count += 1
        raw = {k[1:]: r for k, r in local.items() if k[0] == 'raw'}
        if {k[1:] for k in local if k[0] == 'DURA-L'} != set(raw):
            raise ValueError('Raw/restored strata matrix differs')
        if set(raw) != {(t, v, d, s) for t, _, d, s in raw for v in VIEWS}:
            raise ValueError('Seven-view strata missing')
        if raw['main', 'clean', 'overall', 'ALL']['count'] != 168000:
            raise ValueError('F0 must retain seven target RX and168000 packets/view')
        original = rec['original_raw_score']
        original_raw = {row_key(r): (r['count'], r['confusion']) for r in original['results']}
        current_raw = {k: (r['count'], r['confusion']) for k, r in local.items() if k[0] == 'raw'}
        if (original['run_id'], original['row_id'], original['model_seed']) != (cfg['run_id'], cfg['row_id'], seed) or original_raw != current_raw:
            raise ValueError('Reused raw CM differs from the exact same frozen F')
        pairs = {row_key(r): r for r in score['paired_raw']}
        if len(pairs) != len(score['paired_raw']) or set(pairs) != {k for k in local if k[0] == 'DURA-L'}:
            raise ValueError('Net-correction strata incomplete')
        for k, pair in pairs.items():
            rr, gr = raw[k[1:]], local[k]
            n = rr['count']
            raw_correct = sum(rr['confusion'][i][i] for i in range(6))
            net = sum(gr['confusion'][i][i] - rr['confusion'][i][i] for i in range(6))
            if (gr['count'] != n or pair['count'] != n or pair['net'] != net
                    or pair['repairs'] - pair['damage'] != net or min(pair['repairs'], pair['damage']) < 0
                    or pair['repairs'] > n - raw_correct or pair['damage'] > raw_correct):
                raise ValueError('Stored joint repairs/damage inconsistent with CM net and bounds')
            nets.append(dict(arm=arm, seed=seed, **dict(zip(('readout', 'test', 'view', 'dimension', 'stratum'), k)),
                             count=n, repairs=pair['repairs'], damage=pair['damage'], net=net))
        meta = rec['source_metadata']
        source = meta['complete.json']
        stages = source['segments']
        if (source['epoch'] != 50 or source['updates'] != 11100 or source['target_access'] is not False
                or source['head_fit_updates'] or len(stages) != 3
                or [s['epoch'] for s in stages] != [20, 10, 20]
                or [s['updates'] for s in stages] != [4440, 2220, 4440]):
            raise ValueError('Reused F full source trajectory differs')
        progress = [meta['segment%d/progress.json' % e] for e in (20, 30, 50)]
        if any(p['U_actual'] or p['PL_seen'] or p['PL_accepted'] for p in progress):
            raise ValueError('L-only F actually consumed U')
        seconds = sum(s['seconds'] for s in stages)
        if abs(seconds - source['seconds']) > 1e-8:
            raise ValueError('F stage elapsed sum differs')
        L_evaluations = sum(p['L_actual'] + p['extra_L'] for p in progress)
        if L_evaluations != 11100 * (384 if arm == 'X3_CE' else 768):
            raise ValueError('Registered actual L forward exposure differs')
        cost.append(dict(arm=arm, seed=seed, reused_source_ablation=source_arm, epochs=50, updates=11100,
                         new_F_training=False, shared_worker_seconds=seconds,
                         L_packet_forward_evaluations=L_evaluations, U_IQ_exposures=0,
                         ordinary_PL_seen=0, scope='Reused F historical shared-worker elapsed, not incremental/exclusive GPU time'))
        d = rec['source_physics']
        if (d['status'] != 'FROZEN_SOURCE_L_PHYSICS_DIAGNOSTICS' or d['samples'] != 720 or d['groups'] != 90
                or d['source_U_reads'] or d['target_reads'] or d['fitted_probe_updates']
                or len(d['physical_ids']) != 720 or len(set(d['physical_ids'])) != 720
                or len(d['interventions']) != 11):
            raise ValueError('Source-only physics diagnostics incomplete or unauthorized')
        if physical_ids is None:
            physical_ids = d['physical_ids']
        elif d['physical_ids'] != physical_ids:
            raise ValueError('Source-L physical-ID pairing differs')
        for name in ('same_TX_cross_RX_distance', 'different_TX_same_RX_distance', 'separation_margin'):
            physics[arm, name][seed] = d['geometry'][name]
        for intervention in d['interventions']:
            if intervention['count'] != 720:
                raise ValueError('Physics intervention exposure differs')
            for metric in ('feature_cosine_distance', 'default_head_flip_fraction', 'reference_descriptor_relative_L2'):
                v = intervention[metric]
                if not math.isfinite(v):
                    raise ValueError('Non-finite source physics metric')
                sensitivity[arm, intervention['intervention'], metric][seed] = v
    if found != expected:
        raise ValueError('Missing a three-seed recipe')
    base = {k[3:]: n for k, n in counts.items() if k[:3] == (ARMS[0], SEEDS[0], 'raw')}
    for arm in ARMS:
        for seed in SEEDS:
            for reader in READOUTS:
                if {k[3:]: n for k, n in counts.items() if k[:3] == (arm, seed, reader)} != base:
                    raise ValueError('F/reader/seed physical target strata differ')
    for k in list(values):
        if k[4] == 'clean':
            six = [values[(*k[:4], v, *k[5:])] for v in VIEWS[1:]]
            values[(*k[:4], 'six_practical_mean', *k[5:])] = {m: statistics.mean(r[m] for r in six) for m in ('accuracy', 'macro_f1')}
    per, groups = [], defaultdict(dict)
    for k, item in sorted(values.items()):
        arm, seed, reader, test, view, dim, stratum = k
        for metric, value in item.items():
            per.append(dict(arm=arm, seed=seed, readout=reader, test=test, view=view, dimension=dim,
                            stratum=stratum, metric=metric, percent=100 * value))
            groups[arm, reader, test, view, dim, stratum, metric][seed] = 100 * value
    fields = ('arm', 'readout', 'test', 'view', 'dimension', 'stratum', 'metric')
    summary = [dict(zip(fields, k), **moments([v[s] for s in SEEDS])) for k, v in sorted(groups.items())]
    paired = []
    conditions = {k[3:] for k in values if k[:3] == (ARMS[0], SEEDS[0], 'raw')}
    for condition in sorted(conditions):
        for metric in ('accuracy', 'macro_f1'):
            for arm in ARMS:
                v = [100 * (values[(arm, s, 'DURA-L', *condition)][metric] - values[(arm, s, 'raw', *condition)][metric]) for s in SEEDS]
                paired.append(dict(contrast='DURA-L_minus_raw', arm=arm, readout='paired',
                    **dict(zip(('test', 'view', 'dimension', 'stratum'), condition)), metric=metric, **moments(v)))
            for reader in READOUTS:
                v = [100 * (values[('T1_SUP', s, reader, *condition)][metric] - values[('X3_CE', s, reader, *condition)][metric]) for s in SEEDS]
                paired.append(dict(contrast='T1_SUP_minus_X3_CE', arm='recipe_pair', readout=reader,
                    **dict(zip(('test', 'view', 'dimension', 'stratum'), condition)), metric=metric, **moments(v)))
    ph = [dict(arm=a, source_metric=m, **moments([v[s] for s in SEEDS])) for (a, m), v in sorted(physics.items())]
    sens = [dict(arm=a, intervention=i, source_metric=m, **moments([v[s] for s in SEEDS])) for (a, i, m), v in sorted(sensitivity.items())]
    g = read(folder / 'G_L_ancestry.json')
    order = ['coarse', 'action', 'r17', 'r38', 'r44', 'r46', 'r65', 'r69']
    if (g['allowed_roles'] != 'L' or g['completion']['stages'] != order
            or g['completion']['updates'] != 39520 or g['model_seed'] != SEEDS[0]):
        raise ValueError('All-eight-stage L-only G ancestry required')
    gcost = []
    for stage in order:
        s = g['stages'][stage]
        if (s['status'] != 'SOURCE_FROZEN' or s['updates'] != 4940 or s['U_exposures']
                or s['TX_labels_read'] or s['F_reads'] or s['target_reads']):
            raise ValueError('G ancestry permissions/budget differs')
        gcost.append(dict(G_seed=SEEDS[0], stage=stage, roles='L', model_epochs=20, updates=4940,
                          shared_worker_seconds=s['seconds'], new_G_training=False, U_IQ_exposures=0))
    data = dict(per_seed=per, summary=summary, paired=paired, net_corrections=nets,
                F_cost=cost, G_cost=gcost, source_geometry=ph, source_sensitivity=sens)
    verification = dict(status='ANALYZED_LONLY6', run_id=RUN, actual_F_seeds=list(SEEDS), logical_F_models=6,
        frozen_score_files=6, raw_G_seven_view_cells=84, serialized_CM_recomputed=cm_count,
        net_correction_strata=len(nets), raw_CM_exact_reuse=True, new_F_training=0, new_G_training=0,
        F_source_stages=18, source_U_IQ_exposures=0, physics_panels=6, source_L_samples_per_panel=720,
        source_TX_RX_day_groups=90, source_interventions_per_panel=11, source_fitted_probe_updates=0,
        target_truth_reads_by_analysis=0, target_feedback_used=False, conditional_on_one_fixed_G=True,
        reused_F_shared_worker_seconds=sum(r['shared_worker_seconds'] for r in cost),
        reused_G_shared_worker_seconds=sum(r['shared_worker_seconds'] for r in gcost))
    return data, verification


def run(folder, output):
    data, verification = aggregate(folder)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in data.items():
        write_csv(out / ('Lonly_' + name + '.csv'), rows)
    for name in ('summary', 'paired'):
        write_csv(out / ('Lonly_' + name + '_overall.csv'), [r for r in data[name] if r['dimension'] == 'overall'])
    lookup = {(r['arm'], r['readout'], r['view']): r for r in data['summary'] if r['dimension'] == 'overall' and r['metric'] == 'accuracy'}
    report = ['# PAMF 严格 L-only：F15 纯 CE 与 F20 监督配方的六条完整读出', '',
        'F0 七目标 RX、每景 168000 包，两个配方各三个实际 F seed，共用一个同契约、全八阶段只用 L 的 DURA。全部七景预测固定后独立 truth-last 评分；raw 来自同一修复后权重，CM 完全一致。', '',
        'X3_CE 复用 F15：普通视图 CE + 默认星地拼接 CE（权重 0.68）、LS0，无 CRIC/KL/U/EMA。T1_SUP 复用 F20：保留 T1 的监督 CRIC、ordinary/raw KL、As 两次 CE（As KL0）和 LS0.01，关闭 U/EMA。它是监督正则配方，不是纯 CE。',
        '两者均为 50 轮、11100 更新和原默认联合头，但每更新实际 L-packet 前向为 384/768，增强和损失也不同。因此配方差值不是单组件因果效应，不能与严格结构面板的匹配 CE 消融混用。', '',
        '|F 配方|读出|clean %|六景均值 %|low urban %|', '|---|---|---:|---:|---:|']
    for arm in ARMS:
        for reader in READOUTS:
            f = lambda v: '%.4f ± %.4f' % (lookup[arm, reader, v]['mean'], lookup[arm, reader, v]['sd'])
            report.append('|%s|%s|%s|%s|%s|' % (arm, reader, f('clean'), f('six_practical_mean'), f('practical_low_urban')))
    report += ['', '六景先在每个 seed 内平均，再作三 seed 配对和样本 SD；固定同一 G，只描述 F seed 变化，SD 不是显著性检验。CSV 保留每景、RX/TX/day、Accuracy、Macro-F1、净纠错和全部负差值。',
        '六个同物理 ID 的源侧诊断均只读 720 条 L，覆盖 90 个 TX/RX/day 层和 11 种干预，无 U/目标访问或探针拟合。几何与模拟干预仅说明敏感性、归纳偏置；模拟 PA 不假设身份保持，不能据此证明硬件完整解耦。',
        '此次新增 F/G 训练均为 0。成本表单列复用 F 的完整三段历史和同一个 G 的八阶段历史；共享 worker elapsed 不是独占 GPU 时间、在线部署时延或本次新增成本。结构专用时延与 24 条结构三种子结果仍待其独立面板闭合。']
    (out / 'Lonly_results.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (out / 'Lonly_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.input, args.output)
