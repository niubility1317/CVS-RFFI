"""Both frozen readouts, matched baselines and exact raw-only goal accounting."""
import argparse
import csv
import statistics
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from .analyze import overall
from .dura_reference import METHODS, READOUTS, validate_rows
from .plan import SEEDS, VIEWS
from .hard_pl_early import METHODS as EARLY_METHODS
from .hard_pl_guard import METHODS as GUARD_METHODS
from .hard_pl_joint import METHODS as JOINT_METHODS
from .hard_pl_complement import METHODS as COMP_METHODS
from .hard_pl_path import METHODS as PATH_METHODS
from .hard_pl_reliable_as import METHODS as RAS_METHODS
from .hard_pl_path_reliable_as import METHODS as PATH_RAS_METHODS
from .hard_pl_receiver_extrapolation import METHODS as RXE_METHODS
from .hard_pl_native_as import METHODS as NAS_METHODS
from .hard_pl_balanced_ce import METHODS as BCE_METHODS
from .hard_pl_native_cap2 import METHODS as CAP2_METHODS
from .hard_pl_branch_guard import METHODS as BG_METHODS
from .hard_pl_step_guard import METHODS as SG_METHODS
from .hard_pl_step_no_cap import METHODS as NC_METHODS
from .hard_pl_all_u_rdrop import METHODS as RD_METHODS
from .report import cells


def verdict(data, method, readout):
    pairs = {}
    passed = True
    for arm in ('F20', 'F21'):
        pair = {}
        for view in ('clean', 'six_mean'):
            key = (view, 'overall', 'ALL')
            denominator = sum(data[readout, method, seed][key][1] for seed in SEEDS)
            net = [data[readout, method, seed][key][0] - data[readout, arm, seed][key][0] for seed in SEEDS]
            threshold = 100 * sum(net) >= denominator if view == 'clean' else 200 * sum(net) > 3 * denominator
            pair[view] = dict(net_correct=sum(net), denominator=denominator,
                delta_pp=100. * sum(net) / denominator, per_seed_net=dict(zip(map(str, SEEDS), net)),
                positive_seeds=sum(n > 0 for n in net), threshold_pass=threshold)
            passed = passed and threshold
        pairs[arm] = pair
    return dict(method=method, readout=readout, thresholds_pass=passed, paired_differences=pairs)


def report(rt, output):
    methods=tuple(rt.get('methods',METHODS))
    if methods==METHODS:
        validate_rows(rt['rows'])
    elif methods in (('PAIREDU',), ('UCRIC',), ('UMIX',), ('UGP',), ('UCMR',), ('UDAP',), ('UVB',), ('HPL2','HPLT1U'), EARLY_METHODS, GUARD_METHODS, JOINT_METHODS, COMP_METHODS, PATH_METHODS, RAS_METHODS, PATH_RAS_METHODS, RXE_METHODS, NAS_METHODS, BCE_METHODS, CAP2_METHODS, BG_METHODS, SG_METHODS, NC_METHODS,RD_METHODS, ('HPL_NO_UAS',), ('HPL_LS2',), ('HPL_MS2',), ('HPL_NAMS2',)):
        if methods == ('PAIREDU',):
            from .paired_u import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('UCRIC',):
            from .u_cric import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('UMIX',):
            from .u_mixstyle import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('UGP',):
            from .u_gradprotect import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('UCMR',):
            from .u_cmr import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('UDAP',):
            from .u_dualanchor import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('UVB',):
            from .u_viewbalance import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('HPL2','HPLT1U'):
            from .hard_pl import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('HPL_NO_UAS',):
            from .hard_pl_no_uas import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('HPL_LS2',):
            from .hard_pl_ls import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('HPL_MS2',):
            from .hard_pl_mixstyle import RUN, RESOURCE_GPU_BY_SEED
        elif methods == ('HPL_NAMS2',):
            from .hard_pl_no_as_mixstyle import RUN, RESOURCE_GPU_BY_SEED
        elif methods == GUARD_METHODS:
            from .hard_pl_guard import RUN, RESOURCE_GPU_BY_SEED
        elif methods == JOINT_METHODS:
            from .hard_pl_joint import RUN, RESOURCE_GPU_BY_SEED
        elif methods == COMP_METHODS:
            from .hard_pl_complement import RUN, RESOURCE_GPU_BY_SEED
        elif methods == PATH_METHODS:
            from .hard_pl_path import RUN, RESOURCE_GPU_BY_SEED
        elif methods == RAS_METHODS:
            from .hard_pl_reliable_as import RUN, RESOURCE_GPU_BY_SEED
        elif methods == PATH_RAS_METHODS:
            from .hard_pl_path_reliable_as import RUN, RESOURCE_GPU_BY_SEED
        elif methods == RXE_METHODS:
            from .hard_pl_receiver_extrapolation import RUN, RESOURCE_GPU_BY_SEED
        elif methods == NAS_METHODS:
            from .hard_pl_native_as import RUN, RESOURCE_GPU_BY_SEED
        elif methods == BCE_METHODS:
            from .hard_pl_balanced_ce import RUN, RESOURCE_GPU_BY_SEED
        elif methods == CAP2_METHODS:
            from .hard_pl_native_cap2 import RUN, RESOURCE_GPU_BY_SEED
        elif methods == RD_METHODS:
            from .hard_pl_all_u_rdrop import RUN, RESOURCE_GPU_BY_SEED
        elif methods == NC_METHODS:
            from .hard_pl_step_no_cap import RUN, RESOURCE_GPU_BY_SEED
        elif methods == SG_METHODS:
            from .hard_pl_step_guard import RUN, RESOURCE_GPU_BY_SEED
        elif methods == BG_METHODS:
            from .hard_pl_branch_guard import RUN, RESOURCE_GPU_BY_SEED
        else:
            from .hard_pl_early import RUN, RESOURCE_GPU_BY_SEED
        from .control import validate_rows as validate_followup
        if rt['run_id']!=RUN:
            raise ValueError('Followup report belongs to another run')
        validate_followup(rt['rows'],methods,RESOURCE_GPU_BY_SEED)
    else:
        raise ValueError('Exact declared dual-readout method matrix required')
    if read(Path(rt['run_root']) / 'state.json')['status'] != 'ALL_SCORED':
        raise ValueError('Complete supplemental matrix required before result analysis')
    return _write_report(rt, output, methods)


def report_complete_hpl2(rt, output):
    """Full HPL2 three-seed method; the original six-model experiment continues."""
    from .hard_pl import RUN
    if rt['run_id']!=RUN:raise ValueError('Original fixed six-model hard-PL experiment required')
    return report_complete_method(rt,output,'HPL2')


def report_complete_method(rt, output, method):
    """All three fixed seeds of one declared method; do not open pending-arm scores."""
    from .hard_pl import RUN as HARD_RUN,METHODS as HARD_METHODS,RESOURCE_GPU_BY_SEED
    from .hard_pl_early import RUN as EARLY_RUN
    from .hard_pl_guard import RUN as GUARD_RUN
    from .control import validate_rows as validate_followup
    registered={HARD_RUN:HARD_METHODS,EARLY_RUN:EARLY_METHODS,GUARD_RUN:GUARD_METHODS}
    methods=registered.get(rt['run_id'])
    if methods is None or tuple(rt['methods'])!=methods or method not in methods:
        raise ValueError('Fixed registered six-model method declaration required')
    validate_followup(rt['rows'],methods,RESOURCE_GPU_BY_SEED)
    state=read(Path(rt['run_root'])/'state.json')
    selected=[row for row in rt['rows'] if row['method']==method]
    if (set(state.get('jobs',{}))!={row['row_id'] for row in rt['rows']}
        or any(state['jobs'][row['row_id']]['status']!='SCORED' for row in selected)):
        raise ValueError('All three registered method rows must be scored before any score read')
    done=_write_report(dict(rt,rows=selected),output,(method,))
    out=Path(output);scope=dict(kind='COMPLETE_REGISTERED_METHOD',method=method,models=3,
        registered_full_models=6,report_is_full_run=False,
        full_registered_matrix_scored=state['status']=='ALL_SCORED' and all(v['status']=='SCORED' for v in state['jobs'].values()),
        other_methods_pending=[other for other in methods if other!=method and
            any(state['jobs'][r['row_id']]['status']!='SCORED' for r in rt['rows'] if r['method']==other)])
    analysis=read(out/'analysis.json');analysis['report_scope']=scope;write(out/'analysis.json',analysis,exclusive=False)
    text=(out/'report.md').read_text(encoding='utf-8')
    text=text.replace('# PAMF 与固定 DURA v1.0＋PAMF 目标测试结果',
        '# '+method+' 完整三种子目标测试结果',1)
    text+='\n报告范围仅为原注册 '+method+' 方法的三个种子及全部双读出七场景；原六模型训练与测试矩阵继续，其他方法未完成部分未读取。本报告不代表整个六模型实验完成，也不改变其他已冻结配方、父来源、候选顺序或测试范围。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    done.update(status='ANALYZED_COMPLETE_METHOD',report_scope=scope);write(out/'complete.json',done,exclusive=False)
    return done


def _write_report(rt, output, methods):
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserve existing report output')
    files = []
    for seed in SEEDS:
        for arm in ('F20', 'F21'):
            files.append((arm, seed, Path(rt['baseline_scores'][str(seed)][arm]) / 'scores.json',
                          dict(model_seed=seed, row_id=arm + '-s' + str(seed))))
    for row in rt['rows']:
        c = read(row['score_config'])
        if (c['row_id'], c['model_seed'], c['run_id'], c['readouts']) != (
                row['row_id'], row['model_seed'], rt['run_id'], list(READOUTS)):
            raise ValueError('Supplemental score configuration identity/readouts differ')
        files.append((row['method'], row['model_seed'], Path(c['output_root']) / 'scores.json',
                      {k: c[k] for k in ('run_id', 'row_id', 'model_seed')}))
    data, scores = {}, {}
    for method, seed, path, identity in files:
        value = read(path)
        scores[method, seed] = value
        for readout in READOUTS:
            overall(path, identity, readout)  # Truth-last, identities and 168000 integer counts.
            data[readout, method, seed] = cells(value, readout)
    keys = set(data['raw', 'F20', SEEDS[0]])
    if not keys or any(set(v) != keys for v in data.values()):
        raise ValueError('Both readouts require matched full target strata')
    rows = []
    for key in sorted(keys):
        denominators = {v[key][1] for v in data.values()}
        if len(denominators) != 1:
            raise ValueError('Readout/method/seed stratum denominators differ')
        n = denominators.pop()
        for readout in READOUTS:
            for method in ('F20', 'F21') + methods:
                correct = [data[readout, method, seed][key][0] for seed in SEEDS]
                row = dict(readout=readout, method=method, view=key[0], dimension=key[1], stratum=key[2],
                    count_per_seed=n, correct_total=sum(correct), mean_accuracy_percent=100. * sum(correct)/(3*n),
                    sample_sd_percent=statistics.stdev(100.*c/n for c in correct))
                row.update({'correct_s' + str(seed): c for seed, c in zip(SEEDS, correct)})
                for arm in ('F20', 'F21'):
                    net = [data[readout, method, seed][key][0] - data[readout, arm, seed][key][0] for seed in SEEDS]
                    row['delta_' + arm + '_pp'] = 100. * sum(net) / (3*n)
                    row['positive_seeds_' + arm] = sum(v > 0 for v in net)
                rows.append(row)
    pairs = []
    for (method, seed), value in scores.items():
        present = set()
        for pair in value['paired_raw']:
            if pair['test'] != 'main' or pair['readout'] != 'DURA-LU':
                continue
            key = (pair['view'], pair['dimension'], pair['stratum'])
            if key in present or key not in keys or pair['count'] != data['raw', method, seed][key][1]:
                raise ValueError('DURA/raw paired target coverage differs')
            present.add(key)
            if (any(type(pair[k]) is not int or pair[k] < 0 for k in ('repairs', 'damage'))
                    or pair['repairs'] + pair['damage'] > pair['count']
                    or pair['repairs'] - pair['damage'] != pair['net']
                    or pair['net'] != data['DURA-LU', method, seed][key][0] - data['raw', method, seed][key][0]):
                raise ValueError('DURA repairs/damage disagree with exact correct counts')
            pairs.append(dict(method=method, model_seed=seed, **pair))
        if present != {k for k in keys if k[0] in VIEWS}:
            raise ValueError('Complete seven-view DURA/raw repair evidence required')
    verdicts = [verdict(data, method, readout) for readout in READOUTS for method in methods]
    goal_met = any(v['thresholds_pass'] for v in verdicts if v['readout'] == 'raw')
    summary = [row for row in rows if row['dimension'] == 'overall' and row['stratum'] == 'ALL']
    per_seed = []
    for (method, seed), value in scores.items():
        for r in value['results']:
            if r['test'] == 'main' and r['readout'] in READOUTS and r['dimension'] == 'overall':
                per_seed.append(dict(method=method, model_seed=seed, readout=r['readout'], view=r['view'],
                    count=r['count'], accuracy_percent=100.*r['accuracy'], macro_f1=r['macro_f1']))
    analysis = dict(status='ANALYZED', model_seeds=list(SEEDS), readouts=list(READOUTS),
        summary=summary, verdicts=verdicts, goal_met=goal_met, raw_goal_only=True,
        DURA_version='dura_v1', truth_last=True, target_feedback_for_training=False)
    write(out / 'analysis.json', analysis)
    write(out / 'strata.json', rows)
    write(out / 'paired_raw.json', pairs)
    for name, values in (('strata', rows), ('paired_raw', pairs), ('per_seed', per_seed)):
        with (out / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    text = ['# PAMF 与固定 DURA v1.0＋PAMF 目标测试结果', '',
        '同一固定 unseen-RX 测试集，每景 168000 条物理样本，三个模型种子。表内为准确率均值 ± 样本 SD（%）。两个 readout 分别对同 readout 的监督 F20、普通 PL F21 比较。', '']
    for readout in READOUTS:
        text += [('PAMF 单模型' if readout == 'raw' else '固定 DURA v1.0＋同一 PAMF'), '',
            '| 视图 | F20 监督 | F21 普通 PL | '+' | '.join(methods)+' |', '|---|' + '---:|'*(2+len(methods))]
        for view in (*VIEWS, 'six_mean'):
            values = [next(r for r in summary if r['readout'] == readout and r['method'] == method and r['view'] == view)
                      for method in ('F20', 'F21') + methods]
            text.append('| ' + view + ' | ' + ' | '.join(f"{r['mean_accuracy_percent']:.6f} ± {r['sample_sd_percent']:.6f}" for r in values) + ' |')
        text.append('')
    text += ['原始研发目标继续按 PAMF 单模型的整数正确数验收：对两个对照，clean 均至少 +1 pp，六景平均均严格超过 +1.5 pp。DURA 组合为并列参考，不替代单模型验收。', '']
    for v in verdicts:
        text.append(v['method'] + ' / ' + v['readout'] + '：' + ('满足数值门槛。' if v['thresholds_pass'] else '未同时满足数值门槛。'))
        for arm, pair in v['paired_differences'].items():
            text.append(f"对 {arm}：clean {pair['clean']['delta_pp']:+.6f} pp；六景 {pair['six_mean']['delta_pp']:+.6f} pp。")
        text.append('')
    text += ['逐种子准确率与 Macro-F1 见 per_seed.csv；逐 RX／TX／day／RX×day 见 strata.csv；DURA 相对同一模型 raw 的纠错、损伤与净纠错见 paired_raw.csv。全部原定种子和场景均保留。', '']
    (out / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    done = dict(status='ANALYZED', goal_met=goal_met, raw_goal_only=True, rows=len(rows), paired_rows=len(pairs))
    write(out / 'complete.json', done)
    return done


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(report(read(a.config), a.output))
