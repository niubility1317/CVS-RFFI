"""Fixed E50 source-only dose comparisons on 1% and 5% labeled data."""
import argparse
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from statistics import mean

from experiments.cric_dura_comparison.io import read, write
from . import full_suite, pilot, runner, transfer


RUN = '20261120-phase1-pamf-tri-source-dose-e50-m6-r01'
MODULE = 'experiments.pamf_repl_source_adapt.joint_holdout'
SETTINGS = ('labeled_1pct', 'labeled_5pct')
WEIGHTS = {'labeled_1pct': (6., 3., 4.5), 'labeled_5pct': (6., 4.5, 3.)}


def folds(receivers, days):
    receivers, days = sorted(set(receivers)), sorted(set(days))
    if len(receivers) < 3 or len(days) < 3:
        raise ValueError('Three source receivers and dates are required for the joint holdout pilot')
    receivers_selected = receivers[0], receivers[len(receivers)//2], receivers[-1]
    return {
        f'rx{rx}_day{day}': dict(date=day,
            excluded=[(r, d) for r in receivers for d in days if r == rx or d == day],
            scored=[(rx, d) for d in days])
        for rx, day in zip(receivers_selected, days[:3], strict=True)
    }


def registered(config):
    spec = folds(config['source_rxs'], config['source_days'])[config['source_fold']]
    return pilot.registered_fold(config, excluded_groups=spec['excluded'], scored_groups=spec['scored'])


def prepare(rt):
    root = Path(rt['run_root'])
    ctx = read(rt['context'])
    if any('tests' in h for h in ctx['handles'].values()):
        raise ValueError('Only source contracts are allowed in the dose controller')
    jobs = []
    for setting in SETTINGS:
        spec = dict(setting=setting, model_seed=pilot.SEED)
        anchor = next(r for r in pilot.paper.make_plan()['rows'] if r['row_id'] == transfer.original_row(spec))
        source = pilot.paper.configuration(ctx, anchor, rt['project'], rt['code_commit'])
        for name in folds(source['source_rxs'], source['source_days']):
            jid = f'train__{setting}-{name}'
            folder = root/'F'/setting/name
            path = root/'configs'/f'{jid}.json'
            config = dict(source, run_id=RUN, row_id=jid, output_root=str(folder),
                          source_fold=name, frozen_sources=rt['frozen_sources'])
            write(path, config)
            jobs.append(dict(id=jid, kind='train', config=str(path), completion=str(folder/'complete.json'),
                             successful_statuses=['SOURCE_PILOT_SCORED']))
    write(root/'jobs.json', jobs)
    return jobs


def segment(config, folder, weight, offset, epochs, parent):
    c = deepcopy(config)
    c.update(output_root=str(folder), optimizer_updates=11100, segment_epochs=epochs, epoch_offset=offset,
             step_offset=offset*222, parent_source=str(parent) if parent else None)
    c['hard_pl']['weight'] = c['hard_pl_gradient_guard']['weight'] = weight
    return c


def train_fold(c):
    from experiments.cric_dura_comparison.f_train import train
    from experiments.pamf_paper_completion.model import installed
    from .data import installed_training_fold

    if c['run_id'] != RUN or any(c.get(key) for key in ('target_access', 'tests', 'target_truth', 'views_root', 'checkpoint')):
        raise ValueError('Only registered source input is allowed')
    _, partition = registered(c)
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)

    def stage(folder, weight, offset, epochs, parent=None):
        config = segment(c, folder, weight, offset, epochs, parent)
        holder = {}

        def validate(candidate):
            if candidate != config:
                raise ValueError('Fold data and recipe must remain bound to their source configuration')

        with installed_training_fold(config, partition), installed(config, holder, offset>0, validate=validate):
            train(config, 'cuda:0')
        pilot._check_segment(folder, config)
        if offset:
            state = holder['objective'].state_dict()
            if state['calls'] != epochs*222 or state['U_truth_read'] or state['student_views'] != 6*state['seen']:
                raise ValueError('Expected fixed-budget blind U training')
            write(folder/'repl_state.json', state)
        return config

    prefix = out/'segment20'
    stage(prefix, 6., 0, 20)
    final_arms = {}
    for weight in WEIGHTS[c['paper_row']['data_setting']]:
        arm = out/f'weight_{pilot._weight_name(weight)}'
        middle = arm/'segment30'
        stage(middle, weight, 20, 10, prefix)
        final_arms[weight] = stage(arm/'segment50', weight, 30, 20, middle)
    pilot.evaluate_fold(c, partition, 'cuda:0', final_arms=final_arms)


def choose(evidence):
    variants, eligible_weights = {}, []
    for weight in (4.5, 3.):
        cross_rx, cross_both = [], []
        for spec, result in evidence:
            cells = {(r['rx'],r['date']): Fraction(r['gain']-r['damage'], r['count'])
                     for r in result if r['weight'] == weight}
            if set(cells) != set(spec['scored']):
                raise ValueError('Complete paired source cells required')
            cross_rx.append(mean(delta for (_, date), delta in cells.items() if date != spec['date']))
            cross_both.append(mean(delta for (_, date), delta in cells.items() if date == spec['date']))
        per_fold = [mean((a,b)) for a,b in zip(cross_rx, cross_both, strict=True)]
        macro = mean(per_fold)
        eligible = sum(x>0 for x in per_fold) >= 2 and macro >= Fraction(1,1000) and mean(cross_both) >= 0
        if eligible:
            eligible_weights.append((macro, weight))
        variants[str(weight)] = dict(eligible=eligible, macro_delta=float(macro), fold_delta=list(map(float, per_fold)),
                                     source_cross_rx=list(map(float, cross_rx)), source_cross_both=list(map(float, cross_both)))
    choice = max(eligible_weights, default=(0.,6.))
    return dict(weight=choice[1], candidates=variants, decision='source_improvement' if choice[1] != 6. else 'retain_native')


def analyze(rt):
    root = Path(rt['run_root'])
    evidence = {setting: {} for setting in SETTINGS}
    for job in read(root/'jobs.json'):
        config, result = read(job['config']), read(job['completion'])
        setting, name = config['paper_row']['data_setting'], config['source_fold']
        if (result['status'] != 'SOURCE_PILOT_SCORED' or result['target_access'] or result['U_truth_read']
                or result['setting'] != setting or result['fold'] != name or name in evidence[setting]):
            raise ValueError('Invalid source-only fold evidence')
        evidence[setting][name] = (folds(config['source_rxs'],config['source_days'])[name], result['rows'])
    if any(len(values) != 3 for values in evidence.values()):
        raise ValueError('Three folds required per label setting')
    write(root/'analysis'/'source_decisions.json', dict(status='SOURCE_ADAPTATION_DECISIONS_FROZEN',
        model_seed=pilot.SEED, target_access=False, U_truth_read=False, epoch=50,
        decisions={setting: choose(values.values()) for setting,values in evidence.items()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('publish','dispatch','worker'))
    parser.add_argument('--kind', choices=('train','smoke'))
    for name in ('config','repo','output','context','ssh-config'):
        parser.add_argument('--'+name)
    a = parser.parse_args()
    if a.action == 'publish':
        from .publish import publish
        publish(a.repo,a.output,a.context,a.ssh_config,run_id=RUN,settings=SETTINGS,module=MODULE,gpus=full_suite.GPUS)
        return
    c = read(a.config)
    if a.action == 'dispatch':
        transfer.dispatch(c, prepare_fn=prepare, analyze_fn=analyze, module=MODULE)
    else:
        from experiments.pamf_repl_completion.runner import activate
        activate(c)
        if a.kind == 'smoke':
            runner.smoke(c, fold_validator=registered, setting=SETTINGS[0])
        else:
            train_fold(c)


if __name__ == '__main__':
    main()
