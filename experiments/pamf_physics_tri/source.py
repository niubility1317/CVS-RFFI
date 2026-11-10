"""Adopt the original shuffled E20 and run the matched E10+E20 trajectory."""
import shutil
import time
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_branch_training.source import checked_e20_parent
from experiments.pamf_physics_branch.tri_identity_scope import installed
from experiments.pamf_repl_view_shuffle import plan as tri
from .plan import RUN, SEEDS


def parent(c):
    seed = c['model_seed']
    original = f'main_split-repl-shuffled_time-3view-s{seed}'
    root = Path(c['output_root'])
    if (seed not in SEEDS or c['run_id'] != RUN or c['row_id'] != f'tri-label-tangent-s{seed}'
            or root.parts[-5:] != ('runs', RUN, 'F', c['row_id'], 'source')
            or c['algorithm_id'] != 'pamf_tri_label_tangent_v1'
            or c['physical_relation'] != dict(beta=.5, l_weight=.05, u_weight=0.)
            or c['paper_row']['row_id'] != f'main_split-factorial-ssl-full-permuted-s{seed}'):
        raise ValueError('Unregistered Tri-SIG-TIME physical continuation')
    expected = root.parents[3] / tri.RUN / 'F' / original / 'source' / 'segment20'
    if Path(c['shared_prefix']) != expected:
        raise ValueError('Only the exact original shuffled same-seed E20 is allowed')
    proxy = dict(c, run_id=tri.RUN, row_id=original)
    proxy.pop('shared_prefix')
    tri.validate_config(proxy)
    if any(c.get(k) for k in ('target_inputs', 'initial_checkpoint', 'parent_source')):
        raise ValueError('Source-only prefix continuation required')
    evidence = checked_e20_parent(c)
    original_config = evidence['config']
    matched = ('view_comparison', 'repl_row', 'source_cache', 'source_rxs', 'source_days')
    if (original_config['row_id'] != original or original_config['run_id'] != tri.RUN
            or any(original_config[k] != c[k] for k in matched)
            or any(original_config[k] != c.get(k) for k in original_config if k.startswith('hard_pl'))):
        raise ValueError('Shuffled graph, source, or original REPL options differ')
    return evidence


def train(c):
    return train_trajectory(c, parent(c), installed, ssl=True)


def train_trajectory(c, prefix, scope, *, ssl):
    from experiments.cric_dura_comparison.f_train import train as segment

    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'resolved_config.json', c)
    segments, states, previous = [prefix], [], prefix['folder']
    started = time.monotonic()
    for epochs, offset in ((10, 20), (20, 30)):
        folder = root / f'segment{epochs + offset}'
        cfg = dict(c, output_root=str(folder), segment_epochs=epochs, epoch_offset=offset,
                   step_offset=offset * 222, parent_source=previous)
        holder = {}
        with scope(cfg, holder):
            segment(cfg)
        evidence = read(folder / 'complete.json')
        model = holder['model']
        steps = epochs * 222
        if (evidence['status'] != 'SOURCE_FROZEN' or evidence['updates'] != epochs * 222
                or evidence['target_access'] or evidence['U_truth_read'] or evidence['head_fit_updates']
                or model._physical_L_calls != steps or model._physical_L_samples != 64 * steps):
            raise ValueError('Physical true-label continuation execution differs')
        if ssl:
            state = holder['objective'].state_dict()
            if (state['calls'] != steps or state['student_views'] != 6 * state['seen']
                    or state['EMA_updates'] or state['U_truth_read']):
                raise ValueError('Original Tri-SIG-TIME U execution differs')
        else:
            progress = read(folder / 'progress.json')
            if progress['U_actual'] or progress['PL_seen'] or model._cd_U_seen:
                raise ValueError('Supervised physical training accessed U')
            state = dict(calls=0, seen=0, accepted=0, student_views=0, U_truth_read=False, EMA_updates=0)
        state['physical_L'] = dict(calls=model._physical_L_calls, selected_samples=model._physical_L_samples)
        write(folder / ('repl_state.json' if ssl else 'physical_state.json'), state)
        segments.append(dict(folder=str(folder), **evidence))
        states.append(state)
        previous = str(folder)
        write(root / 'progress.json', dict(status='SEGMENT_COMPLETE', epoch=offset + epochs,
                                         updates=(offset + epochs) * 222), exclusive=False)
    shutil.copyfile(Path(previous) / 'final_ssdg.pth', root / 'final_ssdg.pth')
    write(root / 'source_contract.json', read(c['source_contract']))
    write(root / 'complete.json', dict(status='SOURCE_FROZEN', epoch=50, updates=11100, config=c, segments=segments,
          initialization='same_contract_scratch20_own10_own20', ancestors=[s['folder'] for s in segments],
          source_contract=c['source_contract'], target_access=False, U_truth_read=False, head_fit_updates=0,
          actual_REPL={k: sum(s[k] for s in states) for k in ('calls', 'seen', 'accepted', 'student_views')},
          actual_physical_L={k: sum(s['physical_L'][k] for s in states) for k in ('calls', 'selected_samples')},
          seconds=time.monotonic() - started, cumulative_training_seconds=sum(s['seconds'] for s in segments)))
