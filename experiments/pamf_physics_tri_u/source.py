"""Continue each verified original source-only E20 through E50."""
import shutil
import time
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_physics_tri import source as protected
from experiments.pamf_physics_tri.plan import RUN as PROTECTED_RUN
from .plan import RUN, SEEDS, RECIPE
from .objective import installed


def parent(c):
    seed, root = c['model_seed'], Path(c['output_root'])
    if (seed not in SEEDS or c['run_id'] != RUN or c['row_id'] != f'tri-u-physical-s{seed}'
            or root.parts[-5:] != ('runs', RUN, 'F', c['row_id'], 'source')
            or c.get('algorithm_id') != 'pamf_tri_u_physical_v1' or c.get('physical_relation') != RECIPE):
        raise ValueError('Unregistered U physical relation continuation')
    row = f'tri-label-tangent-s{seed}'
    proxy = dict(c, run_id=PROTECTED_RUN, row_id=row, algorithm_id='pamf_tri_label_tangent_v1',
                 physical_relation=dict(beta=.5, l_weight=.05, u_weight=0.),
                 output_root=str(root.parents[3]/PROTECTED_RUN/'F'/row/'source'))
    return protected.parent(proxy)


def train(c):
    from experiments.cric_dura_comparison.f_train import train as segment
    prefix, root = parent(c), Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'resolved_config.json', c)
    segments, states, previous = [prefix], [], prefix['folder']
    started = time.monotonic()
    for epochs, offset in ((10,20),(20,30)):
        folder = root/f'segment{offset+epochs}'
        cfg = dict(c, output_root=str(folder), segment_epochs=epochs, epoch_offset=offset,
                   step_offset=offset*222, parent_source=previous)
        holder = {}
        with installed(cfg, holder):
            segment(cfg)
        evidence = read(folder/'complete.json')
        state = holder['objective'].state_dict()
        u = state['U_physical_relation']
        if (evidence['status'] != 'SOURCE_FROZEN' or evidence['updates'] != epochs*222
                or evidence['target_access'] or evidence['U_truth_read'] or evidence['head_fit_updates']
                or state['calls'] != epochs*222 or state['student_views'] != 6*state['seen']
                or state['EMA_updates'] or state['U_truth_read'] or u['calls'] != state['calls']
                or not 0 <= u['accepted'] <= u['selected'] <= state['seen'] or not u['valid_anchors']):
            raise ValueError('Original SSL and accepted-U physical relation execution differs')
        write(folder/'repl_state.json', state)
        segments.append(dict(folder=str(folder), **evidence)); states.append(state); previous = str(folder)
        write(root/'progress.json', dict(status='SEGMENT_COMPLETE', epoch=offset+epochs,
                                        updates=(offset+epochs)*222), exclusive=False)
    shutil.copyfile(Path(previous)/'final_ssdg.pth', root/'final_ssdg.pth')
    write(root/'source_contract.json', read(c['source_contract']))
    write(root/'complete.json', dict(status='SOURCE_FROZEN', epoch=50, updates=11100, config=c, segments=segments,
        initialization='same_contract_scratch20_own10_own20', ancestors=[s['folder'] for s in segments],
        source_contract=c['source_contract'], target_access=False, U_truth_read=False, head_fit_updates=0,
        actual_REPL={k:sum(s[k] for s in states) for k in ('calls','seen','accepted','student_views')},
        actual_U_physical={k:sum(s['U_physical_relation'][k] for s in states)
                           for k in ('calls','selected','accepted','valid_anchors','extra_eval_forwards')},
        seconds=time.monotonic()-started, cumulative_training_seconds=sum(s['seconds'] for s in segments)))
