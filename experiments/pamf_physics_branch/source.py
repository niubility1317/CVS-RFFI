"""Real F21 scratch-E20 parent and own 10+20 PTBR source-only continuation."""
from pathlib import Path
import shutil
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_branch_training import plan as old_plan
from experiments.pamf_branch_training.source import checked_parent as old_checked_parent
from .plan import RUN, validate_candidate
from .training_scope import installed


def validate_source(c):
    row = validate_candidate(c['physical_row'])
    return validate_source_config(c, row, RUN)


def validate_source_config(c, row, run_id):
    if (c['run_id'] != run_id or c['row_id'] != row['row_id'] or c['model_seed'] != row['model_seed']
            or c['source_rxs'] != row['source_rxs'] or c['source_days'] != row['source_days']
            or c['num_classes'] != 6 or c['epochs'] != 50 or c['optimizer_updates'] != 11100
            or c['ablation'] != 'F21' or not c['U_use'] or c['own_EMA'] or c['native_U_gate']
            or c['posthoc_head_fit'] or c['paper_row']['structure'] != 'full'
            or c['paper_row']['permuted'] or c['paper_row']['signature_views'] != 3
            or c['hard_pl'] != dict(method='HPL2', threshold=.95, weight=1.)
            or c['hard_pl_label_smoothing'] != 0.
            or c.get('reference_scale', 1.) != 1.
            or c['physical_relation'] != dict(beta=row['beta'], l_weight=.05, u_weight=.05)):
        raise ValueError('PTBR source method, permissions or matched architecture differs')
    if any(c.get(k) for k in ('truth', 'target_truth', 'target_inputs', 'tests',
                              'views_root', 'checkpoint', 'teacher_checkpoint')):
        raise ValueError('Source worker rejects target and teacher handles')
    return row


def checked_parent(c):
    validate_source(c)
    # The pre-existing parent checker verifies the real payload and data roles;
    # only this in-memory validation descriptor uses its registered row name.
    old = next(r for r in old_plan.candidate_rows() if r['row_id'].startswith('pl_lu-'))
    probe = dict(c, run_id=old_plan.RUN, row_id=old['row_id'], branch_row=old,
                 branch_relation=dict(l_weight=.05, u_weight=.05))
    return old_checked_parent(probe)


def train(c):
    from experiments.cric_dura_comparison.f_train import train as segment
    parent = checked_parent(c)
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'resolved_config.json', c)
    segments, states, previous, start = [parent], [], parent['folder'], time.monotonic()
    for epochs, offset in ((10, 20), (20, 30)):
        folder = root / ('segment' + str(offset + epochs))
        cfg = dict(c, output_root=str(folder), segment_epochs=epochs, epoch_offset=offset,
                   step_offset=offset * 222, parent_source=previous)
        holder = {}
        with installed(cfg, holder):
            segment(cfg)
        done = read(folder / 'complete.json')
        state = holder['objective'].state_dict()
        model = holder['model']
        if (done['status'] != 'SOURCE_FROZEN' or done['updates'] != epochs * 222
                or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
                or state['calls'] != epochs * 222 or state['student_views'] != state['seen']
                or state['EMA_updates'] or state['U_truth_read']
                or state['physical_U']['selected_samples'] != epochs * (221 * 128 + 124)
                or not 0 <= state['physical_U']['selected_accepted'] <= state['physical_U']['selected_samples']
                or state['physical_U']['no_grad_forward_packets'] != 2 * state['physical_U']['selected_samples']
                or model._physical_L_calls != epochs * 222
                or model._physical_L_samples != 64 * epochs * 222):
            raise ValueError('PTBR source exposure, trajectory or permissions differ')
        state['physical_L'] = dict(calls=model._physical_L_calls, selected_samples=model._physical_L_samples,
                                   no_grad_forward_packets=2 * model._physical_L_samples)
        write(folder / 'physical_state.json', state)
        states.append(state)
        segments.append(dict(folder=str(folder), **done))
        previous = str(folder)
        write(root / 'progress.json', dict(status='SEGMENT_COMPLETE', epoch=offset + epochs,
                                           updates=(offset + epochs) * 222), exclusive=False)
    shutil.copyfile(Path(previous) / 'final_ssdg.pth', root / 'final_ssdg.pth')
    write(root / 'source_contract.json', read(c['source_contract']))
    write(root / 'complete.json', dict(status='SOURCE_FROZEN', config=c, epoch=50, updates=11100,
          segments=segments, initialization='same_contract_scratch20_own10_own20',
          ancestors=[s['folder'] for s in segments], source_contract=c['source_contract'],
          target_access=False, U_truth_read=False, head_fit_updates=0, physical_states=states,
          seconds=time.monotonic() - start, cumulative_training_seconds=sum(s['seconds'] for s in segments)))
