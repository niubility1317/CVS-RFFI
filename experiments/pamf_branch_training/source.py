"""Matched F20/F21 E20 continuations; inputs are source-only."""
from pathlib import Path
import shutil
import time

from experiments.cric_dura_comparison.io import read, write
from .plan import RUN, SEED, validate_candidate
from .training_scope import installed


def validate_source(c):
    row = validate_candidate(c['branch_row'])
    if (c['run_id'] != RUN or c['row_id'] != row['row_id'] or c['model_seed'] != SEED
            or c['source_rxs'] != row['source_rxs'] or c['source_days'] != row['source_days']
            or c['num_classes'] != 6 or c['epochs'] != 50 or c['optimizer_updates'] != 11100
            or c['branch_relation'] != dict(l_weight=row['l_weight'], u_weight=row['u_weight'])
            or c['ablation'] != ('F20' if row['regime'] == 'sup' else 'F21')
            or c['U_use'] != (row['regime'] == 'pl') or c['own_EMA'] or c['native_U_gate']
            or c['posthoc_head_fit'] or c['paper_row']['structure'] != 'full'
            or c['paper_row']['permuted'] or c['paper_row']['signature_views'] != 3
            or bool(c.get('hard_pl')) != bool(row['u_weight'])):
        raise ValueError('Source method, network, schedule, or data contract differs')
    if any(c.get(k) for k in ('truth', 'target_truth', 'target_inputs', 'tests',
                              'views_root', 'checkpoint', 'teacher_checkpoint')):
        raise ValueError('Source training must not receive target or teacher handles')
    return row


def checked_parent(c):
    """Check actual source contract, scratch ancestry and payload before reuse."""
    import torch
    row = validate_source(c)
    folder = Path(c['shared_prefix'])
    done = read(folder / 'complete.json')
    cfg = done['config']
    if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 20 or done['updates'] != 4440
            or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
            or done['initialization'] != 'scratch' or done['ancestors'] != []
            or cfg['ablation'] != c['ablation'] or cfg['model_seed'] != SEED
            or read(cfg['source_contract']) != read(c['source_contract'])):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH_OR_PROVENANCE')
    payload = torch.load(folder / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (payload['epoch'] != 20 or payload['a1_ema_successful_updates'] != 4440
            or payload['run_id'] != cfg['run_id'] or payload['candidate_id'] != cfg['row_id']
            or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt']
            or payload['args']['use_unlabeled'] or payload['args']['use_ema_teacher']):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    return dict(folder=str(folder), **done)


def train(c):
    from experiments.cric_dura_comparison.f_train import train as segment
    parent = checked_parent(c)
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'resolved_config.json', c)
    segments, states = [parent], []
    previous = parent['folder']
    start = time.monotonic()
    for epochs, offset in ((10, 20), (20, 30)):
        folder = root / ('segment' + str(offset + epochs))
        cfg = dict(c, output_root=str(folder), segment_epochs=epochs, epoch_offset=offset,
                   step_offset=offset * 222, parent_source=previous)
        holder = {}
        with installed(cfg, holder, branch_u_enabled=bool(c['branch_relation']['u_weight'])):
            segment(cfg)
        done = read(folder / 'complete.json')
        if (done['status'] != 'SOURCE_FROZEN' or done['updates'] != epochs * 222
                or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']):
            raise ValueError('Candidate source segment incomplete or unauthorized')
        if c['branch_relation']['u_weight']:
            state = holder['objective'].state_dict()
            if (state['calls'] != epochs * 222 or state['EMA_updates'] or state['U_truth_read']
                    or state['student_views'] != state['seen']):
                raise ValueError('U branch training did not execute as registered')
            write(folder / 'branch_u_state.json', state)
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
          target_access=False, U_truth_read=False, head_fit_updates=0,
          branch_U_states=states, seconds=time.monotonic() - start,
          cumulative_training_seconds=sum(s['seconds'] for s in segments)))
