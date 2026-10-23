"""Reuse the predeclared RXPSM prefix without rewriting any parent artifacts."""
from pathlib import Path
import shutil
import time
import torch

from experiments.cric_dura_comparison.io import read, write
from .plan import validate as validate_scaffold
from .train import installed
from .view_anchor import validate, PARENT_RUN, PARENT_ALGORITHM


def validate_parent_metadata(c, parents, contract, *, validate_config=validate):
    validate_config(c)
    if len(parents) != 2:
        raise ValueError('Fixed source E20/E30 ancestry required')
    previous = None
    for (epochs, offset, end), (folder, done, actual) in zip(((20, 0, 20), (10, 20, 30)), parents):
        pc = done['config']
        validate_scaffold(pc)
        if (str(folder) != str(Path(c['fork_parent_source']) / ('segment%d' % end))
                or done.get('status') != 'SOURCE_FROZEN' or done.get('epoch') != epochs
                or done.get('updates') != epochs * 222 or done.get('target_access') is not False
                or done.get('U_truth_read') is not False or done.get('head_fit_updates') != 0
                or actual != contract or pc.get('model_seed') != c['model_seed']
                or pc.get('run_id') != PARENT_RUN or pc.get('row_id') != 'RXPSM-s' + str(c['model_seed'])
                or pc.get('algorithm_id') != PARENT_ALGORITHM or pc.get('ssl_method') != 'RXPSM'
                or pc.get('view_anchor') or pc.get('ssl') != c['ssl']
                or pc.get('step_offset') != offset * 222 or pc.get('epoch_offset') != offset
                or pc.get('parent_source') != previous):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: fixed source-only prefix')
        previous = str(folder)
    return parents


def source_parents(c, *, validate_config=validate):
    """No target inputs, truth, scores or final50 selection are read."""
    validate_config(c)
    contract = read(c['source_contract'])
    parents = []
    for end in (20, 30):
        folder = Path(c['fork_parent_source']) / ('segment%d' % end)
        done = read(folder / 'complete.json')
        parents.append((folder, done, read(done['config']['source_contract'])))
    validate_parent_metadata(c, parents, contract, validate_config=validate_config)
    previous = None
    for folder, done, _ in parents:
        payload = torch.load(folder / 'final_ssdg.pth', map_location='cpu', weights_only=False)
        pc = done['config']
        if (payload.get('run_id') != pc['run_id'] or payload.get('candidate_id') != pc['row_id']
                or payload.get('epoch') != done['epoch']
                or payload.get('a1_ema_successful_updates') != done['updates']
                or payload['args']['baseline_ckpt'] != ('' if previous is None else previous + '/final_ssdg.pth')
                or payload['args']['teacher_ckpt']):
            raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: fixed parent checkpoint')
        previous = str(folder)
    state = torch.load(parents[-1][0] / 'ssl_state.pt', map_location='cpu', weights_only=False)
    if (state['method'] != 'RXPSM' or state['ema_updates'] != 6660 or state['seen'] != 963900
            or state['physical_seen'] != 481950 or state['calls'] != 3774
            or int(state['counts'].sum()) != 483072 or state.get('view_anchor')):
        raise ValueError('Fixed RXPSM E30 teacher/prototype state required')
    return [dict(folder=str(folder), **done) for folder, done, _ in parents], contract


def train(c, device='cuda:0'):
    parents, contract = source_parents(c)
    from experiments.cric_dura_comparison.f_train import train as segment
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'resolved_config.json', c)
    folder = root / 'segment50'
    cfg = dict(c, output_root=str(folder), segment_epochs=20, epoch_offset=30,
               step_offset=6660, parent_source=str(Path(c['fork_parent_source']) / 'segment30'))
    holder = {}
    start = time.perf_counter()
    with installed(cfg, holder, parent_algorithm=PARENT_ALGORITHM):
        segment(cfg, device=device)
    state = holder['objective'].state_dict()
    if (state['ema_updates'] != 11100 or state['seen'] != 2097900 or state['calls'] != 8214
            or state['view_anchor_calls'] != 4440 or state['view_anchor_physical_seen'] != 567000):
        raise ValueError('Full declared final20 physical-feature budget missing')
    torch.save(state, folder / 'ssl_state.pt')
    done = read(folder / 'complete.json')
    done['actual_adapter_SSL'] = dict(method=c['algorithm_id'], native_U_loop=False, own_EMA=True,
        cumulative_EMA_updates=state['ema_updates'], U_seen=state['seen'],
        physical_U_seen=state['physical_seen'], effective_U_weight=state['effective'],
        effective_U_per_class=state['per_class'].tolist(), U_truth_read=False,
        labeled_only_prototype_updates=True, view_anchor_calls=state['view_anchor_calls'],
        view_anchor_physical_seen=state['view_anchor_physical_seen'])
    write(folder / 'complete.json', done, exclusive=False)
    segments = parents + [dict(folder=str(folder), **done)]
    shutil.copyfile(folder / 'final_ssdg.pth', root / 'final_ssdg.pth')
    write(root / 'source_contract.json', contract)
    seconds = time.perf_counter() - start
    write(root / 'complete.json', dict(status='SOURCE_FROZEN', method_id=c['algorithm_id'],
        scaffold_id=c['scaffold_id'], epoch=50, updates=11100, segments=segments,
        source_contract=c['source_contract'], config=c, initialization='same_contract_scratch20_own10_own20',
        ancestors=[s['folder'] for s in segments], actual_adapter_SSL=done['actual_adapter_SSL'],
        head_fit_updates=0, U_truth_read=False, target_access=False,
        seconds=seconds, incremental_epochs=20, incremental_updates=4440,
        parent_prefix_native_seconds=sum(s['seconds'] for s in parents),
        parent_prefix_training_seconds=c.get('parent_prefix_training_seconds'),
        cumulative_training_seconds=None if c.get('parent_prefix_training_seconds') is None else c['parent_prefix_training_seconds'] + seconds,
        parent_run_id=PARENT_RUN, parent_selection='Fixed RXPSM E30 before target scoring'))
    write(root / 'progress.json', dict(status='SOURCE_FROZEN', epoch=50, updates=11100,
        incremental_epochs=20, incremental_updates=4440, seconds=seconds,
        actual_adapter_SSL=done['actual_adapter_SSL']))
