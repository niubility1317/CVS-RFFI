"""Source-only fixed E30 fork, complete paired-view accounting and cumulative cost."""
from pathlib import Path
import shutil
import time
import torch
from experiments.cric_dura_comparison.io import read, write
from .paired_u import validate, PARENT_RUN, PARENT_ALGORITHM
from .view_anchor_train import source_parents
from .train import installed


def train(c, device='cuda:0'):
    parents, contract = source_parents(c, validate_config=validate)
    from experiments.cric_dura_comparison.f_train import train as segment
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'resolved_config.json', c)
    folder = root/'segment50'
    cfg = dict(c, output_root=str(folder), segment_epochs=20, epoch_offset=30,
               step_offset=6660, parent_source=str(Path(c['fork_parent_source'])/'segment30'))
    holder = {}
    start = time.perf_counter()
    with installed(cfg, holder, parent_algorithm=PARENT_ALGORITHM):
        segment(cfg, device=device)
    state = holder['objective'].state_dict()
    expected = dict(ema_updates=11100, seen=2097900, physical_seen=1615950, calls=8214,
        paired_calls=4440, paired_clean_seen=1134000, paired_physical_seen=1134000, student_U_views=3231900)
    if any(state.get(k)!=v for k,v in expected.items()) or int(state['counts'].sum())!=1051392:
        raise ValueError('Full E50 paired-U/teacher/labeled-prototype exposure differs')
    torch.save(state, folder/'ssl_state.pt')
    done = read(folder/'complete.json')
    actual = dict(method=c['algorithm_id'], native_U_loop=False, own_EMA=True,
        cumulative_EMA_updates=state['ema_updates'], U_seen=state['seen'], physical_U_seen=state['physical_seen'],
        effective_U_weight=state['effective'], effective_U_per_class=state['per_class'].tolist(),
        U_truth_read=False, labeled_only_prototype_updates=True,
        **{k:state[k] for k in ('paired_calls','paired_clean_seen','paired_physical_seen','student_U_views')})
    done['actual_adapter_SSL'] = actual
    write(folder/'complete.json', done, exclusive=False)
    segments = parents+[dict(folder=str(folder), **done)]
    shutil.copyfile(folder/'final_ssdg.pth', root/'final_ssdg.pth')
    write(root/'source_contract.json', contract)
    seconds = time.perf_counter()-start
    write(root/'complete.json', dict(status='SOURCE_FROZEN', method_id=c['algorithm_id'],
        scaffold_id=c['scaffold_id'], epoch=50, updates=11100, segments=segments,
        source_contract=c['source_contract'], config=c, initialization='same_contract_scratch20_own10_own20',
        ancestors=[s['folder'] for s in segments], actual_adapter_SSL=actual,
        head_fit_updates=0, U_truth_read=False, target_access=False, seconds=seconds,
        incremental_epochs=20, incremental_updates=4440, parent_prefix_native_seconds=sum(s['seconds'] for s in parents),
        parent_prefix_training_seconds=c['parent_prefix_training_seconds'],
        cumulative_training_seconds=c['parent_prefix_training_seconds']+seconds,
        parent_run_id=PARENT_RUN, parent_selection='Fixed RXPSM E30; paired-U hypothesis before VIEWANCHOR target access'))
    write(root/'progress.json', dict(status='SOURCE_FROZEN', epoch=50, updates=11100,
        incremental_epochs=20, incremental_updates=4440, seconds=seconds, actual_adapter_SSL=actual))
