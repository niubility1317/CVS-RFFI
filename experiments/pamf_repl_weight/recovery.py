"""Resume the unchanged candidate after its complete E30 proxy failure."""
import gzip
import json
import shutil
import time
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import plan
from .objective import installed, source_validation_proxy


def recover_e30_telemetry(folder, classes):
    folder = Path(folder)
    n = 0; last = None
    with gzip.open(folder/'step_metrics.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line); n += 1
                if item['step'] != n:
                    raise ValueError('E30 source step continuity differs')
                last = item['metrics']
    if n != 2220 or last is None:
        raise ValueError('Completed E30 step telemetry missing')
    seen, accepted = int(last['HPL_seen']), int(last['HPL_accepted'])
    hist = {str(i): int(last[f'HPL_pseudo_class_{i}']) for i in range(classes)}
    bins = {k:int(last['HPL_conf_bin_'+k]) for k in ('below90','90to95','95to99','99plus')}
    progress = read(folder/'progress.json')
    if (int(last['HPL_calls']) != n or int(last['HPL_student_views']) != 6*seen
            or int(last['HPL_EMA_updates']) or progress['PL_seen'] != seen
            or progress['PL_accepted'] != accepted or sum(hist.values()) != accepted or sum(bins.values()) != seen):
        raise ValueError('E30 counters or U permission do not reconcile')
    return dict(calls=n, seen=seen, accepted=accepted, student_views=6*seen, EMA_updates=0,
                U_truth_read=False, U_pseudo_observation=dict(accepted_by_pseudo_class=hist,
                confidence_bins=bins, U_truth_read=False), recovered_from_complete_step_log=True)


def proxy_checkpoint(c, payload, out):
    import torch
    from torch import nn
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.pamf_paper_completion.model import reader_scope
    class Adapter(nn.Module):
        def __init__(self, reader):
            super().__init__(); self.reader = reader
        def forward(self, x):
            return self.reader.classify_features(self.reader.features(x))
    with reader_scope(c['paper_row']):
        reader = reader_from_payload(c, payload, 'cuda:0')
    # Test and run the same unpatched CoordinateSinc path as the live trainer.
    # reader_from_payload installs an inference-safe override by default; keeping
    # it would hide the exact E30 failure that this recovery must verify.
    sinc = reader.core.id_backbone.sinc
    if '_filters' not in sinc.__dict__:
        raise ValueError('Expected frozen-reader Sinc override is missing')
    delattr(sinc, '_filters')
    model = Adapter(reader).cuda().eval()
    cfg = dict(c, output_root=str(out))
    source_validation_proxy(cfg, model)
    del model, reader
    torch.cuda.empty_cache()


def train(c):
    from experiments.cric_dura_comparison.f_train import train as segment
    prefix = plan.parent(c)
    previous, payload = plan.recovered_e30(c)
    previous_state = recover_e30_telemetry(previous['folder'], c['num_classes'])
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    write(root/'resolved_config.json', c)
    write(root/'recovered_e30_state.json', previous_state)
    started = time.monotonic()
    proxy_checkpoint(c, payload, root/'recovered_e30_v')
    del payload
    folder = root/'segment50'
    cfg = dict(c, output_root=str(folder), segment_epochs=20, epoch_offset=30,
               step_offset=6660, parent_source=previous['folder'])
    holder = {}
    with installed(cfg, holder):
        segment(cfg)
    done = read(folder/'complete.json')
    state = holder['objective'].state_dict()
    if (done['status'] != 'SOURCE_FROZEN' or done['updates'] != 4440 or done['target_access']
            or done['U_truth_read'] or done['head_fit_updates'] or state['calls'] != 4440
            or state['student_views'] != 6*state['seen'] or state['U_truth_read'] or state['EMA_updates']):
        raise ValueError('Recovery E50 continuation incomplete')
    write(folder/'repl_state.json', state)
    segments = [prefix, previous, dict(folder=str(folder), **done)]
    shutil.copyfile(folder/'final_ssdg.pth', root/'final_ssdg.pth')
    write(root/'source_contract.json', read(c['source_contract']))
    write(root/'complete.json', dict(status='SOURCE_FROZEN', epoch=50, updates=11100, config=c,
        segments=segments, initialization='same_contract_scratch20_own10_own20',
        ancestors=[s['folder'] for s in segments], source_contract=c['source_contract'],
        target_access=False, U_truth_read=False, head_fit_updates=0,
        actual_REPL={k:sum(s[k] for s in (previous_state, state)) for k in ('calls','seen','accepted','student_views')},
        seconds=time.monotonic()-started, cumulative_training_seconds=sum(s['seconds'] for s in segments),
        recovered_e30_run=plan.PREVIOUS_RUN))
