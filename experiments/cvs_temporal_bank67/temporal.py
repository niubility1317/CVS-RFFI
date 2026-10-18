"""Observe the existing native epoch bank without changing its returned mask."""
import ast
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch

FIELDS = ('rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index')


def plain(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, 'tolist') else list(value)


def load_native_function(path):
    """Compile only three existing tensor/metadata helpers for source-free toys."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    names = ('_as_plain_list', '_meta_from_extra', '_temporal_bank_mask_tensor')
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Pinned native epoch-bank interface is unavailable; no fallback')
    namespace = dict(torch=torch, Any=Any, Dict=Dict, List=List, Tuple=Tuple, Mapping=Mapping)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_temporal_bank_mask_tensor']


class TemporalTrace:
    """Metadata-only trace, external to the student/EMA and the native bank."""
    def __init__(self, emit, steps_per_epoch=222, allowed_rx=(1, 3, 4, 6, 8)):
        self.emit = emit
        self.steps_per_epoch = steps_per_epoch
        self.allowed_rx = set(allowed_rx)
        self.bank = None
        self.calls = self.packets = self.selected = self.eligible = 0
        self.last_bank_size = 0
        self.epochs = {}

    def call(self, native, pseudo, conf, extra, args, device, *, epoch, bank, step):
        if not callable(native):
            raise ValueError('Existing native epoch-bank function required; no fallback')
        if (getattr(args, 'pseudo_temporal_mode', None) != 'epoch_bank'
                or not getattr(args, 'pseudo_temporal_gate', False)
                or float(args.pseudo_temporal_min_conf) != .8
                or int(args.pseudo_temporal_bank_min_streak) != 2):
            raise ValueError('Only the existing native epoch-bank/min-streak2 contract is allowed')
        if step != self.calls + 1 or epoch != (step - 1) // self.steps_per_epoch + 1:
            raise ValueError('Exactly one ordered native U bank call per successful source update')
        if self.bank is None:
            if not isinstance(bank, dict) or bank:
                raise ValueError('Each own20+10 row must start with a new empty native temporal bank')
            self.bank = bank
        elif bank is not self.bank or len(bank) != self.last_bank_size:
            raise ValueError('The row must exclusively retain its own native temporal bank')
        if not isinstance(extra, (tuple, list)) or len(extra) < 2 or not isinstance(extra[1], Mapping):
            raise ValueError('Original hidden U physical metadata is required')
        pred, confidence = [int(v) for v in plain(pseudo)], [float(v) for v in plain(conf)]
        n = len(pred)
        meta = extra[1]
        fields = {k: plain(meta[k]) for k in FIELDS if k in meta}
        if (not 0 < n <= 256 or len(confidence) != n or set(fields) != set(FIELDS)
                or any(len(v) != n for v in fields.values())
                or any(p not in range(6) for p in pred)
                or any(not math.isfinite(c) or not 0 <= c <= 1 for c in confidence)):
            raise ValueError('Full hidden-U metadata/finite six-class teacher predictions required')
        keys = [tuple(int(fields[k][i]) for k in FIELDS) for i in range(n)]
        if any(k[0] not in self.allowed_rx or k[1] not in (1, 2, 3) or k[2] != 1 for k in keys):
            raise ValueError('Only original source RX/day/equalized1 U keys are legal')
        size_before = len(bank)
        before = {key: bank.get(key) for key in set(keys)}
        result = native(pseudo, conf, extra, args, device, epoch=epoch, bank=bank)
        if result.dtype != torch.bool or result.shape != (n,):
            raise ValueError('The original native temporal mask shape changed')
        passed = [bool(v) for v in plain(result)]
        # Replay only native integer state transitions for the visited keys to
        # audit duplicates. The returned mask is always the untouched native mask.
        shadow = dict(before)
        entries = []
        same_epoch = consecutive_same = 0
        histogram = Counter()
        for key, prediction, c, keep in zip(keys, pred, confidence, passed):
            previous = shadow[key]
            streak = 1
            if previous is not None:
                old_prediction, old_epoch, old_streak = map(int, previous)
                same_epoch += int(old_epoch == epoch)
                if old_prediction == prediction and old_epoch == epoch - 1:
                    streak = old_streak + 1
                    consecutive_same += 1
                elif old_prediction == prediction and old_epoch == epoch:
                    streak = old_streak
            after = (prediction, int(epoch), streak) if c >= .8 else previous
            if previous is not None and int(previous[1]) == epoch and after is not None and int(after[2]) > int(previous[2]):
                raise ValueError('A repeated same-epoch visit incremented the temporal streak')
            if keep != (c >= .8 and streak >= 2) or (epoch == 1 and keep):
                raise ValueError('Native cross-epoch mask or first-epoch empty-bank behavior differs')
            shadow[key] = after
            histogram[str(after[2] if after is not None else 0)] += 1
            entries.append(dict(key=list(key), pseudo=prediction, confidence=c,
                previous=list(previous) if previous is not None else None,
                after=list(after) if after is not None else None, temporal_pass=keep))
        if any(bank.get(key) != value for key, value in shadow.items()):
            raise ValueError('Native bank states differ from recorded physical-ID transitions')
        expected_added = sum(before[key] is None and shadow[key] is not None for key in shadow)
        if len(bank) != size_before + expected_added:
            raise ValueError('Native bank changed outside the original U batch keys')
        selected, eligible = sum(passed), sum(c >= .8 for c in confidence)
        self.calls += 1
        self.packets += n
        self.selected += selected
        self.eligible += eligible
        self.last_bank_size = len(bank)
        record = dict(step=step, cumulative_step=4440 + step, epoch=epoch, cumulative_epoch=20 + epoch,
            packets=n, unique_batch_keys=len(set(keys)), bank_size_before=size_before, bank_size_after=len(bank),
            confidence_eligible=eligible, temporal_selected=selected, min_conf=.8, min_streak=2,
            streak_histogram=dict(histogram), same_epoch_repeats=same_epoch,
            cross_epoch_same_prediction=consecutive_same, first_epoch_all_false=True,
            no_same_epoch_streak_growth=True, entries=entries)
        e = self.epochs.setdefault(epoch, dict(epoch=epoch, steps=0, packets=0, temporal_selected=0, confidence_eligible=0))
        for key, value in [('steps', 1), ('packets', n), ('temporal_selected', selected), ('confidence_eligible', eligible)]:
            e[key] += value
        e.update(bank_size=len(bank), streak_histogram=dict(Counter(str(v[2]) for v in bank.values()))) if step % self.steps_per_epoch == 0 else None
        self.emit(record)
        return result

    def summary(self):
        return dict(steps=self.calls, packets=self.packets, temporal_selected=self.selected,
            confidence_eligible=self.eligible, bank_size=self.last_bank_size,
            bank_empty_at_segment_start=True, bank_owned_exclusively=True,
            first_epoch_all_false=True, no_same_epoch_streak_growth=True,
            epoch_summaries=[dict(v) for _, v in sorted(self.epochs.items())],
            physical_key=list(FIELDS), temporal_mask_is_final_pseudo_mask=False,
            native_algorithm_unchanged=True, extra_NN=0, U_TX_truth_read=False)
