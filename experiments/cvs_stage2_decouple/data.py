"""One source-only CPU cache per parent seed; source V is validation-only."""
from contextlib import contextmanager
import time

import torch
from torch.utils.data import Dataset, DataLoader

from . import design as d
from .actions import signal_state, logit_margin

SOURCE_COUNTS = (6300, 27000)


@contextmanager
def _reference_eval(parent):
    modes = [(module, module.training) for module in parent.modules()]
    try:
        parent.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def _preprocessing(dataset):
    chain = []
    current = dataset
    while True:
        chain.append(current)
        if not hasattr(current, "base"):
            break
        current = current.base
    if any(getattr(item, "transform", None) is not None for item in chain):
        raise ValueError("Stage2 cache cannot silently include a dataset transform")
    expected = dict(out_len=256, crop_mode="center", normalize=True, center=False)
    if any(getattr(current, name, None) != value for name, value in expected.items()):
        raise ValueError("Native source preprocessing differs from fixed parent recipe")
    return dict(**expected, normalization="per-packet complex RMS after center crop/pad; epsilon 1e-12",
                identity_source="opaque_id(dataset.index[i])", cached_order="exact dataset index order")


def _cache_dataset(dataset, role, opaque_id, expected_ids, parent=None, device="cpu", batch_size=128):
    if role not in ("L_s", "V"):
        raise ValueError("Only visible source L or source V may be materialized")
    if (role == "L_s") != (parent is not None):
        raise ValueError("Only source L may receive a parent forward/cache statistics")
    preprocessing = _preprocessing(dataset)
    ids = [opaque_id(item) for item in dataset.index]
    if len(ids) != len(dataset) or len(set(ids)) != len(ids) or sorted(ids) != list(expected_ids):
        raise ValueError("Cached physical IDs differ from exact registered role")
    x_rows, labels, domains, receivers, days = [], [], [], [], []
    for i, item in enumerate(dataset.index):
        x, y, domain, meta = dataset[i][:4]
        if (int(y) != int(item.tx_i) or int(meta["rx_i"]) != int(item.rx_i)
                or int(meta["day_i"]) != int(item.day_i)):
            raise ValueError("Dataset item metadata does not match physical index order")
        x = torch.as_tensor(x).detach().cpu().float().contiguous()
        if x.shape != (2, 256) or not bool(torch.isfinite(x).all()):
            raise ValueError("Nonfinite or malformed source IQ")
        x_rows.append(x)
        labels.append(int(y))
        domains.append(int(domain))
        receivers.append(int(item.rx_i))
        days.append(int(item.day_i))
    cache = dict(x=torch.stack(x_rows), y=torch.tensor(labels, dtype=torch.long),
                 domain=torch.tensor(domains, dtype=torch.long), rx=torch.tensor(receivers, dtype=torch.long),
                 day=torch.tensor(days, dtype=torch.long), ids=ids, role=role,
                 preprocessing=preprocessing)
    if role == "L_s":
        states, parent_margins, correct = [], [], []
        with _reference_eval(parent):
            for start in range(0, len(ids), batch_size):
                batch = cache["x"][start:start + batch_size].to(device)
                y = cache["y"][start:start + batch_size].to(device)
                logits = parent(batch)
                states.append(signal_state(batch).detach().cpu().float())
                parent_margins.append(logit_margin(logits, y).detach().cpu().float())
                correct.append((logits.argmax(1) == y).detach().cpu())
        cache["state"] = torch.cat(states)
        cache["parent_margin"] = torch.cat(parent_margins)
        cache["parent_correct"] = torch.cat(correct)
        if not bool(torch.isfinite(cache["state"]).all() and torch.isfinite(cache["parent_margin"]).all()):
            raise ValueError("Nonfinite parent source L cache")
    return cache


def build_source_cache(c, parent, device):
    """Read native role indices, materialize only L/V, then release native U.

    ``parent`` is the strictly loaded identity view.  The returned dictionary
    owns CPU tensors only and retains no native dataset, U iterator or parent.
    The caller writes one seed preparation artifact reused by C0/C1/C2/C3.
    """
    d.validate(c)
    expected = d.read(d.SOURCE)
    started = time.perf_counter()
    from experiments.cvs_phase1_stack.runtime import installed
    with installed(c) as native:
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        from cvsrffi.game_tracking.data import opaque_id
        args = d.make_args(c, str(device))
        if (not args.a1_source_screen_only or args.use_unlabeled or args.use_ema_teacher
                or args.use_concat_sat_channel_aug):
            raise ValueError("Source cache attempted an unregistered U/teacher/satellite/target path")
        context = native._build_ssdg_wisig_data(args, torch.device(device))
        if context["named_test_loaders"] or role_ids_from_native(context) != expected["role_ids"]:
            raise ValueError("Physical source roles must EXACT_MATCH; target loaders must be absent")
        # U is inspected exclusively by role_ids_from_native through dataset
        # indices. No U sample, iterator, label tensor or U cache is returned.
        labeled = _cache_dataset(context["train_loader"].dataset, "L_s", opaque_id,
                                 expected["role_ids"]["L_s"], parent, device)
        validation = _cache_dataset(context["val_loader"].dataset, "V", opaque_id,
                                    expected["role_ids"]["V"])
        if (len(labeled["ids"]), len(validation["ids"])) != SOURCE_COUNTS:
            raise ValueError("Stage2 source cache must contain L_s=6300 and V=27000")
        del context
    return dict(labeled=labeled, validation=validation, contract=dict(expected,native_role_comparison='EXACT_MATCH'),
                diagnostics=dict(source_roles="EXACT_MATCH", labeled_count=len(labeled["ids"]),
                                 validation_count=len(validation["ids"]), unlabeled_sample_reads=0,
                                 target_loaders_constructed=False, V_parent_forwards=0,
                                 V_statistics_fitted=False, native_context_retained=False,
                                 elapsed_seconds=time.perf_counter() - started))


class CachedValidationDataset(Dataset):
    """Read-only sequential V view compatible with native source validation."""
    def __init__(self, cache):
        if cache.get("role") != "V":
            raise ValueError("The validation loader accepts only the source V cache")
        count = len(cache["ids"])
        if cache["x"].shape != (count, 2, 256) or cache["x"].device.type != "cpu":
            raise ValueError("Validation cache must hold CPU IQ [N,2,256]")
        if any(cache[name].shape != (count,) for name in ("y", "rx", "day", "domain")):
            raise ValueError("Validation metadata dimensions differ")
        if any(name in cache for name in ("state", "parent_margin", "parent_correct", "u0")):
            raise ValueError("Validation must not carry fitted/parent training statistics")
        self.cache = cache

    def __len__(self):
        return len(self.cache["ids"])

    def __getitem__(self, index):
        c = self.cache
        # Clone IQ to keep item-level consumers from editing the frozen cache.
        return (c["x"][index].clone(), int(c["y"][index]), int(c["domain"][index]),
                dict(rx_i=int(c["rx"][index]), day_i=int(c["day"][index])))


def cached_loader(vcache, batch_size=256):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive validation batch size required")
    return DataLoader(CachedValidationDataset(vcache), batch_size=batch_size, shuffle=False,
                      num_workers=0, drop_last=False, generator=torch.Generator().manual_seed(0))
