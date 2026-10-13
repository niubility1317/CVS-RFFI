"""Focused mathematical, label-isolation, and independent-fit checks."""
import argparse
import inspect
import json
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from . import transforms as t
from . import sets as s


def _raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected negative check to reject the input")


def checks(device="cpu"):
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(9417)
    P, _ = torch.linalg.qr(torch.randn(160, 2, generator=generator, dtype=torch.float64))
    B = torch.randn(5, 2, 2, generator=generator, dtype=torch.float64) * .025
    c = torch.randn(5, 2, generator=generator, dtype=torch.float64) * .3
    B[0].zero_(); c[0].zero_()
    h = torch.randn(40, 160, generator=generator, dtype=torch.float64)
    r = torch.arange(len(h)) % 5; dest = (r+1) % 5; third = (r+3) % 5
    same = t.shared_transform(h, r, r, P, B, c)
    forward = t.shared_transform(h, r, dest, P, B, c)
    inverse = t.shared_transform(forward, dest, r, P, B, c)
    composed = t.shared_transform(forward, dest, third, P, B, c)
    direct = t.shared_transform(h, r, third, P, B, c)
    for actual, expected in [(same, h), (inverse, h), (composed, direct)]:
        if not torch.allclose(actual, expected, atol=1e-11, rtol=1e-11):
            raise AssertionError("Receiver coordinate identity/inverse/composition failed")
    if "TX" in inspect.signature(t.shared_transform).parameters or "y" in inspect.signature(t.shared_transform).parameters:
        raise AssertionError("Shared transformation accepts identity labels")
    # Exact shared receiver effects are fitted without any held-TX/day endpoint.
    centers = {}; fitted_features = []
    for cell in range(24):
        latent = torch.randn(1, 160, generator=generator, dtype=torch.float64)
        latent[:, :3] += cell*.1
        values = []
        for rx in range(5):
            observed = t.shared_transform(latent, 0, rx, P, B, c)[0]
            values.append((observed, torch.tensor([cell*5+rx]), 1))
            fitted_features.append(observed)
        centers[(cell, 0, 0)] = values
    fitted = t._fit_coordinates(centers, torch.stack(fitted_features), 924)
    held = torch.randn(60, 160, generator=generator, dtype=torch.float64)
    observed = t.shared_transform(held, 0, 3, P, B, c)
    predicted = t.shared_transform(held, 0, 3, fitted["P"], fitted["B"], fitted["c"])
    zero_mse = float((held-observed).square().mean())
    fitted_mse = float((predicted-observed).square().mean())
    if not fitted_mse < zero_mse*.05:
        raise AssertionError("Shared fit failed to recover synthetic independent endpoints")
    for key in ("B", "random_B"):
        if float(torch.linalg.matrix_norm(fitted[key], ord=2).max()) > .150000001:
            raise AssertionError("Near-identity spectral constraint failed")
    if not torch.allclose(torch.linalg.svdvals(fitted["B"]), torch.linalg.svdvals(fitted["random_B"]), atol=1e-12):
        raise AssertionError("Random coordinate control does not match B spectrum")
    if not torch.allclose(fitted["c"].norm(dim=1), fitted["random_c"].norm(dim=1), atol=1e-12):
        raise AssertionError("Random coordinate control does not match translation magnitude")
    random_composed = t.shared_transform(t.shared_transform(h, r, dest, fitted["P"], fitted["random_B"], fitted["random_c"]), dest, third, fitted["P"], fitted["random_B"], fitted["random_c"])
    random_direct = t.shared_transform(h, r, third, fitted["P"], fitted["random_B"], fitted["random_c"])
    if not torch.allclose(random_composed, random_direct, atol=1e-11, rtol=1e-11):
        raise AssertionError("Random common coordinate control lacks exact composition")
    # State fitting ignores every held observation, including its scale/outliers.
    raw = torch.randn(80, 12, generator=generator)
    fit = torch.arange(80) < 40
    _, first_state = t._state_bins(raw, fit)
    changed = raw.clone(); changed[~fit] = changed[~fit]*10000+900000
    _, second_state = t._state_bins(changed, fit)
    for key in ("mean", "scale", "axis", "cuts"):
        if not torch.equal(first_state[key], second_state[key]):
            raise AssertionError("Held source observations entered state fitting")

    logits = torch.randn(12, 6, generator=generator, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(12) % 6
    singleton = F.one_hot(labels, 6).bool()
    if not torch.allclose(s.set_loss(logits, singleton), F.cross_entropy(logits, labels, reduction="none"), atol=1e-12):
        raise AssertionError("Singleton set loss differs from ordinary CE")
    full = torch.ones_like(singleton)
    full_loss = s.set_loss(logits, full)
    full_grad = torch.autograd.grad(full_loss.sum(), logits, retain_graph=True)[0]
    if not torch.equal(full_loss, torch.zeros_like(full_loss)) or not torch.equal(full_grad, torch.zeros_like(full_grad)):
        raise AssertionError("Whole-class set must have exactly zero loss and gradient")
    partial = singleton.clone(); partial[:, 4] = True
    formula = -(logits.softmax(1)*partial).sum(1).log()
    if not torch.allclose(s.set_loss(logits, partial), formula, atol=1e-12):
        raise AssertionError("Partial-set logsumexp formula differs")
    _raises(lambda: s.set_loss(logits, singleton.float()))
    _raises(lambda: s.set_loss(logits, singleton[:, :5]))
    _raises(lambda: s.set_loss(logits, torch.zeros_like(singleton)))
    _raises(lambda: s.class_set_mask([[6]]))
    _raises(lambda: s.class_set_mask([[-1]]))
    _raises(lambda: s.class_set_mask([[]]))
    _raises(lambda: s.class_set_mask([[True]]))
    p = torch.tensor([[.96,.01,.01,.01,.005,.005], [.50,.46,.01,.01,.01,.01], [1/6]*6], dtype=torch.float64)
    prefixes = s.prefix_sets(p)
    if prefixes.sum(1).tolist() != [1, 2, 6]:
        raise AssertionError("Minimal .95 cumulative prefixes differ")
    other = p.clone(); other[1] = torch.tensor([.50,.01,.46,.01,.01,.01])
    union, one, two = s.teacher_sets(p, other)
    if union.sum(1).tolist() != [1, 3, 6] or not (union >= one).all() or not (union >= two).all():
        raise AssertionError("Conservative teacher-view union differs")
    student = torch.randn(3, 6, generator=generator, dtype=torch.float64, requires_grad=True)
    teacher1 = p.clone().requires_grad_(); teacher2 = other.clone().requires_grad_()
    paired = s.matched_losses(student, teacher1, teacher2)
    sum(paired["mean_losses"].values()).backward()
    if teacher1.grad is not None or teacher2.grad is not None or student.grad is None:
        raise AssertionError("Teacher graph isolation or student gradient failed")
    if paired["participation"].tolist() != [True, True, False]:
        raise AssertionError("Matched small-set mask differs")
    for key, values in paired["losses"].items():
        if values[2] != 0 or not torch.equal(paired["mean_losses"][key], values.sum()/3):
            raise AssertionError("Hard/soft/set must share participation and full denominator")
    wrong = torch.tensor([0, 2, 3], dtype=torch.long)
    summary = s._summary(p, other, union, one, two, torch.ones(3, dtype=torch.bool), wrong)
    hidden = s._summary(p, other, union, one, two, torch.ones(3, dtype=torch.bool), None)
    if summary["false_exclusion_count"] != 0 or summary["ambiguous_top1_error_truth_inclusion"] != 1:
        raise AssertionError("Independent lawful-source set coverage recount differs")
    if hidden["truth_coverage"] is not None or hidden["top1_errors"] is not None:
        raise AssertionError("Hidden U diagnostic exposed correctness")
    # Exercise the real diagnose entry with the existing ten-field hidden-U schema.
    # The teacher has no label input, and a guarded L/V cache rejects early label reads.
    class ToyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.calls = 0
        def forward(self, x):
            self.calls += 1
            return x[:, 0, :6]
    teacher = ToyTeacher().eval()
    class GuardedSource(dict):
        def __init__(self, data, minimum_calls):
            super().__init__(data); self.minimum_calls = minimum_calls
        def __getitem__(self, key):
            if key == "y" and teacher.calls < self.minimum_calls:
                raise AssertionError("Source truth was read before teacher-set construction")
            return super().__getitem__(key)
    entries = [(tx, rx, day, packet) for tx in range(6) for rx in t.d.RX
               for day in t.d.DAYS for packet in range(3)]
    count = len(entries)
    target_y = torch.tensor([v[0] for v in entries])
    q = torch.full((count, 6), .008)
    q[torch.arange(count), target_y] = .96
    x = torch.zeros(count, 2, 256); x[:, 0, :6] = q.log()
    source = dict(x=x, y=target_y, rx=torch.tensor([v[1] for v in entries]),
                  day=torch.tensor([v[2] for v in entries]),
                  ids=[f"source-{i}" for i in range(count)])
    hidden_u = {key: value for key, value in source.items() if key != "y"}
    hidden_u.update(ids=[f"U-{i}" for i in range(count)], domain=torch.arange(count) % 5,
                    eq=torch.zeros(count, dtype=torch.long), sig=torch.arange(count),
                    base_index=torch.arange(count), role="U_s", preprocessing="synthetic_fixture")
    calls_per_role = 2*((count+t.d.BATCH_SIZE-1)//t.d.BATCH_SIZE)
    context = dict(model=teacher, seed=47, L=GuardedSource(source, calls_per_role),
                   V=GuardedSource(dict(source, ids=[f"V-{i}" for i in range(count)]), calls_per_role*2),
                   U=hidden_u, probabilities={role:q for role in ("L_s", "V", "U_s")},
                   folds=torch.arange(count) % 3)
    root = t.d.ROOT/"local_artifacts"; root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sets_diagnostic_fixture_", dir=root) as directory:
        report = s.diagnose(context, Path(directory)/"sets", "cpu")
        saved = torch.load(Path(directory)/"sets"/"audit.pt", weights_only=False)
        if report["roles"]["U_s"]["summary"]["truth_coverage"] is not None or "legal_source_TX_after_set_construction" in saved["roles"]["U_s"]:
            raise AssertionError("Full diagnostic audit exposed hidden U correctness")
        if report["extra_identity_forward_samples"] != count*6 or report["identity_updates"]:
            raise AssertionError("Fixed source-view forward count or identity-update count differs")
        if report["roles"]["L_s"]["L_fold2"]["count"] != count//3:
            raise AssertionError("Held-fold coverage summary does not use the independent source fold")
        # Full held-TX/day transform entry: actual source-ID endpoints and V health.
        entries_t = [(tx, rx, day, packet) for tx in range(6) for rx in t.d.RX
                     for day in t.d.DAYS for packet in range(9)]
        n = len(entries_t); tx_t = torch.tensor([v[0] for v in entries_t])
        rx_t = torch.tensor([v[1] for v in entries_t]); day_t = torch.tensor([v[2] for v in entries_t])
        folds_t = torch.tensor([v[3] % 3 for v in entries_t])
        x_t = torch.zeros(n, 2, 256); h_t = torch.zeros(n, 160, dtype=torch.float64)
        for i, (tx, rx, day, packet) in enumerate(entries_t):
            state = packet//3
            x_t[i, 0] = 1.
            x_t[i, 0, :64] = 1+state*.1
            x_t[i, 0, 64:128] = 1-state*.05
            latent = torch.zeros(1, 160, dtype=torch.float64)
            latent[0, tx] = 1.; latent[0, 6] = int(day)*.1
            latent[0, 7] = state*.1; latent[0, 8] = packet%3*.002
            latent[0, 158] = tx*.1+state*.3; latent[0, 159] = int(day)*.2
            h_t[i] = t.shared_transform(latent, 0, list(t.d.RX).index(rx), P, B, c)[0]
        cache_t = dict(x=x_t, y=tx_t, rx=rx_t, day=day_t,
                       ids=[f"transform-L-{i}" for i in range(n)])
        model_t = torch.nn.Module(); model_t.head = torch.nn.Linear(160, 6, bias=False).eval().requires_grad_(False)
        state_before = {name:value.clone() for name,value in model_t.state_dict().items()}
        from .common import state_q
        context_t = dict(model=model_t, seed=71, L=cache_t,
                         V=dict(cache_t, ids=[f"transform-V-{i}" for i in range(n)]),
                         features={role:{"joint160":h_t.float()} for role in ("L_s", "V")},
                         state_q_raw=state_q(x_t), folds=folds_t)
        report_t = t.diagnose(context_t, Path(directory)/"transform", "cpu")
        saved_t = torch.load(Path(directory)/"transform"/"audit.pt", weights_only=False)
        if len(report_t["blocks"]) != 18 or report_t["fit_updates"] != 7200:
            raise AssertionError("Independent shared-transform source matrix is incomplete")
        for block in saved_t["blocks"]:
            indices = block["fit_indices"]
            if torch.isin(tx_t[indices], torch.tensor(block["summary"]["held_TX"])).any() or (day_t[indices] == block["summary"]["held_day"]).any():
                raise AssertionError("Held TX/day entered shared-coordinate fitting")
            if block["summary"]["source_V_health"]["fit"] or block["summary"]["source_V_health"]["performance_gate"]:
                raise AssertionError("Source V entered fit or promotion")
        if any(not torch.equal(value, state_before[name]) for name,value in model_t.state_dict().items()) or any(p.grad is not None for p in model_t.parameters()):
            raise AssertionError("Shared-transform diagnostic changed frozen identity state")
    # No parameters/buffers, graph, labels, or private teacher state are persisted in deployment.
    return dict(status="PASS", device=device, source_cache_opened=False, target_access=False,
                identity_updates=0, coordinate_identity_inverse_composition=True,
                random_common_coordinate_spectrum_norm_and_composition=True,
                independent_synthetic_endpoint_MSE=dict(zero=zero_mse, fitted=fitted_mse),
                held_state_outliers_do_not_change_fit=True, no_TX_operator_argument=True,
                singleton_CE_equivalence=True, whole_set_zero_loss_and_gradient=True,
                partial_set_exact_formula=True, invalid_sets_rejected=True,
                cumulative_prefix_and_union=True, teacher_stop_gradient=True,
                shared_mask_and_denominator=True, lawful_source_coverage_recounted=True,
                hidden_U_correctness_unavailable=True, ten_field_hidden_U_diagnose_entry=True,
                labels_joined_after_teacher_sets=True, full_source_audit_recounted=True,
                full_18_fit_transform_diagnose_entry=True, held_TX_day_fit_exclusion=True,
                source_V_health_never_fit_or_promote=True, frozen_identity_unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args(); result = checks(args.device)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text+"\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
