"""Held-TX/day test of a train-only shared receiver coordinate system.

All endpoints are state-conditioned *distribution means* of distinct source
packets.  They are not paired observations of the same transmitted waveform.
The fitted operator accepts only a feature and source/destination RX indices.
"""
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from . import design as d

RECIPE = dict(feature="joint160", rank=2, max_B_spectral_norm=.15,
              max_c_relative_to_fit_feature_norm=.25, anchor_RX=d.RX[0],
              fit_steps=400, learning_rate=.02, state_bins=3,
              TX_holdouts=[[0, 1], [2, 3], [4, 5]],
              day_holdouts=list(d.DAYS), independent_fit_folds=[0, 1],
              endpoint_fold_pairs=[[0, 1], [1, 2]],
              state="received amplitude q; training-only standardization/PCA/quantiles",
              endpoint="same TX/day/state-bin, distinct physical packets across RX/folds",
              controls=["zero", "shared_translation", "shared_lowrank",
                        "random_common_coordinates", "equal_amplitude_random_displacement",
                        "per_TX_translation_reference"])
CRITERIA = dict(each_fit_mean_MSE_improvement_over_zero=.02,
                each_fit_mean_MSE_improvement_over_random=.02,
                each_fit_positive_holdout_blocks=6, holdout_blocks=9,
                minimum_endpoint_coverage=.8, max_coordinate_condition=1.36,
                nontrivial_mean_delta_norm=1e-6)


def shared_transform(h, r, s, P, B, c):
    """T_rs(h)=A_s A_r^-1(h-b_r)+b_s in a rank-k shared subspace.

    Feature tensors use row vectors; A_r=I+P B_r P^T and b_r=P c_r.
    RX indices may be scalar integers or one index per feature row. No TX input.
    """
    if h.ndim != 2 or P.ndim != 2 or h.shape[1] != P.shape[0]:
        raise ValueError("Feature/shared-coordinate dimensions differ")
    if B.ndim != 3 or c.shape != B.shape[:2] or B.shape[1] != B.shape[2] or P.shape[1] != B.shape[1]:
        raise ValueError("Receiver coordinate dimensions differ")
    z = h @ P
    eye = torch.eye(P.shape[1], dtype=h.dtype, device=h.device)
    ar = eye + B[r]; ass = eye + B[s]
    latent = torch.linalg.solve(ar, (z - c[r]).unsqueeze(-1)).squeeze(-1)
    target = (ass @ latent.unsqueeze(-1)).squeeze(-1) + c[s]
    return h + (target - z) @ P.T


def _state_bins(raw_q, fit):
    q = raw_q.double().detach()
    mean = q[fit].mean(0); scale = q[fit].std(0, unbiased=False).clamp_min(1e-4)
    z = (q - mean) / scale
    _, _, vh = torch.linalg.svd(z[fit], full_matrices=False)
    axis = vh[0]
    # Resolve PCA sign deterministically without labels from a held group.
    if axis[axis.abs().argmax()] < 0:
        axis = -axis
    score = z @ axis
    cuts = torch.quantile(score[fit], torch.tensor([1/3, 2/3], dtype=torch.float64))
    return torch.bucketize(score.contiguous(), cuts), dict(mean=mean, scale=scale,
                                                           axis=axis, cuts=cuts)


def _centers(h, cache, folds, bins, txs, days, fold):
    result = {}
    for tx in txs:
        for day in days:
            for state in range(RECIPE["state_bins"]):
                values = []
                for rx in d.RX:
                    mask = ((cache["y"] == tx) & (cache["day"] == day) &
                            (cache["rx"] == rx) & (folds == fold) & (bins == state))
                    if not mask.any():
                        break
                    values.append((h[mask].mean(0), torch.where(mask)[0], int(mask.sum())))
                if len(values) == len(d.RX):
                    result[(tx, day, state)] = values
    return result


def _fit_coordinates(centers, fit_h, seed):
    if not centers:
        raise ValueError("No complete source state-conditioned fit cell")
    stacked = torch.stack([torch.stack([v[0] for v in values]) for values in centers.values()])
    drift = (stacked[:, 1:] - stacked[:, :1]).reshape(-1, stacked.shape[-1])
    _, singular, vh = torch.linalg.svd(drift, full_matrices=False)
    P = vh[:RECIPE["rank"]].T.detach()
    # The subspace and all normalization use the training TX/day/fold only.
    z = stacked @ P
    B = torch.zeros(len(d.RX), 2, 2, dtype=torch.float64, requires_grad=True)
    c = torch.zeros(len(d.RX), 2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([B, c], lr=RECIPE["learning_rate"])
    cap_c = float(fit_h.norm(dim=1).mean()) * RECIPE["max_c_relative_to_fit_feature_norm"]
    losses = []
    for _ in range(RECIPE["fit_steps"]):
        optimizer.zero_grad(set_to_none=True)
        # Anchor is fixed to identity: its feature is the common coordinate.
        predicted = torch.einsum("nc,rkc->nrk", z[:, 0], torch.eye(2, dtype=z.dtype)[None] + B) + c
        loss = (predicted[:, 1:] - z[:, 1:]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Shared receiver coordinate fit is not finite")
        loss.backward(); optimizer.step()
        with torch.no_grad():
            u, sv, vh_b = torch.linalg.svd(B)
            B.copy_((u * sv.clamp_max(RECIPE["max_B_spectral_norm"])[:, None]) @ vh_b)
            c.mul_((cap_c / c.norm(dim=1).clamp_min(1e-12)).clamp_max(1)[:, None])
            B[0].zero_(); c[0].zero_()
        losses.append(float(loss.detach()))
    B = B.detach(); c = c.detach()
    translation = (stacked - stacked[:, :1]).mean(0)
    generator = torch.Generator().manual_seed(seed)
    random_edges = torch.randn(len(d.RX), len(d.RX), stacked.shape[-1],
                               generator=generator, dtype=torch.float64)
    random_edges = F.normalize(random_edges, dim=-1)
    random_B = []; random_c = []
    for rx in range(len(d.RX)):
        left, _ = torch.linalg.qr(torch.randn(2, 2, generator=generator, dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(2, 2, generator=generator, dtype=torch.float64))
        spectrum = torch.linalg.svdvals(B[rx])
        random_B.append((left*spectrum[None]) @ right.T)
        direction = F.normalize(torch.randn(2, generator=generator, dtype=torch.float64), dim=0)
        random_c.append(direction*c[rx].norm())
    random_B = torch.stack(random_B); random_c = torch.stack(random_c)
    return dict(P=P, B=B, c=c, translation=translation, random_edges=random_edges,
                random_B=random_B, random_c=random_c,
                drift_singular_values=singular, fit_loss=torch.tensor(losses),
                c_norm_cap=cap_c, parameters=4*(2*2+2), deployment_parameters=0,
                fit_cell_keys=list(centers), fit_centers=stacked)


def _margin(h, y, head_weight, head_scale):
    logits = F.normalize(h.float(), dim=1) @ F.normalize(head_weight.float(), dim=1).T
    logits = logits * head_scale
    other = logits.clone(); other[torch.arange(len(y)), y] = -torch.inf
    return logits[torch.arange(len(y)), y] - other.max(1).values


def _evaluate(centers_a, centers_b, model, per_tx_translation, head_weight, head_scale):
    sources = []; targets = []; routes = []; labels = []; keys = []; counts = []
    for key in sorted(set(centers_a) & set(centers_b)):
        for r in range(len(d.RX)):
            for s in range(len(d.RX)):
                if r == s:
                    continue
                a = centers_a[key][r]; b = centers_b[key][s]
                if set(a[1].tolist()) & set(b[1].tolist()):
                    raise ValueError("Endpoint physical samples overlap")
                sources.append(a[0]); targets.append(b[0]); routes.append([r, s])
                labels.append(key[0]); keys.append(key); counts.append([a[2], b[2]])
    if not sources:
        return None
    h = torch.stack(sources); target = torch.stack(targets)
    route = torch.tensor(routes); r, s = route.T; y = torch.tensor(labels)
    fitted = shared_transform(h, r, s, model["P"], model["B"], model["c"])
    delta = fitted - h
    predictions = dict(zero=h, shared_translation=h + model["translation"][s] - model["translation"][r],
                       shared_lowrank=fitted,
                       random_common_coordinates=shared_transform(h, r, s, model["P"], model["random_B"], model["random_c"]),
                       equal_amplitude_random_displacement=h + model["random_edges"][r, s]*delta.norm(dim=1, keepdim=True),
                       per_TX_translation_reference=h + per_tx_translation[y, s] - per_tx_translation[y, r])
    errors = {k: (v-target).square().mean(1) for k, v in predictions.items()}
    margins = {k: _margin(v, y, head_weight, head_scale) for k, v in predictions.items()}
    return dict(source=h, target=target, routes=route, TX=y, keys=keys,
                endpoint_counts=torch.tensor(counts), predictions=predictions,
                MSE=errors, margins=margins, delta_norm=delta.norm(dim=1))


def _reference_translation(h, cache, folds, bins, train_days, fit_fold):
    """Label-conditioned reference sees held TX on non-held days; not shared fit."""
    reference = torch.zeros(6, len(d.RX), h.shape[-1], dtype=h.dtype)
    all_centers = _centers(h, cache, folds, bins, range(6), train_days, fit_fold)
    for tx in range(6):
        values = [v for k, v in all_centers.items() if k[0] == tx]
        if values:
            stacked = torch.stack([torch.stack([a[0] for a in v]) for v in values])
            reference[tx] = (stacked - stacked[:, :1]).mean(0)
    return reference


def diagnose(ctx, output: Path, device):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    L = ctx["L"]; h = ctx["features"]["L_s"]["joint160"].double().detach()
    folds = ctx["folds"]; raw_q = ctx["state_q_raw"]
    from .common import state_q
    V = ctx["V"]; vh = ctx["features"]["V"]["joint160"].double().detach()
    vraw_q = state_q(V["x"]).double()
    vfolds = torch.arange(len(V["ids"])) % 2
    if h.requires_grad or h.shape != (len(L["ids"]), 160) or not torch.isfinite(h).all():
        raise ValueError("Detached finite source identity features required")
    head = ctx["model"].head
    head_weight = head.weight.detach().cpu()
    # A positive scale changes numerical margin units, not the sign or geometry.
    head_scale = 1.  # Report unscaled cosine margins, independent of logit scale.
    records = []; audit = []; fit_results = []
    for fit_fold in RECIPE["independent_fit_folds"]:
        blocks = []
        for holdout_index, held_txs in enumerate(RECIPE["TX_holdouts"]):
            train_txs = [tx for tx in range(6) if tx not in held_txs]
            for held_day in d.DAYS:
                train_days = [day for day in d.DAYS if day != held_day]
                fit = ((folds == fit_fold) & torch.isin(L["y"], torch.tensor(train_txs)) &
                       torch.isin(L["day"], torch.tensor(train_days)))
                bins, state_fit = _state_bins(raw_q, fit)
                fit_centers = _centers(h, L, folds, bins, train_txs, train_days, fit_fold)
                model = _fit_coordinates(fit_centers, h[fit], ctx["seed"] + 4200 + fit_fold*100 + holdout_index*10 + int(held_day))
                reference = _reference_translation(h, L, folds, bins, train_days, fit_fold)
                vscore = ((vraw_q-state_fit["mean"])/state_fit["scale"]) @ state_fit["axis"]
                vbins = torch.bucketize(vscore.contiguous(), state_fit["cuts"])
                va = _centers(vh, V, vfolds, vbins, held_txs, [held_day], 0)
                vb = _centers(vh, V, vfolds, vbins, held_txs, [held_day], 1)
                vhealth = _evaluate(va, vb, model, reference, head_weight, head_scale)
                vsummary = dict(endpoint_pairs=len(vhealth["source"]) if vhealth else 0,
                                MSE={name: float(vhealth["MSE"][name].mean()) if vhealth else None for name in RECIPE["controls"]},
                                fit=False, performance_gate=False)
                endpoints = []
                for fa, fb in RECIPE["endpoint_fold_pairs"]:
                    ca = _centers(h, L, folds, bins, held_txs, [held_day], fa)
                    cb = _centers(h, L, folds, bins, held_txs, [held_day], fb)
                    endpoint = _evaluate(ca, cb, model, reference, head_weight, head_scale)
                    if endpoint is not None:
                        endpoint.update(endpoint_folds=[fa, fb]); endpoints.append(endpoint)
                expected = len(held_txs)*RECIPE["state_bins"]*len(RECIPE["endpoint_fold_pairs"])*len(d.RX)*(len(d.RX)-1)
                actual = sum(len(e["source"]) for e in endpoints)
                mse = {name: float(torch.cat([e["MSE"][name] for e in endpoints]).mean()) if endpoints else None
                       for name in RECIPE["controls"]}
                margin = {name: float(torch.cat([e["margins"][name] for e in endpoints]).mean()) if endpoints else None
                          for name in RECIPE["controls"]}
                coordinate_singular = torch.linalg.svdvals(torch.eye(2, dtype=torch.float64)[None] + model["B"])
                # The full 160-D A also has 158 unit singular values.
                condition = coordinate_singular[:, 0].clamp_min(1)/coordinate_singular[:, -1].clamp_max(1)
                delta = float(torch.cat([e["delta_norm"] for e in endpoints]).mean()) if endpoints else 0.
                positive = bool(endpoints and mse["shared_lowrank"] < mse["zero"] and mse["shared_lowrank"] < mse["random_common_coordinates"])
                block = dict(fit_fold=fit_fold, held_TX=held_txs, held_day=held_day,
                             training_TX=train_txs, training_days=train_days, fit_count=int(fit.sum()),
                             fit_state_cells=len(fit_centers), endpoint_pairs=actual,
                             expected_endpoint_pairs=expected, coverage=actual/expected,
                             MSE=mse, mean_unscaled_cosine_margin=margin,
                             positive_vs_zero_and_random=positive,
                             coordinate_condition=condition.tolist(), max_condition=float(condition.max()),
                             drift_effective_rank=int((model["drift_singular_values"] > model["drift_singular_values"].max()*1e-6).sum()),
                             rank2_drift_energy_fraction=float(model["drift_singular_values"][:2].square().sum()/model["drift_singular_values"].square().sum().clamp_min(1e-30)),
                             source_V_health=vsummary,
                             B_spectral_norm=torch.linalg.matrix_norm(model["B"], ord=2).tolist(),
                             c_norm=model["c"].norm(dim=1).tolist(), mean_delta_norm=delta,
                             per_TX_reference_extra_labels="held TX on training days only; never shared-operator fit")
                blocks.append(block); records.append(block)
                audit.append(dict(summary=block, fit_indices=torch.where(fit)[0], state_fit=state_fit,
                                  bins=bins, model=model, per_TX_translation=reference, endpoints=endpoints,
                                  source_V_health=vhealth, source_V_bins=vbins))
        mse = {name: sum(b["MSE"][name]*b["endpoint_pairs"] for b in blocks if b["MSE"][name] is not None)/max(1, sum(b["endpoint_pairs"] for b in blocks)) for name in RECIPE["controls"]}
        zero_improvement = 1-mse["shared_lowrank"]/max(mse["zero"], 1e-30)
        random_improvement = 1-mse["shared_lowrank"]/max(mse["random_common_coordinates"], 1e-30)
        positive = sum(b["positive_vs_zero_and_random"] for b in blocks)
        coverage = sum(b["endpoint_pairs"] for b in blocks)/sum(b["expected_endpoint_pairs"] for b in blocks)
        fit_results.append(dict(fit_fold=fit_fold, MSE=mse, improvement_over_zero=zero_improvement,
                                improvement_over_random=random_improvement, positive_blocks=positive,
                                endpoint_coverage=coverage,
                                passed=bool(zero_improvement >= .02 and random_improvement >= .02 and positive >= 6 and coverage >= .8 and all(b["max_condition"] <= 1.36 for b in blocks) and sum(b["mean_delta_norm"] for b in blocks)/9 > 1e-6)))
    torch.save(dict(recipe=RECIPE, criteria=CRITERIA, source_ids=L["ids"], source_TX=L["y"],
                    source_RX=L["rx"], source_day=L["day"], folds=folds,
                    features=h, raw_state_q=raw_q, source_V=dict(ids=V["ids"], TX=V["y"], RX=V["rx"],
                                                               day=V["day"], features=vh,
                                                               raw_state_q=vraw_q, endpoint_folds=vfolds),
                    blocks=audit), output/"audit.pt")
    passed = all(r["passed"] for r in fit_results)
    return dict(direction="transform", recipe=RECIPE, identity_updates=0, target_access=False,
                U_truth_read=False, source_V_fit=False, operator_accepts_TX=False,
                applicability=dict(passed=passed, criteria=CRITERIA,
                                   reason="Both independent source fits meet held-TX/day endpoint conditions" if passed else "Shared RX coordinates did not meet every preregistered independent held-TX/day condition"),
                fit_results=fit_results, blocks=records, elapsed_seconds=time.monotonic()-started,
                fit_updates=len(audit)*RECIPE["fit_steps"], auxiliary_parameters=24,
                identity_backward_calls=0, deployment_parameters=0,
                random_control="Common near-identity coordinates share fitted P, exactly matched per-RX B singular values and c norms, randomized orientations; separate equal-amplitude edge displacement is descriptive only",
                audit_artifact="audit.pt", caveat="Cycle identities are mathematical construction; only independent endpoint prediction tests transfer. State q and feature drift may still include receiver/channel conditions.")
