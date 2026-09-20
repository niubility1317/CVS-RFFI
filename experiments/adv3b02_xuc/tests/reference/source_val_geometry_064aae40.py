# Frozen regression reference from commit 064aae40; executed with test-injected globals.
def _evaluate_source_val_tail_geometry(model, data_ctx, device, args) -> Dict[str, Any]:
    """Evaluate tail/proxy geometry on one fixed source-val protocol."""

    if direct_metric_acceptance_loss is None:
        return {"status": "FAILED", "reason": "direct_metric_acceptance_loss_unavailable"}
    was_training = bool(model.training)
    features: List[torch.Tensor] = []
    sat_features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    domains: List[torch.Tensor] = []
    sat_scenarios = list(getattr(args, "sat_train_protocol_scenario_list", []))
    sat_generator = make_torch_generator(device, int(getattr(args, "seed", 0)) + 91079) if make_torch_generator is not None else None
    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_ctx["val_loader"], start=1):
                if int(args.eval_max_batches) > 0 and batch_idx > int(args.eval_max_batches):
                    break
                x, y, extra = move_batch(batch, device)
                if args.rc4_satellite_family == "practical":
                    from cvsrffi.practical_adapter import set_source_evaluation_context
                    set_source_evaluation_context(extra[1])
                d = domain_from_extra(extra, data_ctx["domain_label_map"], device)
                out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=d)
                features.append(out["z_id"].detach().float())
                if sat_scenarios and apply_sat_channel_for_scenario is not None:
                    scenario = sat_scenarios[(batch_idx - 1) % len(sat_scenarios)]
                    x_sat, _ = apply_sat_channel_for_scenario(
                        x,
                        scenario,
                        args,
                        gen=sat_generator,
                        return_meta=False,
                    )
                    sat_out = model(
                        _safe_iq_tensor(x_sat),
                        y_tx=None,
                        grl_lambda=1.0,
                        return_aux=True,
                        domain_labels=d,
                    )
                    sat_features.append(sat_out["z_id"].detach().float())
                labels.append(y.detach().long())
                if d is not None:
                    domains.append(d.detach().long())
        if not features:
            return {"status": "FAILED", "reason": "source_val_geometry_empty"}
        z = torch.cat(features, dim=0)
        y = torch.cat(labels, dim=0)
        d = torch.cat(domains, dim=0) if len(domains) == len(features) else None
        cuda_devices = [int(device.index or 0)] if getattr(device, "type", "cpu") == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(getattr(args, "seed", 0)) + 91073)
            if (
                bool(getattr(args, "direct_metric_multiview_separate", False))
                and len(sat_features) == len(features)
                and sat_features
                and multiview_direct_metric_acceptance_loss is not None
            ):
                z_sat = torch.cat(sat_features, dim=0)
                _loss, info = multiview_direct_metric_acceptance_loss(
                    z,
                    z_sat,
                    y,
                    d,
                    clean_weight=float(args.direct_metric_clean_weight),
                    sat_weight=float(args.direct_metric_sat_weight),
                    pair_weight=float(args.direct_metric_sat_pair_weight),
                    sat_pair_target_rad=math.radians(float(args.direct_metric_sat_pair_target_deg)),
                    **_direct_metric_kwargs(args),
                )
            else:
                _loss, info = direct_metric_acceptance_loss(z, y, d, **_direct_metric_kwargs(args))
        result = {str(key): float(value) if isinstance(value, (int, float)) else value for key, value in info.items()}
        result.update(
            {
                "status": "COMPLETE" if float(info.get("active", 0.0)) > 0.0 else "FAILED",
                "protocol": "fixed_source_val_multiview_local_component_v2",
                "sample_count": int(z.size(0)),
                "seed": int(getattr(args, "seed", 0)) + 91073,
                "satellite_scenarios": sat_scenarios,
                "multiview_sample_count": int(torch.cat(sat_features, dim=0).size(0)) if sat_features else 0,
            }
        )
        if result["status"] != "COMPLETE":
            result["reason"] = "source_val_direct_metric_inactive"
        return result
    finally:
        model.train(was_training)
