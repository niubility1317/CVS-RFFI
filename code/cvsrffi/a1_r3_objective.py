"""Source-only R3 self/swap/shared objective for checkpoint-free A1 training.

Factor and reconstruction mathematics come from the pinned R3 implementation.
No TX labels enter this auxiliary API. Eta metadata is explicitly unavailable.
"""
from __future__ import annotations

import torch
from .a1_budget_schedule import reference_epoch

from .phase1_fcr_losses import compute_cross_losses, mrstft_loss, phase_increment_loss
from .phase1_fcr_schedule import permission_for_role, stage_for_epoch
from .phase1_fcr_types import FCRPairBatch


def r3_pair_objective(*, model, clean_iq, domains, physical_ids, role,
                      args, epoch, batch_idx, optimizer_step, apply_sat_fn):
    permission = permission_for_role(role)
    if not permission.optimizer_step:
        raise ValueError('R3 auxiliary training accepts source L_s/U_s only')
    if not bool(getattr(args, 'use_a1_r3', False)):
        return clean_iq.new_zeros(()), {}
    n = int(clean_iq.size(0))
    if len(physical_ids) != n:
        raise ValueError('R3 requires one physical ID for each source row')
    schedule_epoch = reference_epoch(args, epoch)
    stage = stage_for_epoch(schedule_epoch, optimizer_step=optimizer_step)
    # Match the original source curriculum; every auxiliary row is a paired view.
    scenarios = (('leo_clear_weak',) if schedule_epoch <= 40 else
                 ('leo_low_elev_weak', 'leo_rain_weak') if schedule_epoch <= 90 else
                 ('leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak'))
    scenario = scenarios[(epoch + batch_idx - 1) % len(scenarios)]
    generator = torch.Generator(device=clean_iq.device)
    generator.manual_seed(int(args.seed) + epoch * 1000003 + batch_idx * 97
                          + (0 if permission.role == 'L_s' else 500000003))
    with torch.no_grad():
        leo_iq, _ = apply_sat_fn(clean_iq, scenario, args, gen=generator, return_meta=True)
        leo_iq = torch.nan_to_num(leo_iq.float(), nan=0., posinf=0., neginf=0.)
    # Complex STFT and physical factors must never enter ComplexHalf under AMP.
    with torch.autocast(device_type=clean_iq.device.type, enabled=False):
        clean_iq = clean_iq.float()
        clean = model.forward_r3_factors(clean_iq)
        leo = model.forward_r3_factors(leo_iq)
        fcr = model.a1_r3

        def cross_decode(source, destination):
            response = fcr.fingerprint_operator(source.content.s_hat.detach(), destination.fingerprint)
            return fcr.decoder(source.content.s_hat, response.delta_f, destination.nuisance)

        c2l, l2c = cross_decode(clean, leo), cross_decode(leo, clean)
        ids = tuple(str(value) for value in physical_ids)
        index = torch.arange(n, device=clean_iq.device)
        invalid = torch.full_like(index, -1)
        valid = torch.ones(n, device=clean_iq.device, dtype=torch.bool)
        eta = torch.zeros_like(leo.factors.z_n_parts['eta_pred'])
        pair = FCRPairBatch(
            clean_iq=clean_iq, leo_iq=leo_iq, labels=invalid,
            label_mask=~valid, receiver_id=domains, day_id=invalid,
            nuisance=eta, nuisance_valid=torch.zeros_like(eta, dtype=torch.bool),
            physical_sample_id=ids, pair_id=ids,
            clean_crop_offset=torch.zeros_like(index), leo_crop_offset=torch.zeros_like(index),
            nuisance_pair_index=index, content_pair_index=invalid, fingerprint_pair_index=invalid,
            pair_valid_mask={'nuisance': valid, 'content': ~valid, 'fingerprint': ~valid})
        active = stage.active & frozenset({'self', 'swap', 'shared', 'eta'})
        cross = compute_cross_losses(
            clean_factors=clean.factors, leo_factors=leo.factors,
            clean_self=clean.decode, leo_self=leo.decode,
            clean_to_leo=c2l, leo_to_clean=l2c, pair=pair,
            reencode_clean_to_leo=None, reencode_leo_to_clean=None,
            config=fcr.config, active=active)
        zero = clean.factors.z_s.sum() * 0.
        physical = (0.5 * (mrstft_loss(clean_iq, clean.decode.mu_iq)
                    + mrstft_loss(leo_iq, leo.decode.mu_iq)
                    + phase_increment_loss(clean_iq, clean.decode.mu_iq)
                    + phase_increment_loss(leo_iq, leo.decode.mu_iq))
                    if 'self' in active else zero)
        raw = {'self': cross.components['self'] + physical,
               'swap': cross.components['swap'], 'shared': cross.components['shared'],
               'eta': cross.components['eta']}
        weighted = {name: value * stage.scales[name] if name in active else zero
                    for name, value in raw.items()}
        # The structural control executes the same forwards/BN updates/RNG path.
        scale = float(args.a1_r3_aux_scale)
        total = sum(weighted.values(), zero) * scale
        prefix = 'train/r3_' + permission.role.lower() + '_'
        logs = {prefix + name + '_raw': value.detach() for name, value in raw.items()}
        logs.update({prefix + name + '_weighted': (value * scale).detach()
                     for name, value in weighted.items()})
        logs.update({prefix + 'pairs': float(n), prefix + 'eta_valid': 0.,
                     prefix + 'loss': total.detach(),
                     prefix + 'aux_scale': scale,
                     prefix + 'self_active': float('self' in active and scale > 0),
                     prefix + 'swap_active': float(stage.scales['swap'] > 0 and scale > 0),
                     prefix + 'shared_active': float(stage.scales['shared'] > 0 and scale > 0)})
    return total, logs
