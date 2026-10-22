"""The project default concat generator, with a disclosed first-batch start."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from baseline_origin_sat_view import BaselineOriginSatViewAugment
from sat_channel import SatSimConfig, apply_sat_gnd_channel_batch
from training_controls import sat_channel_config_for_scenario


def default_apply(x, scenario, args, *, gen=None, return_meta=False):
    config = sat_channel_config_for_scenario(scenario)
    config.update(fs_hz=25e6, fc_hz=2.462e9)
    y, metadata, _ = apply_sat_gnd_channel_batch(x, SatSimConfig(**config), gen=gen, return_meta=return_meta)
    return y.to(device=x.device, dtype=x.dtype), metadata


class DefaultConcat:
    def __init__(self, seed=2027):
        self.generator = BaselineOriginSatViewAugment(seed=seed, apply_fn=default_apply,
            schedule='1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak')

    def __call__(self, x, progress, batch_index, stage=1):
        if not 0 <= progress < 1:
            raise ValueError('Fixed-budget progress must be in [0,1)')
        # Map E100, E200 and fixed-update budgets onto the same Core90 stages.
        virtual_epoch = min(200, int(progress * 200) + 1)
        with torch.no_grad():
            view = self.generator.transform(x, args=None, epoch=virtual_epoch,
                batch_idx=int(batch_index) + (int(stage) - 1) * 1000000)
        if view.x.shape != x.shape or not torch.isfinite(view.x).all():
            raise FloatingPointError('Invalid default satellite view')
        return view.x.detach(), dict(scenario=view.scenario, applied=bool(view.applied),
            probability=view.view_prob, virtual_epoch=virtual_epoch,
            stage_index=view.stage_index, samples=len(x), weight=.68,
            concat_sat_ce_only=True, consistency_weight=0.)
