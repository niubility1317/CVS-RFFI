"""Original CVS batch-level channel sampler and exact concat CE timing."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from experiments.cvs_phase1_overlay.contract import SCHEDULE, SCENES, LEO
from experiments.cvs_phase1_overlay.model import native_modules


class OriginalLEO:
    def __init__(self, output):
        native_modules()
        from concat_sat_channel_aug import ConcatSatChannelAugment
        from cvsrffi.practical_adapter import apply_practical
        self.args = SimpleNamespace(practical_fs_hz=LEO['fs_hz'], practical_fc_hz=LEO['fc_hz'],
            practical_route='residual', practical_equalization=False, practical_equalizer_method='zf',
            practical_receiver_seed=2027, output_dir=str(output))
        self.augment = ConcatSatChannelAugment(scenarios=SCENES, p=.30, seed=2027,
            apply_fn=apply_practical, schedule=SCHEDULE)

    def __call__(self, x, batch, epoch, step):
        from cvsrffi.practical_adapter import set_training_context
        meta = batch['meta']
        set_training_context(dict(base_index=[m['sample_id'] for m in meta],
            rx_i=[m['rx_i'] for m in meta], day_i=[m['day_i'] for m in meta]), epoch, 'L')
        return self.augment.transform(x, args=self.args, epoch=epoch, batch_idx=step)


def loss_for_batch(model, x, y, domain, batch, augment, epoch, step):
    if augment is None:
        logits = model(x, y=y, domain_labels=domain)
        ce = F.cross_entropy(logits, y)
        return ce, dict(clean_ce=ce, satellite_ce=None, satellite_weight=0.,
            view='disabled', view_probability=0., channel_applied=False, concat_samples=0)
    view = augment(x, batch, epoch, step)
    logits = model(torch.cat((x, view.x)), y=torch.cat((y, y)),
                   domain_labels=torch.cat((domain, domain)))
    clean, satellite = logits.split(len(x))
    ce = F.cross_entropy(clean, y)
    sat = F.cross_entropy(satellite, y) if epoch >= 80 else None
    loss = ce if sat is None else ce + .68*sat
    return loss, dict(clean_ce=ce, satellite_ce=sat, satellite_weight=.68 if sat is not None else 0.,
        view=view.scenario, view_probability=view.view_prob, channel_applied=view.applied,
        concat_samples=len(x))
