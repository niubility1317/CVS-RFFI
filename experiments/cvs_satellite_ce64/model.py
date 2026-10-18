"""Reuse the original six labeled forwards; replace satellite CE only."""
from pathlib import Path
from types import MethodType
import torch
from . import design as d
from .objective import captured_native_views, replacement, validate_head, validate_objective
from experiments.cvs_noise_shape_factorial43 import model as original

classifier, source_domain_count = original.classifier, original.source_domain_count


def hook(self, **kw):
    validate_head(self.id_backbone.encoder.core.id_backbone.cls_head)
    with captured_native_views(self) as (logits, observations):
        delta, info = original.hook(self, **kw)
    if len(logits) != 5 or len(observations) != 1:
        raise ValueError('Original hook must execute ordinary1/mixed0/mixed1/raw0/raw1 and one channel call')
    mask = (torch.arange(128, device=kw['labels'].device) + kw['optimizer_step']).remainder(2) != 0
    correction, record = replacement(logits[1][mask], logits[2][mask], kw['labels'][mask],
                                     self._satellite_ce_arm, self._satellite_ce_objective)
    if not correction.requires_grad or not logits[1].requires_grad or not logits[2].requires_grad:
        raise ValueError('Satellite CE replacement must retain live student/default-head gradients')
    native = self._full_records[-1]
    base = native['total_L']
    record.update(step=kw['optimizer_step'], cumulative_step=kw['optimizer_step'] + 4440,
                  base_L_before_ce=base, total_L=base + record['satellite_CE_correction'],
                  gradient_alive=True, forward_count=len(logits), channel_calls=len(observations))
    self._satellite_ce_records.append(record)
    native.update(base_L_before_ce=base, satellite_CE_original=record['old_sat_CE'],
                  satellite_CE_modified=record['new_sat_CE'],
                  satellite_CE_correction=record['satellite_CE_correction'], total_L=record['total_L'])
    info.update({'satellite_ce/' + key: record[key] for key in
                 ('old_sat_CE', 'new_sat_CE', 'satellite_CE_correction', 'temperature', 'weight')})
    return delta + correction, info


hook._accepts_labeled_inputs = True


def attach(model, config):
    d.validate(config)
    validate_objective(config['arm'], config['ce_objective'])
    model._satellite_ce_records = []
    model._satellite_ce_arm = config['arm']
    model._satellite_ce_objective = dict(config['ce_objective'])
    model._source_labeled_feature_hook = MethodType(hook, model)
    model._feature_learning_report['satellite_ce64'] = dict(config['ce_objective'])
    return model


def install(model, config):
    return attach(original.install(model, d.old_config(config['arm'])), config)


def build(config, device='cpu'):
    return attach(original.build(d.old_config(config['arm']), device), config)


def own_model(config, device='cpu'):
    from .source import parent_payload
    from .protocol import final_metadata, checkpoint
    d.validate(config)
    parent = parent_payload(config)
    root = Path(config['output_root'])
    final_metadata(config, d.read(root / 'completion.json'), d.read(root / 'source_frozen.json'),
                   d.read(root / 'initialization.json'), d.read(root / 'source_contract.json'),
                   d.read(d.SOURCE)['role_ids'])
    cp = torch.load(root / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    checkpoint(config, cp)
    if set(cp['model']) != set(parent['model']) or set(cp['ema_model']) != set(parent['ema_model']):
        raise ValueError('Own default-head/student/EMA keys differ')
    model = build(config, device)
    model.load_state_dict(cp['model'], strict=True)
    return model.eval().requires_grad_(False)
