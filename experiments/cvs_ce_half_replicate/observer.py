"""Keep native IQ/U/EMA and account for the reused-logit satellite CE replacement."""
import gzip
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from . import design as d, model
from .source import parent_payload, segment_train
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original

class Observer(Original):
    def __init__(self, c):
        self.c = d.validate(c)
        self.optimizer = None
        self.out = Path(c['output_root'])
        self.parent = parent_payload(c)

    def initialize(self, m, c, out):
        super().initialize(m, c, out)
        actual = dict(config=c, channel=m._full_channel.audit(),
                      modules={k: str(v.__file__) for k, v in sys.modules.items()
                               if k in ['experiments.cvs_ce_half_replicate.objective',
                                        'experiments.cvs_noise_shape_factorial43.model',
                                        'SSDG.train_ssdg', 'DataAugmentation',
                                        'leo_practical.channel', 'leo_practical.residual']})
        d.write(self.out / 'satellite_ce_actual.json', actual)
        print(json.dumps(dict(stage='ACTUAL_MODULES_AND_SATELLITE_CE', **actual)), flush=True)

    def epoch(self, m, ctx, rows, out, device):
        super().epoch(m, ctx, rows, out, device)
        if len(m._satellite_ce_records) != len(rows) * d.STEPS:
            raise ValueError('Exactly one satellite CE replacement per original labeled update')
        d.write(out / 'satellite_ce_progress.json', self.ce_summary())

    def ce_summary(self):
        records = self.model._satellite_ce_records
        sums = {k: sum(r[k] for r in records) for k in (
            'old_sat_CE0', 'old_sat_CE1', 'old_sat_CE', 'new_sat_CE0',
            'new_sat_CE1', 'new_sat_CE', 'satellite_CE_delta_unscaled',
            'satellite_CE_correction', 'base_L_before_ce', 'total_L',
            'satellite_slots', 'mixed_slots')}
        n = len(records)
        return dict(steps=n, **{k + '_sum': v for k, v in sums.items()},
                    **{k + '_mean': v / n if n else 0. for k, v in sums.items()},
                    arm=self.c['arm'], **self.c['ce_objective'],
                    label_smoothing=.01, T_squared=False, default_head_scale=30.,
                    mixed_group_weight=.5, satellite_fraction=.5,
                    satellite_objective_coefficient=.25, per_pass_coefficient=.125,
                    gradient_alive_steps=sum(bool(r['gradient_alive']) for r in records),
                    correction_nonzero_steps=sum(r['satellite_CE_correction'] != 0 for r in records),
                    forward_reuse=True, extra_NN=0, extra_U_NN=0,
                    extra_parameters=0, all_labeled_satellite_CE_retained=True,
                    two_separate_CE_mean=True, clean_CE_unchanged=True,
                    ordinary_CE_unchanged=True, mixed_RDrop_KL_unchanged=True,
                    sample_quality_weighting=False, confidence_gate=False)

    def finish(self):
        result = super().finish()
        if len(self.model._satellite_ce_records) != d.UPDATES:
            raise ValueError('Complete2220 satellite CE replacement records required')
        with gzip.open(self.out / 'satellite_ce_steps.jsonl.gz', 'xt', encoding='utf8', compresslevel=3) as f:
            for r in self.model._satellite_ce_records:
                f.write(json.dumps(r, allow_nan=False) + '\n')
        result.update(new_epochs=10, parent_epochs=20, cumulative_epochs=30,
                      new_updates=2220, parent_updates=4440, cumulative_updates=6660,
                      satellite_ce_summary=self.ce_summary(), source_RX_holdout=None,
                      TX_RX_scene_state_lock=[dict(TX=k[0], RX=k[1], scene=k[2], state=k[3], lock=k[4], count=v)
                                              for k, v in sorted(self.model._full_coverage.items())])
        if (result['L_NN'] != 1704960 or result['L_As'] != 142080
                or result['Ag_packets'] or result['extra_U_NN'] or result['head_fit_updates'] or result['U_truth_read']):
            raise ValueError('Original sixL/nominalAs/oldU/default-head budget differs')
        native=sys.modules['SSDG.train_ssdg'];scratch_args=SimpleNamespace(**self.parent['args'])
        d.write(self.out/'ordinary_augmentation_actual.json',dict(base=native.build_aug_base_cfg(scratch_args),schedule_frozen_at=13,args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},schedule_args_epochs=self.parent['args']['epochs'],source='own scratch20 checkpoint args',no_ordinary_before_practical=True,RDrop_same_IQ=True))
        result['native_NN']=sum(result[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'))
        d.write(self.out / 'result.json', result)
        return result

def train(c):
    observer = Observer(c)
    segment_train(c, design=d, model_api=model, observer=observer)
    return observer.finish()
