"""Keep original optimizer/EMA/ordinary schedule; alter only normal PA return."""
from contextlib import contextmanager
import gzip
import json
import sys
from pathlib import Path
from . import design as d, model
from .augmentation import Wrapper
from .source import parent_payload, segment_train
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original

class Observer(Original):
    def __init__(self, c):
        self.c = d.validate(c)
        self.optimizer = None
        self.out = Path(c['output_root'])
        self.parent = parent_payload(c)
        self.PA_records = []
        self.augmentors_created = 0

    def initialize(self, m, c, out):
        super().initialize(m, c, out)
        actual = dict(config=c, channel=m._full_channel.audit(),
                      modules={k: str(v.__file__) for k, v in sys.modules.items()
                               if k in ['experiments.cvs_pa_waveform60.augmentation',
                                        'experiments.cvs_noise_shape_factorial43.model',
                                        'SSDG.train_ssdg', 'DataAugmentation',
                                        'leo_practical.channel', 'leo_practical.residual']})
        d.write(self.out / 'PA_waveform_actual.json', actual)
        print(json.dumps(dict(stage='ACTUAL_MODULES_AND_PA_SHADOW', **actual)), flush=True)

    @contextmanager
    def native_scope(self, native):
        make = native.make_augmentor
        def wrapped(cfg):
            self.augmentors_created += 1
            if self.augmentors_created != 1:
                raise ValueError('Only the original one ordinary augmentor is changed')
            return Wrapper(make(cfg), self.PA_records)
        native.make_augmentor = wrapped
        try:
            with super().native_scope(native):
                yield
        finally:
            native.make_augmentor = make

    def epoch(self, m, ctx, rows, out, device):
        super().epoch(m, ctx, rows, out, device)
        steps = len(rows) * d.STEPS
        if (self.augmentors_created != 1 or len(self.PA_records) != steps
                or any(r['actual_PA_injected_packets'] or not r['exact_native_nonPA_prefix']
                       or r['effective_p_pa'] <= 0 or r['native_no_pa']
                       for r in self.PA_records)):
            raise ValueError('PA shadow must execute once per ordinary L step only')
        d.write(out / 'PA_shadow_progress.json', self.PA_summary())

    def PA_summary(self):
        return dict(ordinary_L_calls=len(self.PA_records),
                    ordinary_L_packets=sum(r['count'] for r in self.PA_records),
                    shadow_PA_calls=sum(r['shadow_PA_calls'] for r in self.PA_records),
                    shadow_PA_selected_packets=sum(r['shadow_PA_selected_packets'] for r in self.PA_records),
                    shadow_PA_changed_packets=sum(r['shadow_PA_changed_packets'] for r in self.PA_records),
                    shadow_PA_MSE_sum=sum(r['shadow_PA_MSE_sum'] for r in self.PA_records),
                    actual_PA_injected_packets=sum(r['actual_PA_injected_packets'] for r in self.PA_records),
                    full_native_draws=True, native_no_pa=False, native_p_pa_unchanged=True,
                    frozen_ordinary_schedule_epoch=13, scope='ordinary_L_E21_E30_only',
                    parent_E1_E20_PA=True, U_changed=False, As_changed=False, raw_slots_changed=False)

    def finish(self):
        result = super().finish()
        if len(self.PA_records) != d.UPDATES or self.augmentors_created != 1:
            raise ValueError('Complete actual ordinary-L PA shadow accounting required')
        with gzip.open(self.out / 'PA_shadow_steps.jsonl.gz', 'xt', encoding='utf8', compresslevel=3) as f:
            for r in self.PA_records:
                f.write(json.dumps(dict(r, cumulative_step=r['step'] + 4440), allow_nan=False) + '\n')
        result.update(new_epochs=10, parent_epochs=20, cumulative_epochs=30,
                      new_updates=2220, parent_updates=4440, cumulative_updates=6660,
                      PA_shadow=self.PA_summary(), source_RX_holdout=None,
                      TX_RX_scene_state_lock=[dict(TX=k[0], RX=k[1], scene=k[2], state=k[3], lock=k[4], count=v)
                                              for k, v in sorted(self.model._full_coverage.items())])
        expected_As = 142080 if self.c['As'] else 0
        if (result['L_NN'] != 1704960 or result['L_As'] != expected_As
                or result['Ag_packets'] or result['extra_U_NN'] or result['head_fit_updates']
                or result['U_truth_read'] or result['PA_shadow']['actual_PA_injected_packets']):
            raise ValueError('Original sixL/nominalAs/oldU/default-head budget differs')
        d.write(self.out / 'result.json', result)
        return result

def train(c):
    observer = Observer(c)
    segment_train(c, design=d, model_api=model, observer=observer)
    return observer.finish()
