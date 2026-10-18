"""Full original costs/EMA plus the second complete physical draw; no U truth."""
from contextlib import contextmanager
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from . import design as d, model
from .source import parent_payload, segment_train
from experiments.cvs_satellite_dropout18.observer import Observer as Original


def joint(counter):
    return [dict(TX=k[0], RX=k[1], scene=k[2], state=k[3], lock=k[4], count=v)
            for k, v in sorted(counter.items())]


class Observer(Original):
    def __init__(self, c):
        self.c = d.validate(c)
        self.optimizer = None
        self.out = Path(c['output_root'])
        self.parent = parent_payload(c)
        self.physics_records = 0
        self.physics_file = None

    def initialize(self, m, c, out):
        super().initialize(m, c, out)
        self.physics_file = gzip.open(self.out / 'full_as_pair_physics.jsonl.gz', 'xt', encoding='utf8', compresslevel=3)
        model.physics_sink(m, self.write_physics)
        actual = dict(config=c, original_channel=m._full_channel.audit(),
            second_channel=dict(same_configurations=True, same_train_seed=True,
                namespace_suffix=model.SECOND_SUFFIX, hardware='same object from first process',
                full_receiver_state_recomputed=True, EQ_enabled=False, EQ_G=1),
            modules={k: str(v.__file__) for k, v in sys.modules.items() if k in (
                'experiments.cvs_full_channel_pair68.model', 'experiments.cvs_satellite_dropout18.model',
                'experiments.cvs_clock_factorial30.channel', 'SSDG.train_ssdg',
                'DataAugmentation', 'leo_practical.channel', 'leo_practical.residual')})
        d.write(self.out / 'full_as_pair_actual.json', actual)
        print(json.dumps(dict(stage='ACTUAL_FULL_AS_PAIR', **actual)), flush=True)

    def write_physics(self, record):
        if (record['step'] != self.physics_records + 1 or len(record['view0']) != 64
                or len(record['view1']) != 64 or self.physics_file is None):
            raise ValueError('Exactly two ordered complete64 physical records per L update')
        self.physics_file.write(json.dumps(record, allow_nan=False) + '\n')
        self.physics_records += 1

    @contextmanager
    def native_scope(self, native):
        try:
            with super().native_scope(native):
                yield
        finally:
            if hasattr(self, 'model'):
                model.physics_sink(self.model)
            if self.physics_file is not None:
                self.physics_file.close()
                self.physics_file = None

    def epoch(self, m, ctx, rows, out, device):
        super().epoch(m, ctx, rows, out, device)
        n = len(rows) * d.STEPS
        if (len(m._pair_records) != n or m._pair_generated != n * 64
                or sum(m._pair_coverage.values()) != n * 64 or sum(m._pair_strata.values()) != n * 64
                or self.physics_records != n or m._pair_sessions != m._full_channel.sessions):
            raise ValueError('Two complete As generations/fullsource/session coverage differs')
        if self.physics_file is not None:
            self.physics_file.flush()
        d.write(out / 'full_as_pair_progress.json', self.pair_summary())

    def pair_summary(self):
        records = self.model._pair_records
        keys = ('original_mixed_KL', 'ordinary_half_KL', 'satellite_half_KL', 'used_mixed_KL',
            'removed_satellite_KL', 'satellite_CE0', 'satellite_CE1', 'As_IQ_pair_MSE')
        n = len(records)
        return dict(method='INDEP_CE6', steps=n,
            **{k + '_mean': sum(r[k] for r in records) / n if n else 0. for k in keys},
            generated_As0=self.model._full_channel.packets, generated_As1=self.model._pair_generated,
            generated_As_total=self.model._full_channel.packets + self.model._pair_generated,
            As_forward_NN=n * 128, L_NN=n * 768, total_L_CE_scale=1.25,
            mixed_group_weight=.5, As_fraction=.5, As_CE_coefficient=.25,
            per_pass_As_CE_coefficient=.125, ordinary_raw_KL_unchanged=True,
            As_KL=False, original_dropout=True, dropout_bypassed_elements=sum(r['bypassed_elements'] for r in records),
            same_hardware_steps=sum(r['same_hardware'] for r in records),
            full_physics_steps=self.physics_records, view0_namespace_unchanged=True,
            second_from_same_raw=True, second_from_first_As=False,
            session_count=len(self.model._pair_sessions), extra_NN=0, extra_U_NN=0,
            extra_parameters=0, extra_losses=0, U_truth_read=False, target_access=False)

    def finish(self):
        result = super().finish()
        m = self.model
        if (len(m._pair_records) != d.UPDATES or self.physics_records != d.UPDATES
                or m._pair_generated != 142080 or m._full_channel.packets != 142080
                or result['L_NN'] != 1704960 or result['Ag_packets']
                or result['extra_U_NN'] or result['head_fit_updates'] or result['U_truth_read']):
            raise ValueError('Complete fixed sixL/twoAs/no-new-U budget required')
        with gzip.open(self.out / 'full_as_pair_steps.jsonl.gz', 'xt', encoding='utf8', compresslevel=3) as f:
            for record in m._pair_records:
                f.write(json.dumps(record, allow_nan=False) + '\n')
        second = dict(calls=d.UPDATES, packets=m._pair_generated, sessions=m._pair_sessions,
            session_count=len(m._pair_sessions), dropped=0, confidence_filter=False,
            all_original_source_RX=list(d.RX), ordinary_before_satellite=False,
            formal_test_changed=False,
            strata=[dict(RX=k[0], day=k[1], scene=k[2], state=k[3], lock=k[4], count=v)
                for k, v in sorted(m._pair_strata.items())])
        result.update(new_epochs=10, parent_epochs=20, cumulative_epochs=30,
            new_updates=2220, parent_updates=4440, cumulative_updates=6660,
            L_As_total=284160, L_As_forward_NN=284160,
            L_As_note='L_As counts original view0 generation; L_As_total includes second full draw; As NN unchanged',
            L_channel_second=second, full_as_pair_summary=self.pair_summary(),
            TX_RX_scene_state_lock=joint(m._full_coverage + m._pair_coverage),
            TX_RX_scene_state_lock_view0=joint(m._full_coverage),
            TX_RX_scene_state_lock_view1=joint(m._pair_coverage), source_RX_holdout=None,
            SAME_control=dict(run=d.SAME_RUN, config=d.reference_config('AS_CE'), reused=True),
            parent_history_difference=self.c['parent_history_difference'])
        result['native_NN'] = sum(result[k] for k in ('student_train_NN', 'student_eval_NN', 'teacher_train_NN', 'teacher_eval_NN'))
        native = sys.modules['SSDG.train_ssdg']
        parent_args = SimpleNamespace(**self.parent['args'])
        d.write(self.out / 'ordinary_augmentation_actual.json', dict(
            base=native.build_aug_base_cfg(parent_args), schedule_frozen_at=13,
            args={k: v for k, v in self.parent['args'].items() if k.startswith('aug_')},
            schedule_args_epochs=self.parent['args']['epochs'], source='own AS_CE scratch20 checkpoint args',
            no_ordinary_before_practical=True, RDrop_same_IQ_ordinary_raw=True,
            As_views_same_IQ=False, As_cross_view_KL=False))
        d.write(self.out / 'result.json', result)
        return result


def train(c):
    observer = Observer(c)
    segment_train(c, design=d, model_api=model, observer=observer)
    return observer.finish()
