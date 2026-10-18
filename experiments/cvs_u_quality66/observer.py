"""Preserve native L/U and count one additional U student/fixed teacher pair."""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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
        bank = model.prepare_frozen_teachers(m, c, payloads={'B01': self.parent})
        bank.assert_frozen()
        actual = dict(config=c, channel=m._full_channel.audit(), U_channel=m._clean_teacher_channel.audit(),
            frozen_teacher_sources=c['teacher_sources'], frozen_teachers=bank.summary(),
            frozen_teachers_training_only=True, student_parameters=sum(p.numel() for p in m.parameters()),
            student_state_keys=len(m.state_dict()), native_U_teacher='own evolving EMA, unchanged',
            modules={k: str(v.__file__) for k, v in sys.modules.items() if k in [
                'experiments.cvs_u_quality66.teaching', 'experiments.cvs_u_quality66.model',
                'experiments.cvs_u_quality66.channel', 'SSDG.train_ssdg', 'DataAugmentation',
                'leo_practical.channel', 'leo_practical.residual']})
        d.write(self.out / 'u_quality_actual.json', actual)
        print(json.dumps(dict(stage='ACTUAL_MODULES_AND_U_QUALITY_TEACHER', **actual)), flush=True)

    def epoch(self, m, ctx, rows, out, device):
        super().epoch(m, ctx, rows, out, device)
        if len(m._clean_teacher_records) != len(rows) * d.STEPS:
            raise ValueError('Exactly one matched quality-teaching record per original U update')
        bank = model.teacher_bank(m)
        b = bank.summary()
        if b['registered_in_student'] or any(t.training or any(p.requires_grad for p in t.parameters()) for t in bank.teachers.values()):
            raise ValueError('Frozen teacher left eval/no-gradient training-only scope')
        progress = d.read(out / 'reference_progress.json')
        progress.update(extra_U_NN=m._clean_teacher_packets,
            extra_student_U_NN=m._clean_teacher_packets,
            extra_frozen_teacher_U_NN=b['total_forward_NN'], frozen_teacher_NN=b['forward_NN'],
            U_As=m._clean_teacher_channel.packets, existing_U_only=False, native_U_unchanged=True)
        d.write(out / 'reference_progress.json', progress)
        d.write(out / 'u_quality_progress.json', self.teaching_summary())

    def teaching_summary(self):
        records = self.model._clean_teacher_records
        n = len(records)
        packets = sum(r['packets'] for r in records)
        sums = {k: sum(r[k] for r in records) for k in (
            'KL', 'weighted_KL', 'native_packets', 'original_selected', 'extra_original_selected',
            'extra_student_NN', 'extra_frozen_teacher_NN')}
        quality = {}
        for name in ('q_teacher', 'q_observation', 'effective_q'):
            sub = [r[name] for r in records]
            count = sum(t['count'] for t in sub)
            mass = sum(t['mass'] for t in sub)
            positive = sum(t['positive'] for t in sub)
            quality[name] = dict(count=count, mass=mass, mean=mass / count if count else 0.,
                positive=positive, positive_fraction=positive / count if count else 0.,
                histogram=[sum(t['histogram'][i] for t in sub) for i in range(5)])
        mass_keys = ('q_teacher_mass', 'observation_weighted_mass', 'mass_uniform_mass', 'difference')
        mass_match = {k + '_sum': sum(r['mass_match'][k] for r in records) for k in mass_keys}
        mass_match.update(max_absolute_difference=max((r['mass_match']['absolute_difference'] for r in records), default=0.),
            uniform_multiplier_step_mean=sum(r['mass_match']['uniform_multiplier'] for r in records) / n if n else 0.)
        grouped = defaultdict(lambda: dict(count=0, none_snr_count=0, q_teacher_mass=0., q_observation_mass=0.,
            mass_uniform_mass=0., observation_weighted_mass=0., q_mass=0., positive_q=0,
            finite_snr_count=0, signal_to_added_noise_db_sum=0.))
        for r in records:
            for s in r['strata']:
                a = grouped[(s['scene'], s['state'], s['lock'])]
                for key in ('count', 'none_snr_count', 'q_teacher_mass', 'q_observation_mass',
                            'mass_uniform_mass', 'observation_weighted_mass', 'q_mass', 'positive_q'):
                    a[key] += s[key]
                finite_count = s['count'] - s['none_snr_count']
                if finite_count:
                    a['finite_snr_count'] += finite_count
                    a['signal_to_added_noise_db_sum'] += s['signal_to_added_noise_db_mean'] * finite_count
        strata = []
        for (scene, state, lock), a in sorted(grouped.items()):
            strata.append(dict(scene=scene, state=state, lock=lock, **a,
                q_mean=a['q_mass'] / a['count'], q_teacher_mean=a['q_teacher_mass'] / a['count'],
                q_observation_mean=a['q_observation_mass'] / a['count'],
                signal_to_added_noise_db_mean=a['signal_to_added_noise_db_sum'] / a['finite_snr_count'] if a['finite_snr_count'] else None))
        bank = model.teacher_bank(self.model)
        return dict(steps=n, arm=self.c['arm'], selected_teacher='FROZEN_AS20_EMA', packets_sum=packets,
            **{k + '_sum': v for k, v in sums.items()},
            extra_student_NN=sums['extra_student_NN'], extra_frozen_teacher_NN=sums['extra_frozen_teacher_NN'],
            frozen_teacher_NN={'B01': sum(r['frozen_teacher_NN']['B01'] for r in records)},
            KL_step_mean=sums['KL'] / n if n else 0., weighted_KL_step_mean=sums['weighted_KL'] / n if n else 0.,
            KL_packet_sum=sum(r['KL'] * r['packets'] for r in records),
            weighted_KL_packet_sum=sum(r['weighted_KL'] * r['packets'] for r in records),
            unweighted_KL_packet_sum=sum(r['unweighted_KL'] * r['packets'] for r in records),
            q_mass_sum=sum(r['q_mass'] for r in records), positive_q_sum=sum(r['positive_q'] for r in records),
            none_snr_count=sum(r['none_snr_count'] for r in records), quality_statistics=quality,
            mass_match=mass_match, strata=strata, frozen_teachers=bank.summary(), teacher_sources=self.c['teacher_sources'],
            all_frozen_eval=True, teacher_parameters_registered_in_student=False,
            student_state_keys=len(self.model.state_dict()), student_parameters=sum(p.numel() for p in self.model.parameters()),
            native_U_mask_unchanged=True, all_labeled_As_CE_retained=True, U_TX_truth_read=False,
            source_global_disable=False, student_urban_confidence_gate=False,
            subset_rule='first_min_128_native_U_batch_no_padding', temperature=1., tau=.95,
            max_weight=.16, ramp_epochs=5, selected_count_normalization=False,
            observation_proxy_is_recoverability_truth=False, extra_teacher_training_updates=0, extra_deployment_parameters=0)

    def finish(self):
        result = super().finish()
        records = self.model._clean_teacher_records
        model.teacher_bank(self.model).assert_frozen()
        if len(records) != d.UPDATES:
            raise ValueError('Complete2220 matched quality-teaching records required')
        summary = self.teaching_summary()
        extra, frozen = summary['extra_student_NN'], summary['extra_frozen_teacher_NN']
        if (extra != 284120 or frozen != 284120 or summary['native_packets_sum'] != 567000
                or self.model._clean_teacher_packets != extra
                or summary['frozen_teacher_NN'] != {'B01': 284120}
                or summary['frozen_teachers']['forward_NN'] != summary['frozen_teacher_NN']
                or summary['packets_sum'] != extra
                or any(s['count'] != extra for s in summary['quality_statistics'].values())):
            raise ValueError('Exact first128/tail124 extra U or single frozen teacher cost differs')
        with gzip.open(self.out / 'u_quality_steps.jsonl.gz', 'xt', encoding='utf8', compresslevel=3) as f:
            for r in records:
                f.write(json.dumps(r, allow_nan=False) + '\n')
        # Native student encoder counters include the additional student view.
        # Remove it once, then add both extra forwards explicitly below.
        result['observed_student_train_NN_including_extra'] = result['student_train_NN']
        result['student_train_NN'] -= extra
        result['native_extra_U_NN'], result['native_U_As'] = result['extra_U_NN'], result['U_As']
        result.update(extra_U_NN=extra, extra_student_U_NN=extra, extra_frozen_teacher_U_NN=frozen,
            U_As=extra, U_steps=d.UPDATES, U_channel=self.model._clean_teacher_channel.summary(),
            u_quality_summary=summary, new_epochs=10, parent_epochs=20, cumulative_epochs=30,
            new_updates=2220, parent_updates=4440, cumulative_updates=6660, source_RX_holdout=None,
            frozen_teacher_training_updates=0, extra_deployment_parameters=0,
            training_only_teachers=dict(summary['frozen_teachers'], frozen_parameters_and_buffers_unchanged=True),
            TX_RX_scene_state_lock=[dict(TX=k[0], RX=k[1], scene=k[2], state=k[3], lock=k[4], count=v)
                for k, v in sorted(self.model._full_coverage.items())])
        if (result['L_NN'] != 1704960 or result['L_As'] != 142080 or result['Ag_packets']
                or result['head_fit_updates'] or result['U_truth_read'] or result['native_extra_U_NN']):
            raise ValueError('Original sixL/nominalAs/oldU/default-head budget differs')
        native = sys.modules['SSDG.train_ssdg']
        scratch = SimpleNamespace(**self.parent['args'])
        d.write(self.out / 'ordinary_augmentation_actual.json', dict(base=native.build_aug_base_cfg(scratch),
            schedule_frozen_at=13, args={k: v for k, v in self.parent['args'].items() if k.startswith('aug_')},
            schedule_args_epochs=self.parent['args']['epochs'], source='own scratch20 checkpoint args',
            no_ordinary_before_practical=True, RDrop_same_IQ=True))
        result['native_NN'] = sum(result[k] for k in ('student_train_NN', 'student_eval_NN', 'teacher_train_NN', 'teacher_eval_NN'))
        result['training_total_NN'] = result['native_NN'] + extra + frozen
        if result['native_NN'] != 3162962 or result['training_total_NN'] != 3731202:
            raise ValueError('Matched native plus one U student/teacher pair forward budget differs')
        d.write(self.out / 'result.json', result)
        return result


def train(c):
    observer = Observer(c)
    segment_train(c, design=d, model_api=model, observer=observer)
    return observer.finish()
