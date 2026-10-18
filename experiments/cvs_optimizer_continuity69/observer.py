"""Keep corresponding parent AdamW states; native data, EMA and sixL stay intact."""
from collections import Counter
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from . import design as d, model
from .optimizer_state import restore_optimizer, state_summary
from .source import parent_payload, segment_train
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original


class Observer(Original):
    def __init__(self, c):
        self.c = d.validate(c)
        self.optimizer = None
        self.out = Path(c['output_root'])
        self.parent = parent_payload(c)
        self.optimizer_epochs = []
        self.optimizer_file = None

    def data(self, native, ctx, out):
        super().data(native, ctx, out)
        self.optimizer_file = gzip.open(self.out / 'optimizer_state_epochs.jsonl.gz', 'xt', encoding='utf8', compresslevel=3)

    def initialize(self, m, c, out):
        super().initialize(m, c, out)
        if (getattr(m, '_source_unlabeled_selection_hook', None) is not None
                or getattr(m, '_source_unlabeled_feature_hook', None) is not None):
            raise ValueError('No new U hook in optimizer continuity control')
        d.write(self.out / 'optimizer_keep_actual.json', dict(config=c,
            modules={k: str(v.__file__) for k, v in sys.modules.items() if k in (
                'SSDG.train_ssdg', 'DataAugmentation', 'leo_practical.channel', 'leo_practical.residual',
                'experiments.cvs_optimizer_continuity69.optimizer_state',
                'experiments.cvs_rdrop_sinc_units.optimizer')},
            native_U_temporal_mode='batch_neighbor', new_U_rules=0, extra_NN=0,
            original_default_head=True, sixL_RDrop_SupCon_unchanged=True))

    def optimizer_created(self, opt):
        # This callback is inside the native fresh AdamW constructor, after
        # CENTER_UNITS hooks attach and before any epoch or optimizer update.
        if opt.state:
            raise ValueError('The new factory must expose an empty optimizer before KEEP load')
        summary = restore_optimizer(opt, self.parent['optimizer'], expected_steps=4440,
            named_parameters=self.model.named_parameters(), parent_model=self.parent['model'])
        self.optimizer_initialization = dict(state_entries=len(opt.state), reset=False,
            parent_state_loaded=True, schedule='cosine10', lr=opt.param_groups[0]['lr'],
            weight_decay=opt.param_groups[0]['weight_decay'], parent_optimizer_steps=4440,
            parent_checkpoint=self.c['checkpoint_sources'][0],
            fresh_sinc_unit_steps=opt._sinc_unit_steps,
            fresh_sinc_physical_clip_calls=opt._sinc_physical_clip_calls,
            restored=summary, extra_NN=0, optimizer_updates_during_load=0)
        if opt._sinc_unit_steps or opt._sinc_physical_clip_calls or opt._sinc_unit_lr_records:
            raise ValueError('Parent moment load must preserve empty new-segment Sinc counters')
        d.write(self.out / 'optimizer_initialization.json', self.optimizer_initialization)

    @contextmanager
    def native_scope(self, native):
        try:
            with super().native_scope(native):
                yield
        finally:
            if self.optimizer_file is not None and not self.optimizer_file.closed:
                self.optimizer_file.close()

    def optimizer_progress(self, epoch):
        opt = self.optimizer
        histogram = Counter()
        inherited, added = 0, 0
        for group, old in zip(opt.param_groups, self.parent['optimizer']['param_groups']):
            for parameter, pid in zip(group['params'], old['params']):
                before = self.parent['optimizer']['state'].get(pid)
                after = opt.state.get(parameter)
                if before is None:
                    if after is not None:
                        added += 1
                    continue
                if after is None:
                    raise ValueError('Inherited active parameter lost AdamW state')
                previous = float(before['step'].item() if torch.is_tensor(before['step']) else before['step'])
                current = float(after['step'].item() if torch.is_tensor(after['step']) else after['step'])
                delta = int(current - previous)
                if current - previous != delta or delta < 0 or delta > epoch * d.STEPS:
                    raise ValueError('Per-parameter AdamW step did not continue within actual updates')
                histogram[str(delta)] += 1
                inherited += 1
        return dict(epoch=epoch, cumulative_epoch=epoch + 20, successful_updates=epoch*d.STEPS,
            parent_successful_updates=4440, optimizer_state=state_summary(opt),
            inherited_state_entries=inherited, new_state_entries=added,
            inherited_step_delta_histogram=dict(sorted(histogram.items(), key=lambda x: int(x[0]))),
            sinc_unit_steps=opt._sinc_unit_steps, sinc_physical_clip_calls=opt._sinc_physical_clip_calls,
            last_unit_step=opt._sinc_unit_lr_records[-1] if opt._sinc_unit_lr_records else None,
            no_extra_optimizer_updates=True, extra_NN=0)

    def epoch(self, m, ctx, rows, out, device):
        super().epoch(m, ctx, rows, out, device)
        record = self.optimizer_progress(len(rows))
        if len(self.optimizer_epochs) + 1 != len(rows):
            raise ValueError('Exactly one optimizer-state telemetry record per completed epoch')
        self.optimizer_epochs.append(record)
        self.optimizer_file.write(json.dumps(record, allow_nan=False) + '\n')
        self.optimizer_file.flush()
        d.write(out / 'optimizer_keep_progress.json', record)

    def finish(self):
        result = super().finish()
        m = self.model
        if (len(self.optimizer_epochs) != 10 or result['optimizer_steps'] != 2220
                or result['L_NN'] != 1704960 or result['L_As'] != (142080 if self.c['As'] else 0)
                or result['extra_U_NN'] or result['U_As'] or result['U_truth_read']
                or result['head_fit_updates'] or result['Ag_packets']):
            raise ValueError('Original sixL/nativeU/true-label/default-head budget differs')
        actual = d.read(self.out / 'actual_native_args.json')
        args = actual.get('args', actual)
        if args.get('pseudo_temporal_mode') != 'batch_neighbor' or not args.get('pseudo_temporal_gate'):
            raise ValueError('Original native batch-neighbor U rule changed')
        ema = d.read(self.out / 'EMA_initialization.json')
        if ema.get('origin') != 'strict own stage20 EMA' or ema.get('extra_teacher'):
            raise ValueError('Only strict corresponding own EMA is allowed')
        final = self.optimizer_progress(10)
        d.write(self.out / 'optimizer_state_final.json', final)
        result.update(optimizer_keep_summary=dict(initialization=self.optimizer_initialization,
                final=final, epochs=len(self.optimizer_epochs), inherited_per_parameter_steps=True,
                fresh_cosine10=True, fresh_Sinc_segment_counters=True, old_LR_restored=False),
            U_steps=d.UPDATES, native_U_mask_components='original batch-neighbor/confidence/domain/strong masks',
            new_epochs=10, parent_epochs=20, cumulative_epochs=30, new_updates=2220,
            parent_updates=4440, cumulative_updates=6660, extra_neural_forwards=0,
            extra_deployment_parameters=0, source_RX_holdout=None,
            own_EMA_successful_updates=2220, own_EMA_parent_restored=True,
            TX_RX_scene_state_lock=[dict(TX=k[0], RX=k[1], scene=k[2], state=k[3], lock=k[4], count=v)
                for k, v in sorted(m._full_coverage.items())])
        result['native_NN'] = sum(result[k] for k in ('student_train_NN', 'student_eval_NN', 'teacher_train_NN', 'teacher_eval_NN'))
        result['training_total_NN'] = result['native_NN']
        if result['native_NN'] != 3162962:
            raise ValueError('Optimizer state continuity must add no neural forwards')
        native = sys.modules['SSDG.train_ssdg']
        parent_args = SimpleNamespace(**self.parent['args'])
        d.write(self.out / 'ordinary_augmentation_actual.json', dict(base=native.build_aug_base_cfg(parent_args),
            schedule_frozen_at=13, args={k: v for k, v in self.parent['args'].items() if k.startswith('aug_')},
            schedule_args_epochs=self.parent['args']['epochs'], source='own scratch20 checkpoint args',
            no_ordinary_before_practical=True, RDrop_same_IQ=True))
        d.write(self.out / 'result.json', result)
        return result


def train(c):
    observer = Observer(c)
    segment_train(c, design=d, model_api=model, observer=observer)
    return observer.finish()
