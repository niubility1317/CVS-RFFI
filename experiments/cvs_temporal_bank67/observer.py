"""Native epoch-bank telemetry only; preserve P43 L/U/EMA and forward costs."""
from contextlib import contextmanager
import gzip
import json
import sys
from pathlib import Path
from . import design as d, model
from .source import parent_payload, segment_train
from .temporal import TemporalTrace
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original


class Observer(Original):
    def __init__(self, c):
        self.c = d.validate(c)
        self.optimizer = None
        self.out = Path(c['output_root'])
        self.parent = parent_payload(c)
        self.temporal_file = None
        self.temporal = TemporalTrace(self._emit_temporal, d.STEPS, d.RX)

    def _emit_temporal(self, record):
        if self.temporal_file is None or self.temporal_file.closed:
            raise ValueError('Native temporal trace must have its own open immutable output')
        self.temporal_file.write(json.dumps(record, allow_nan=False) + '\n')

    def data(self, native, ctx, out):
        super().data(native, ctx, out)
        self.temporal_file = gzip.open(self.out / 'temporal_steps.jsonl.gz', 'xt', encoding='utf8', compresslevel=3)

    def initialize(self, m, c, out):
        super().initialize(m, c, out)
        if (getattr(m, '_source_unlabeled_selection_hook', None) is not None
                or getattr(m, '_source_unlabeled_feature_hook', None) is not None):
            raise ValueError('The native temporal replacement cannot include another U mechanism')
        d.write(self.out / 'temporal_actual.json', dict(config=c,
            native_function='_temporal_bank_mask_tensor', native_algorithm_unchanged=True,
            modules={k: str(v.__file__) for k, v in sys.modules.items() if k in (
                'SSDG.train_ssdg', 'experiments.cvs_temporal_bank67.temporal',
                'experiments.cvs_temporal_bank67.model', 'leo_practical.channel', 'leo_practical.residual')},
            temporal_mode='epoch_bank', new_empty_bank_at_segment_start=True,
            student_parameters=sum(p.numel() for p in m.parameters()), student_state_keys=len(m.state_dict()),
            native_U_teacher='strict own stage20 EMA, original updates', extra_NN=0,
            labeled_CE_RDrop_SupCon_unchanged=True, U_TX_truth_read=False))

    @contextmanager
    def native_scope(self, native):
        original_bank = getattr(native, '_temporal_bank_mask_tensor', None)
        if not callable(original_bank):
            raise ValueError('Pinned native epoch-bank interface is absent; no batch-neighbor fallback')

        def observed(pseudo, conf, extra, args, device, *, epoch, bank):
            if (not hasattr(self, 'teacher') or self.teacher is self.model or self.teacher.training
                    or not self.model.training):
                raise ValueError('Temporal confirmation must use the original live own EMA context')
            return self.temporal.call(original_bank, pseudo, conf, extra, args, device,
                epoch=epoch, bank=bank, step=self.model._feature_l_steps)

        try:
            with super().native_scope(native):
                native._temporal_bank_mask_tensor = observed
                try:
                    yield
                finally:
                    native._temporal_bank_mask_tensor = original_bank
        finally:
            if self.temporal_file is not None and not self.temporal_file.closed:
                self.temporal_file.close()

    def epoch(self, m, ctx, rows, out, device):
        super().epoch(m, ctx, rows, out, device)
        summary = self.temporal.summary()
        if summary['steps'] != len(rows) * d.STEPS or summary['packets'] != len(rows) * 56700:
            raise ValueError('One native temporal-bank trace per original U batch is required')
        if summary['epoch_summaries'][0]['temporal_selected'] != 0:
            raise ValueError('The first local epoch must have no cross-epoch confirmation')
        self.temporal_file.flush()
        d.write(out / 'temporal_progress.json', summary)

    def finish(self):
        result = super().finish()
        summary = self.temporal.summary()
        if (summary['steps'] != d.UPDATES or summary['packets'] != 567000
                or len(summary['epoch_summaries']) != 10
                or any(e['steps'] != d.STEPS or e['packets'] != 56700 for e in summary['epoch_summaries'])
                or summary['epoch_summaries'][0]['temporal_selected'] != 0
                or result['optimizer_steps'] != 2220
                or result['L_NN'] != 1704960 or result['L_As'] != (142080 if self.c['As'] else 0)
                or result['extra_U_NN'] or result['U_As'] or result['U_truth_read']
                or result['head_fit_updates'] or result['Ag_packets']):
            raise ValueError('Native epoch-bank/original L/U/default-head budget differs')
        actual = d.read(self.out / 'actual_native_args.json')
        args = actual.get('args', actual)
        if (args.get('pseudo_temporal_mode') != 'epoch_bank'
                or args.get('pseudo_temporal_bank_min_streak') != 2
                or args.get('pseudo_temporal_min_conf') != .8
                or not args.get('pseudo_temporal_gate')):
            raise ValueError('The actual native temporal mode/min-streak/threshold differs')
        ema = d.read(self.out / 'EMA_initialization.json')
        if ema.get('origin') != 'strict own stage20 EMA' or ema.get('extra_teacher'):
            raise ValueError('Original own EMA must be strictly restored without another teacher')
        result.update(temporal_summary=summary, U_steps=d.UPDATES,
            native_U_mask_components='original confidence/domain/strong masks AND native physical-ID epoch bank',
            temporal_algorithm_unchanged=True, new_epochs=10, parent_epochs=20, cumulative_epochs=30,
            new_updates=2220, parent_updates=4440, cumulative_updates=6660,
            extra_neural_forwards=0, extra_deployment_parameters=0, source_RX_holdout=None,
            own_EMA_successful_updates=2220, own_EMA_parent_restored=True)
        result['native_NN'] = sum(result[k] for k in ('student_train_NN', 'student_eval_NN', 'teacher_train_NN', 'teacher_eval_NN'))
        result['training_total_NN'] = result['native_NN']
        if result['native_NN'] != 3162962:
            raise ValueError('Replacing the temporal confirmation must not add neural forwards')
        d.write(self.out / 'temporal_progress.json', summary)
        d.write(self.out / 'result.json', result)
        return result


def train(c):
    observer = Observer(c)
    segment_train(c, design=d, model_api=model, observer=observer)
    return observer.finish()
