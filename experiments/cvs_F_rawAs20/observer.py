"""Original U/ownEMA and pure F; verify actual raw inputs and complete budgets."""
from contextlib import contextmanager
from pathlib import Path
import gzip
import json
import sys
import torch
from . import design as d, model as api
from .source import ordinary_factory
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_fullsource_concat20.source import RecordingLoader
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope


class Observer:
    def __init__(self, c):
        self.c = c
        self.optimizer = None
        self.ordinary_records = []

    def data(self, native, ctx, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        if ctx['num_domains'] != 15:
            raise ValueError('All five source RX and three days required')
        self.sequence = gzip.open(Path(out) / 'training_sequence.jsonl.gz', 'xt',
                                  encoding='utf8', compresslevel=3)
        for key, role in (('train_loader', 'L'), ('unlabeled_loader', 'U')):
            ctx[key] = RecordingLoader(ctx[key], role, self.sequence)

    def initialize(self, m, c, out):
        self.model = m
        self.out = Path(out)
        if (m._legacy_mode != 'ORIGINAL' or m._legacy_alpha != 0.
                or any('supcon' in k or 'risk_logq' in k or 'class_teacher' in k
                       for k in vars(m))):
            raise ValueError('Pure F forbids RDrop/CROSS_RX/risk/KD/external teacher')
        d.write(self.out / 'actual_modules.json', {
            k: str(v.__file__) for k, v in sys.modules.items()
            if k in ('SSDG.train_ssdg', 'DataAugmentation',
                     'leo_practical.channel', 'leo_practical.residual')})
        if m._rawas_channel is not None:
            d.write(self.out / 'actual_channel.json', m._rawas_channel.audit())

    def capture_raw(self, raw, labels, step):
        witness = self.model._rawas_witness
        if (witness['raw'] is not None or step != self.model._rawas_steps + 1
                or not self.model.training or not torch.is_grad_enabled()):
            raise ValueError('Exactly one ordinary shadow immediately before native L is required')
        witness.update(raw=raw.detach().clone(), labels=labels.detach().clone(),
                       extra=None, phase='first', step=step,
                       first_seen=0, extra_seen=0)

    @contextmanager
    def native_scope(self, native):
        deepcopy = native.deepcopy
        factory = native.make_augmentor
        self.ordinary_factory = ordinary_factory(factory, self.ordinary_records, self.capture_raw)
        native.make_augmentor = self.ordinary_factory

        def copied(value, *args, **kwargs):
            result = deepcopy(value, *args, **kwargs)
            if value is getattr(self, 'model', None):
                self.ema = result
                enc = result.id_backbone.encoder
                self.ema_offset = dict(train=enc._full_train_packets, eval=enc._full_eval_packets)
            return result

        native.deepcopy = copied
        try:
            with native_scope(self, 'CENTER_UNITS'):
                yield
        finally:
            native.deepcopy = deepcopy
            native.make_augmentor = factory
            if hasattr(self, 'sequence'):
                self.sequence.close()

    def epoch(self, m, ctx, rows, out, device):
        ep = len(rows)
        n = ep * 222
        u = max(0, ep - 13) * 56700
        enc = m.id_backbone.encoder
        ema = self.ema.id_backbone.encoder
        if (m._feature_l_steps != n or m._legacy_calls != n
                or m._rawas_steps != n or m._rawas_L_packets != n * 256
                or len(self.ordinary_records) != n or self.ordinary_factory.calls != 1
                or self.optimizer._sinc_physical_clip_calls != n
                or self.optimizer._sinc_unit_steps != n
                or m._rawas_witness['raw'] is not None
                or enc._full_train_packets != n * 256 + u
                or ema._full_train_packets - self.ema_offset['train'] != 0
                or ema._full_eval_packets - self.ema_offset['eval'] != u):
            raise ValueError('Actual pure F/raw inputs/L/U/EMA/Sinc budget differs')
        if m._rawas_channel is not None and m._rawas_channel.packets != n * 128:
            raise ValueError('Every original legal As observation must be used')
        d.write(out / 'reference_progress.json', dict(
            epoch=ep, steps=n, L_packets=m._rawas_L_packets,
            As_packets=0 if m._rawas_channel is None else m._rawas_channel.packets,
            ordinary_shadow_packets=n * 128, teacher_packets=0,
            U_packets=u, U_student_NN=u, own_EMA_eval_packets=u,
            raw_first_forward_verified_steps=n, source_RX=list(d.FIT),
            RDrop=False, CROSS_RX=False, U_truth_read=False, target_access=False))

    def finish(self):
        out = self.out
        m = self.model
        cp = torch.load(out / 'final_ssdg.pth', map_location='cpu', weights_only=False)
        done = d.read(out / 'completion.json')
        initial = torch.load(out / 'initial_smoke.pt', map_location='cpu', weights_only=False)['model']
        state = cp['model']
        if (done['epoch'] != 20 or done['optimizer_steps'] != 4440
                or done['final_source_metrics']['source_val_count'] != 27000
                or cp.get('ema_model') is None or cp['args']['baseline_ckpt']
                or cp['args']['teacher_ckpt'] or len(m._rawas_records) != 4440
                or len(self.ordinary_records) != 4440):
            raise ValueError('Own scratch20/default-head/full-source execution incomplete')
        head = 'id_backbone.encoder.core.id_backbone.cls_head.weight'
        if state[head].shape != (6, 160):
            raise ValueError('Default classifier changed')
        changed = {
            k: dict(l2=float((value.float() - initial[k].float()).norm()),
                    initial_l2=float(initial[k].float().norm()))
            for k, value in state.items()
            if k in dict(m.named_parameters()) and not torch.equal(value, initial[k])}
        if (head not in changed or not any('.sinc.' in k for k in changed)
                or not any('.sequence_pool.' in k for k in changed)
                or not any('.core.id_backbone.' in k and '.cls_head.' not in k
                           and '.sinc.' not in k for k in changed)):
            raise ValueError('Sinc, signature, backbone and default head must actually update')
        if (any(r['returned_raw_MSE'] != 0. or r['full_native_draws'] != 1
                or r['count'] != 128 for r in self.ordinary_records)
                or any(not r['raw_first_forward_verified'] or r['encoder_raw_MSE'] != 0.
                       or r['hook_raw_MSE'] != 0. for r in m._rawas_records)):
            raise ValueError('Raw native L input witness differs')
        m.eval()
        with torch.no_grad():
            feat = m.id_backbone.encoder.features(torch.zeros(
                2, 2, 256, device=next(m.parameters()).device))
        if feat.shape != (2, 160) or not bool(torch.isfinite(feat).all()):
            raise ValueError('Actual deployment joint160 required')
        for name, records in (
                ('mechanism_steps', m._rawas_records),
                ('ordinary_draw_steps', self.ordinary_records),
                ('unit_steps', self.optimizer._sinc_unit_lr_records)):
            with gzip.open(out / (name + '.jsonl.gz'), 'xt', encoding='utf8', compresslevel=3) as f:
                for record in records:
                    f.write(json.dumps(record, allow_nan=False) + '\n')
        native = sys.modules['SSDG.train_ssdg']
        args = type('ActualArgs', (), cp['args'])()
        d.write(out / 'ordinary_augmentation_actual.json', dict(
            use_aug=cp['args']['use_aug'], base_config=native.build_aug_base_cfg(args),
            args={k: value for k, value in cp['args'].items() if k.startswith('aug_')},
            ordinary_shadow_packets=568320, ordinary_output_used_by_student=False,
            first_L_input='pre-call raw clone', ordinary_calls=4440,
            U_augmentor_wrapped=False, U_weak_strong_unchanged=True,
            practical_input_is_raw=True, ordinary_before_As=False,
            As_after_normalization=False, physical_channel_reused_unchanged=True,
            whole_training_RNG_matched=False))
        enc = m.id_backbone.encoder
        ema = self.ema.id_backbone.encoder
        result = dict(
            status='PURE_F_SOURCE_TRAINED', config=self.c, steps=4440,
            joint160_shape=list(feat.shape), changed_parameters=changed,
            trainable_parameters=sum(p.numel() for p in m.parameters()),
            student_train_packets=enc._full_train_packets,
            student_eval_packets=enc._full_eval_packets,
            L_packets=m._rawas_L_packets, teacher_packets=0,
            own_EMA_train_packets=ema._full_train_packets - self.ema_offset['train'],
            own_EMA_eval_packets=ema._full_eval_packets - self.ema_offset['eval'],
            own_EMA_inherited_counter_offset=self.ema_offset,
            U_packets=396900, ordinary_shadow_packets=568320,
            raw_first_forward_verified_steps=4440,
            channel=None if m._rawas_channel is None else m._rawas_channel.summary(),
            source_metrics=done['final_source_metrics'], CROSS_RX=False,
            RDrop=False, external_teacher=False, KD=False,
            head_fit_updates=0, target_access=False)
        d.write(out / 'training_result.json', result)
        return result


def train(c):
    observer = Observer(c)
    shared_train(c, design=d, model_api=api, observer=observer)
    return observer.finish()
