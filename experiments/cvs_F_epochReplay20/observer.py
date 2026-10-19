"""Observe complete native ordinary/As replay/U/ownEMA and physical Sinc budgets."""
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
        self.epoch_records = []

    def data(self, native, ctx, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        if ctx['num_domains'] != 15:
            raise ValueError('All five source RX and three days required')
        self.sequence = gzip.open(Path(out) / 'training_sequence.jsonl.gz', 'xt',
                                  encoding='utf8', compresslevel=3)
        for key, role in (('train_loader', 'L'), ('unlabeled_loader', 'U')):
            ctx[key] = RecordingLoader(ctx[key], role, self.sequence)

    def initialize(self, m, c, out):
        self.model, self.out = m, Path(out)
        if (m._legacy_mode != 'ORIGINAL' or m._legacy_alpha != 0.
                or any('supcon' in k or 'risk_logq' in k or 'class_teacher' in k
                       for k in vars(m)) or m._epochreplay_channel.calls):
            raise ValueError('Pure F forbids RDrop/CROSS_RX/risk/KD/external teacher')
        d.write(self.out / 'actual_modules.json', {
            k: str(v.__file__) for k, v in sys.modules.items()
            if k in ('SSDG.train_ssdg', 'DataAugmentation',
                     'leo_practical.channel', 'leo_practical.residual')})
        d.write(self.out / 'actual_channel.json', m._epochreplay_channel.audit())

    def capture_ordinary(self, raw, ordinary, labels, step):
        witness = self.model._epochreplay_witness
        if (witness['raw'] is not None or step != self.model._epochreplay_steps + 1
                or not self.model.training or not torch.is_grad_enabled()):
            raise ValueError('One original ordinary call immediately before native L required')
        witness.update(raw=raw.detach().clone(), ordinary=ordinary.detach().clone(),
                       labels=labels.detach().clone(), used_As=None, phase='first',
                       step=step, first_seen=0, extra_seen=0)

    @contextmanager
    def native_scope(self, native):
        deepcopy, factory = native.deepcopy, native.make_augmentor
        self.ordinary_factory = ordinary_factory(
            factory, self.ordinary_records, self.capture_ordinary)
        native.make_augmentor = self.ordinary_factory

        def copied(value, *args, **kwargs):
            result = deepcopy(value, *args, **kwargs)
            if value is getattr(self, 'model', None):
                self.ema = result
                enc = result.id_backbone.encoder
                self.ema_offset = dict(train=enc._full_train_packets, eval=enc._full_eval_packets)
                channel = result._epochreplay_channel
                if (channel.calls or channel.packets or channel.fresh.packets
                        or channel.replay.stats()['cumulative_used']
                        or channel.replay.stats()['current_cache_entries']):
                    raise ValueError('Own EMA deepcopy must start with a new empty replay channel')
            return result

        native.deepcopy = copied
        try:
            with native_scope(self, 'CENTER_UNITS'):
                yield
        finally:
            native.deepcopy, native.make_augmentor = deepcopy, factory
            if hasattr(self, 'sequence'):
                self.sequence.close()

    def epoch(self, m, ctx, rows, out, device):
        ep = len(rows)
        n, u = ep * 222, max(0, ep - 13) * 56700
        enc, ema = m.id_backbone.encoder, self.ema.id_backbone.encoder
        channel, ema_channel = m._epochreplay_channel, self.ema._epochreplay_channel
        if (m._feature_l_steps != n or m._legacy_calls != n
                or m._epochreplay_steps != n or m._epochreplay_L_packets != n * 256
                or len(m._epochreplay_records) != n or len(m._epochreplay_replay_records) != n
                or len(self.ordinary_records) != n or self.ordinary_factory.calls != 1
                or self.optimizer._sinc_physical_clip_calls != n
                or self.optimizer._sinc_unit_steps != n
                or m._epochreplay_witness['raw'] is not None
                or enc._full_train_packets != n * 256 + u
                or ema._full_train_packets - self.ema_offset['train'] != 0
                or ema._full_eval_packets - self.ema_offset['eval'] != u
                or channel.calls != n or channel.packets != n * 128
                or channel.fresh.calls != n or channel.fresh.packets != n * 128
                or ema_channel.calls or ema_channel.packets or ema_channel.fresh.packets):
            raise ValueError('Actual pure F/ordinary/used-As/U/EMA/Sinc budget differs')
        records = m._epochreplay_records[(ep - 1) * 222:n]
        if (len(records) != 222 or any(r['epoch'] != ep for r in records)
                or any(r['first_encoder_calls'] != 1 or r['extra_encoder_calls'] != 1
                       or not r['ordinary_first_forward_verified'] for r in records)):
            raise ValueError('Each native epoch requires exactly 222 witnessed two-L steps')
        unique = hit = cross_step = 0
        observed = {}
        for r in records:
            if (r['fresh_meta_summary']['packets'] != 128 or r['used_meta_summary']['packets'] != 128
                    or r['online_As']['all']['n'] != 128 or len(r['packets']) != 128):
                raise ValueError('Full fresh and used L128 observations must be retained')
            for p in r['packets']:
                pid = tuple(p['sample_id'])
                current = (r['step'], p['slot'])
                origin = (p['source_step'], p['source_slot'])
                if p['cache_hit']:
                    if pid not in observed or origin != observed[pid][:2]:
                        raise ValueError('Replay hit must refer to the current epoch first observation')
                    if p['used_scene'] != observed[pid][2]:
                        raise ValueError('Replayed scene changed within current physical-ID epoch')
                    hit += 1
                    cross_step += int(origin[0] < r['step'])
                else:
                    if pid in observed or origin != current or p['used_scene'] != p['fresh_scene']:
                        raise ValueError('New physical-ID observation must use the current full As')
                    observed[pid] = (*origin, p['used_scene'])
                    unique += 1
        replay = channel.replay.stats()
        replay_steps = m._epochreplay_replay_records[(ep - 1) * 222:n]
        if (replay['steps'] != n or replay['epoch'] != ep or replay['epochs_started'] != ep
                or replay['reset_count'] != ep - 1 or replay['current_cache_entries'] != len(observed)
                or replay['cumulative_generated'] != n * 128 or replay['cumulative_used'] != n * 128
                or replay['miss_packets'] + replay['hit_packets'] != n * 128
                or sum(r['miss_packets'] for r in replay_steps) != unique
                or sum(r['hit_packets'] for r in replay_steps) != hit
                or any(r['epoch_cache_reset'] != (i == 0)
                       or r['reset_count'] != ep - 1
                       or r['fresh_generated_packets'] != 128 or r['used_packets'] != 128
                       for i, r in enumerate(replay_steps))):
            raise ValueError('Actual epoch reset, cache origins and fresh/used cumulative budgets differ')
        self.epoch_records.append(dict(epoch=ep, steps=222, cumulative_steps=n,
            fresh_packets=222 * 128, used_packets=222 * 128,
            unique_packets=unique, hit_packets=hit, cross_step_hit_packets=cross_step,
            unique_physical_IDs=len(observed), expected_reset_count=ep - 1,
            replay=replay, epoch_provenance_reconstructed=True))
        if unique + hit != 222 * 128:
            raise ValueError('Epoch replay packet accounting differs')
        d.write(out / 'reference_progress.json', dict(
            epoch=ep, steps=n, L_packets=m._epochreplay_L_packets,
            As_packets=channel.packets, fresh_As_packets=channel.fresh.packets,
            used_As_packets=channel.packets, ordinary_packets=n * 128, teacher_packets=0,
            U_packets=u, U_student_NN=u, own_EMA_eval_packets=u,
            ordinary_first_forward_verified_steps=n, source_RX=list(d.FIT), replay=replay,
            RDrop=False, CROSS_RX=False, U_truth_read=False, target_access=False))
        d.write(out / 'replay_epochs.json', self.epoch_records)

    def finish(self):
        out, m = self.out, self.model
        cp = torch.load(out / 'final_ssdg.pth', map_location='cpu', weights_only=False)
        done = d.read(out / 'completion.json')
        initial = torch.load(out / 'initial_smoke.pt', map_location='cpu', weights_only=False)['model']
        state = cp['model']
        if (done['epoch'] != 20 or done['optimizer_steps'] != 4440
                or done['label_epochs'] != 13 or done['pseudo_epochs'] != 7
                or done['final_source_metrics']['source_val_count'] != 27000
                or cp.get('ema_model') is None or cp['args']['baseline_ckpt']
                or cp['args']['teacher_ckpt'] or cp['a1_ema_successful_updates'] != 4440
                or len(m._epochreplay_records) != 4440 or len(self.ordinary_records) != 4440
                or len(m._epochreplay_replay_records) != 4440 or len(self.epoch_records) != 20
                or set(initial) != set(state) or any('replay' in k for k in state)):
            raise ValueError('Own scratch20/full-source/cache-free checkpoint execution incomplete')
        if (cp['run_id'] != self.c['run_id'] or cp['candidate_id'] != self.c['row_id']
                or cp['checkpoint_selection'] != 'final_only'
                or done['config'] != self.c or done['target_access'] or done['head_fit_updates']):
            raise ValueError('Final own default-head checkpoint identity differs')
        head = 'id_backbone.encoder.core.id_backbone.cls_head.weight'
        if state[head].shape != (6, 160):
            raise ValueError('Existing jointly trained classifier changed')
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
        if (any(r['returned_ordinary_MSE'] != 0. or r['full_native_draws'] != 1
                or r['count'] != 128 or not r['ordinary_output_used_by_student']
                for r in self.ordinary_records)
                or any(not r['ordinary_first_forward_verified'] or r['encoder_ordinary_MSE'] != 0.
                       or r['hook_raw_MSE'] != 0. or r['encoder_used_As_MSE'] != 0.
                       for r in m._epochreplay_records)):
            raise ValueError('Actual ordinary/raw-hook/used-As encoder witness differs')
        enc, ema = m.id_backbone.encoder, self.ema.id_backbone.encoder
        if (enc._joint160_seen_calls == 0 or enc._joint160_last_shape[1] != 160
                or enc._joint160_seen_packets != enc._full_train_packets + enc._full_eval_packets):
            raise ValueError('Observed real encoder output must prove finite joint160')
        for name, records in (
                ('mechanism_steps', m._epochreplay_records),
                ('replay_steps', m._epochreplay_replay_records),
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
            ordinary_packets=568320, ordinary_output_used_by_student=True,
            first_L_input='exact native ordinary output', ordinary_calls=4440,
            ordinary_output_returned_unmodified=True, U_augmentor_wrapped=False,
            U_weak_strong_unchanged=True, practical_input_is_raw=True,
            ordinary_before_As=False, As_after_normalization=False,
            physical_channel_reused_unchanged=True, fresh_channel_draws_retained=True,
            whole_training_RNG_matched=False))
        result = dict(
            status='PURE_F_SOURCE_TRAINED', config=self.c, steps=4440,
            joint160_actual=dict(calls=enc._joint160_seen_calls,
                packets=enc._joint160_seen_packets, last_shape=enc._joint160_last_shape,
                finite_checked_each_forward=True, extra_verification_NN=0),
            changed_parameters=changed,
            trainable_parameters=sum(p.numel() for p in m.parameters()),
            student_train_packets=enc._full_train_packets,
            student_eval_packets=enc._full_eval_packets, L_packets=m._epochreplay_L_packets,
            teacher_packets=0, own_EMA_train_packets=ema._full_train_packets - self.ema_offset['train'],
            own_EMA_eval_packets=ema._full_eval_packets - self.ema_offset['eval'],
            own_EMA_inherited_counter_offset=self.ema_offset, own_EMA_successful_updates=4440,
            U_packets=396900, U_hidden_steps=1554, ordinary_packets=568320,
            ordinary_first_forward_verified_steps=4440,
            fresh_As_packets=m._epochreplay_channel.fresh.packets,
            used_As_packets=m._epochreplay_channel.packets,
            channel=m._epochreplay_channel.fresh.summary(),
            replay=m._epochreplay_channel.replay.stats(), replay_epochs=self.epoch_records,
            cache_in_state_dict=False, source_metrics=done['final_source_metrics'],
            CROSS_RX=False, RDrop=False, external_teacher=False, KD=False,
            head_fit_updates=0, target_access=False)
        d.write(out / 'training_result.json', result)
        return result


def train(c):
    observer = Observer(c)
    shared_train(c, design=d, model_api=api, observer=observer)
    return observer.finish()
