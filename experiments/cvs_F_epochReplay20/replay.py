"""Reuse each physical L packet's first full As IQ within one training epoch.

The caller always generates the original fresh Practical batch first.  This
object never opens data, generates a channel, accepts labels, or runs a model.
Existing source-builder/native-hook role checks remain the data authority.
"""
from copy import deepcopy
from numbers import Integral

import torch


SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
SOURCE_RX = (1, 3, 4, 6, 8)
PACKETS = 128
STEPS_PER_EPOCH = 222


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _pid(record, namespace):
    if not isinstance(record, dict):
        raise ValueError('Original Practical packet record required')
    value = record.get('sample_id')
    if (not isinstance(value, (list, tuple)) or len(value) != 5
            or not all(_integer(v) for v in value)):
        raise ValueError('Five ordered physical metadata integers required')
    pid = tuple(int(v) for v in value)
    eq, rx, day, sig, index = pid
    if eq != 1 or rx not in SOURCE_RX or day not in (1, 2, 3) or sig < 0 or index < 0:
        raise ValueError('Original source EQ/RX/day/physical metadata required')
    if (record.get('namespace') != namespace
            or record.get('session_id') != f'rx{rx}:day{day}:eq{eq}'
            or record.get('scene') not in SCENES
            or not isinstance(record.get('state_end'), str) or not record['state_end']
            or not isinstance(record.get('receiver_lock_status'), str)
            or not record['receiver_lock_status']):
        raise ValueError('Current fresh training Practical namespace/state required')
    return pid


class EpochReplay:
    """Ephemeral CPU IQ cache, deliberately outside all model state_dicts."""

    def __init__(self):
        self.cache = {}
        self.epoch = 0
        self.steps = 0
        self.reset_count = 0
        self.epochs_started = 0
        self.generated_packets = 0
        self.used_packets = 0
        self.miss_packets = 0
        self.hit_packets = 0
        self.hit_changed_packets = 0
        self.hit_squared_error = 0.
        self.hit_elements = 0
        self.hit_max_abs = 0.

    def __deepcopy__(self, memo):
        result = type(self)()
        memo[id(self)] = result
        return result

    def audit(self):
        return dict(mode='first_fullAs_per_physicalID_per_epoch',
                    key_order=['eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index'],
                    cache_device='cpu', cache_contains_detached_IQ_and_raw=True,
                    fresh_fullAs_generated_every_step=True,
                    cache_reset='before the first step of every new epoch',
                    reset_count_excludes_first_epoch=True,
                    within_batch_first_slot_wins=True,
                    labels_or_predictions_consumed=False,
                    data_roles_checked_by='existing source builder and native L hook',
                    evaluation_uses_cache=False, persistent_model_state=False)

    def summary(self):
        return dict(steps=self.steps, epoch=self.epoch,
                    reset_count=self.reset_count, epochs_started=self.epochs_started,
                    current_cache_entries=len(self.cache),
                    cumulative_generated=self.generated_packets,
                    cumulative_used=self.used_packets,
                    fresh_generated_packets=self.generated_packets,
                    used_packets=self.used_packets,
                    miss_packets=self.miss_packets, hit_packets=self.hit_packets,
                    unique_epoch_pid_realizations=self.miss_packets,
                    hit_changed_packets=self.hit_changed_packets,
                    hit_fresh_used_MSE=(self.hit_squared_error / self.hit_elements
                                        if self.hit_elements else 0.),
                    hit_fresh_used_max_abs=self.hit_max_abs)

    def stats(self):
        return self.summary()

    def apply(self, fresh, records, raw, *, epoch, step):
        if (not _integer(epoch) or not _integer(step)
                or not 1 <= epoch <= 20 or not 1 <= step <= 4440
                or epoch != (step - 1) // STEPS_PER_EPOCH + 1
                or step != self.steps + 1):
            raise ValueError('Exactly ordered scratch E1..20/step1..4440 required')
        if (not torch.is_tensor(fresh) or not torch.is_tensor(raw)
                or fresh.shape != (PACKETS, 2, 256) or raw.shape != fresh.shape
                or fresh.dtype != torch.float32 or raw.dtype != torch.float32
                or fresh.device != raw.device
                or not bool(torch.isfinite(fresh).all())
                or not bool(torch.isfinite(raw).all())):
            raise ValueError('Finite matching float32 raw/fresh source L128 required')
        if not isinstance(records, (tuple, list)) or len(records) != PACKETS:
            raise ValueError('All128 fresh Practical packet records required')
        namespace = f'FcrossRX/train/e{epoch}/step{step}/L_S'
        ids = [_pid(record, namespace) for record in records]
        changed_epoch = epoch != self.epoch
        if changed_epoch and step != (epoch - 1) * STEPS_PER_EPOCH + 1:
            raise ValueError('Epoch cache must start at its actual first native step')
        # Build locally before committing any counters/cache entries, so a bad
        # raw duplicate or malformed record cannot consume an ordered step.
        cache = {} if changed_epoch else self.cache.copy()
        raw_cpu, fresh_cpu = raw.detach().cpu(), fresh.detach().cpu()
        values, used_records, source_steps, source_slots, flags = [], [], [], [], []
        hit_changed = 0
        hit_error = 0.
        hit_max = 0.
        for slot, (pid, record) in enumerate(zip(ids, records)):
            hit = pid in cache
            if hit:
                entry = cache[pid]
                if not torch.equal(raw_cpu[slot], entry['raw']):
                    raise ValueError('Same physicalID raw IQ changed within its epoch')
            else:
                entry = dict(raw=raw_cpu[slot].clone(), iq=fresh_cpu[slot].clone(),
                             record=deepcopy(record), step=int(step), slot=slot)
                cache[pid] = entry
            used = entry['iq']
            if hit:
                difference = fresh_cpu[slot].double() - used.double()
                hit_changed += int(not torch.equal(fresh_cpu[slot], used))
                hit_error += float(difference.square().sum())
                hit_max = max(hit_max, float(difference.abs().max()))
            selected = deepcopy(entry['record'])
            selected.update(replay_source_step=entry['step'],
                            replay_source_slot=entry['slot'], cache_hit=hit,
                            first_step=entry['step'], first_slot=entry['slot'],
                            used_namespace=entry['record']['namespace'],
                            replay_current_step=int(step), replay_current_slot=slot)
            values.append(used)
            used_records.append(selected)
            source_steps.append(entry['step'])
            source_slots.append(entry['slot'])
            flags.append(hit)
        output = torch.stack(values).to(device=fresh.device).detach()
        hits = sum(flags)
        misses = PACKETS - hits
        self.cache = cache
        if changed_epoch:
            self.reset_count += int(self.epoch != 0)
            self.epochs_started += 1
        self.epoch, self.steps = int(epoch), int(step)
        self.generated_packets += PACKETS
        self.used_packets += PACKETS
        self.miss_packets += misses
        self.hit_packets += hits
        self.hit_changed_packets += hit_changed
        self.hit_squared_error += hit_error
        self.hit_elements += hits * 512
        self.hit_max_abs = max(self.hit_max_abs, hit_max)
        summary = dict(epoch=int(epoch), step=int(step),
                       epoch_cache_reset=changed_epoch,
                       fresh_generated_packets=PACKETS, used_packets=PACKETS,
                       miss_packets=misses, hit_packets=hits,
                       reset_count=self.reset_count,
                       current_cache_entries=len(cache),
                       cumulative_generated=self.generated_packets,
                       cumulative_used=self.used_packets,
                       source_steps=source_steps, source_slots=source_slots,
                       cacheflags=flags,
                       used_namespaces=[v['namespace'] for v in used_records],
                       used_scenes=[v['scene'] for v in used_records],
                       used_states=[v['state_end'] for v in used_records],
                       used_locks=[v['receiver_lock_status'] for v in used_records],
                       fresh_scenes=[v['scene'] for v in records],
                       hit_changed_packets=hit_changed,
                       hit_fresh_used_MSE=hit_error / (hits * 512) if hits else 0.,
                       hit_fresh_used_max_abs=hit_max)
        return output, used_records, summary
