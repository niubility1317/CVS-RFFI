"""Full nominal practical(raw hidden U), with the original physical L sessions."""
from collections import Counter
from dataclasses import asdict
import math
import numpy as np
import torch
from . import design as d


def serial(v):
    if isinstance(v, np.generic): return serial(v.item())
    if isinstance(v, np.ndarray): return serial(v.tolist())
    if isinstance(v, dict): return {str(k): serial(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [serial(x) for x in v]
    if isinstance(v, complex): return [v.real, v.imag]
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


class UPractical:
    def __init__(self, role='train'):
        if role != 'train':
            raise ValueError('Only original hidden source U receives this training view')
        from experiments.cvs_phase1_overlay.model import native_modules
        native_modules()
        import leo_practical.channel as ch
        import leo_practical.residual as residual
        self.ch, self.residual = ch, residual
        self.calls, self.packets = 0, 0
        self.coverage, self.sessions = Counter(), {}
        self.configs = {s: ch.Config(fs_hz=25000000., fc_hz=2462000000., scenario=s,
                                    processing_route='residual', mode='post_sync',
                                    equalization_enabled=False, reference_snr_db=30., boundary='reflect')
                        for s in d.SCENES}

    def __deepcopy__(self, memo):
        result = type(self)()
        memo[id(self)] = result
        return result

    def audit(self):
        return dict(channel_module_file=self.ch.__file__, residual_module_file=self.residual.__file__,
                    configurations={s: asdict(c) for s, c in self.configs.items()},
                    all_original_RX=list(d.FIT), ordinary_before_satellite=False,
                    formal_test_changed=False, hardware_seed=d.HARDWARE_SEED,
                    channel_seed=d.U_SEED, channel_stream='P66/U_S', dropped=0)

    def apply(self, x, meta, epoch, step):
        if (epoch not in range(21, 31) or step not in range(4441, 6661)
                or epoch != (step - 1) // 222 + 1 or x.ndim != 3
                or x.shape[1:] != (2, 256) or not 0 < len(x) <= 128
                or not bool(torch.isfinite(x).all())):
            raise ValueError('Finite original U IQ/cumulative E21..30 and fixed <=128 extra slot only')
        keys = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
        if any(k not in meta or len(meta[k]) != len(x) for k in keys):
            raise ValueError('Physical source metadata required')
        v = {k: meta[k].detach().cpu().tolist() if hasattr(meta[k], 'detach') else list(meta[k]) for k in keys}
        if (set(map(int, v['rx_i'])) - set(d.FIT) or set(map(int, v['day_i'])) - {1, 2, 3}
                or set(map(int, v['eq_i'])) != {1}):
            raise ValueError('Original full source RX/day/equalized roles required')
        raw = np.asarray(x.detach().cpu().tolist(), dtype=np.float64)
        out, records = [], []
        namespace = f'FcrossRX/P66/train/e{epoch}/step{step}/U_S'
        for i in range(len(x)):
            pid = tuple(int(v[k][i]) for k in keys)
            rx, day = int(v['rx_i'][i]), int(v['day_i'][i])
            session = f'rx{rx}:day{day}:eq{int(v["eq_i"][i])}'
            rng = np.random.default_rng(self.ch.stable_seed(d.U_SEED, namespace, 'scene', pid))
            scene = str(rng.choice(d.TRAIN_SCENES))
            cfg = self.configs[scene]
            seed = self.ch.stable_seed(d.U_SEED, namespace, scene, pid)
            hardware = self.ch.receiver_for_session(cfg, d.HARDWARE_SEED, session)
            if self.sessions.setdefault(session, asdict(hardware)) != asdict(hardware):
                raise ValueError('Hardware changed within label-independent physical session')
            normalized, _ = self.ch.normalize_input(raw[i, 0] + 1j * raw[i, 1])
            signal, evidence = self.residual.ResidualChannel(cfg, seed, hardware).process(normalized)
            if not np.isfinite(signal).all() or evidence['channel_equalization_applied']:
                raise ValueError('Original full residual/no EQ/finite output required; no dropping')
            self.coverage[(rx, day, scene, evidence['state_end'], evidence['receiver_lock_status'])] += 1
            out.append(np.stack((signal.real, signal.imag)))
            records.append(serial(dict(evidence, scene=scene, sample_id=pid, session_id=session, namespace=namespace)))
        self.calls += 1
        self.packets += len(x)
        a = np.ascontiguousarray(out, dtype=np.float32)
        return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape).to(x.device), records

    def summary(self):
        return dict(calls=self.calls, packets=self.packets, session_count=len(self.sessions), sessions=self.sessions,
                    strata=[dict(RX=k[0], day=k[1], scene=k[2], state=k[3], lock=k[4], count=n)
                            for k, n in sorted(self.coverage.items())],
                    all_original_source_RX=list(d.FIT), dropped=0, confidence_filter=False,
                    ordinary_before_satellite=False, formal_test_changed=False)


def tensor_iq(a, device):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape).to(device)
