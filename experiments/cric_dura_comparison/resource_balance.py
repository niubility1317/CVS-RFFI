"""Admission-only resource policy. Never changes or terminates a science worker."""
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


class LoadPolicy:
    def __init__(self, config):
        if set(config.get('boost_gpus', [])) != {0, 6} or config.get('max_gpu_workers') != 26:
            raise ValueError('Only approved GPU0/6 pilot with 26 total workers is supported')
        self.host = {}
        self.devices = {}
        self.history = {}
        self.disabled = {}
        self.trials = {}
        self.cpu_previous = None
        self.now = 0
        self.last_error = None

    def observe(self, host, devices, now):
        self.host = dict(host)
        self.devices = devices
        self.now = now
        for gpu, data in devices.items():
            history = self.history.setdefault(gpu, [])
            history.append((now, data['util']))
            self.history[gpu] = [(t, u) for t, u in history if now-t <= 180]

    def host_ok(self):
        h = self.host
        required = ('cpu_busy_pct', 'available_ram_gib', 'iowait_pct')
        if any(not isinstance(h.get(k), (int, float)) or not math.isfinite(h[k]) for k in required):
            return False
        return 0 <= h['cpu_busy_pct'] < 75 and h['available_ram_gib'] >= 64 and 0 <= h['iowait_pct'] < 5

    def mean_util(self, gpu):
        values = self.history.get(gpu, [])
        return statistics.mean(v for _, v in values) if values else 100

    def order(self, gpus):
        return sorted(gpus, key=lambda g: (self.mean_util(g), g))

    def admits(self, gpu, job, active, jobs, apps, free, reserve):
        from .multiprocess_control import gpu_admits
        if not self.host_ok():
            return False
        own = [jobs[k] for k, v in active.items() if v.get('gpu') == gpu]
        owned_counts = {}
        for item in active.values():
            g = item.get('gpu')
            if g is not None:
                owned_counts[g] = owned_counts.get(g, 0)+1
        # Count non-owned apps too; reserve slots for workers not yet in CUDA.
        cuda_pids = {pid for d in self.devices.values() for pid in d.get('cuda_pids', [])}
        owned_pids = {item['pid'] for item in active.values() if item.get('gpu') is not None}
        total = len(cuda_pids | owned_pids)
        # capacity() is fresher than the rolling sample. Count any newly seen
        # apps conservatively; a reserved owned PID may temporarily count twice.
        total += max(0, apps-len(self.devices.get(gpu, {}).get('cuda_pids', [])))
        if total >= 26:
            return False
        if job.get('exclusive'):
            return gpu_admits(gpu, job, active, jobs, apps, free, reserve)
        # Reserve startup / later-stage peaks, not just the current allocation.
        budget = {'F': 4096, 'G': 10240, 'G_student': 10240, 'predict': 2048}
        if job['kind'] not in budget or any(j['kind'] not in budget for j in own):
            return False
        total_mem = self.devices.get(gpu, {}).get('total_mib', 0)
        if sum(budget[j['kind']] for j in own)+budget[job['kind']]+3072 > total_mem:
            return False
        if len(own) < 3:
            return gpu_admits(gpu, job, active, jobs, apps, free, reserve)
        history = self.history.get(gpu, [])
        warm = len(history) >= 5 and history[-1][0]-history[0][0] >= 120
        if (gpu not in (0, 6) or gpu in self.disabled or not warm or
                self.mean_util(gpu) >= 85 or len(own) != 3 or apps != 3 or
                job['kind'] != 'F' or any(j['kind'] != 'F' or j.get('exclusive') for j in own)):
            return False
        return gpu_admits(gpu, job, active, jobs, apps, free, reserve, 4, 4)

    def evaluate_slowdown(self, gpu, ratios):
        if len(ratios) >= 3 and statistics.median(ratios) >= 4/3:
            self.disabled[gpu] = 'same-phase slowdown >=4/3; no estimated fourth-slot throughput gain'

    @staticmethod
    def training_records(job):
        config = json.loads(Path(job['config']).read_text())
        records = []
        for segment in ('segment20', 'segment30', 'segment50'):
            path = Path(config['output_root'])/segment/'metrics_epoch.jsonl'
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    duration = float(row['epoch_time_s'])
                    if math.isfinite(duration) and duration > 0:
                        records.append((row['phase'], duration))
                except (ValueError, KeyError):
                    continue  # Ignore only an incomplete live record.
        return records

    def start_trial(self, gpu, active, jobs):
        cohort = {}
        for key, item in active.items():
            if item.get('gpu') != gpu:
                continue
            records = self.training_records(jobs[key])
            if not records:
                continue
            phase = records[-1][0]
            previous = [d for p, d in records if p == phase][-5:]
            if len(previous) >= 3:
                cohort[key] = dict(count=len(records), phase=phase, seconds=statistics.median(previous))
        self.trials[gpu] = dict(started_at=self.now, cohort=cohort, assessed=False)

    def check_trials(self, jobs):
        for gpu, trial in self.trials.items():
            if trial['assessed']:
                continue
            ratios = []
            for key, old in trial['cohort'].items():
                new = self.training_records(jobs[key])[old['count']:]
                duration = [d for p, d in new if p == old['phase']]
                if len(duration) >= 3:
                    ratios.append(statistics.median(duration[-5:])/old['seconds'])
            if len(ratios) >= 3:
                self.evaluate_slowdown(gpu, ratios)
                trial.update(assessed=True, same_phase_ratios=ratios,
                             throughput_proxy=(4/3)/statistics.median(ratios))

    def poll(self, jobs):
        try:
            cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
            host = {}
            if self.cpu_previous:
                delta = [b-a for a, b in zip(self.cpu_previous, cpu)]
                total = sum(delta)
                if total > 0:
                    host.update(cpu_busy_pct=(total-delta[3]-delta[4])/total*100,
                                iowait_pct=delta[4]/total*100)
            self.cpu_previous = cpu
            mem = {l.split(':')[0]: int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
            host['available_ram_gib'] = mem['MemAvailable']/1048576
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.total',
                                               '--format=csv,noheader,nounits'], text=True, timeout=10)
            devices = {}
            uuids = {}
            for line in output.splitlines():
                index, uuid, util, total = [v.strip() for v in line.split(',')]
                gpu = int(index)
                devices[gpu] = dict(util=float(util), total_mib=int(total), apps=0, cuda_pids=[])
                uuids[uuid] = gpu
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                             '--format=csv,noheader,nounits'], text=True, timeout=10)
            for line in apps.splitlines():
                pid, uuid = [v.strip() for v in line.split(',')]
                if uuid in uuids:
                    devices[uuids[uuid]]['apps'] += 1
                    devices[uuids[uuid]]['cuda_pids'].append(int(pid))
            self.observe(host, devices, time.time())
            self.last_error = None
            self.check_trials(jobs)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            self.host = {}  # Fail closed for new admissions; healthy workers continue.
            self.last_error = str(error)

    def snapshot(self):
        return dict(host=self.host, device_mean_util={g: self.mean_util(g) for g in self.history},
                    disabled_fourth_slots=self.disabled, trials=self.trials, telemetry_error=self.last_error,
                    policy='GPU0/6 <=4 F-only; others <=3; total<=26; CPU<75; RAM>=64GiB; IOwait<5; util<85; warmup120s')
