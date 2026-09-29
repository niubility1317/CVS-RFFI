"""Synthetic numerical/runtime check; no project samples or target metrics."""
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path('E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt')
sys.path.insert(0, str(ROOT/'code'))
from cvsrffi.d92_branch_local_margin import fit_branch_local_margin

DEST = Path('E:/type10-7/.codex_tmp/local_margin_synthetic_v2_20260929.json')
assert not DEST.exists(), 'Measurement exists; preserve and review'
rows = []
with threadpool_limits(limits=2):
    for k in (1, 14, 20):
        c = 26
        rng = np.random.default_rng(20260929 + k)
        classes = [f'class-{i:02d}' for i in range(c)]
        labels = np.repeat(np.arange(c), k)
        arrays = {}
        for name, dim in [('z_id', 160), ('fft', 96), ('t_emb', 160), ('f_emb', 160), ('pa_local', 160)]:
            centers = rng.normal(size=(c, dim))
            arrays[name] = np.repeat(centers, k, axis=0) + rng.normal(size=(c*k, dim))
        logs = []
        tracemalloc.start()
        begin = time.perf_counter()
        try:
            state = fit_branch_local_margin(**arrays, support_labels=labels,
                support_ids=[f'physical-{i:04d}' for i in range(c*k)], classes=classes,
                log_callback=logs.append)
            audit = state.audit_dict()
            scalar = {key: value for key, value in audit.items()
                      if value is None or isinstance(value, (str, bool, int, float))}
            scalar['final_fit'] = {key: value for key, value in audit['final_fit'].items()
                if value is None or isinstance(value, (str, bool, int, float))}
            rows.append(dict(shape=[c*k, c], status='VERIFIED', elapsed_seconds=time.perf_counter()-begin,
                scalar_audit=scalar, callback_count=len(logs), last_sweep=logs[-1] if logs else None,
                traced_peak_allocation_bytes=tracemalloc.get_traced_memory()[1]))
        except Exception as error:
            rows.append(dict(shape=[c*k, c], status='FAILED', elapsed_seconds=time.perf_counter()-begin,
                error_type=type(error).__name__, error=str(error), callback_count=len(logs),
                failure_context=error.audit_dict() if hasattr(error, 'audit_dict') else None))
        tracemalloc.stop()
        print(json.dumps(rows[-1], allow_nan=False), flush=True)
    result = dict(scope='SYNTHETIC_ONLY_NO_PROJECT_DATA', python=sys.executable,
        conda_prefix=os.environ.get('CONDA_PREFIX'), numpy=np.__version__, threads=threadpool_info(), rows=rows)
with DEST.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(str(DEST), flush=True)
if not all(row['status'] == 'VERIFIED' for row in rows):
    raise SystemExit(1)
