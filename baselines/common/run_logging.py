"""Machine-readable startup information mirrored into human training logs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys


def startup_record(config, output):
    import torch
    record = dict(time=datetime.now(timezone.utc).isoformat(), config=config,
        argv=sys.argv, cwd=os.getcwd(), pid=os.getpid(), python=sys.executable,
        python_version=platform.python_version(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        release_commit=os.environ.get('COMPARISON_RELEASE_COMMIT', 'not_provided'),
        checkpoint_selection='last.pt / fixed_final_epoch', target_feedback=False)
    Path(output, 'startup.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[STARTUP] '+json.dumps(record,ensure_ascii=False),flush=True)
    return record
