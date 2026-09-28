"""Copy this branch's explicitly owned D92 artifacts to the workspace front door."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WS=Path('E:/type10-7')
paths=['docs/D92_SUPPORT_UPGRADE_20260928.md','code/cvsrffi/stage2_d92_support_cv.py',
    'tools/predict_d92_support_cv.py','tools/build_d92_confirmation_data.py','tools/evaluate_d92_registration_proxy.py',
    'tools/run_d92_confirmation.py','tools/publish_d92_confirmation.py','tools/score_d92_confirmation.py',
    'tools/summarize_d92_confirmation.py','tools/read_d92_run.py','tools/sync_d92_run_record.py',
    'configs/d92_registration_proxy_20260928.json','configs/d92_confirmation_20260928.json',
    'configs/d92_confirmation_data_20260928.json','configs/d92_confirmation_data_record_20260928.json',
    'configs/d92_scv_frozen_20260928.json','configs/d92_confirmation_recovery_20260928.json',
    'tests/test_d92_confirmation_score.py','tests/test_run_d92_confirmation.py','tests/test_summarize_d92_confirmation.py']
for name in paths:
    src=ROOT/name;dst=WS/name;dst.parent.mkdir(parents=True,exist_ok=True)
    # Each name is unique to this task; original D92 and shared wrappers are not mirrored.
    shutil.copyfile(src,dst)
    if src.read_bytes()!=dst.read_bytes():raise ValueError('Mirror mismatch: '+name)
    text=dst.read_text(encoding='utf-8')
    if '\ufffd' in text:raise ValueError('Encoding corruption: '+name)
print('VERIFIED explicit owned mirrors:',len(paths))
