from pathlib import Path, PurePosixPath
import json

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261012-phase1-e20-peer-audit-manysig-m3-r01'
SEEDS = (2026092701, 2026092702, 2026092703)
SOURCE_RUN = '20261012-phase1-e20-error-audit-manysig-m3-r01'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
SOURCE = PurePosixPath(PROJECT) / 'runs' / SOURCE_RUN
BASE = ROOT / 'local_artifacts' / RUN
REPORT = ROOT / 'automation_reports/CV-SincNet' / RUN
OWNER = 'codex/root/e20-peer-audit-20261012'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def config():
    return dict(run_id=RUN, model_seeds=list(SEEDS), roles=dict(L=6300, V=27000),
                input_root=SOURCE.as_posix(), input_run=SOURCE_RUN, launch_owner=OWNER,
                inputs={str(s): str(SOURCE/f'E20-s{s}'/'source_predictions.pt') for s in SEEDS},
                readouts='only L/V.parent original native E20; endpoints are not consumed',
                teachers=['LOO2 uniform probability mean of the other two seeds per student',
                          'ALL3 uniform probability mean, one teacher only'],
                confidence=0.95, ece_bins=15,
                consensus='other two teachers argmax agree and each confidence>=.95',
                training_updates=0, teacher_fitting=False, V_backward=False, target_access=False,
                U_access=False, identity_parameters_updated=0, GPU_processes=0,
                claims='known-source diagnosis only; not unseen RX, not benefit on U or target, not deployment oracle',
                output=str(BASE), selection='all fixed three seeds; no weighting, threshold fit, ranking or selection')
