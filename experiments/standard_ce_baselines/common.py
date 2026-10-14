import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261010-phase1-standard-ce-baselines-manysig-m56-r01'
RELEASE='standard_ce_baselines_20261010_r01'
OWNER='codex/root/standard-ce-baselines-20261010'
SEEDS=(2026092701,2026092702,2026092703,2026092704)
VIEWS=('clean',)
EXTENSION_RUN='20261012-phase1-standard-ce-seed-extension-manysig-m52-r01'
EXTENSION_RELEASE='standard_ce_seed_extension_20261012_r01'
EXTENSION_OWNER='codex/root/standard-ce-seed-extension-20261012'
EXTENSION_SEEDS=(2026092705,2026092706,2026092707,2026092708)
EXTENSION_CONFIG='experiments/standard_ce_baselines/configs/seed_extension_20261012'
SIG_RUN='20261014-phase1-sigtime-purece-e200-clean-manysig-m8-r01'
SIG_RELEASE='sigtime_purece_e200_20261014_r01'
SIG_OWNER='codex/root/sigtime-purece-eight-20261014'
SIG_SEEDS=tuple(range(2026092701,2026092709))
SIG_CONFIG='experiments/standard_ce_baselines/configs/sigtime_20261014'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
