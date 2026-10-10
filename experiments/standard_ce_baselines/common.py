import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261010-phase1-standard-ce-baselines-manysig-m56-r01'
RELEASE='standard_ce_baselines_20261010_r01'
OWNER='codex/root/standard-ce-baselines-20261010'
SEEDS=(2026092701,2026092702,2026092703,2026092704)
VIEWS=('clean',)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
