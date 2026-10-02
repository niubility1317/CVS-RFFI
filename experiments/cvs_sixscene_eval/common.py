import json
from pathlib import Path
RUN='20261002-phase1-cvs-residual-sixscene-manysig-m8-r01'
RELEASE='cvs_residual_sixscene_20261002_r01'
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
VIEWS=('clean',*SCENES)
SEEDS=(2026092701,2026092702,2026092703,2026092704)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
