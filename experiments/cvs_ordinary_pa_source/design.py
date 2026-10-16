import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-ordinary-PA-heldsource-manysig-s2026092701-m5-r01'
RELEASE='cvs_ordinary_PA_heldsource_20261016_r01'
OWNER='codex/root/ordinary-PA-heldsource-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
FIRST='20261016-phase1-ordinary-augmentation-RX1-e20-s2026092701-m2-r01'
REST='20261016-phase1-ordinary-augmentation-sourceRX-confirm-e20-s2026092701-m8-r01'
RX=(1,3,4,6,8);SEED=2026092701;DRAW_SEED=20261016171

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def rows():return [dict(row_id=f'PA-RX{r}-s{SEED}',held_RX=r) for r in RX]

def origin(rx,arm='N0'):
    if rx not in RX or arm not in ('N0','N1'):raise ValueError('Fixed source origin only')
    return Path(PROJECT)/'runs'/(FIRST if rx==1 else REST)/f'{arm}-RX{rx}-s{SEED}'/'source'

def config(row):
    if row not in rows():raise ValueError('Only five fixed source folds')
    return dict(schema='ordinary_PA_held_source_v1',run_id=RUN,launch_owner=OWNER,**row,
        output_root=(BASE/row['row_id']).as_posix(),checkpoint=(origin(row['held_RX'])/'final_ssdg.pth').as_posix(),
        checkpoint_origin=origin(row['held_RX']).as_posix(),recipe_origin=origin(row['held_RX'],'N1').as_posix(),
        model_seed=SEED,source_contract=SOURCE,preparation=PREPARATION,
        role='L_s',source_only=True,target_access=False,V_used=False,U_used=False,
        held_L_packets=1260,TX=list(range(6)),days=[1,2,3],batch_size=128,draws=16,draw_seed=DRAW_SEED,
        draw_policy='Private stream per heldRX/draw/batch; FULL and NO_PA share exact stream; no TX seed/session',
        conditions=['RAW','FULL','NO_PA'],ordinary_epoch=13,
        physics='Unchanged nativeRMS IQ256/25MHz; original ordinary E13 only; no clipping/PA amplification/satellite',
        paired_control='Final PA flag only; exact upstream IQ and non-PA output equality required',
        readout='Frozen final20 defaultcos joint160 SIG_TIME RAW_HZ/NoExplicitCFO',
        new_models=0,optimizer_updates=0,head_fit_updates=0,automatic_training=False,
        expected_unique_forward_packets=41580,expected_checkpoint_smoke_packets=2,
        primary='NO_PA minus FULL retention on fixed RAW-correct source L',
        criterion=dict(mean_retention_gain_pp_min=.5,positive_RX_min=4,worst_retention_gain_pp_min=-.5),
        interpretation='Conditional frozen-head input effect only, not training causation or Ag efficacy')

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed PA source diagnostic differs')

def qualify(records):
    if [v['held_RX'] for v in records]!=list(RX):raise ValueError('All five source folds required')
    values=[v['retention_gain_pp'] for v in records]
    checks=dict(mean_retention=sum(values)/5>=.5-1e-9,positive_RX=sum(v>0 for v in values)>=4,
        worst_RX=min(values)>=-.5-1e-9)
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,folds=records,
        mean_retention_gain_pp=sum(values)/5,source_only=True,target_access=False,
        proves_training_gain=False,automatic_training=False,goal_completed=False)
