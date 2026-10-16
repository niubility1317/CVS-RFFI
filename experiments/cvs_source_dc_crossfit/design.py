import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-source-DC-crossfit-manysig-m1-r01'
RELEASE='cvs_source_DC_crossfit_20261016_r01'
OWNER='codex/root/source-DC-crossfit-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE=PROJECT+'/runs/20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01/C0-RX1-s2026092701/source'
FIT=(3,4,6,8);TX=tuple(range(6));DAYS=(1,2,3)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows():return [dict(row_id='DC-fitRX3_4_6_8',arm='DC')]
def config(row):
    if row not in rows():raise ValueError('One fixed source statistic row only')
    return dict(schema='source_DC_cross_TX_day_v1',run_id=RUN,launch_owner=OWNER,**row,
        output_root=(BASE/row['row_id']).as_posix(),source_contract=SOURCE,source_preparation=PREPARATION,reference_metadata=REFERENCE,
        source_RX=list(FIT),source_days=list(DAYS),source_TX=list(TX),held_RX1_used=False,L_packets=5040,packets_per_cell=70,
        descriptor='mean_n(I,Q)/sqrt(mean_n(I^2+Q^2)); full256, original source L only; descriptor units are original native packet-RMS',
        cell_effect='TX/RX/day mean descriptor minus sameTX/day mean across4 fitRX; no TX hardware inversion',
        crossfit='18 folds: leave one TX AND one day out; predict held cell RX contrast from other5TX x other2days',
        null='zero RX contrast; squared Euclidean prediction risk',
        primary=dict(aggregate_risk_reduction_min=.1,positive_folds_min=12,fold_count=18,
            envelope='p95 of componentwise-max pair differences of fitted RX mean contrasts must exceed actual old ordinary E13 dc_offset_max'),
        on_fail='Close fixed RX-DC synthesis direction, no window/magnitude/coordinate/TX/day/RX/threshold rescue',
        on_pass='Only supports separately preregistered bounded IQ training; no Ag efficacy or target/As promotion',
        fixed_identity_network='F+CROSS_RX SupCon/SIG_TIME_X3/CENTER_25K/CENTER_UNITS/joint160/defaultcosine unchanged; no model used here',
        device='CPU',CPU_threads=2,CUDA_VISIBLE_DEVICES='',new_models=0,model_seed=None,model_forwards=0,checkpoint_loads=0,
        optimizer_updates=0,head_fit_updates=0,U_used=False,V_used=False,target_access=False,source_only=True,
        channel_generation=False,formal_test_changed=False,automatic_training=False,goal_completed=False)
def validate(c):
    if c!=config(rows()[0]):raise ValueError('Fixed DC source statistic config differs')
