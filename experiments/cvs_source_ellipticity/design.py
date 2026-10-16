import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-source-ellipticity-crossfit-manysig-m1-r01'
RELEASE='cvs_source_ellipticity_crossfit_20261016_r01'
OWNER='codex/root/source-ellipticity-crossfit-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE=PROJECT+'/runs/20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01/C0-RX1-s2026092701/source'
FIT=(3,4,6,8);TX=tuple(range(6));DAYS=(1,2,3)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows():return [dict(row_id='ELLIP-fitRX3_4_6_8',arm='ELLIP')]
def config(row):
    if row not in rows():raise ValueError('Only registered ellipticity row')
    return dict(schema='source_ellipticity_cross_TX_day_v1',run_id=RUN,launch_owner=OWNER,**row,
        output_root=(BASE/row['row_id']).as_posix(),source_contract=SOURCE,source_preparation=PREPARATION,reference_metadata=REFERENCE,
        source_RX=list(FIT),source_days=list(DAYS),source_TX=list(TX),held_RX1_used=False,L_packets=5040,packets_per_cell=70,
        descriptor='abs(mean((x-mean(x))**2))/mean(abs(x-mean(x))**2); one full256 AC ellipticity magnitude, not complex orientation',
        cell_effect='TX/RX/day mean descriptor minus sameTX/day mean across4 fitRX; no TX/RX hardware inversion',
        crossfit='18fold simultaneously leave one TX AND one day; estimate from other5TX x other2days',
        primary=dict(nonzero_zero_effect_risk=True,aggregate_risk_reduction_min=.1,positive_folds_min=12,fold_count=18,RX_pair_ellipticity_p95_min=.01),
        rationale='Scalar phase/gain/DC changes leave this descriptor invariant. Other old compound operators may change it; no claim their output support is disjoint.',
        on_fail='Close this fixed full256 RX ellipticity-guided synthesis direction; no complex-orientation/window/RX/TX/day/threshold rescue',
        on_pass='Only supports separately registered bounded widely-linear received-IQ training; does not establish causal RX hardware, identity preservation or Ag efficacy',
        fixed_identity_network='F+CROSS_RX/SIG_TIME_X3/joint160/defaultcosine unchanged; no model used',
        device='CPU',CPU_threads=2,CUDA_VISIBLE_DEVICES='',new_models=0,model_forwards=0,checkpoint_loads=0,
        optimizer_updates=0,head_fit_updates=0,U_used=False,V_used=False,target_access=False,source_only=True,
        channel_generation=False,formal_test_changed=False,automatic_training=False,goal_completed=False)
def validate(c):
    if c!=config(rows()[0]):raise ValueError('Fixed ellipticity statistic config differs')
