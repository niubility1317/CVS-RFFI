import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-source-repeatcorr-crossfit-manysig-m1-r01'
RELEASE='cvs_source_repeatcorr_crossfit_20261016_r01'
OWNER='codex/root/source-repeatcorr-crossfit-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE=PROJECT+'/runs/20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01/C0-RX1-s2026092701/source'
FIT=(3,4,6,8);TX=tuple(range(6));DAYS=(1,2,3)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows():return [dict(row_id='repeatcorr-fitRX3_4_6_8',arm='repeatcorr')]
def config(row):
    if row not in rows():raise ValueError('One fixed repeat-correlation source row')
    return dict(schema='source_repeat_difference_lag1_crossfit_v1',run_id=RUN,launch_owner=OWNER,**row,output_root=(BASE/row['row_id']).as_posix(),
        source_contract=SOURCE,source_preparation=PREPARATION,reference_metadata=REFERENCE,source_RX=list(FIT),source_TX=list(TX),source_days=list(DAYS),held_RX1_used=False,L_packets=5040,packets_per_cell=70,
        observation=dict(IQ_length=256,window=[80,160],period=20,cycle_count=4,phase='unit pooled adjacent-cycle complex correlation; no amplitude fit',
            residuals='(cycle1-q*cycle0)/sqrt2 and (cycle3-q*cycle2)/sqrt2; disjoint pairs',
            descriptor='Complex normalized lag1 correlation within each20-point difference; no joining across residual-pair boundaries or demeaning',
            validity='Residual/preamble power ratio>1e-12 and positive adjacent covariance denominator. Undefined rho stored0 with valid=false; any undefined packet fails; no dropping.',
            claim='Repeated-difference residual, not verified pure RX noise; TX transients, propagation and preexisting processing can remain'),
        cell_effect='Mean complex rho in each TX/RX/day minus mean over4 RX in same TX/day',
        crossfit='18fold simultaneously leaveTX AND day; predict from other5TX x other2days; null zero RX effect; 3 cyclic RX mismatches',
        primary=dict(all_valid_packets=True,coherence_p10_min=.8,each_RX_median_residual_ratio_max=.1,
            aggregate_risk_reduction_min=.1,positive_folds_min=14,each_RX_gain_min=0.,correct_over_best_wrong_min=.02,TX_half_profile_cosine_min=.8,
            median_RX_profile_absolute_rho_min=.05,RX_pair_profile_distance_p95_min=.05),
        on_fail='Close fixed repeat-difference lag1 coloration direction; no period/window/lag/statistic/TX/day/RX/threshold rescue',
        on_pass='Only supports separately registered bounded colored-versus-white IQ training on F+CROSS_RX/E20; not automatic training, pureRX/label-preservation or Ag efficacy proof',
        fixed_network='F+CROSS_RX/SIG_TIME_X3/CENTER_25K/CENTER_UNITS/joint160/defaultcos30/NoExplicitCFO unchanged; no model here',
        device='CPU',CPU_threads=2,CUDA_VISIBLE_DEVICES='',new_models=0,model_seed=None,model_forwards=0,checkpoint_loads=0,optimizer_updates=0,head_fit_updates=0,
        U_used=False,V_used=False,target_access=False,source_only=True,channel_generation=False,formal_test_changed=False,automatic_training=False,goal_completed=False)
def validate(c):
    if c!=config(rows()[0]):raise ValueError('Fixed repeat-difference source config differs')
