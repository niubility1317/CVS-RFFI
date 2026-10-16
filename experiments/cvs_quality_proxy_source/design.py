"""Fixed source-only finite-waveform receiver-quality proxy diagnostic."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261017-diagnostic-FcrossRX-qualityproxy-fitL-manysig-m1-r01';RELEASE='cvs_FcrossRX_qualityproxy_fitL_20261017_r01';OWNER='codex/root/FcrossRX-qualityproxy-20261017';BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE=PROJECT+'/runs/20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01/C0-RX1-s2026092701/source'
ENGINE=PROJECT+'/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code'
FIT=(3,4,6,8);TX=tuple(range(6));DAYS=(1,2,3);SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');PRIMARY=SCENES[-2:];SEED=20261017891;RECEIVER_SEED=20261016102
ROW='quality-proxy-fitRX3_4_6_8'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows():return [dict(row_id=ROW,arm='quality_proxy')]
def config(row):
 if row!=rows()[0]:raise ValueError('One fixed source-physics diagnostic')
 return dict(schema='source_quality_proxy_finite_waveform_v1',run_id=RUN,launch_owner=OWNER,**row,output_root=(BASE/ROW).as_posix(),source_contract=SOURCE,source_preparation=PREPARATION,reference_metadata=REFERENCE,engine=ENGINE,source_RX=list(FIT),source_TX=list(TX),source_days=list(DAYS),source_L=5040,held_RX1_used=False,U_used=False,V_used=False,target_access=False,source_only=True,model_seed=None,new_models=0,checkpoint_loads=0,model_forwards=0,optimizer_updates=0,teacher_forwards=0,head_fit_updates=0,CPU_threads=2,CUDA_VISIBLE_DEVICES='',epochs=0,automatic_training=False,goal_complete=False,
 baseline='F+CROSS_RX sameTX/SIG_TIME_X3/unifiedSinc/NoExplicitCFO/joint160/defaultcos30/E20 remains;no model instantiated here',
 physical=dict(fs_hz=25000000.,fc_hz=2462000000.,IQ_length=256,scenes=list(SCENES),draws=2,seed=SEED,receiver_seed=RECEIVER_SEED,namespace='source_fit_quality/drawD',processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect',session='rx/day/eq1 only,neverTX',formal_test_unchanged=True),
 measurements=dict(proxy='mean(abs(x)^2)*sum(abs(H)^2)/expected_added_noise_variance',whole='mean(abs(causal_Hx_reflect)^2)/sameexpected_added_noise_variance',interior='sameHx indices[len(H)-1:256],no artificial-prefix dependence;statistic only,no output trimming',meaning='Frozen-H signal-to-added-noise estimate,not totalSNR,not exact timevarying full-route quality or RX hardware identification',conditions=['native_proxy','explicit_proxy_replay','whole_quality_total_effect','interior_quality_total_effect','whole_quality_fixed_receiver_state_and_scale'],fixed_control='Only receiver_quality return(status,scale)held to native proxy;native propagation/noise/IQ/phase/CFO/AGC code and RNG draws unchanged;zero IQ effect is sanity control',physics_records=60480,native_process_calls=302400,launch_synthetic_process_calls=10),
 primary=dict(scenes=list(PRIMARY),each_draw_median_abs_bias_min_db=1.,each_draw_whole_lock_flip_min=.05,each_draw_whole_vs_interior_quality_median_abs_difference_max_db=.5,each_draw_boundary_robust_flip_fraction_min=.5,all_interior_sample_count_min=128),on_fail='Close fixed finite-waveform quality-proxy receiver-switch premise;no thresholds/draws/seed/scenes/window/RX rescue or training',on_pass='Only supports separately registered frozen-default-head recoverability probe;not Ag/As identification gain or automatic E20',formal_test_changed=False)
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed source/quality/no-model design differs')
