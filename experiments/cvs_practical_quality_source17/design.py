"""Source-only falsification of the practical white-input receiver-quality proxy."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-practical-quality-fullL-manysig-m1-r01';RELEASE='cvs_F_practical_quality_fullL_20261017_r01';OWNER='codex/root/F-practical-quality-source';LOCAL='local_artifacts/F_practical_quality_source_20261017'
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json';PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701';BASE='PROJECT/runs/'+RUN
FIT=(1,3,4,6,8);SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CHANNEL_SEED=20261017431;RECEIVER_SEED=20261016423

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(v,rt):
 if not v.startswith('PROJECT/'):raise ValueError('Existing registered project handle required')
 return Path(rt['project'])/v[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,output_root=BASE,L=6300,source_RX=list(FIT),source_days=[1,2,3],source_TX=list(range(6)),source_RX_holdout=None,physical_roles='L6300/U56700hiddenTX/V27000',scenes=list(SCENES),all_L_each_scene=True,observations=18900,channel_seed=CHANNEL_SEED,receiver_seed=RECEIVER_SEED,fs_hz=25000000.,fc_hz=2462000000.,reference_snr_db=30.,mode='post_sync',route='residual',EQ=False,boundary='reflect',noise_or_propagation_changed=False,IQ_augmentations_stacked=False,quality_comparison='white expected Pin*sum|h|^2 versus finite-window mean|H*x|^2;both divided by same injected reference noise floor;not true total SNR',same_raw_same_latent=True,receiver_scope='same existing RX/day/eq physical session acrossTX',normalization='same existing normalize_input before As;no output RMS',preregistered_threshold=dict(min_effective_quality_change_fraction=.05,min_each_RX_fraction=.03,effective_change='lock class changes OR max(scale_old/scale_new,scale_new/scale_old)>=1.10'),on_fail='Close this fixed quality-coupling candidate;no window/SNR/threshold/seed/RX rescue',on_pass='Only physical mismatch evidence;matched E30 identity accuracy must still prove benefit;not Ag/As/joint success',CPU_threads=2,GPU='none',network_forwards=0,checkpoint_loads=0,source_cache_loads=1,optimizer_updates=0,teacher_forwards=0,head_fit_updates=0,U_truth_read=False,U_statistically_used=False,V_statistically_used=False,target_access=False,automatic_training=False)
def validate(c):
 if c!=config():raise ValueError('Source handles/matrix/physics/budget changed')
 return c
