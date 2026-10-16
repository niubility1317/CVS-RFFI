"""One frozen source phase-curvature association question;0 models or IQ augmentation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-crossRX-phasecurvature-fitL-manysig-m1-r01';RELEASE='cvs_F_phase_curvature_fitL_20261017_r01';OWNER='codex/root/F-phase-curvature-source-basis-20261017';LOCAL='local_artifacts/F_phase_curvature_source_basis_20261017'
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json';PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
BASE='PROJECT/runs/'+RUN;FIT=(3,4,6,8);WINDOW=(80,160);EPS=1e-12;DIM=156

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias,runtime):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered projectrelative handle required')
 return Path(runtime['project'])/alias[len('PROJECT/'):]

def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root=BASE,source_contract=SOURCE,preparation=PREPARATION,fit_L=5040,fit_RX=list(FIT),days=[1,2,3],TX=list(range(6)),samples_per_cell=70,window=list(WINDOW),triplet_lag=1,triplet_count_per_packet=78,coordinates=DIM,numerical_epsilon=EPS,arithmetic='Complex128/Float64',observation='c[n]=z[n+1]^2*conj(z[n])*conj(z[n+2]);q=c/(abs(c)+1e-12);78 interleaved real/imag coordinates for n80..157;allpoints/no filtering or unwrapping',cell_summary='Mean of all70 observed vectors;population variance percoordinate;allraw packet observations retained',crossfit='18heldTX/day groups;train excludes BOTH heldTX and heldday;other5TX*2day*4RX cellmeans;center4RX within eachTX/day;meanRX displacement predictsheldgroup;no fittedscale/projection/classifier',criteria=dict(min_pooled_risk_gain=.10,min_each_RX_risk_gain=0.,min_positive_TXday_groups=15),risk_definition='Mean squared Euclidean error over18groups*4RX*156coordinates;zeroRX displacement baseline;perRX equally weighted;positive group strictlygain>0',on_fail='Closefixedphasebasis;no statistic/lag/window/TX/day/RX/threshold/seed rescue;no IQtransform or E20',on_pass='SourceRX-associated phase-shape evidence only;separate physically justified boundedIQ operator/identityprotection and matchedE20 still required;no automatictraining',source_only=True,model_constructed=False,checkpoint_loads=0,source_cache_loads=1,network_forwards=0,optimizer_steps=0,head_fit_updates=0,teacher_forwards=0,IQ_views_generated=0,IQ_modified=False,U_truth_read=False,U_statistically_used=False,V_statistically_used=False,held_RX1_statistically_used=False,target_access=False,automatic_training=False,device='CPU',threads=2,NoExplicitCFO_preserved=True,phase_or_CFO_not_identity_feature=True,baseline_boundary='FutureF+CROSS_RX/3SIG_TIME/unifiedSinc/joint160/defaultcos/E20/originalU unchanged;no model used here',limitations=['Measuredwaveformcurvature includesTX/channel/RX/filter/equalizer effects;not identifiableRX oscillatorphasenoise','Sourceassociation is not hardwareRXreplacement,IQ identitypreservation,heldRXrecognition,DG,As orjointproof','CacheL6300/unusedV/features deserialized;only originalfitL5040 statisticallyused,U opaqueIDs only','Not periodiclag20 residual or oldtrainable featurecurvature branch;no oldroute reactivation'])
def validate(c):
 if c!=config():raise ValueError('Fixed phasebasis roles/observation/threshold/budget differ')
 return c
