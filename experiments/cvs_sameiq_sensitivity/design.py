import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-diagnostic-F-CSIL-inputJacobian-s2026092701-m1-r01'
RELEASE='cvs_F_CSIL_inputJacobian_probe_20261019_r01'
LOCAL='local_artifacts/F_sameIQ_sensitivity_20261019';OWNER='codex/root/sameIQ-sensitivity'
REFERENCE_RELEASE='cvs_F_nontrue_CSIL_E20_20261019_r01';REFERENCE_COMMIT='1e7cc414f73f093dfbdccee0e4ba7cf90c7a5324'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE=Path(PROJECT)/'runs'/RUN;N=360;BATCH=90
VIEWS=('clean','d0_mid','d0_low','d1_mid','d1_low')
def config():return dict(schema='sameIQ_input_sensitivity_source_v1',run_id=RUN,owner=OWNER,model_seed=2026092701,student='pureF ORIGINAL/defaultcos30/scratchFINAL20 B1',student_run='20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01',student_row='B1-s2026092701',teacher='frozen CSIL native/defaulthead/scale5/scratchlast200',unique_source_V=N,selection='First4 original V panel entries per TX x sourceRX x day; no prediction selection',source_RX=[1,3,4,6,8],source_days=[1,2,3],classes=['14-10','14-7','20-15','20-19','6-15','8-20'],batch=BATCH,views=list(VIEWS),observations='existing V1800 full residual IQ at condition0; original two cached draws',student_mode='eval for source geometry; train only synthetic double-backward smoke',target='normalize d logit_y/d IQ; signed cosine; one true-label row, not full Jacobian',wrong_target='same teacher derivative row (y+1)%6; same correct-teacher mask',teacher_mask='teacher argmax==legal source V TX; denominator all90',finite_difference_relative_L2=.001,physical_parameter_metric='Sinc q gradient /25000; remaining native coordinates',expected_NN=10804,synthetic_CP_smoke_NN=4,source_base_forward_packets=3600,finite_difference_packets=7200,new_physical_channel_draws=0,optimizer_updates=0,head_fit_updates=0,teacher_or_student_state_update=False,U_truth_read=False,target_access=False,qualification=dict(teacher_minus_student_urban_min_pp=1.,urban_body_true_CE_cos_min=0.,urban_body_true_minus_wrong_cos_min=.01,clean_body_true_CE_cos_min=-.05,scope='fixed source mechanistic rationale only; never E50/performance evidence',on_fail='close fixed signed true-row alignment rationale; no norm/label/threshold rescue',automatic_training=False))
def validate(c):
 if c!=config():raise ValueError('Fixed source/sensitivity/provenance/zero-update matrix differs')
 return c
