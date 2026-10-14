from pathlib import Path
from experiments.cvs_source_feature_ce_gradient import design as base
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,PARAMETERS,SEEDS,ARMS,GROUPS,rows,prior,source_config,model_api=(getattr(base,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','PARAMETERS','SEEDS','ARMS','GROUPS','rows','prior','source_config','model_api'))
RUN='20261015-phase1-source-teacher-confidence-manysig-m6-r01';RELEASE='cvs_source_teacher_confidence_20261015_r01';OWNER='codex/root/source-teacher-confidence-20261015';BASE=Path(PROJECT)/'runs'/RUN
QUALIFICATION=dict(source_only=True,all_six_models_required=True,role='V',groups_must_be_nonempty=True,helpful_retention_strictly_greater_than_harmful_retention=True,no_automatic_training_release=True)
def config(row):
 c=base.config(row);c.update(schema='source_teacher_confidence_v1',run_id=RUN,roles=['L_s','V'],counts=dict(L_s=6300,V=27000),shuffle_seeds=dict(L_s=2026101501,V=2026101503),strong_dropout_seeds=dict(L_s=2026101502,V=2026101504),V_forward=True,V_gradient=False,rule='FP32 maxsoftmax(EMAweak) >= maxsoftmax(studentstrong), detached, ties retained',gap_threshold=0.,loss_denominator='full batch count, not selected count',coefficient_interpretation='Fixed .1 diagnostic derivative in both frozen arms; ZERO original training coefficient remains0',source_qualification=QUALIFICATION.copy(),output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed zero-gap source experiment differs')
