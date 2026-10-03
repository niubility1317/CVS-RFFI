"""Model-free fixed evaluation matrix and query permission checks."""
import json
from pathlib import Path
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
SOURCE_RUN='20261003-phase1-cvs-reference-residual-identity-manysig-m8-r01'
SOURCE_RELEASE='cvs_reference_residual_identity_20261003_r01'
SOURCE_COMMIT='8a3871650c2a35ab5c8a63d9e6d0634946292542'
RUN='20261003-phase1-cvs-reference-residual-clean-manysig-m8-r01'
RELEASE='cvs_reference_residual_clean_20261003_r01'
VARIANTS=('reference_residual_scalar','reference_residual_inverse')
SEEDS=(2026092701,2026092702,2026092703,2026092704)
SOURCE_ROOT=PROJECT+'/runs/'+SOURCE_RUN
RUNTIME=PROJECT+'/runs/'+RUN
LOGS=PROJECT+'/logs/'+RUN
CAPSULE=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule'
TRUTH=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
SOURCE_CONTRACT=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
CLASSES=['14-10','14-7','20-15','20-19','6-15','8-20']
QUERY_COUNT=168000
OWNER='codex/root/reference-residual-clean-20261003'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def expected_rows():return {f'{v}-s{s}':(v,s) for v in VARIANTS for s in SEEDS}
def prediction_config(rid):
    v,s=expected_rows()[rid]
    return dict(method='cvs_reference_residual_clean',variant=v,model_seed=s,row_id=rid,source_output=SOURCE_ROOT+'/'+rid+'/source',
        source_release_commit=SOURCE_COMMIT,source_contract=SOURCE_CONTRACT,freeze_file=SOURCE_ROOT+'/frozen_source_matrix.json',
        p1_capsule=CAPSULE,output_root=RUNTIME+'/'+rid,views=['clean'],device='cuda:0',batch_size=256,
        authorization='preregistered fixed scalar/inverse comparison; all8E200 checkpoints; no selection or target feedback')
def validate_config(c):
    rid=c.get('row_id')
    if rid not in expected_rows() or c!=prediction_config(rid):raise ValueError('Fixed clean prediction config differs')
    return c
def make_spec():
    return dict(run_id=RUN,runtime_root=RUNTIME,log_root=LOGS,launch_owner=OWNER,source_commit=SOURCE_COMMIT,
        source_root=SOURCE_ROOT,freeze_file=SOURCE_ROOT+'/frozen_source_matrix.json',p1_truth=TRUTH,
        rows=[dict(row_id=rid,variant=v,model_seed=s,source_output=SOURCE_ROOT+'/'+rid+'/source',output_root=RUNTIME+'/'+rid,
            config=PROJECT+'/releases/'+RELEASE+'/experiments/cvs_reference_residual_clean/configs/'+rid+'.json') for rid,(v,s) in sorted(expected_rows().items())])
def validate_spec(spec):
    if spec!=make_spec():raise ValueError('Fixed8-row clean matrix differs')
    for row in spec['rows']:validate_config(read(row['config']))
    return spec
