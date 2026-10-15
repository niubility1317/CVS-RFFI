import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('E:/type10-7')
RUN='20261016-phase1-current-response-coverage-manysig-m1-r01'
OWNER='codex/root/source-response-coverage-20261016'
BASE=PROJECT/'local_artifacts/cvs_response_coverage_20261016/run'
EVIDENCE=PROJECT/'automation_reports/CV-SincNet/20261011-phase1-rx-spectral-crossfit-manysig-m1-r01/evidence/remote_results.json'
TRAIN_REGISTRATION=ROOT/'experiments/cvs_mixed_iq20/train_registration.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def config():
    return dict(run_id=RUN,owner=OWNER,output=str(BASE),source_evidence=str(EVIDENCE),
        source_role='L_s-derived fixed aggregate only',source_count=6300,
        primary='rectangular/DCT1to8',folds=18,RX=5,coordinates=8,
        response='H(w)=1+a*cos(w), abs(a)<=0.06',radius=.06,FFT_bins=256,pooled_bins=32,
        primary_grid=dict(radii=17,angles=2048),resolution_control=dict(radii=9,angles=1024),
        direction_gap_upper=.5,substantial_direction_lower=.8,resolution_tolerance=.01,
        scope='Ideal power-response proxy, not finite-IQ or identity-performance evidence',
        no_new_IQ_reads=True,no_checkpoint=True,U_used=False,V_used=False,target_access=False,
        model_forwards=0,optimizer_updates=0,head_fit_updates=0,device='CPU',threads=2,
        automatic_training=False,training_parameters_selected=False)


def validate(c):
    if c!=config():raise ValueError('Fixed source-capacity analysis configuration differs')
    return c
