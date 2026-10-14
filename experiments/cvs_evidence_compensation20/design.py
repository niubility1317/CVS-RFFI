"""Fixed frozen opportunity census before any new training."""
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01'
RELEASE='cvs_SIGTIME_evidence_compensation_sourceVU_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
SOURCE_RX=(1,3,4,6,8)
SAMPLING_SEED=2026101501
GATE=dict(U_EVIDENCE_active_min=567,U_EVIDENCE_new_vs_SELF_min=100,
    V_EVIDENCE_active_min=100,V_EVIDENCE_active_precision_min=.98,
    V_each_RX_EVIDENCE_active_min=10,V_each_RX_EVIDENCE_active_precision_min=.95,
    interpretation='Frozen source opportunity/reliability proxy only; no hiddenU precision or training efficacy proof')


def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',gate_recipe_row='BASE_SSL-s2026092701',
        model_seed=2026092701,sampling_seed=SAMPLING_SEED,
        roles=['V_proxy','U_s'],counts={'V_proxy':27000,'U_s':56700},
        batch_size=256,V_sampler='fixed random permutation; no physical-pair ordering',
        U_sampler='original native shuffle loader; no replacement',
        teacher='deepcopy of same own BASE_CE final20 student; no EMA state',
        student_mode='eval',teacher_mode='eval',beta=.5,pi=.9,coefficient=.1,
        original_acceptance_gate=True,new_acceptance_gate=False,source_falsification=GATE,
        U_truth_read=False,V_gradient=False,target_access=False,
        analysis_code_commit='release_commit.txt')


def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed compensation census request or actual Git commit differs')
    return c
