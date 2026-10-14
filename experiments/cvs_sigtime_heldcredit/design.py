"""One bounded source falsification, fixed existing 25 kHz coordinates."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261015-phase1-SIGTIME-center-heldcredit-sourceL-manysig-s2026092701-r01'
RELEASE='cvs_SIGTIME_center_heldcredit_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/cvs_SIGTIME_credit50_20261014_r01'
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
ORDINALS=(0,17,35,52)
HELD_ORDINALS=(68,69)
SOURCE_RX=(1,3,4,6,8)
COORDINATES=('RAW_HZ','CENTER_25K')
MODES=('SAME_L','HELD_L')
GATE=dict(minimum_CENTER_held_active_cohorts=2,
          minimum_CENTER_changed_kernel_cohorts=2,
          credit_tolerance=1e-6,
          interpretation='Source availability only. At least one held accepted group in each of two fixed cohorts; at least two physical Sinc kernel updates. No target selection or training-efficacy claim.')

def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
                source_role='L_s',model_seed=2026092701,source_row='BASE-s2026092701',
                coordinate_modes=list(COORDINATES),evaluation_modes=list(MODES),
                anchor_ordinals=list(ORDINALS),held_ordinals=list(HELD_ORDINALS),
                scale_hz=25000.,source_falsification=GATE,
                analysis_code_commit='release_commit.txt')

def validate(c):
    expected=request(); expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c != expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed source request or actual Git commit differs')
    return c

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
