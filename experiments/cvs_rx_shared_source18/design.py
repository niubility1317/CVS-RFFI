"""One source-only question; no IQ operator or candidate promotion."""
import json
from pathlib import Path
from experiments.cvs_iq_circularity16.design import PROJECT, SOURCE, PREPARATION

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261018-diagnostic-F-sharedRX-sourceLtoV-manysig-m1-r01'
RELEASE = 'cvs_F_sharedRX_sourceLtoV_20261018_r01'
OWNER = 'codex/root/F-sharedRX-source-20261018'
BASE = PROJECT + '/runs/' + RUN
LOCAL = ROOT / 'local_artifacts/F_sharedRX_source_LtoV_20261018'
RX = (1, 3, 4, 6, 8)
ROW = 'SHARED_RX_L_TO_V'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def config():
    return dict(
        run_id=RUN, launch_owner=OWNER, source_contract=SOURCE,
        preparation=PREPARATION, output_root=BASE, RX=list(RX), days=[1, 2, 3],
        TX=list(range(6)), source_RX_holdout=None, L=6300, V=27000,
        L_per_cell=70, V_per_cell=300, U_used=False, U_truth_read=False,
        IQ_points=256, window=[80, 160], period=20, epsilon=1e-6,
        observable='Exactly existing CFO-compensated centered circularity magnitude; no changed coordinate or window',
        fit='L only; each RX/TX/day cell mean centered over all5RX;18leaveTXanddayout mean predictors',
        evaluation='All original V in90cells,each physical sample in one heldTX/day evaluation;no V subdivision or persistent update',
        baseline='Zero RX contrast;V contrast centers overRX within each TX/day for analysis only',
        criteria=dict(min_relative_risk_gain=.20, min_positive_folds=15,
                      max_each_RX_SSE_increase=0., numerical_SSE_tolerance=1e-20),
        wrong_RX='All44 fiveRX derangements descriptive only;no uniqueness condition',
        on_fail='Stop deriving augmentation from this fixed all5RX L-to-V scalar source basis;no window/coordinate/RX/threshold rescue',
        on_pass='Independent physical-source association only;no hardware identification,label preservation,IQ operator,training or target promotion',
        historical_result='Original4RX circularity FAIL preserved;this source confirmation implements no closed IQ candidate',
        protocol='Existing source roles/handles reused;physical L,V,U-ID exclusion only;no new data validation or hash',
        network_forwards=0, checkpoint_loads=0, optimizer_steps=0, teacher_forwards=0,
        target_access=False, automatic_training=False, CPU_threads=2, GPU='none',
        future_training_budget='Current authorized E30,full5RX,F+CROSS_RX,unchanged defaultcos andU;no model trained in this row',
    )


def validate(value):
    if value != config():
        raise ValueError('Fixed full-source confirmation differs')
