"""Fixed before reading new source phase observations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
RUN = '20261016-diagnostic-source-phase-crossfit-manysig-m1-r01'
RELEASE = 'cvs_source_phase_crossfit_20261016_r01'
OWNER = 'codex/root/source-phase-crossfit-20261016'
BASE = PROJECT + '/runs/' + RUN
SOURCE = PROJECT + '/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION = PROJECT + '/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
WINDOWS = ((0, 80), (80, 160), (160, 240))
RX = (1, 3, 4, 6, 8)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def config():
    return dict(run_id=RUN, launch_owner=OWNER, source_only=True, target_access=False,
        source_contract=SOURCE, preparation=PREPARATION, output_root=BASE,
        role='L_s', count=6300, RX=list(RX), days=[1, 2, 3], TX_indices=list(range(6)),
        equalized=1, IQ_length=256, sampling_rate_hz=25000000, packet_RMS='existing native source input',
        windows=[list(v) for v in WINDOWS], primary='w80_160',
        CFO='same wrapped lag20 estimator as existing known-excitation diagnostic; alias period 1.25MHz',
        phase_coordinates='10 full-rank integer closure coordinates on 12 known L-STF tones, null constant/linear phase',
        availability=dict(min_lag20_coherence=.9, min_occupied_fraction=.95,
            min_cycle_repeatability_ratio=10., min_eligible_per_TX_RX_day_coordinate=20),
        crossfit='18 folds; exclude held TX AND held day from both RX and pooled means; equal cell weight',
        controls='four cyclic wrong-RX mean assignments, same eligible observations',
        criterion=dict(min_risk_gain=.02, min_positive_folds=14, min_RX_risk_gain=0.,
            min_gain_over_best_wrong_RX=.01, min_fit_resultant=.5, min_RX_split_cosine=.8),
        secondary_windows='report only, never replace primary or tune thresholds',
        new_models=0, checkpoint_loads=0, network_forwards=0, identity_updates=0,
        U_label_access=False, V_fit=False, target_predictions=0, automatic_training=False,
        device='CPU', threads=2,
        interpretation='RX-associated phase repeatability of received IQ only; not identified RX hardware or proven identity benefit')


def validate(value):
    if value != config():
        raise ValueError('Fixed source phase configuration differs')
