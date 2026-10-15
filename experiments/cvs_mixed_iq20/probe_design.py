"""One frozen, zero-update source check of the already specified IQ operator."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
RUN = '20261015-phase1-sigtime-rdrop-mixedIQ-source-s2026092701-r01'
RELEASE = 'cvs_sigtime_rdrop_mixedIQ_source_20261015_r01'
OWNER = 'codex/root/mixed-IQ-source-20261015'
BASE = Path(PROJECT) / 'runs' / RUN
SOURCE = PROJECT + '/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION = PROJECT + '/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
DEPENDENCY = PROJECT + '/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02'
LEGACY = PROJECT + '/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
FOLD = PROJECT + '/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
PARENT_RUN = '20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01'
PARENT = PROJECT + '/runs/' + PARENT_RUN + '/E20_RDROP-s2026092701/source'
CONDITIONS = ('RAW', 'G_E1_REFLECT', 'G_E20_REFLECT', 'G_E1_EDGE', 'G_E20_EDGE')
FORWARDS = 27000 * len(CONDITIONS) + 2


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def config():
    return dict(run_id=RUN, launch_owner=OWNER, source_only=True, query_read=False,
                checkpoint=PARENT + '/final_ssdg.pth', parent_model_seed=2026092701,
                parent_training_epochs=20, parent_training_updates=4440,
                conditions=list(CONDITIONS), source_role='V', source_count=27000,
                augmentation_seed=20261015031, stream='source_probe',
                source_RX=[1, 3, 4, 6, 8], source_days=[1, 2, 3], equalized=1,
                coefficient_radius=.06, reflect_is_candidate=True, edge_is_control=True,
                source_gate=dict(max_accuracy_drop_pp=1., reference='RAW',
                                 candidates=['G_E1_REFLECT', 'G_E20_REFLECT']),
                forward_packets=FORWARDS, optimizer_updates=0, EMA_updates=0,
                head_fit_updates=0, U_opened=False, target_predictions=0,
                formal_test_IQ_changed=False, extra_training_epochs=0,
                device='CPU', threads=2,
                interpretation='Source safety/falsification only; no learned or target generalization gain')
