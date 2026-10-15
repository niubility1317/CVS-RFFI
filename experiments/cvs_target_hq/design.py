import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
FORMAL_VIEWS=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
TRUTH=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
PARENT_RUN='20261015-phase1-sigtime-branch-align-e50-manysig-s2026092701-m8-r01'
PARENT_SOURCE=Path(PROJECT)/'runs'/PARENT_RUN/'A0-s2026092701/source'
RUN='20261015-phase1-frozen-ema-target-hq-manysig-s2026092701-n4200-r01'
RELEASE='cvs_frozen_ema_target_hq_20261015_r01'
OWNER='codex/side/target-hq-20261015'
BASE=Path(PROJECT)/'runs'/RUN
COUNT=4200
SAMPLE_SEED=2026101513
SCENE='practical_low_urban'
VIEWS=('full','independent_same_noise_power','HQ_independent_minus6dB','HQ_shared_minus6dB')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def parent_config():
    from experiments.cvs_urban_branch50 import design as parent
    return parent.config(next(r for r in parent.rows() if r['arm']=='A0'))

def sample_positions(ids,count=COUNT,seed=SAMPLE_SEED):
    ids=list(map(str,ids))
    if len(set(ids))!=len(ids) or not 0<count<=len(ids) or any(not p for p in ids):
        raise ValueError('Unique fixed anonymous physical IDs and bounded sample count required')
    # A reproducible label-blind random sample, not an output/package seal.
    order=sorted(range(len(ids)),key=lambda i:hashlib.sha256((str(seed)+'/'+ids[i]).encode()).digest())
    return sorted(order[:count])

def configuration():
    return dict(run_id=RUN,owner=OWNER,stage='Phase1',scope='TARGET_HQ_COUNTERFACTUAL_NOT_FORMAL',
        parent_checkpoint=(PARENT_SOURCE/'final_ssdg.pth').as_posix(),parent_run=PARENT_RUN,parent_row='A0-s2026092701',
        model_component='same checkpoint ema_model including its default cosine head',
        physical_sample_count=COUNT,selection='4200 smallest seeded anonymous-ID ranks; label/quality/output blind',
        sample_seed=SAMPLE_SEED,scene=SCENE,views=list(VIEWS),noise_improvement_db=6.,
        original_full='original fixed received array used unchanged; regenerated reference checked numerically',
        formal_test_IQ_modified=False,receiver_actions='original full-view state, compensation and AGC frozen',
        optimizer_updates=0,EMA_updates=0,head_fit_updates=0,training_epochs_added=0,
        target_results_for_training_selection=False,automatic_expansion=False)
