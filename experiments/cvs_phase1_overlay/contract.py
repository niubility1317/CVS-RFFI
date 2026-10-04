"""Frozen first-round 2x2 design; no target scores enter configuration."""
from pathlib import Path
from types import SimpleNamespace
import json
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

ROOT = Path(__file__).resolve().parents[2]
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
RUN = '20261004-phase1-reference-overlay-r1-manysig-m16-r01'
RELEASE = 'cvs_reference_overlay_r1_20261004_r01'
SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
ARMS = ('ce', 'leo', 'mixstyle', 'leo_mixstyle')
SOURCE = PROJECT + '/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
CAPSULE = PROJECT + '/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule'
TRUTH = PROJECT + '/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
CLASSES = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
SCENES = ['practical_high', 'practical_mid', 'practical_low_urban']
SCHEDULE = '1@0.30:practical_high;41@0.60:practical_mid,practical_low_urban;91@0.80:practical_high,practical_mid,practical_low_urban'
MIXSTYLE = dict(p=.18, alpha=.1, eps=1e-6, strength=.70,
    layers=['time_down', 't1'], mix='same_tx_crossdomain', fallback='skip', domain='rx_day')
LEO = dict(schedule=SCHEDULE, concat_start=1, satellite_ce_start=80, clean_weight=1.,
    satellite_weight=.68, consistency_weight=0., processing_route='residual', mode='post_sync',
    equalization=False, fs_hz=25000000., fc_hz=2462000000., receiver_seed=2027,
    seed=2027, sampling='native batch Bernoulli and one scene per applied batch',
    physical_id_namespace='source:<opaque source-contract physical ID>')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    p.write_text(text, encoding='utf-8', newline='\n')


def validate_config(c):
    required = dict(method='cvs_phase1_overlay', variant='reference_response', epochs=200,
        batch_size=128, lr=.0002, lr_min=1e-6, weight_decay=.0001, split_seed=392005,
        drop_last=False, domain_backbone=False, extra_losses=[], selection='fixed_last_epoch',
        numerical_policy=FULL_FP32_POLICY, mixstyle_recipe=MIXSTYLE, leo_recipe=LEO,
        source_contract=SOURCE, dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl')
    if c.get('arm') not in ARMS or type(c.get('model_seed')) is not int or c['model_seed'] not in SEEDS:
        raise ValueError('Unregistered arm/seed')
    if any(c.get(k) != v for k, v in required.items()):
        raise ValueError('Frozen source contract differs')
    for k, v in c.items():
        if v and (k.startswith(('target', 'p1_', 'p2_')) or k in
                  ('checkpoint', 'resume', 'teacher', 'ancestors', 'checkpoint_sources', 'initial_checkpoint', 'ema_checkpoint')):
            raise ValueError('Scratch source-only training rejects ' + k)
    expected = PROJECT+'/runs/'+RUN+'/'+c['arm']+'-s'+str(c['model_seed'])+'/source'
    if c.get('output_root') != expected:
        raise ValueError('Output differs from exclusive registered row')
    return c


def build_model(arm):
    if arm not in ARMS:
        raise ValueError('Unregistered arm')
    from experiments.cvs_phase1_overlay.model import Phase1IdentityAdapter
    return Phase1IdentityAdapter(SimpleNamespace(num_classes=6, input_len=256,
        use_mixstyle='mixstyle' in arm, mixstyle_p=MIXSTYLE['p'], mixstyle_alpha=MIXSTYLE['alpha'],
        mixstyle_eps=MIXSTYLE['eps'], mixstyle_strength=MIXSTYLE['strength'],
        mixstyle_layers=','.join(MIXSTYLE['layers']), mixstyle_mix=MIXSTYLE['mix'],
        mixstyle_fallback=MIXSTYLE['fallback'], mixstyle_use_domain_label=True))
