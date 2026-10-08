"""Fixed 2x2 design. No target result participates in configuration."""
from pathlib import Path
from experiments.cvs_phase1_overlay.contract import (
    PROJECT, SOURCE, CAPSULE, TRUTH, CLASSES, SEEDS, LEO, FULL_FP32_POLICY, read, write)

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261008-phase1-receiver-residual-manysig-m16-r01'
RELEASE = 'cvs_receiver_residual_20261008_r01'
METHOD = 'cvs_receiver_residual'
ARMS = ('baseline', 'displacement', 'contribution', 'combined')
SOURCE_RXS = (1, 3, 4, 6, 8)
VIEWS_ROOT = PROJECT + '/runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
SCENES = ('practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
VIEWS = ('clean', *SCENES)
COUNT = 168000
RECIPE = dict(warmup_epochs=40, ramp_epochs=20, nuisance_input='received_response_29',
    hidden=64, rank=16, displacement_weight=1., gauge_weight=.1,
    contribution_weight=.2, contribution_clip=2., contribution_temperature=1.,
    correction_relative_norm_cap=.25, gate_relative_cap=.25,
    centroid_role='L_s_only', centroid_refresh='current_eval_features_each_epoch_from_E40',
    centroid_ema=False, cross_tx_aggregate='coordinatewise_median_excluding_supervised_TX',
    auxiliary_views='clean_only', counterfactual='single_branch_deletion_at_current_base_head')


def ramp(epoch):
    return min(1., max(0., (epoch - RECIPE['warmup_epochs']) / RECIPE['ramp_epochs']))


def row_id(arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Unregistered arm/seed')
    return arm + '-s' + str(seed)


def config(arm, seed):
    rid = row_id(arm, seed)
    return dict(method=METHOD, variant='reference_response', arm=arm, model_seed=seed,
        source_contract=SOURCE, dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',
        output_root=PROJECT+'/runs/'+RUN+'/'+rid+'/source', device='cuda:0',
        epochs=200, batch_size=128, lr=.0002, lr_min=1e-6, weight_decay=.0001,
        split_seed=392005, data_seed=392005, loader_seed=seed,
        augmentation_seed=2027, receiver_seed=2027, drop_last=False,
        domain_backbone=False, mixstyle=False, pseudo_labels=False, ema=False,
        extra_losses=['cross_tx_displacement', 'class_gauge'] if arm == 'displacement'
            else ['branch_contribution'] if arm == 'contribution'
            else ['cross_tx_displacement', 'class_gauge', 'branch_contribution'] if arm == 'combined' else [],
        selection='fixed_last_epoch', numerical_policy=FULL_FP32_POLICY,
        leo_recipe=LEO, mechanism_recipe=RECIPE, concat_sat_ce_only=True,
        lambda_sat_cls=.68, lambda_sat_cons=0.)


def validate_config(c):
    if c.get('arm') not in ARMS or c.get('model_seed') not in SEEDS:
        raise ValueError('Unregistered source row')
    expected = config(c['arm'], c['model_seed'])
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('Source configuration differs from fixed recipe')
    forbidden = ('checkpoint', 'resume', 'teacher', 'ancestors', 'checkpoint_sources',
                 'initial_checkpoint', 'ema_checkpoint', 'truth')
    for k, v in c.items():
        if v and (k.startswith(('target', 'p1_', 'p2_')) or k in forbidden):
            raise ValueError('Scratch source-only training rejects '+k)
    return c


def rows():
    return [config(a, s) for s in SEEDS for a in ARMS]


def build_model(arm):
    from experiments.cvs_receiver_residual.model import ReceiverResidualCVS
    return ReceiverResidualCVS(arm)
