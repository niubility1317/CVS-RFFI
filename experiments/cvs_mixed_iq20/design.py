"""Executable design, not a registered training launcher or performance result."""
import argparse
import json
from pathlib import Path


ARMS = ('R_LEGACY', 'B00', 'B10', 'B01', 'B11')
SCENES = ('practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def plan():
    common = dict(initialization='scratch', parent_training_epochs=0, total_epochs=20,
                  label_epochs=13, pseudo_epochs=7, steps_per_epoch=222,
                  optimizer_steps=4440, L_batch=128, U_batch=256,
                  model_seed=2026092701, split_seed=392005, data_seed=392005,
                  generalization_seed=20261015031, practical_seed=20261015032,
                  receiver_seed=2026092701, evaluation_seed=392005,
                  source_roles=dict(L_s=6300, U_s=56700, V=27000),
                  source_RX=[1, 3, 4, 6, 8], source_days=[1, 2, 3], TX_count=6,
                  checkpoint_selection='final_only', head='existing cosine6x160 scale30',
                  feature='existing SIG_TIME RAW_HZ joint160, NoExplicitCFO',
                  label_smoothing=.01, RDrop_alpha=1., U_rule='unchanged native oldgate+ownEMA',
                  new_U_augmentation=False, head_fit_updates=0,
                  test_views=['clean', *SCENES])
    rows = []
    for arm in ARMS:
        ag, sat = arm in ('B10', 'B11'), arm in ('B01', 'B11')
        legacy = arm == 'R_LEGACY'
        forwards = 2 if legacy else 6
        rows.append(dict(common, arm=arm, legacy_L_augmentation=legacy,
                         A_g=ag, A_s=sat, compose_A_s_A_g=False,
                         labeled_forward_packets=4440 * 128 * forwards,
                         physical_L_exposures=4440 * 128,
                         views_per_physical_sample=1 if legacy else 3,
                         RDrop_forwards_per_IQ_view=2,
                         objective='original smoothed RDrop on one old-augmented IQ' if legacy
                         else '.25 RDrop(raw) + .25 RDrop(g) + .5 RDrop(s); inactive slot=raw'))
    return dict(status='DESIGN_AND_PRIMITIVES_ONLY', training_launcher_implemented=False,
                experiments_launched=0, source_identity_preservation_test='PENDING',
                rows=rows, first_seed_only=True, training_scenes=list(SCENES),
                satellite_sampler='.8 legacy-three uniform + .2 all-six uniform; native state priors, no curriculum',
                training_scene_probabilities=dict(zip(SCENES, [1/30, 1/30, .3, 1/30, .3, .3])),
                response_radius=.06, response_boundary='reflect',
                practical=dict(route='residual', mode='post_sync', equalization_enabled=False,
                               reference_snr_db=30., fs_hz=25_000_000,
                               quality_adaptation=True, phase_tracking=True, IQ_compensation=True),
                composition='B11 mixes separate G and S packets, never a cascade',
                primary='B11', factorial=['B00', 'B10', 'B01', 'B11'],
                clean_comparisons=['B11-B00', 'B11-R_LEGACY', 'B10-B00', 'B11-B10'],
                satellite_comparison='B11-B01', interaction='B11-B10-B01+B00',
                exploration_targets=dict(clean_net_gain_pp=.5, low_urban_accuracy_gt=.60,
                                         suburban_max_gap_to_own_clean_pp=2.,
                                         suburban_absolute_drop_max_pp=.5),
                source_falsification=dict(default_head_augmentation_drop_max_pp=1.,
                                         clean_V_mean_drop_max_pp=.5,
                                         worst_source_RX_drop_max_pp=1.,
                                         B10_clean_source_gain_over_B00_min_pp=.5,
                                         B11_clean_source_gain_over_both_controls_min_pp=0.,
                                         B11_low_source_gain_over_B01_min_pp=1.,
                                         B11_urban_source_gain_over_B01_min_pp=0.,
                                         target_metrics_read=False),
                required_before_real_run=['source no-update candidate-IQ checks',
                                          'native L-only integration with same-IQ RDrop evidence',
                                          'complete no-query smoke and protocol negatives',
                                          'one P0/P1 code review and experiment registration',
                                          'immutable release and automatic7view truth-last pipeline'],
                no_test_IQ_change=True, no_target_feedback=True,
                no_cross_seed_claim=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    text = json.dumps(plan(), ensure_ascii=False, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text, encoding='utf-8')
        print(json.dumps(dict(status=plan()['status'], path=str(a.output),
                              rows=len(ARMS), models_launched=0)))
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
