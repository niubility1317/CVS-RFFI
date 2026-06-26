from types import SimpleNamespace

from cvsrffi.presets import apply_experiment_preset
from train import capture_explicit_preset_sensitive_args, restore_explicit_preset_sensitive_args


def _rxrobust_args():
    return SimpleNamespace(
        exp_group="s3_rxrobust_no_dac",
        epochs=170,
        arch_family="cvsincnet",
        model_variant="lite_d",
        branch_ablation="no_dac",
        domain_branch_ablation="no_stats",
        domain_enhancer="rcn_stats",
        domain_enhancer_strength=0.35,
        id_time_stability_mode="off",
        id_freq_stability_mode="off",
        domain_time_stability_mode="off",
        domain_freq_stability_mode="off",
        pa_orders="1,3,5",
        lambda_dom=0.0,
        lambda_adv=0.16,
        lambda_orth=0.0,
        lambda_cons=0.03,
        lambda_group_ce=0.025,
        lambda_proto=0.004,
        lambda_supcon_id=0.004,
        lambda_fishr=0.0,
        lambda_sat_cls=0.0,
        lambda_sat_cons=0.0,
        lambda_feature_norm_guard=0.0001,
        lambda_cls_pa=0.0,
        lambda_pa_joint_inv=0.0,
        lambda_pa_kl=0.0,
        lambda_pa_reg=0.0,
        group_ce_top_frac=0.20,
        groupdro_tau=0.30,
        groupdro_cap=0.42,
        aug_p_pa=0.0,
        aug_p_dac=0.0,
        concat_sat_ce_weight=0.35,
        sat_view_prob=0.35,
        sat_cons_start_epoch=999,
        mixstyle_p=0.10,
        mixstyle_strength=0.38,
        mixstyle_late_min_p=0.02,
        mixstyle_late_min_strength=0.18,
        feature_norm_guard_mode="l2",
        feature_norm_guard_target=0.0,
        use_aug=False,
        use_mixstyle=False,
        enable_pa_aux=False,
        enable_dac_aux=False,
        aug_enable_pa_normal=False,
        use_sat_consistency=False,
        use_concat_sat_channel_aug=False,
        stage1_epochs=1,
        stage2_epochs=2,
        stage3_ramp_epochs=3,
        late_stable_start=80,
        late_stable_ramp_epochs=25,
        late_adv_min_scale=0.35,
        late_cons_min_scale=0.20,
        late_cls_aux_min_scale=0.25,
        late_reg_aux_min_scale=0.25,
        late_joint_inv_min_scale=0.08,
        late_kl_min_scale=0.16,
        late_group_ce_min_scale=0.35,
        late_aug_min_scale=0.45,
    )


def test_rxrobust_profile_keeps_default_values_without_cli_override():
    args = _rxrobust_args()
    apply_experiment_preset(args)

    assert args.lambda_adv == 0.45
    assert args.lambda_cons == 0.08
    assert args.lambda_group_ce == 0.10
    assert args.group_ce_top_frac == 0.35
    assert args.late_adv_min_scale == 0.70
    assert args.late_group_ce_min_scale == 0.80


def test_lowshot_cli_values_survive_rxrobust_profile():
    args = _rxrobust_args()
    explicit = capture_explicit_preset_sensitive_args(
        args,
        [
            "--lambda_adv",
            "0.16",
            "--lambda_dom",
            "0.0",
            "--lambda_orth",
            "0.0",
            "--lambda_cons",
            "0.03",
            "--lambda_group_ce",
            "0.025",
            "--group_ce_top_frac",
            "0.20",
            "--late_adv_min_scale",
            "0.35",
            "--late_group_ce_min_scale",
            "0.35",
        ],
    )

    apply_experiment_preset(args)
    restore_explicit_preset_sensitive_args(args, explicit)

    assert args.lambda_adv == 0.16
    assert args.lambda_dom == 0.0
    assert args.lambda_orth == 0.0
    assert args.lambda_cons == 0.03
    assert args.lambda_group_ce == 0.025
    assert args.group_ce_top_frac == 0.20
    assert args.late_adv_min_scale == 0.35
    assert args.late_group_ce_min_scale == 0.35


def test_lowshot_regularizer_values_restore_after_preset_mutation():
    args = _rxrobust_args()
    explicit = capture_explicit_preset_sensitive_args(
        args,
        [
            "--lambda_proto=0.004",
            "--lambda_supcon_id",
            "0.004",
            "--lambda_fishr=0.0",
            "--lambda_sat_cons=0.0",
            "--lambda_feature_norm_guard",
            "0.0001",
            "--lambda_cls_pa=0.0",
            "--lambda_pa_joint_inv=0.0",
            "--lambda_pa_kl=0.0",
            "--lambda_pa_reg=0.0",
            "--aug_p_pa=0.0",
            "--aug_p_dac=0.0",
            "--concat_sat_ce_weight",
            "0.35",
            "--sat_view_prob=0.35",
            "--sat_cons_start_epoch",
            "999",
            "--mixstyle_p=0.10",
            "--mixstyle_strength",
            "0.38",
            "--mixstyle_late_min_p=0.02",
            "--mixstyle_late_min_strength=0.18",
            "--feature_norm_guard_mode",
            "l2",
            "--feature_norm_guard_target=0.0",
        ],
    )

    args.lambda_proto = 0.016
    args.lambda_supcon_id = 0.022
    args.lambda_fishr = 0.002
    args.lambda_sat_cons = 0.006
    args.lambda_feature_norm_guard = 0.0
    args.lambda_cls_pa = 0.20
    args.lambda_pa_joint_inv = 0.06
    args.lambda_pa_kl = 0.02
    args.lambda_pa_reg = 0.10
    args.aug_p_pa = 0.14
    args.aug_p_dac = 0.12
    args.concat_sat_ce_weight = 1.0
    args.sat_view_prob = 1.0
    args.sat_cons_start_epoch = 1
    args.mixstyle_p = 0.25
    args.mixstyle_strength = 0.75
    args.mixstyle_late_min_p = 0.08
    args.mixstyle_late_min_strength = 0.40
    args.feature_norm_guard_mode = "hinge"
    args.feature_norm_guard_target = 8.0

    restore_explicit_preset_sensitive_args(args, explicit)

    assert args.lambda_proto == 0.004
    assert args.lambda_supcon_id == 0.004
    assert args.lambda_fishr == 0.0
    assert args.lambda_sat_cons == 0.0
    assert args.lambda_feature_norm_guard == 0.0001
    assert args.lambda_cls_pa == 0.0
    assert args.lambda_pa_joint_inv == 0.0
    assert args.lambda_pa_kl == 0.0
    assert args.lambda_pa_reg == 0.0
    assert args.aug_p_pa == 0.0
    assert args.aug_p_dac == 0.0
    assert args.concat_sat_ce_weight == 0.35
    assert args.sat_view_prob == 0.35
    assert args.sat_cons_start_epoch == 999
    assert args.mixstyle_p == 0.10
    assert args.mixstyle_strength == 0.38
    assert args.mixstyle_late_min_p == 0.02
    assert args.mixstyle_late_min_strength == 0.18
    assert args.feature_norm_guard_mode == "l2"
    assert args.feature_norm_guard_target == 0.0


def test_lowshot_boolean_overrides_survive_preset_mutation():
    args = _rxrobust_args()
    explicit = capture_explicit_preset_sensitive_args(
        args,
        [
            "--no_use_aug",
            "--no_use_mixstyle",
            "--no_enable_pa_aux",
            "--no_enable_dac_aux",
            "--no_aug_enable_pa_normal",
            "--no_use_sat_consistency",
            "--no_use_concat_sat_channel_aug",
        ],
    )

    args.use_aug = True
    args.use_mixstyle = True
    args.enable_pa_aux = True
    args.enable_dac_aux = True
    args.aug_enable_pa_normal = True
    args.use_sat_consistency = True
    args.use_concat_sat_channel_aug = True

    restore_explicit_preset_sensitive_args(args, explicit)

    assert args.use_aug is False
    assert args.use_mixstyle is False
    assert args.enable_pa_aux is False
    assert args.enable_dac_aux is False
    assert args.aug_enable_pa_normal is False
    assert args.use_sat_consistency is False
    assert args.use_concat_sat_channel_aug is False


def test_lowshot_architecture_values_survive_preset_mutation():
    args = _rxrobust_args()
    args.domain_enhancer = "off"
    args.domain_enhancer_strength = 0.0
    explicit = capture_explicit_preset_sensitive_args(
        args,
        [
            "--arch_family",
            "cvsincnet",
            "--model_variant=lite_d",
            "--branch_ablation",
            "no_dac",
            "--domain_branch_ablation=no_stats",
            "--domain_enhancer",
            "off",
            "--domain_enhancer_strength=0.0",
            "--id_time_stability_mode=off",
            "--id_freq_stability_mode=off",
            "--domain_time_stability_mode=off",
            "--domain_freq_stability_mode=off",
            "--pa_orders",
            "1,3,5",
        ],
    )

    args.arch_family = "resnet18_1d"
    args.model_variant = "lite_b"
    args.branch_ablation = "none"
    args.domain_branch_ablation = "none"
    args.domain_enhancer = "rcn_stats"
    args.domain_enhancer_strength = 0.35
    args.id_time_stability_mode = "phase_delta"
    args.id_freq_stability_mode = "dsq"
    args.domain_time_stability_mode = "phase_delta"
    args.domain_freq_stability_mode = "dsq"
    args.pa_orders = "1,3,5,7"

    restore_explicit_preset_sensitive_args(args, explicit)

    assert args.arch_family == "cvsincnet"
    assert args.model_variant == "lite_d"
    assert args.branch_ablation == "no_dac"
    assert args.domain_branch_ablation == "no_stats"
    assert args.domain_enhancer == "off"
    assert args.domain_enhancer_strength == 0.0
    assert args.id_time_stability_mode == "off"
    assert args.id_freq_stability_mode == "off"
    assert args.domain_time_stability_mode == "off"
    assert args.domain_freq_stability_mode == "off"
    assert args.pa_orders == "1,3,5"
