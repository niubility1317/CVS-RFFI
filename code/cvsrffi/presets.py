from __future__ import annotations


def set_pa_weights(args, *, cls_pa: float, joint_inv: float, imp_inv: float, pa_kl: float, pa_reg: float, pa_select: float, pa_mono: float):
    args.lambda_cls_pa = float(cls_pa)
    args.lambda_pa_joint_inv = float(joint_inv)
    args.lambda_pa_imp_inv = 0.0
    args.lambda_pa_kl = float(pa_kl)
    args.lambda_pa_reg = float(pa_reg)
    args.lambda_pa_select = 0.0
    args.lambda_pa_mono = 0.0


def zero_pa_path(args):
    args.enable_pa_aux = False
    args.aug_enable_pa_normal = False
    args.aug_p_pa = 0.0
    set_pa_weights(args, cls_pa=0.0, joint_inv=0.0, imp_inv=0.0, pa_kl=0.0, pa_reg=0.0, pa_select=0.0, pa_mono=0.0)


def zero_dac_path(args):
    args.enable_dac_aux = False
    args.aug_p_dac = 0.0
    args.lambda_cls_dac = 0.0
    args.lambda_dac_reg = 0.0
    args.lambda_dac_select = 0.0
    args.lambda_dac_mono = 0.0


def set_dac_weights(args, *, cls_dac: float, dac_reg: float, dac_select: float, dac_mono: float):
    args.lambda_cls_dac = float(cls_dac)
    args.lambda_dac_reg = float(dac_reg)
    args.lambda_dac_select = 0.0
    args.lambda_dac_mono = 0.0


def parse_branch_ablation_flags(branch_ablation: str) -> frozenset[str]:
    raw = str(branch_ablation or "none").lower().replace(";", ",").replace("+", ",")
    aliases = {
        "none": "",
        "base": "",
        "off": "",
        "no_time_branch": "no_time",
        "no_dac_branch": "no_dac",
        "no_pa_branch": "no_pa",
        "no_freq_branch": "no_freq",
        "no_spectral": "no_freq",
        "no_spec": "no_freq",
        "no_stat": "no_stats",
        "no_spectral_stats": "no_stats",
        "no_dac_pa": "no_dac,no_pa",
        "no_physical": "no_dac,no_pa",
        "time_only": "no_dac,no_pa,no_freq,no_stats",
        "freq_only": "no_time,no_dac,no_pa,no_stats",
        "no_defect_branches": "no_dac,no_pa",
    }
    expanded = []
    for item in raw.split(","):
        item = item.strip()
        if item == "":
            continue
        item = aliases.get(item, item)
        expanded.extend([z.strip() for z in item.split(",") if z.strip()])
    return frozenset(expanded)


def apply_experiment_preset(args):
    g_raw = str(args.exp_group).strip()
    alias_map = {
        # new stagewise names
        "s1_core_only": "s1_core_only",
        "s2_pure_aux_no_select": "s2_pure_aux_no_select",
        "s3_stagewise_pa_focus": "s3_stagewise_pa_focus",
        "s4_stagewise_full_dual": "s4_stagewise_full_dual",
        "s3_stable_no_dac": "s3_stable_no_dac",
        "s4_late_stable_full": "s4_late_stable_full",
        "s3_rxrobust_no_dac": "s3_rxrobust_no_dac",
        "s4_rxrobust_full": "s4_rxrobust_full",
        # backward-compatible aliases from older versions
        "g1_true_no_pa": "s1_core_only",
        "g2_pa_aux_only": "s3_stagewise_pa_focus",
        "g3_pa_main_only": "s3_stagewise_pa_focus",
        "g4_pa_main_plus_aux": "s3_stagewise_pa_focus",
        "g5_full_dual_puredefect": "s4_stagewise_full_dual",
    }
    if g_raw not in alias_map:
        valid = ", ".join(sorted(alias_map.keys()))
        raise ValueError(f"Unknown exp_group={g_raw}. Valid values: {valid}")
    g = alias_map[g_raw]
    args.exp_group = g

    # Shared defaults
    args.use_aug = True
    args.aug_enable_class_signature = False
    args.aug_scale_min = 0.10
    args.aug_scale_max = 0.35
    args.aug_warmup_epochs = 3
    args.aug_ramp_epochs = 15
    args.aug_ramp_curve = 1.25
    args.aux_warmup_epochs = 3
    args.aux_ramp_epochs = 15
    args.robust_temp = 1.0
    args.select_margin = 0.03
    args.mono_margin = 0.00
    args.aug_pa_mp_sigma = 0.04
    args.aug_pa_mem_sigma = 0.03
    args.aug_pa_ampm_max = 0.15
    args.aug_pa_iq_img_max = 0.010

    # Stagewise defaults: pure defect-only views, mixed DAC+PA view may still include channel.
    args.aug_dac_only_apply_anti_shortcut = False
    args.aug_dac_only_apply_channel = False
    args.aug_pa_only_apply_anti_shortcut = False
    args.aug_pa_only_apply_channel = False
    args.aug_dac_pa_apply_anti_shortcut = True
    args.aug_dac_pa_apply_channel = True
    args.aug_defect_strength_mode = "tiered"
    args.aug_dac_only_tiers = "0.15,0.35,0.55"
    args.aug_pa_only_tiers = "0.15,0.35,0.60"

    # Clean slate before per-group setup.
    args.lambda_cross_zero = 0.0
    args.stage1_epochs = int(getattr(args, "stage1_epochs", 15))
    args.stage2_epochs = int(getattr(args, "stage2_epochs", 45))
    args.stage3_ramp_epochs = int(getattr(args, "stage3_ramp_epochs", 20))
    args.late_stable_start = int(getattr(args, "late_stable_start", 0))
    args.late_stable_ramp_epochs = int(getattr(args, "late_stable_ramp_epochs", 12))
    args.late_adv_min_scale = float(getattr(args, "late_adv_min_scale", 0.75))
    args.late_cons_min_scale = float(getattr(args, "late_cons_min_scale", 0.55))
    args.late_cls_aux_min_scale = float(getattr(args, "late_cls_aux_min_scale", 0.35))
    args.late_reg_aux_min_scale = float(getattr(args, "late_reg_aux_min_scale", 0.35))
    args.late_joint_inv_min_scale = float(getattr(args, "late_joint_inv_min_scale", 0.12))
    args.late_kl_min_scale = float(getattr(args, "late_kl_min_scale", 0.25))
    args.late_group_ce_min_scale = float(getattr(args, "late_group_ce_min_scale", 0.75))
    args.late_aug_min_scale = float(getattr(args, "late_aug_min_scale", -1.0))

    # Base defaults for DAC/PA paths.
    args.enable_dac_aux = False
    args.enable_pa_aux = False
    args.aug_enable_pa_normal = False
    args.aug_p_dac = 0.0
    args.aug_p_pa = 0.0
    args.lambda_cls_dac = 0.0
    args.lambda_dac_reg = 0.0
    args.lambda_dac_select = 0.0
    args.lambda_dac_mono = 0.0
    set_pa_weights(args, cls_pa=0.0, joint_inv=0.0, imp_inv=0.0, pa_kl=0.0, pa_reg=0.0, pa_select=0.0, pa_mono=0.0)

    if g == "s1_core_only":
        args.exp_desc = "S1 仅主任务：cls + dom/adv/orth/cons，不启用 DAC/PA 辅助"
        args.stage1_epochs = max(args.stage1_epochs, 999999)
        args.stage2_epochs = max(args.stage2_epochs, 999999)
    elif g == "s2_pure_aux_no_select":
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.18
        args.aug_p_dac = 0.22
        args.lambda_cls_dac = 0.10
        args.lambda_dac_reg = 0.25
        args.lambda_dac_select = 0.0
        args.lambda_dac_mono = 0.0
        set_pa_weights(args, cls_pa=0.30, joint_inv=0.10, imp_inv=0.00, pa_kl=0.04, pa_reg=0.18, pa_select=0.0, pa_mono=0.0)
        args.exp_desc = "S2 纯辅助无选择性：纯 DAC/PA-only 视图 + 轻 reg/joint_inv/kl，不启用 select/mono/cross_zero"
    elif g == "s3_stagewise_pa_focus":
        args.enable_dac_aux = False
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.18
        set_pa_weights(args, cls_pa=0.32, joint_inv=0.10, imp_inv=0.00, pa_kl=0.04, pa_reg=0.18, pa_select=0.06, pa_mono=0.04)
        args.exp_desc = "S3 PA 重点阶段式：主视图温和 PA + 纯 PA-only 辅助，仅保留 joint/kl/reg"
    elif g == "s4_stagewise_full_dual":
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.18
        args.aug_p_dac = 0.22
        args.lambda_cls_dac = 0.10
        args.lambda_dac_reg = 0.25
        args.lambda_dac_select = 0.0
        args.lambda_dac_mono = 0.0
        set_pa_weights(args, cls_pa=0.30, joint_inv=0.10, imp_inv=0.00, pa_kl=0.04, pa_reg=0.18, pa_select=0.08, pa_mono=0.05)
        args.lambda_cross_zero = 0.0
        args.exp_desc = "S4 双缺陷阶段式：joint 特征去域 + 纯 DAC/PA-only 辅助，移除 select/mono/cross_zero"
    elif g == "s3_stable_no_dac":
        total_epochs = int(max(1, getattr(args, "epochs", 100)))
        args.enable_dac_aux = False
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.16
        args.stage1_epochs = max(12, int(round(total_epochs * 0.08)))
        args.stage2_epochs = max(args.stage1_epochs + 24, int(round(total_epochs * 0.38)))
        args.stage3_ramp_epochs = max(8, int(round(total_epochs * 0.08)))
        args.late_stable_start = max(52, int(round(total_epochs * 0.65)))
        args.late_stable_ramp_epochs = max(12, int(round(total_epochs * 0.18)))
        args.late_adv_min_scale = 0.70
        args.late_cons_min_scale = 0.50
        args.late_cls_aux_min_scale = 0.30
        args.late_reg_aux_min_scale = 0.30
        args.late_joint_inv_min_scale = 0.08
        args.late_kl_min_scale = 0.18
        args.late_group_ce_min_scale = 0.75
        args.late_aug_min_scale = 0.22
        set_pa_weights(args, cls_pa=0.24, joint_inv=0.08, imp_inv=0.00, pa_kl=0.03, pa_reg=0.12, pa_select=0.0, pa_mono=0.0)
        args.exp_desc = "S3 stable no-DAC: PA-only auxiliary path with late loss decay for OOD stability"
    elif g == "s4_late_stable_full":
        total_epochs = int(max(1, getattr(args, "epochs", 100)))
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.14
        args.aug_p_dac = 0.12
        args.lambda_cls_dac = 0.05
        args.lambda_dac_reg = 0.10
        args.stage1_epochs = max(12, int(round(total_epochs * 0.08)))
        args.stage2_epochs = max(args.stage1_epochs + 24, int(round(total_epochs * 0.36)))
        args.stage3_ramp_epochs = max(8, int(round(total_epochs * 0.08)))
        args.late_stable_start = max(50, int(round(total_epochs * 0.65)))
        args.late_stable_ramp_epochs = max(12, int(round(total_epochs * 0.18)))
        args.late_adv_min_scale = 0.70
        args.late_cons_min_scale = 0.45
        args.late_cls_aux_min_scale = 0.25
        args.late_reg_aux_min_scale = 0.25
        args.late_joint_inv_min_scale = 0.08
        args.late_kl_min_scale = 0.16
        args.late_group_ce_min_scale = 0.75
        args.late_aug_min_scale = 0.20
        set_pa_weights(args, cls_pa=0.22, joint_inv=0.08, imp_inv=0.00, pa_kl=0.03, pa_reg=0.12, pa_select=0.0, pa_mono=0.0)
        args.lambda_cross_zero = 0.0
        args.exp_desc = "S4 late-stable full: reduced DAC/PA auxiliary pressure and late loss decay"
    elif g == "s3_rxrobust_no_dac":
        total_epochs = int(max(1, getattr(args, "epochs", 100)))
        args.enable_dac_aux = False
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.14
        args.lambda_adv = 0.45
        args.lambda_cons = 0.08
        args.lambda_group_ce = 0.10
        args.group_ce_top_frac = 0.35
        args.group_ce_min_domains = 4
        args.stage1_epochs = max(16, int(round(total_epochs * 0.08)))
        args.stage2_epochs = max(args.stage1_epochs + 40, int(round(total_epochs * 0.40)))
        args.stage3_ramp_epochs = max(16, int(round(total_epochs * 0.10)))
        args.late_stable_start = max(120, int(round(total_epochs * 0.68)))
        args.late_stable_ramp_epochs = max(30, int(round(total_epochs * 0.18)))
        args.late_adv_min_scale = 0.70
        args.late_cons_min_scale = 0.45
        args.late_cls_aux_min_scale = 0.25
        args.late_reg_aux_min_scale = 0.25
        args.late_joint_inv_min_scale = 0.08
        args.late_kl_min_scale = 0.16
        args.late_group_ce_min_scale = 0.80
        args.late_aug_min_scale = 0.20
        set_pa_weights(args, cls_pa=0.20, joint_inv=0.06, imp_inv=0.00, pa_kl=0.02, pa_reg=0.10, pa_select=0.0, pa_mono=0.0)
        args.exp_desc = "S3 RX-robust no-DAC: PA-only aux + hard-domain CE for weak unseen receivers"
    elif g == "s4_rxrobust_full":
        total_epochs = int(max(1, getattr(args, "epochs", 100)))
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.12
        args.aug_p_dac = 0.08
        args.lambda_adv = 0.45
        args.lambda_cons = 0.08
        args.lambda_group_ce = 0.08
        args.group_ce_top_frac = 0.35
        args.group_ce_min_domains = 4
        args.lambda_cls_dac = 0.03
        args.lambda_dac_reg = 0.06
        args.stage1_epochs = max(16, int(round(total_epochs * 0.08)))
        args.stage2_epochs = max(args.stage1_epochs + 40, int(round(total_epochs * 0.40)))
        args.stage3_ramp_epochs = max(16, int(round(total_epochs * 0.10)))
        args.late_stable_start = max(120, int(round(total_epochs * 0.68)))
        args.late_stable_ramp_epochs = max(30, int(round(total_epochs * 0.18)))
        args.late_adv_min_scale = 0.70
        args.late_cons_min_scale = 0.45
        args.late_cls_aux_min_scale = 0.25
        args.late_reg_aux_min_scale = 0.25
        args.late_joint_inv_min_scale = 0.08
        args.late_kl_min_scale = 0.16
        args.late_group_ce_min_scale = 0.80
        args.late_aug_min_scale = 0.20
        set_pa_weights(args, cls_pa=0.18, joint_inv=0.06, imp_inv=0.00, pa_kl=0.02, pa_reg=0.10, pa_select=0.0, pa_mono=0.0)
        args.lambda_cross_zero = 0.0
        args.exp_desc = "S4 RX-robust full/no-stats friendly: weak DAC/PA aux + hard-domain CE for low RX groups"
    else:
        raise ValueError(f"Internal exp_group dispatch failure: {g}")

    if not args.enable_pa_aux or not args.enable_dac_aux:
        args.lambda_cross_zero = 0.0

    return args


def apply_slim_ablation_preset(args):
    g = str(getattr(args, "slim_group", "none") or "none").lower().strip()
    table = {
        "none": {
            "desc": "不额外覆盖结构预设，完全使用手动配置。",
        },
        "balanced": {
            "model_variant": "lite_c",
            "branch_ablation": "none",
            "exp_group": "s4_stagewise_full_dual",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "默认推荐：Lite-C 全分支，兼顾精度、参数量和推理延迟。",
        },
        "no_dac": {
            "model_variant": "lite_c",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stagewise_pa_focus",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "优先减时延：去掉 DAC 分支，保留 PA 分支与频域摘要。",
        },
        "no_stats": {
            "model_variant": "lite_c",
            "branch_ablation": "no_stats",
            "exp_group": "s4_stagewise_full_dual",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "轻量裁剪：保留频域卷积，移除频谱统计投影，主要测试小幅时延收益。",
        },
        "lite_b": {
            "model_variant": "lite_b",
            "branch_ablation": "none",
            "exp_group": "s4_stagewise_full_dual",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "参数优先：Lite-B 全分支，适合先测结构压缩对精度的影响。",
        },
        "lite_b_no_dac": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stagewise_pa_focus",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "推荐第二梯队：Lite-B + 去 DAC，进一步压缩参数并降低推理时延。",
        },
        "lite_d_no_dac": {
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stagewise_pa_focus",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "mixstyle_p": 0.15,
            "desc": "更激进的小模型：Lite-D + 去 DAC，适合主力瘦身实验。",
        },
        "lite_e_time_only": {
            "model_variant": "lite_e",
            "branch_ablation": "time_only",
            "exp_group": "s1_core_only",
            "use_mixstyle": False,
            "desc": "极限小模型：Lite-E + 仅时间分支，用来测最小参数/最低时延边界。",
        },
    }
    table.update({
        "balanced_stable": {
            "model_variant": "lite_c",
            "branch_ablation": "none",
            "exp_group": "s4_late_stable_full",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "Lite-C full branches with reduced auxiliary pressure and late-stage loss decay.",
        },
        "lite_b_no_dac_stable": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stable_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "Recommended stable deployment candidate: Lite-B + structural no-DAC.",
        },
        "lite_d_no_dac_stable": {
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stable_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "mixstyle_p": 0.2,
            "desc": "Compact stable candidate: Lite-D + structural no-DAC.",
        },
        "rxrobust_balanced": {
            "model_variant": "lite_c",
            "branch_ablation": "none",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s4_rxrobust_full",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 120,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.10,
            "mixstyle_late_min_strength": 0.45,
            "desc": "Lite-C full branches with hard-domain CE for weak receiver groups.",
        },
        "rxrobust_no_stats": {
            "model_variant": "lite_c",
            "branch_ablation": "no_stats",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s4_rxrobust_full",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 90,
            "mixstyle_late_ramp_epochs": 35,
            "mixstyle_late_min_p": 0.05,
            "mixstyle_late_min_strength": 0.30,
            "desc": "Best low-RX direction from 4.26 logs: no handcrafted stats + hard-domain CE.",
        },
        "rxrobust_lite_b_no_dac": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 120,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.08,
            "mixstyle_late_min_strength": 0.40,
            "desc": "Lite-B no-DAC with hard-domain CE and delayed late stabilization.",
        },
        "rxrobust_lite_d_no_dac": {
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.20,
            "mixstyle_late_start": 110,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.06,
            "mixstyle_late_min_strength": 0.35,
            "desc": "Lite-D no-DAC compact hard-domain CE candidate.",
        },
        "rxrobust_no_dac_no_stats": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac,no_stats",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 90,
            "mixstyle_late_ramp_epochs": 30,
            "mixstyle_late_min_p": 0.04,
            "mixstyle_late_min_strength": 0.30,
            "desc": "Tests whether removing DAC and stats together helps weak RX while staying compact.",
        },
        "rxrobust_lite_b_no_dac_refined": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 120,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.08,
            "mixstyle_late_min_strength": 0.40,
            "desc": "R05 refined default: best 4.27 deployment route with late MixStyle annealing.",
        },
        "rxrobust_lite_b_no_dac_mix015": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.65,
            "mixstyle_p": 0.15,
            "mixstyle_late_start": 110,
            "mixstyle_late_ramp_epochs": 35,
            "mixstyle_late_min_p": 0.05,
            "mixstyle_late_min_strength": 0.35,
            "desc": "R05 conservative MixStyle: lower p/strength for no-stats-sensitive domains.",
        },
        "rxrobust_lite_b_no_dac_domain020": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "domain_enhancer_strength": 0.20,
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 120,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.08,
            "mixstyle_late_min_strength": 0.40,
            "desc": "R05 with weaker RCN enhancer injection; tests over-domainization.",
        },
        "rxrobust_lite_d_no_dac_refined": {
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.70,
            "mixstyle_p": 0.18,
            "mixstyle_late_start": 110,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.05,
            "mixstyle_late_min_strength": 0.32,
            "desc": "R06 refined compact route with gentler MixStyle for lower latency models.",
        },
    })
    for name, group_ce, desc in [
        ("rxrobust_lite_b_no_dac_gce006", 0.06, "R05 refined with weaker hard-domain CE."),
        ("rxrobust_lite_b_no_dac_gce014", 0.14, "R05 refined with stronger hard-domain CE."),
    ]:
        cfg = dict(table["rxrobust_lite_b_no_dac_refined"])
        cfg["lambda_group_ce"] = float(group_ce)
        cfg["desc"] = desc
        table[name] = cfg
    if g not in table:
        valid = ", ".join(sorted(table.keys()))
        raise ValueError(f"Unknown slim_group={g}. Valid values: {valid}")
    cfg = table[g]
    for key, value in cfg.items():
        if key == "desc":
            continue
        setattr(args, key, value)
    args.slim_group = g
    args.slim_desc = cfg.get("desc", "")
    return args


def apply_slim_post_preset_overrides(args):
    """Reapply slim-group values that intentionally override exp_group defaults."""
    g = str(getattr(args, "slim_group", "none") or "none").lower().strip()
    if g == "rxrobust_lite_b_no_dac_gce006":
        args.lambda_group_ce = 0.06
    elif g == "rxrobust_lite_b_no_dac_gce014":
        args.lambda_group_ce = 0.14
    return args


def apply_model_variant_training_defaults(args):
    variant = str(getattr(args, "model_variant", "base") or "base").lower().strip()
    args.lambda_probe = 0.0
    args.lambda_pa_imp_inv = 0.0
    args.lambda_cross_zero = 0.0
    args.lambda_dac_select = 0.0
    args.lambda_pa_select = 0.0
    args.lambda_dac_mono = 0.0
    args.lambda_pa_mono = 0.0
    if variant == "lite_c":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-C streamlined trainer"
    elif variant == "lite_d":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-D compact trunk"
    elif variant == "lite_e":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-E tiny trunk"
    elif variant == "lite_f":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-F CEN31 distill balanced student"
    elif variant == "lite_g":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-G CEN31 distill low-latency student"
    elif variant == "lite_h":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-H CEN31 distill time-only nano student"
    return args


def align_training_with_branch_ablation(args):
    ablated = parse_branch_ablation_flags(getattr(args, "branch_ablation", "none"))
    notes = []
    if "no_dac" in ablated:
        zero_dac_path(args)
        notes.append("no_dac->disable_dac_aux")
    if "no_pa" in ablated:
        zero_pa_path(args)
        notes.append("no_pa->disable_pa_aux")
    if "no_time" in ablated:
        args.use_mixstyle = False
        notes.append("no_time->disable_mixstyle")
    if notes:
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | " + ",".join(notes)
    return args

