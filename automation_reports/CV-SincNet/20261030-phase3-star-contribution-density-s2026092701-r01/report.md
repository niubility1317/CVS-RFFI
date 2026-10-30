# PAMF 融合前四贡献密度：源侧预登记

Run `20261030-phase3-star-contribution-density-s2026092701-r01`，实际执行代码 `edc28ad00baace5e4b9567fbf3391154bf6c34b4`，唯一owner `codex-phase3-contribution-density-20261030`。完整固定方法见 `docs/PAMF_PHASE3_CONTRIBUTION_DENSITY_PLAN_20261030.md`，矩阵唯一维护于experiment.json。一行/两policy：唯一主contribution640_density_ratio，固定控制fused160_density_ratio，控制精确复用父density codeb8163a03e8d3d299127c7914272a2d866f92c89a主model/六τ/预测，不能重fit/recal或按新NV换主。

原F0为完整Tri-SIG-TIME SSL E50/11100/default6head/232993；source split392005、原L6300/Us56700/V27000与DURA v1.0祖先保持。新主只读protected-cache code7dd90671968ddf58aab5a5b302b4d56371c8b9c9原F0四贡献，不加载R1/unused adapter。Source L6300旧六类各1050，OE6134只outside-six/publicRXday，无真实OE TX/K；knownV27000仅模型冻结后known风险42校准；novel3150三个设备各1050仅最终独立truth。源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。sourceRX/日期已见、novel历史曝光，不能称新RX或新未知盲测。Us/target U/Q/正式RX24-18新读取0。

全部输入沿已有固定G10 MIX01 P1：600km/25MHz/2.45GHz/residual/post_sync/noEQ/refSNR30dB/reflect、receiver2027/scene2026103001/channel2026104001，原单记录单received与DURA缓存不变；P0不是clean。没有重抽信道、clean/P0实值或G前向。

首步actual原F0 checkpoint syntheticG10两条无query smoke，同pass直接native hooks捕四贡献/原logits/CFO mask，finally清hooks、状态保持，PASS即继续。每分支FP64unit、zero分支保0、全zero报错，concat全局unit→FP32[640]；core统一FP64再归一化。主L/OE拟合D640收缩Gaussian mixture，90known/32background/25Lloyd/seed3019/covshrink.1，真实mode质量/体积/尺度及Jacobian进入densityratio。新D640与固定D160控制的方差收缩、自由度/存储/计算不同，不能声称纯单项融合因果或同成本。

主模型冻结后V27000仅前向/固定known42阈值；主τ冻结后才novel3150前向。真实新增F30150+synthetic2，G/NN/optimizer更新0。新pass默认身份/ID/metadata与控制逐条核对，父control τ/model/score不改变、不影响主拟合。仅保存聚合state，不部署source逐行物理bank。全部两policy预测完整后独立CUDAempty/BLAS1 scorer唯一读novel truth，重建statraw/margins并报宏AK/RU/H、CK/原F0损失、AUROC/OSCR及全部TX/RX/day/scene/配对。

源晋级=known42 AND 整体sourceRU≥90%，是早筛；其他source切片完整诊断，source通过不替代最终跨未见RX/逐主场景验收。控制不能替主，不按NV调分支/权重/容量/预算/τ。完整拒识、无标签适应/发现/未来匿名注册目标ACTIVE，AN/HOW均N/A。

- 环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-contribution-density-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_contribution_density_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-contribution-density-s2026092701-r01-release/runtime.json`。
- 独占runroot `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-contribution-density-s2026092701-r01`；一个GPU forwardworker、actualfree≥4096MiB、每卡新增own≤2，随后独立CPU评分；其他用户停止/修改/重启0。
- 实际focused验证：{"core": {"tests": 11, "subtests": 23, "seconds": 1.11, "includes_original160_regression": true}, "run_score": {"unique_tests_verified": 8, "subtests": 13, "first_invocation_passed": 7, "first_seconds": 10.6, "matrix_roundoff_targeted_regression_passed": 1, "targeted_seconds": 4.21, "legacy_np_trapz_deprecation_warnings": 432}, "launcher": {"tests": 8, "seconds": 2.92}, "AST_files": 9, "real_IQ_weights_truth_read_for_local_tests": 0}；9AST/runtime PASS，唯一直接P0/P1 review PASS。一次N607preflight PASS，仅metadata/文件/显存，IQ/private truth值0。
- 单release archive SHA比较/远端compile，首步实际checkpoint smoke后立即继续，独立PID/CWD/argv/GPU/log和最终产物核验；不增加成员hash/receipt/审批。
- 预期launch/pipeline/config/log、smoke、models/25步fit、τ/calibration、known/novel预测、独立score complete/results/observations；大体积特征/原log/权重不入Git。
- 非法来源/角色、非有限、身份错位、碰撞或技术失败保留，不评分不完整预测、不自动重试；UNKNOWN先核原PID/root。此处仅预登记，没有新性能结果。
