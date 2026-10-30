# PAMF 多中心密度候选：源侧预登记

Run `20261030-phase3-star-density-s2026092701-r01`，实际执行代码 `b8163a03e8d3d299127c7914272a2d866f92c89a`，唯一owner `codex-phase3-density-20261030`。方法固定于 `docs/PAMF_PHASE3_DENSITY_PLAN_20261030.md`，矩阵唯一维护于 experiment.json：一个原模型seed、known_only匹配无OE控制、唯一主density_ratio，均使用相同源L拟合的已知分布。背景capacity32/25Lloyd/seed2026103019、90已知域mode、共享协方差shrink.1、径向modevariance收缩160；概率质量/体积/尺度与白化行列式真实参与似然，不是旧8NN换名。

L6300（原六类各1050、每TX×5RX×3day70）提供已知真标签，OE6134只提供outside-six角色与公共RX/day。源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23；OE真实TX/K不进入算法。knownV27000仅冻结分布后固定known风险42/六阈值校准；novel3150（19-15/19-16/7-1各1050）在两套预测全固定后独立评分。Us/target U/Q/正式RX24-18不读；sourceRX已见、novel已有开发曝光，不称未见RX或新未知盲测。

全部数据沿已有G10MIX01/P1：600km/25MHz/2.45GHz/residual/post_sync/noEQ/refSNR30dB/reflect，receiver2027/scene2026103001/channel2026104001；不生成新IQ、不抽第二份信道、P0不是clean。原F0为Tri-SIG-TIME SSL E50/11100/default6head/232993，source split392005、原L6300/Us56700/V27000祖先不变。严格加载原F0后，两条合成G10无query smoke为首步，PASS立即继续源缓存拟合；原参数/buffers/默认身份保持，G不加载。

源特征复用protected-cache实际code7dd90671968ddf58aab5a5b302b4d56371c8b9c9中原F0 rawfused160；V/novel特征/defaultlogits复用local-support code84cdc012f7d9641b283c407f6048bfe4e6e21cf6的original_F0/predict。两来源的原F0/DURA与source角色句柄完全一致；不加载旧拒识raw_scores/scores/τ或source评分，local-support不作为训练父。fit/query同FP64单位化。新真实F前向0、合成F2、G0；原预训练和缓存成本保留。

分布参数只拟合L/OE；control先冻结且没有OE参数，主再读取OE并冻结。随后V只用既定known风险校准，两个τ全冻结后才读取novel特征。已知身份直接复用原F0默认头，与同P1 baseline逐条保持。只导出聚合o/W/mode均值/尺度/权重/六τ，不部署source逐行ID/feature/IQ。

独立CUDAempty/BLAS1 CPU scorer在全部两policy预测完整后唯一读取novel truth，给出宏AK/RU/H、CK、原F0损失、AUROC/OSCR、逐TX/RX/day/scene与配对。源晋级只为known42且整体source宏RU≥90%的早筛，其他切片完整诊断；它不代表已达到最终未见RX/逐主场景门槛。控制不能替主、结果不回流任何分布/阈值/seed/预算。通过后才另预登记跨RX星地开发验证，不自动扩多seed。完整无标签适应/发现/未来注册目标仍ACTIVE，AN/HOW均N/A。

- 环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-density-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_density_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-density-s2026092701-r01-release/runtime.json`。
- 独占runroot `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-density-s2026092701-r01`；一个GPU worker仅原F0 smoke与随后源统计/预测，actualfree≥4096MiB、每卡新增own≤2；独立CPU评分。其他用户停止/修改/重启0。
- 聚焦实际验证：{"core": {"unique_tests_verified": 7, "subtests": 13, "first_invocation_passed": 6, "first_invocation_seconds": 1.17, "targeted_floating_sum_assertion_recheck_passed": 1, "targeted_seconds": 0.58, "production_algorithm_changed_for_assertion": false}, "run_score": {"tests": 11, "subtests": 19, "seconds": 6.34, "legacy_np_trapz_deprecation_warnings": 528}, "launcher": {"tests": 8, "seconds": 3.01}, "AST_files": 8, "real_IQ_weights_truth_read_for_local_tests": 0}；八AST及真实runtime校验通过，唯一直接P0/P1审查PASS。一次N607预检已PASS，只读metadata/文件/显存、IQ/private truth值0。
- 一个release archive、一次local/remote SHA比较与远端compile；实际首步checkpoint smoke、启动后PID/CWD/argv/GPU/log及终态独立读回。统计拟合25次聚类不冒称神经更新或训练epoch。
- 预期launch/pipeline/config/log、smoke、聚合state/fittrace、六阈值、两policy source预测、独立score/results/complete及observations。数据/权重/原始log/预测不入Git。
- 非法来源/角色、非有限、碰撞或技术失败保留；不对不完整预测评分，不自动重试或调参，UNKNOWN先核原PID/root；不操作其他owner。此处仅预登记，尚无实际新性能结果。
