# PAMF 旧类与未知类联合邻域诊断预登记

Run `20261030-phase3-star-joint-geometry-s2026092701-r01`，实际执行代码 `3023c26b4bb2b2f74148303362bd615d854d18ff`；唯一 owner `codex-phase3-joint-geometry-20261030`。方法与单行矩阵见 experiment.json 和 `docs/PAMF_PHASE3_JOINT_GEOMETRY_PLAN_20261030.md`。

唯一比较为原 F0 unit160 与主 R1 protected160，固定同池 oldL6300＋novel3150、共9450既有物理记录。L六旧设备各1050，novel三个设备各1050，每设备×source RX1/3/4/6/8×day1/2/3各70；接收机1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。全部来自同一固定G10MIX01/P1，沿用600km/25MHz/2.45GHz/residual/post_sync/no additionalEQ/refSNR30dB/reflect；本轮不生成IQ。

Source RX已经在训练出现，旧L也参与过训练，source novel19-15/19-16/7-1已有历史开发曝光，不能称新RX/TX盲测。source角色pool是采集划分而非模型检测结果。V/OE/Us/target U/Q/正式RX24-18新读取0；未知TX/K/配额不用于选择近邻。

父R1为protected-open实际执行7dd90671968ddf58aab5a5b302b4d56371c8b9c9、2220parts步/native0；原Tri-SIG-TIME SSL F0 E50/11100/defaulthead/232993与DURA v1.0祖先匹配source contract392005。只严格加载R1完整新parts66177，核对实际原source_data_handle及缓存/预测身份；F0/G权重不加载。首步实际adapter checkpoint两条合成特征smoke，PASS即继续L6300映射；novel3150两表示逐值复用。真实/合成F前向均0，G0，更新0。

独立CPU先对两表示×联合9450或novel3150参考×all-other/crossRX/crossday/both四条件固定全部16套近邻。FP64 cosine/self exclusion/opaqueID稳定ties，不使用真TX/K。之后才读取L公开标签及唯一novel私有truth，报告9×9混淆、逐TX宏1NN、novel落known/同novelTX/异novelTX、known落novel、union相对novel-only配对，以及R1相对原F0配对。缺类/候选N/A，方差/秩仅描述性，不拟合阈值/模型/原型/聚类。

这些是同池源几何诊断，不是拒识R_U、未来A_N、无标签域适应或匿名注册成绩；eligiblefalse/finalfalse，全目标ACTIVE。主R1不按控制成绩替换。v3跨RX星地CK90/RU90、AK距原同输入闭集≤2pp及未来AN80/REG旧损≤2pp保持。

- 环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-joint-geometry-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_joint_geometry_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-joint-geometry-s2026092701-r01-release/runtime.json`。
- 独占 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-joint-geometry-s2026092701-r01`；一个新增GPU adapter worker，actualfree≥4096MiB、每卡≤2新增own；独立CUDAempty/BLAS1 CPU scorer；其他用户停止/修改/重启0。
- 聚焦export 5+19subtests/6.04s、scorer5/5.18s、launcher9+19subtests/3.38s通过。六AST/runtime校验通过；唯一直接P0/P1 review PASS；一次N607预检PASS（句柄/文件/显存，IQ/truth值0）。发布仅一个archive SHA比较/compile；启动后核PID/CWD/argv/GPU/log与实际smoke。
- 预期产物launch/pipeline/config/log、union.npz、smoke/export complete、score results/complete和observations；IQ/特征/权重/原日志不入Git。
- 非法来源/角色、非有限、碰撞或技术错误保留，不评分不完整导出，不自动重试；UNKNOWN先核原PID/root，外部任务保护。此处仅预登记，尚未宣称实际启动/性能。
