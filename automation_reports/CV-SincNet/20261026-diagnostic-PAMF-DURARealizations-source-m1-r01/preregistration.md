# 同包多信道实现：冻结源L诊断

PLANNED。只用原source L6300、原seed2026092701 E30默认头和同一固定原F0 G-LU，训练/optimizer/U/V/目标读取均0。当前AsymmetricHG四条健康训练不变。该诊断回答恢复视图是否随信道随机实现变化，不是新半监督性能或新恢复器版本。

## 固定矩阵与权限

三个原源场景：lowSuburban、midUrban、lowUrban。每个L物理包、每场景生成固定H0并恢复G0；对完全相同H0再恢复一次作重复控制；另生成fresh1/fresh2/fresh3三个独立固定namespace的H/G。固定接收机日hardware session和参数，改变信道随机实现。F视图raw+3场景×4实现×2阶段(H/G)=25。全部实现保留，不根据L成绩挑随机种子或调整幅度。

固定namespace为历史`PAMF/restored_dose/U/fixed`，fresh为`PAMF/realization_diversity/L/fresh1`至`fresh3`。前者U仅历史命名空间字符，本run输入角色严格L_s，物理ID/y沿原可见L，不打开U数据或标签。原source_channel residual/post_sync/noEQ/reflect/SNR30、fs25e6/fc2462e6、channelSeed20261016421/hardwareSeed20261016423保持；沿原物理生成器normalize_input，不对G输出另做归一化。

G使用同一冻结版本和InferenceEngine(reuse=True,graph=True,native=False)、FP32/TF32off。重复控制的两次G调用各用相同H0的独立clone，输出clone后比较，避免复用buffer覆盖。只查询一个F seed01默认头；共享来源加载函数会载入三原E30，其他两个不做正式L视图前向，不冒充三种子结果。原E20/E30来源及G契约沿已有严格检查，不加载任何研究F50或目标选模父权重。

## 指标与完成条件

先真实E30/defaulthead/G的L2 no-query smoke，再完整L循环。记录同H重复G的逐包最大绝对误差及精确相同数；freshG相对fixedG的relativeL2、real cosine、RMS比例、相对raw差异、H差异与G最大绝对差。每个实现均报告默认头H/G accuracy与相对固定实现的净纠错，另报同H到同G的纠正/新增错；按overall/TX/RX/day分层。几何分布给出mean/min/p10/median/p90/max。源L识别是训练域代理，不能证明TX细节物理真实性、U伪标签正确率或目标泛化。

完整25×6300 F预测、3×3×6300×6 geometry及3×6300重复误差固定后CPU独立重算375 viewcells、180 H→G cells、45 repeatcells、810 geometrycells。模型/G参数及全部registered buffers不变，原L完整覆盖6300/唯一IDs；U/V/目标/optimizer/另训头均0。没有性能晋级门槛或源最好实现选择，不自动构建训练U缓存、启动新的F训练或访问目标。

逻辑评估F157500包、G94500包，另有真实L2 smoke各4包；不把graph内部warmup当独立样本，实际walltime包含加载、CPU信道生成和推断。私有输出为预测/几何NPZ，公开仅aggregate，无IQ缓存或新权重。固定源F/G都来自既有训练，新增训练轮次0。

## 环境、资源与停止

N607普通账户、既有Python/Torch2.1、CPU2线程，无安装。CWD PROJECT/releases/20261026-diagnostic-PAMF-DURARealizations-source-release-r01；命令PYTHON -u -m experiments.pamf_channel_pl.realization_diversity_source dispatch --config该release/runtime.json；不可覆盖runroot PROJECT/runs/20261026-diagnostic-PAMF-DURARealizations-source-m1-r01。

GPU0..7择一，启动时统计全局CUDA+pending，现有进程总数<2且free≥10000MiB才准入；按最少负载/最多空闲选择，保护现有四路训练与其他owner。一个archive一次本地/远端SHA比较、remote compile和独立PID/CWD/argv/GPU/loggrowth；若已结束以完整产物独立核实，不凭退出码判断成功。首次技术/协议/非有限/状态/覆盖错误只退出本单进程、保留产物，无自动重提或停止其他任务。

3项聚焦检查/local compile已通过；唯一一次独立P0/P1检查PASS，无P0/P1。代码和aggregate进Git，runtime/log/NPZ保持私有。原同候选三个学生种子的raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%目标不变，当前尚未达成。
