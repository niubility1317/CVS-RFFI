# Phase3 原型骨干冻结评估恢复（2026-10-31）

新 run `20261031-phase3-body-prototype-s2026092701-r02` 只预测和评分，精确复用 r01 已完成的 SOURCE/TARGET_PCD 两份模型，不新增神经更新、不更改科学配方或依据目标成绩重训。原 r01 以 FAILED_PRESERVED 保留，尚无性能结论；两臂各444更新的训练代码仍是 `a33612e978948da25bd733bcb9edba3ac9d74e9a`，不能将新执行 commit 写成旧权重的训练 commit。

原故障发生在第一次学生加载、评估 smoke/源密度拟合/V校准/novel与Q预测之前。`load_trained_model` 将全部 `named_buffers()` 当作 state_dict 必含键，但 response.bank_imag/bank_real/tone_bins 和三 view 的 window_order 是 nonpersistent buffer，不保存到 checkpoint。远端只读实际两个权重确认：缺失键恰好为这6项，持久 buffers 与 Sinc 差异均为0，Sinc48参数完整。证据保留在 r01 `load_failure_diagnosis.json` 与完整 predict traceback；无query/truth/optimizer读取或更新。

一次针对性修复只比较进入 state_dict 的持久 named buffers 和 Sinc 参数；完整 strict load、schema、实际祖先、配置和原模型保护继续执行。本地新增 nonpersistent buffer 重载回归先失败，再通过；持久 buffer/Sinc篡改仍拒绝。原 888 步训练与其模型、日志、元数据不改写，r02只从原训练路径加载。

r02 显式区分执行身份与训练身份：eval/density/calibration/prediction/score 使用 r02及新commit；所有学生/全局训练元数据保留 r01及a336。runtime 仍是原14字段、train9/eval6，train_config 是已完成 r01输入，launcher禁止新训练命令，实际只运行 predict→CPU score。失败后未在原 root 重启或覆盖。

完整方法、源数据、U384、未来Q864 clean/star、三DURA六列、阈值36值、原4af对照、宏指标和科学门槛全部保持[原计划](PAMF_PHASE3_BODY_PROTOTYPE_PLAN_20261031.md)。SOURCE/TARGET两原模型均冻结再拟合各源GMM/四组源V阈值，所有预测完成后才独立三truth评分，Utruth0。原训练父signal12818/学生454656/G384及888更新只作为继承成本；本r02新optimizer0，实际评估signal90352/F0与G1728另列，不重复记训练。原失败加载和只读诊断的构造/IO开销披露，不冒算为新增IQ。

唯一 owner root；新 run/release/output 与原路径互斥。新commit、定点复审、一次新root/preflight、单archive SHA比较/compile和实际checkpoint no-query smoke 后运行；不为未改代码重复全审。仍使用N607剩余显存≥4096MiB、一GPUworker顺序预测、每卡新增≤2，不干预其他owner；CPUscore CUDAempty。技术停止规则和完整goal ACTIVE保持。相同加载故障在本次修复后再现则保存证据并停止盲目重发，不做第二轮猜测修补。
