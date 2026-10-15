# SIG_TIME 原 EMA 导出：复用既有三种子 E20

状态 PLANNED。这是独立性能导出比较，不称新颖训练机制，也不救援LOCKED的固定主资格。用户新增报告优先比较同轨迹FINAL/EMA/TAIL_AVG；当前source RX1两已完成条件EMA分别净修正9/12个错误，均只有1seed/1RX，不当正式三seed收益。完整tail缺失时不为平均单独重训。本轮复用 `20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r02` 的三个固定BASE_SSL，不按target挑父；原实际LS0、SIG_TIME/CENTER25K、13L+7U/defaultcos，source-only scratch20/4440steps的own EMA decay=.999。当前四RX源折与正式全5RX契约不同，绝不加载其权重。

新训练为0epoch/0optimizer steps。所有feature、默认W、norm和gain一起取原final checkpoint内`ema_model`，原学生FINAL预测/评分保留复用；不重训/替换头，不跨seed平均，不改变teacher。模型元数据和完整继承来源通过已有source_proof/validate_checkpoint在反序列化前核对。原训练成本3×20/4440保持单列。

科学规则在新source数据前固定：先seed2026092701在原完整V27000比较FINAL与EMA（同一批V包、两模型无梯度eval）。EMA错误数≤FINAL、meanCE≤FINAL，且至少错误数严格减少或CE减少>1e-8；否则SOURCE_FALSIFIED、0target。通过后同时冻结三个预定义ownEMA，再逐row自动clean168000预测、独立新CPU truth-last评分。后两seed的source结果完整报告，不用于挑种子或改参数；不存在decay/窗口/parent scan。

实际source L6300/U56700隐藏/V27000、原source RX1/3/4/6/8和days1/2/3/六TX不变；V仅健康/资格，不用于梯度或模型统计拟合。source数据构造保持source-screen-only，不建target；已有VALIDATED_ONCE句柄复用。独立scorer只在当前row prediction完整后连接truth，原学生correctness IDs必须与当前EMA逐项相同才能配对，完整overall/RX/TX/day共18指标/row；3seed新504000决定/54指标，旧504000决定/54指标复用。每row source27000个独立V读取、54000实际模型前向+2zero-smoke，predict原入口另2zero-smoke，0U读取/新训练/拟合头。

目标仍是默认头clean三seed均值84%，本通用导出措施本身不完成创新机制目标。报告accuracy/sampleSD/MacroF1/最差RX/day、逐seed配对与修正/新增错误，旧负结果和最差RX正迹象保留。基准已暴露，不称独立盲测，结果不回流当前健康三条件训练、结构/门槛/重跑。只测clean，不扩星地或Phase2。

Git commit固定实际代码/config；owner `codex/root/SIGTIME-ownEMA-export-20261015`。CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_ownEMA_export_clean_20261015_r01`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_sigtime_ema_export20/runner.py experiments/cvs_sigtime_ema_export20/configs/matrix.json`；不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-ownEMA-export-clean-manysig-m3-r01`。普通N607/CVS-RFFI Python，单pipeline worker/一GPU，按最少现有进程后最大空闲选择，总所有owner≤3，当前健康dropout任务不干预。一次preflight、单archive本远SHA比较、remotecompile、真实checkpoint source-only smoke PASS后直接资格/冻结/逐row预测评分；唯一下一run P0/P1审查，启动后独立PID/CWD/cmd/GPU/log-growth。技术停止规则和完整矩阵见experiment.json，无额外审批或校验链。

本地11项聚焦检查及compile PASS。唯一下一run审查发现source builder漏传device，已最小修复；实际接口签名的无数据回归修复前失败、修复后通过，独立定向复核PASS。源码/config提交后按预登记直接发布，无额外审批。
