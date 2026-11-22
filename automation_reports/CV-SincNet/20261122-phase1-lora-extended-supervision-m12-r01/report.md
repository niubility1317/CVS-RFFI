# LoRa 延长纯监督与晚启动半监督对照

## 设计

原始监督第 70 至 84 轮，三个种子的源 clean 验证分别增加 0.82、0.73 和 0.66 个百分点。相同源划分上继续纯监督 28 轮，再增加 2.86、1.94 和 2.64 个百分点。三个标注训练准确率在第 112 轮分别为 88.50%、77.89%、85.43%。这组源曲线支持再延长一期纯监督。

沿用完整 LoRa 的 2800 L、25200 盲 U 和 12000 V。每个种子从同源第 112 轮 `continued_sup` 接续 28 轮 × 88 步，使用原配方的标注 clean CE 和 0.35 practical CE。学习率从既有续训的 `1.5e-4` 降至 `6e-5`，余弦衰减至 `1.5e-5`。保留 AdamW 重新初始化。第 140 轮固定权重，再从该权重各接续相同的 28 轮 × 88 步。三路共享 L 顺序、practical 配方、优化器和学习率日程。只有普通 PL 和自适应 Tri 使用 U。主要对照是第 168 轮三路之间的比较。历史第 84 轮启动半监督也提供参照，其总预算和学习率有别。

自适应 Tri 在半监督前重新计算第 140 轮教师的源 V 置信度、视图一致性和源验证精确率，沿用已冻结留源阈值规则。硬 CE 系数仍为 `min(0.40, 0.08 × 合格类别数)`，沿用每批每类最多三条、跨 RX 供体和同包一致性。普通 PL 继续采用 0.95 的在线硬伪标签。报告监督第 112 至 140 轮及三路末轮的 L/V 准确率、交叉熵和学习率，并按类别、种子、首轮和整段报告 U 采用率及离线正确率。固定读出全部九个末轮识别器，覆盖同一批 30000 条目标包的 clean 与六个 practical 视图。固定全矩阵后独立评分。

## 发布

N607 普通账户，沿用已验证的 LoRa 环境。CWD 为 `CV-SincNet/releases/20261122-phase1-lora-extended-supervision-m12-r01-release`，产物为 `CV-SincNet/runs/20261122-phase1-lora-extended-supervision-m12-r01`，唯一发布人为当前 Codex 任务。发布用 `python -m experiments.pamf_lora_main.full_adaptive_publish`，传入原数据、原冷启动监督和前次同预算 release 的 receipt（`--data-submit`、`--supervised-submit`、`--prior-submit`）。发布器记录实际 Git commit 和进程。监督阶段权重及日志在 `warmup/F`，三路权重和日志在 `F`，源教师门槛在 `policy`，盲结果在 `predictions`。GPU0–6，单卡最多一个本任务进程，本任务同时最多四个。正式评分用 `python -m experiments.pamf_lora_main.full_adaptive_score` 读同一 release handle，输出 `scores.json` 与 `u_offline_scores.json`。

训练只使用已登记源数据。读取先前纯监督权重时核对源契约、种子、完整预算、零 U 记录和冻结状态。训练样本绑定异常、权重来源异常、迭代数错误、非有限数以及预测不完整属于技术失败。流程保留已有产物，按技术失败停止；指标低不改变已固定矩阵。
