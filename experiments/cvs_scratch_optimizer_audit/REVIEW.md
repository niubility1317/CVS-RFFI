# 独立 P0/P1 发布正确性审查

结论：**PASS；发现的 1 项 P1 已修复并完成针对性回归，未关闭 P0/P1 为 0。**

审查对象为本包 DESIGN、design/run/geometry/checks/preflight/dispatch/publish/collect，以及 `20261012-phase1-scratch-optimizer-audit-manysig-m3-r01` 登记。仅审查会导致诊断执行错误、越权、无法产出或覆盖的问题；不以性能推测增设门槛。

## 已关闭的 P1

原 `_batch` 在调用 `model.groups` 冻结无关分支之前收集 autograd 参数。独立 CPU 原生构建复现：初始清单有 183 个参数，分组后仅 67 个身份参数可训练，其余 116 个已被冻结。正式 fresh checkpoint 路径因此会将 requires_grad=False 参数送入求导，或在临时优化器完整集合核验处失败。旧集成 fixture 提前建立过优化器，未覆盖该顺序。

实现已改为先执行 `model.groups`，再建立身份参数及浅/中/其余分组。新增 fresh build＋strict state load 后直接调用 batch 的六模型 CUDA 回归已读回 PASS：`local_artifacts/optimizer_audit_fresh_integration_cuda.json` 的 `fresh_model_optimizer_setup_regression=true`，六批、48 次临时优化器调用，原模型参数/buffer/grad/mode 保持、持久更新为 0。主线程将本证据同步至登记；无需重复未变的检查。

## 正确性核验

- 三个 worker 配置与登记逐项相等。固定 calibrated36 和 identity9 的 DELAY_JOINT/CAP_JOINT/DELAYCAP_JOINT 全三种子，共 18 个 final20/4440 checkpoint，不按目标排名选取。原配置、完整教师来源、source freeze、自身校准和已保存优化器状态通过既有来源校验加载。
- 只从合法源 L/U 构造目标；U 的 TX 真值隐藏，教师概率按原来源缓存匹配物理 ID。V 共反序列化后立即丢弃，无 V 前向、梯度、拟合或选择。参照 L 与当前主 L 不重叠；其原训练可见及可能与辅助 A/B 重叠的限制已披露。没有 target loader 或评分路径。
- CE/KL/style/原始 R/IDENTITY 梯度由当前原目标求导；FULL 为 IDENTITY 加 cap 标量乘原始 R，并与完整目标直接梯度核对。UNCAPPED 保留原始 R；STYLE_NEW 从加权 style 梯度中减去对应 clean CE 分量。固定诊断位置从 1332 开始，ramp 已为 1。复数几何使用共轭内积实部，范数使用绝对值平方，零范数余弦为 null。
- 每个目标/模式独立克隆全部优化器身份参数与状态，调用真实 `torch.optim.AdamW.step`；不向原参数分配 grad，不写回临时权重或优化器。saved_final 加载保存的组学习率、矩及计数；reset_peak 使用空状态和原登记峰值学习率。None 梯度保持 PyTorch 的跳过更新语义。实数/复数 AdamW 与手算坐标更新检查一致。
- 原模型参数、buffer、已有 grad、模式和私有随机流在诊断正常路径核对保持；原优化器字典在模型全部批次后比较未变。异常使该 fresh worker 失败并保留产物，不继续产生成功完成状态或改写原 checkpoint。
- 每模型 12 批、每批 2 模式×4 目标，共 216 批、1728 次临时 AdamW、0 次持久身份更新。每批实测 12 次模型前向、1144 条 L 读出、256 条 U 读出；正式总计 2592 次前向、247104 条 L、55296 条 U。预检额外 30 次前向、1180 条 L、256 条 U、8 次临时步单列。日志/CSV/JSONL、collector 和登记预算相符。
- 输出为源标量诊断及物理 ID，不保存新模型、优化器、IQ 或特征图。位移与参照梯度的内积明确是一阶方向导数，未冒充更新后实测 CE；保存/重置两模式学习率也不同，不把其差异单独归因于动量。
- source-only 登记明确不访问 query。三 worker 共用容量锁，每 GPU 至多 3 个总进程/预留，包含其他任务；独占 owner 和新 run 输出，拒绝覆盖，失败不自动重试，不干预健康旧任务。61 个运输路径均存在，发布/预检/调度入口绑定新包。

## 已读验证证据

- 登记 `evidence/optimizer_audit_core_cpu.json`、`optimizer_audit_core_cuda.json`：PASS，实数/复数原生 AdamW、None 梯度、源状态保持及复数几何。
- `optimizer_audit_integration_cuda.json`：原六模型集成 PASS；该旧 fixture 不能单独关闭上述 fresh 路径缺陷。
- `local_artifacts/optimizer_audit_fresh_integration_cuda.json`：修复后六模型直接诊断回归 PASS，逐项读回零持久更新和八种临时条件，关闭该 P1。
- 独立只读核对：三行登记配置、18 个固定来源、预算及运输入口一致。

审查者只进行本地代码/证据读取和一次 CPU 参数集合复现，仅创建本 REVIEW.md；未修改实现、使用 GPU、执行远端操作或读取目标成绩。此 PASS 是发布正确性结论，不代表正式源诊断已完成或性能目标已达成。

审查时间（UTC）：2026-10-12T00:13:18.282062+00:00
