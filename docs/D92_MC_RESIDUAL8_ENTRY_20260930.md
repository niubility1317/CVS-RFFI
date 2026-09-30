# MC-Residual8 support 入口与独立摘要

日期：2026-09-30。实现状态：`IMPLEMENTED_SYNTHETIC_ENTRY_TESTS_VERIFIED`。本 agent 完成 Python AST 语法检查；root 已串行执行新 evaluator/summary 的全部 10 个合成测试，10/10 PASS，耗时 17.62 s。发布和 launch 由 root 执行；本文不表示实际 pilot 已运行或完成。

## 固定评价协议

依据 [冻结设计](D92_JOINT_AFTER_PROTOTYPE_DESIGN_20260930.md)，唯一主候选为 `R_MC_seq`，对照为 `R0` 和 `R_MC_reset_init`。Phase1 固定，使用同一个 160 parent 合法 support pilot：旧类 6 个，K=1/5/10/20，新增类数=0/2/5/10/20；保留物理三折 OOF 和全部 one-shot proxy anchor。没有 query 输入或选参 API。

`R0` 是原 LocalRidge；B0 不是地面 A，A 与 B−A 均为 N/A。报告独立重算 B0→B、B→C 旧类、C 新类、H、新旧绝对差、旧类列变化和全类竞争损失。所有 C 评分逐样本面对全部注册类；不使用真实类别数量配额或全局重排。OOF 是 support 诊断，不能称为 query 结果。摘要输出完整 K×新增类数及模型/接收机场景分层，seq 始终是预声明主线。

B 每折 R0 初始头产生教师；C 每折教师使用冻结 B 的 U/V，仅在对应旧 inner-train 上拟合。seq/reset_init 共用同一个 C 准备与教师；reset_init 将自己的初始点与 proximal anchor 设为 U=0、V=固定 DCT V0，并按自己的初始保持风险定义上限，仍保留 B 教师。Nnew=0 直接复用 B，不额外拟合 C。true K1 只做数值诊断；proxy trainK1 精确等于 R0，不声称 K1 收益。

## 核心接口和计数

模块：`cvsrffi.d92_margin_constrained_residual8_local_ridge`。

- `prepare_mc_residual8_training` 接收五个原特征块、support 标签/物理 ID、注册类/旧类、可选 inherited/context/log_callback/state_callback。
- `fit_mc_residual8_local_ridge(prepared, mode='B'|'C_seq'|'C_reset_init', baseline_state=..., log_callback=..., state_callback=...)` 返回不可变 U/V 状态及 `score/predict/audit_dict`。
- `state_callback(key, arrays)` 保存完整数值数组并返回 JSON 引用；没有回调时 core 的 `state_records()` 可提供内存记录。正式 evaluator 始终使用外置归档。

固定事件为 `MC_INNER_PREPARED`、`MC_INITIAL`、`MC_GRADIENT`、`MC_TRIAL`、`MC_STEP`、`MC_FIT`。outer proxy anchor 用 `outer_trial`，core 回溯试探用 `trial`，避免覆盖。所有数值/布尔型经过严格 native JSON 转换；非有限、未知类型或不支持的数组精度报错，不使用 `default=str`。

固定结构计数为 episodes=160、trueK1=40、OOF=120、proxy anchors=1400、sequence paths=1760、baseline heads=3168、mc preparations=3168、mc states=4576、额外诊断拟合=0。`trained_mc_stage_count` 只计至少接受一次更新的阶段；零更新有信息阶段仍支付初始前向/反向成本。

实际总头为 baseline + preparation.inner + preparation.teacher + stage.inner + stage.final；分解同样核算。B 准备的 `initial_inner_head_fit_count/initial_inner_factorization_count` 是已计入 preparation.inner 的解释分项，不能再加一次。B 初始目标逻辑评价记 1，复用准备缓存而不新增头。C 教师只计一次，seq/reset 的初始 student 头分别计费。

动态计数保留初始目标、反向、接受/拒绝/尝试 trial、尝试迭代、MC forward、prepared distance、teacher score 与最终 score。`derivative_triangular_solve_count` 必须等于 `task_derivative_triangular_solve_count + keep_derivative_triangular_solve_count`。真实 head/factor 次数来自每折 audit，退化零核/单类不猜成满额分解。完整 pilot 的 head 上界为 42336、伴随三角上界为 44928，均不是实测值。

## 无损参数轨迹和日志

11776 个 float64 参数不转成反复复制的 JSON 数组。每次初始化、梯度、试探和最终状态写入独立 `state_arrays/00000000.npz` 序列，使用排他创建。accepted step 引用对应 trial 文件。梯度文件含 U/V、总梯度、保持梯度和投影方向；final 文件含最后接受的 U/V 及 anchor；教师文件含完整 teacher_scores、teacher_q 和 held_old_mask。所有坐标保留，没有抽样或截断。

`state_manifest.json` 列每个文件的逻辑 key、阶段/parent 命名空间、路径、实际文件字节、数组 shape/dtype/nbytes、norm/min/max 摘要及归档耗时，并按 B/C 准备和各拟合阶段汇总数量/字节。没有成员 SHA、receipt 链或额外审批。大体积 NPZ 保留原产物路径，不要求进入 Git 或复制进报告。

正式输出同时保留 `training.log` 的 CVS 风格详细文本、`training_events.jsonl` 全结构事件、`training_events_compact.jsonl/.csv`、`fit_stages.jsonl/.csv`、完整 parent trace 和紧凑 parent JSONL/CSV。文本逐步包含任务/保持/proximal/总损失、梯度、方向约束、步长、三项接受条件、拒绝原因和实际成本。source validation 为 N/A，原因 `SOURCE_ACCESS_FORBIDDEN`。`artifact_manifest.json` 列实际输出文件及字节，排除该清单自身；失败保留已写 NPZ 和不完整 state 清单，不覆盖或自动重试。

归档字节和时间属于训练日志成本，不能称为部署传输。94208 B 仅为 U/V 参数，不是完整分类器大小；实际 head、原/适配 support、raw/标签继承、teacher/cache、RSS 等按各自测量口径保留。未测的 GPU、部署包、增量传输及独立 kernel 时间记 N/A。

## 摘要独立核对

摘要先核实完整四 row、实际 source/checkpoint/cache bindings、固定配置和禁止访问字段，再打开 support held score。它不只信完成 marker，也不重新拟合。

逐阶段核对折内物理隔离、教师旧类头来源、教师 q=clip(true−max_wrong,0,1)、全部 NPZ 清单/可加载性/精度/有限值/字节与数值摘要。加载完整 U/V/梯度坐标后独立重算：两 Frobenius 球投影、总梯度归一化、保持半空间方向投影、真实位移、物理 proximal、先跨折合并再计算任务/旧类 RMS、固定物理松弛及保持上限。

每个 trial 必须同时满足 Armijo、总目标不增和真实保持风险上限；球投影后 g·Δ 可能为正，因此不能只凭方向性质推断目标不增。摘要核对 `min(L_before, armijo_rhs)+tol`、三个 pass flag、拒绝原因、接受 step 对应文件、停止理由以及最终状态等于最后接受缓存。拒绝 trial 的参数不会成为最终状态；不按外层结果选择最好步。

full/compact 事件须与 parent solver trace 一致；实际准备、两个伴随、头/分解和最终评分计数独立累加后再与 row/run marker 交叉核对。归档文件不得缺失、额外或未引用。报告只保留 scalar/ref，全坐标仍指向原 NPZ。

## 合成验证与依赖

新增测试覆盖真实 writer 的 NumPy 标签/类 ID/布尔边界、full/compact/text/CSV/NPZ、共享 B/C 教师与 N0/K1、独立参数投影、teacher/risk/keep limit/计数/停止/接受条件/事件篡改、NPZ 内容与路径清单篡改，以及 incomplete pilot 在打开 held score/归档前停止。只使用合成数据。

建议 root 单次执行：

```text
python -m pytest -q tests/test_evaluate_d92_mc_residual8_probe.py tests/test_summarize_d92_mc_residual8_probe.py
```

新入口沿用 registration diagnostic、branch support/local-ridge evaluator 和对应 summary helpers；新 core 依赖 branch/local/channel/prototype 数值模块及 NumPy/SciPy。没有导入旧未跟踪 residual 模块，没有新增 orchestration 工具；发布须包含这些既有辅助依赖。

root 实际验证命令即上述两个 test 文件，结果为 10 passed in 17.62s。证据：[stdout](E:/type10-7/.codex_tmp/pytest_utf8_1790771643704337400.stdout)、[stderr](E:/type10-7/.codex_tmp/pytest_utf8_1790771643704337400.stderr)。没有新增失败或测试后的代码变更；无需重复该检查。core/driver 验证由 root 另行汇总，本文只登记本入口的实际结果。
