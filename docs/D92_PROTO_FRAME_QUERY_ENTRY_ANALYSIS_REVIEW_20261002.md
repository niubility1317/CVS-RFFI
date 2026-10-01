# ProtoFrame query producer 与 support analyzer 直接正确性审查

## 结论与范围

在本次限定源码范围内，未发现新的 P0/P1。审查对象仅为另一作者的 `tools/evaluate_d92_proto_frame_joint_benchmark.py`、`tools/analyze_d92_proto_frame_joint_probe.py` 及对应 literal synthetic tests。没有复审数学 core、primitives 或本作者的独立 query scorer，也没有新增发布门槛。

本次只读取源码和人工合成 fixture，没有读取真实配置、索引、交接、数据、模型、packet、cache、日志、预测或评分产物；没有运行数值测试、Conda、Git、SSH 或任何实验。结论限于下面列出的直接接口和保存状态契约，不代表真实输入已核实、query 已评分或识别性能已改善。

## Query producer

| 核查项 | 直接源码证据与结论 |
| --- | --- |
| 输入与来源 | `_inputs`（第 71 行）绑定显式 run/row/release、cache/capsule/checkpoint 与原生 Ground packet；冻结 center 仅构造固定几何。输出独占且不得位于原输入内。这里只使用已授权 received feature cache；不加载 encoder/checkpoint，不读取 query truth。 |
| 完整 trainK | `predict`（第 242 行）对每个 split 创建新的 `state_by_name`；只从 `support` 选择 B 的六旧类或 C 的全部注册类。query 只在两阶段拟合完成后进入 singleton 推断循环（第 364 行）。context 的 parentK、trainK 都是该完整 split 的 K。 |
| 实际 B→C 与 new0 | B 当次 fit 后，C preparation 的 `inherited` 直接取本 split B 对象；`_stage_check`（第 224 行）核对 `state.prior is actual_b` 和 final prior ref。new0 在 prepare/fit 前直接复用 B 对象与 audit，不产生 C 二次拟合。R0 的 new0 同样直接复用其 B。 |
| A 与 R0 | A 调用冻结 Ground native float32 六类头，保留原生列序和原生 ties；R0_B 与 R0_C 分别用当前完整旧/all support 独立拟合。没有把 R0 当成 A，也没有把其他 split/run 的适应状态当作 prior。 |
| 每样本全部注册类 | `_structured_score`（第 190 行）沿实际 C score 路径取得 old/new/gate，按 raw 组内胜者与组间 gap 生成公开预测，并与 `c.predict(**one)` 比较。五流具有同一个 opaque query 顺序；C 与公共 alias 写入同一记录。没有 query role、quota、视图聚合或批次联合决策。 |
| 固定预测与完成状态 | 每条记录只有 split/query ID、完整类列、scores 和 prediction。五流、结构决策记录、训练状态档案写出后，才 finalize archive 并写 `predictions_complete.json`（第 426 行起）；producer 不连接 truth，不计算 accuracy。 |
| 成本与失败 | 候选 prep/stage/score 分账，操作 SUM、峰值 MAX；R0 forward factor/solve 与实际 EDF 附加 solve 单列（第 318 行）。public predict 与原 R0 score 未返回的内部工作量保留 N/A，嵌套 callback 时间不再次加到 fit wall。异常路径（第 461 行）保留已返回工作、失败 audit/数值 ref、活动 query 的已完成调用；未知失败工作明确未知，没有 automatic retry 或伪造 complete。 |

对应 tests 覆盖真实 core 的完整 support A/B/C/R0 五流与同次继承（第 138 行）、query 置换与他样本变化不改变当前样本（第 198 行）、query 信息字段在训练前拒绝、C 失败保留 B 与失败数值状态（第 240 行）。这些是合成测试源码范围，不是本审查者执行的真实数据证据。

## Support analyzer

| 核查项 | 直接源码证据与结论 |
| --- | --- |
| 全矩阵先闭合 | `validate_spec`（第 158 行）保留四行、两 model×两 cohort、每行完整 40 parent 的 K×new/两 receiver-scene 选择。`verify_run`（第 602 行）逐行验证后核 160 parent、1800 path 和 supervisor 计数；`analyze`（第 766 行）只有完整验证返回后才调用 `derive_tables`。不存在已完成前缀的指标汇总。 |
| 物理 OOF/proxy | `verify_parent`（第 348 行）按每类 physical ID 排序建立 OOF；覆盖 `min(K,3)` 个 fold 与全部 K 个 one-shot anchor。每个 path 分别验证 parentK、实际 trainK、heldK、train/held 不交及固定 ID 集。proxy 的 trainK=1，不冒充 parentK 的训练量或更新条件。 |
| actual B 与 inner prior | `_verify_stage`（第 264 行）核每个 inner fold 的 train/held 分割；old teacher 只能来自该 fold 的 old inner-train，并保留实际 B theta。最终 C anchor、Q、`actual_B_*` 数组和 stage/prep 的实际 prior ref 必须与该 path B 一致。new0 的候选/基线 stage 集合无第二次 fit，固定 scores/predictions 精确复用 B。 |
| 同物理 A/B/C | 固定 stream 在支持标签 join 前保存；A 只面对同 path 的物理 old-held，使用原生六类列与 float32 scores。B old-held 与 C old/new held 均核对固定档案、类列和预测。C 的公开结构预测被保留；机器舍入造成共同偏移 ties 时，不用 rounded old argmax 替换 actual B 的旧组胜者。 |
| 指标顺序 | `score_fixed`（第 654 行）仅用于全部 row 验证后的保存预测。OOF 在物理 parent 内 pooled；proxy 每个 anchor 分别算 H、绝对 gap 等，再由 `derive_tables`（第 686 行）在该 parent 内平均，之后才跨 parent 汇总。没有从全局旧/新均值反推 H 或绝对 gap；所有 proxy 均保留。K1 无 held 指标，记 N/A。 |
| 资源与档案 | `verify_work`（第 110 行）从实际 operation audits 重算 SUM/MAX，核 free-intercept Ridge 的实际 RHS 列与五方向账；`Archives`（第 189 行）检查真实文件、数值字段、refs 与完整档案闭包。parent→row→supervisor 账逐层读回；未测 public predict 内部工作、源验证、星载能耗等保持 N/A。失败仅生成独立 `analysis_failed.json`，保留 partial，不修改原 run 或伪造完成 summary。 |

对应 tests 明确覆盖 parent-first H/gap 与缺失 A（第 162 行）、全 160 parent 的 K1 与全部 proxy（第 175 行）、第四行未完成时禁止进入任何指标 join（第 240 行），以及真实 production callback、numeric archive、native A 和 actual prior 篡改拒绝。

## 验证状态与边界

Root 告知：query producer 22 cases、support analysis 25 cases、query control 13 cases、修复前独立 scorer 33 cases 与此前 111 cases 合计 204 distinct synthetic cases 已通过。该数值验证由 root 串行执行，本审查者没有执行或读取测试日志。之后独立 query scorer 的实际 NPZ/prior 两项修复不属于本次审查范围，须使用 root 的新验证结果；不能把先前 33 cases 当作修复后通过证据。

Support analyzer 在 summary 中明确限定为保存 metadata、numeric archive、固定 decision 与资源账校验，不是独立 kernel/gate 数学重放。它不能据此证明完整数值算法、真实来源或 query 泛化性能；本次也不扩大为这些验证。当前结论是以上两个源码入口未发现新的直接 P0/P1，root 保持唯一整合、数值验证和 launch owner。
