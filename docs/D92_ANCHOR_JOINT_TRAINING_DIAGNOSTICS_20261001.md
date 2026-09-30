# AJLR 完整训练诊断采集器

日期：2026-10-01。状态：采集器已实现，AST 和 UTF-8 静态检查完成；主 Agent 在项目 `ssr-gpu` 环境串行运行的 10 项合成检查全部通过，证据 `.codex_tmp/pytest_utf8_1790793621645724600`。完整实数据采集尚未执行。当前健康任务不停止、不热改、不重启；本文不声称真实诊断已运行或方法已获得收益。

实现位于 [collect_d92_anchor_joint_training_diagnostics.py](../tools/collect_d92_anchor_joint_training_diagnostics.py)，测试位于 [test_collect_d92_anchor_joint_training_diagnostics.py](../tests/test_collect_d92_anchor_joint_training_diagnostics.py)。方法依据为 [AJLR 数学设计](D92_JOINT_AFTER_FCR8_DESIGN_20261001.md)第 2 至 7 节、[当前核心](../code/cvsrffi/d92_anchor_joint_local_ridge.py)、[入口说明](D92_ANCHOR_JOINT_ENTRY_20261001.md)与[独立摘要实现](../tools/summarize_d92_anchor_joint_probe.py)。不读取设计中链接的外层成绩来派生训练诊断。

## 1. 完成边界与两阶段读取

输入必须是状态为 `COMPLETE_ANCHOR_JOINT_PROBE_VERIFIED` 的完整独立摘要，且记录 `query_rows_used=source_rows_used=0`。固定结构为 160 parent、40 个真实 K1 parent、120 个 OOF parent、1400 个 proxy anchors、1800 条实际路径、3240 个 R0 heads、3240 个 AJLR preparations 和 3240 个 AJLR stages。上述为当前入口已冻结的结构数量；有信息阶段、实际更新数、试探数、head/factor/adjoint 费用只能取完整实际计数，不能把上界填成实测值。`training_stage_count` 也必须为 3240。

完整摘要负责数学验证与科学权限。采集器只检查完成状态、读取范围及派生流完整性，不再次验证 closed solve、SVD、VJP、Armijo 数学，不重新拟合、评分或选参数，不生成额外 hash、receipt 或签名。完成条件不满足时，采集器在打开训练流或 NPZ 之前拒绝。

第一阶段 `snapshot(summary_root, run_root=None)` 是只读快照。它选择 `summary.json` 的允许字段：状态、run/commit、coverage、算法、资源与资源分层、训练源、归档计数以及零 query/source 使用。它通过顶层 JSON 选择解析跳过 `statistics` 等外层成绩字段，不反序列化这些值。随后完整扫描四个 row 的 `training_objectives.jsonl`、`training_events_compact.jsonl` 和 `fit_stages.jsonl`。

入口 `compact_event` 省略了训练物理 ID、prior preparation 元数据和实测分布摘要，包括核心已经测得的 block angles。快照因此额外完整扫描训练专用 `training_events.jsonl`：只选择训练身份、`prior_folds`、`final_problem` 和训练 objective/final-fit 元数据；只补回缺失的实测分布摘要与有限的均值描述。训练 objective 保存的是标量和 NPZ 引用，没有在 JSON 中物化 score 矩阵。原始 parent 结果 `fit_trace.jsonl`、outer support-held 原始特征和 score、query、历史结果、总索引和根交接均不打开。快照不读取任何 NPZ。

第二阶段 `extract(snapshot, run_root=None)` 仅加载快照引用的训练 NPZ，派生描述统计，返回可序列化 JSON。只允许 `state_arrays/` 内相对路径，拒绝目录逃逸；以 `allow_pickle=False` 读取数值数组并使用有限大小的只读缓存。不导入 AJLR fit/score/evaluator，不调用任何新的 adapter forward、距离函数、SVD、Cholesky、线性求解或 scorer。原始大数组留在原位置，派生输出保留全部引用。

第三个本地操作 `write_outputs(new_output, derived)` 创建全新目录并写紧凑 JSONL/CSV、摘要和说明。远端两个阶段均为脚本经 SSH stdin 执行、结果经 stdout 返回；没有 SCP、远端部署或远端写文件。快照单独留在本地，提取失败后可重用快照，不必重新运行训练或摘要。只读提取不是训练干预。

## 2. CLI 与 API

主 Agent 先独立核实完整 160 parent 和 support summary，再按顺序调用。以下路径为占位符，不是已执行命令：

```text
python tools/collect_d92_anchor_joint_training_diagnostics.py --summary-root REMOTE_VERIFIED_SUMMARY --run-root REMOTE_COMPLETE_RUN --snapshot-output NEW_LOCAL_SNAPSHOT.json --ssh-host HOST --ssh-config CONFIG --remote-python NUMPY_PYTHON

python tools/collect_d92_anchor_joint_training_diagnostics.py --snapshot NEW_LOCAL_SNAPSHOT.json --run-root REMOTE_COMPLETE_RUN --output NEW_LOCAL_DIAGNOSTICS --ssh-host HOST --ssh-config CONFIG --remote-python NUMPY_PYTHON
```

`--snapshot-output` 不能与 `--output` 或 `--snapshot` 同时使用；快照目的文件的父目录须已存在。`--snapshot` 阶段不需要重复指定 summary。SSH config 和 NumPy 可用的远端 Python 路径必须显式给出。远端命令参数使用 `shlex.join`，快照 JSON 作为 Python 字符串字面量经 stdin 传入，不拼接进 shell 命令。内部 `--snapshot-stdout`/`--extract-stdout` 禁止输出路径，只供 read-only transport。远端直接一步采集被拒绝，确保 root 先取得可检查的本地快照。

本地完整产品可直接调用：

```text
python tools/collect_d92_anchor_joint_training_diagnostics.py --summary-root LOCAL_VERIFIED_SUMMARY --run-root LOCAL_COMPLETE_RUN --output NEW_LOCAL_DIAGNOSTICS

python tools/collect_d92_anchor_joint_training_diagnostics.py --snapshot LOCAL_SNAPSHOT.json --run-root LOCAL_COMPLETE_RUN --output NEW_LOCAL_DIAGNOSTICS
```

API 为：

```python
captured = snapshot(summary_root, run_root=None)
derived = extract(captured, run_root=None)
write_outputs(new_output, derived)
# 本地便利接口等价于上述前两步。
derived = collect(summary_root, run_root=None)
```

`run_root` 只重定位四个 row 的 `row_id/probe` 训练归档，不改变摘要绑定或配方。新快照与输出目录均禁止覆盖。状态分别为 `COMPLETE_AJLR_TRAINING_READONLY_SNAPSHOT` 和 `COMPLETE_AJLR_TRAINING_DIAGNOSTICS_DERIVED`。CLI 最后只打印状态和数量，不把大数组或全部结果打印进人工终端。

主 Agent 的建议聚焦测试命令：

```text
python -m pytest tests/test_collect_d92_anchor_joint_training_diagnostics.py -q
```

使用项目既有 `ssr-gpu` 环境串行执行。本 Agent 未执行 pytest、Conda、SSH、Git 或实数据拟合。测试全部由手工构造的临时 JSON/NPZ 完整产品驱动，不导入核心 fit 或读取真实 run。

## 3. 目标、参数与梯度字段

AJLR 只有 `B_AJLR` 和 `C_AJLR_seq`；R0 是已拟合的对照头。没有 C reset、teacher、soft keep、hinge margin loss 或 guard。采集器使用 AJLR 当前公式：先跨所有 folds 按物理类别汇总 CE sums/counts，得到 class means，再跨类别取 RMS，目标为 `RMSCE + 0.5||Z||²`。保留完整 `class_ce_sums`、`class_ce_counts`、`class_ce_means`、`RMSCE`、CE/task、prox 和 total，不把各 fold RMS 简单平均。

每个曲线位置包含已保存的原始坐标 U、函数坐标 Z、W 的 norm/min/max/坐标数/非零数，以及 U 相对实际 anchor 的距离。U 保持原始 736×8 坐标，Z 按实际保留 rank，W.shape[1] 决定 rank。`coordinate_parameter_count`、可训练参数数、实际更新参数数分别保留；无监督 K1 仍有 head 和坐标，但训练参数数与更新参数数依实际分支记录。V0 固定，不产生 V 梯度、参数球或 product-ball 字段。rank=0 的空数组 norm 为 0，min/max 为 null，不发生空数组归约错误。

梯度的 CE 部分为 `g_Z−Z`，近端部分严格为 Z，**不除以物理样本数 N**。字段包含 total/CE/proximal 梯度与方向的数值描述、CE 与 proximal 的 dot/cosine/conflict、两项沿实际总下降方向的斜率及方向与归一化负总梯度的偏差。任一参与向量范数为 0 时 cosine 与有关方向比值为 null，不能写成 0 来暗示无冲突。空 rank 边界同样安全。

曲线完整保留 INITIAL、每次 GRADIENT、所有接受/拒绝 TRIAL、每次 STEP 和 FINAL。真实/代理 K1 的 INITIAL/FINAL 也保留；没有 adapter objective 时损失字段为 null，并注明 `NO_ADAPTER_SUPERVISION`，不丢掉 final closed-head 工作。事件 scalar payload 保留所有当前可用标量，未来的整数 `*_count`、`*_calls`、`*_solves` 也保留。

每个 TRIAL 描述实际 ΔZ/ΔU、名义函数坐标步长、gradient-dot-delta、comparison tolerance、Armijo RHS/余量和目标非增余量。拒因由已验证的两个接受标志描述为 `ARMIJO`、`OBJECTIVE_INCREASE` 或二者同时失败；不引入不存在的 keep 拒绝。固定最多 4 次更新、每次 12 次二分试探。预算耗尽与零更新不写成收敛；最后接受 cache 仍是最终训练坐标。

## 4. 函数、核、谱与实际 B 先验

准备记录保留 H、W、anchor 与完整 singular-value NPZ 引用、实际 rank、谱比值、retained condition number、rank threshold、dictionary RMS、实测 whitening residual/tolerance。没有重新运行 SVD 或白化认证。零字典的 rank=0 是已测量分支；没有把它与未测量混同。

实测 `pre_tangent_displacement_mean_squared/rms`、`coordinate_squared_norm` 和 reconstruction error 是当前训练 support 上的**切向投影、饱和和归一化之前**增量。已接受 ΔZ 的 path length 与 `4×1/8=0.5` 上界沿用这一范围。它们不是实际 block/interaction/kernel/score 位移，也不能从 U norm 推断分类保持。

每个 inner head 保留固定 τ、γ、s0、q、actual trace 和数值残差，分别比较初末；固定 γ 不保证 actual trace 恒等于 s0。当前 U 改变时，旧物理参考点重新映射，固定的是参考测度和 nuisance，不是冻结 RKHS 中心向量。核/交叉核 norm 与相对初始变化、所用 mixed distance 变化、archived adapted b/a 几何变化、center_mean 变化、prior score 不变量、残差 score norm、拟合旧参考均值和目标减先验的旧参考均值均可描述。相对变化的参考 norm 为 0 时为 null，并给出原因。

核心已经测得的 block-angle 分布从完整训练元数据补回。归档没有保存 tangent/kappa 或 adapted-only 距离分布，这两项保持 null；采集器不会另运行 forward 补测。mixed distance 是 head 实际使用的距离，不冒充 adapted-only distance。τ=0 精确保留 0，并注明 `ZERO_BANDWIDTH_USES_ORIGINAL_EQUIVALENCE`；原始旧尺度不足时 τ/γ 的合法 null 保留 `NO_OLD_KERNEL_INFORMATION`，不合成 epsilon、bandwidth floor 或 jitter。

C 的每折 prior head 来自本阶段实际 B 的冻结 U_B 和该折旧 inner-train；其引用、实测 forward、监督 score 描述保存在 `prior_heads`。它不是外部教师，旧监督标签也不是独立验证。最终 C NPZ 内的 `prior_B_` 数组保存实际完整 B；final head 记录来自该真实函数先验与闭式残差。M/Y/E 不进行第二次样本中心化，也不新增 intercept、组偏置或候选 prior 系数。

## 5. 注册 head 与后续 adapter 能分开观察到什么

对于有信息 C，INITIAL 的每折已保存 M_held、Z=0 fitted scores、标签和 head 引用。采集器在相同 inner-held 集合、相同全注册类竞争下比较 M_held 与 Z=0 scores，记录 score RMS 变化、winner 变化、正确→错误和错误→正确；并分别列出旧/新类别。这是“固定实际 U_B 时的注册残差 head 影响”，已包括新列零 padding 在全类 argmax 中的竞争。旧类列内准确率另列，不能替代全类准确率。

后续比较该 Z=0 初始 inner score 与最终缓存的 inner score，描述 adapter 改变及其伴随 closed-head 重解。τ/γ/q/M 固定的数值比较和 actual trace/geometry/kernel 变化一起保留，但不能称独立消融，也不能把某一项写成性能因果。

**完整 final C 在 Z=0 时的头没有保存。** 它仅在 adapter 优化结束后拟合一次 final head。因此完整 support 上“先注册 head、再改 adapter”的单独函数影响为 N/A；不从 inner folds 拼成完整头，不再拟合。无信息 C 也没有 inner Z0 objective，相关字段给出无法观测的原因。最终完整头可以描述实际 prior 与 residual 的组成，却不能推出未保存的 Z0 head 或外层成绩变化。

监督 accuracy、margin、逐类 CE、winner 及正确性转换只用于训练解释。inner-held 标签已经直接监督 adapter，actual B 也由当前合法 outer-train 训练，不能称独立泛化验证。采集器不读取独立 support-held scores，也不输出 outer accuracy/H/新旧结果。

## 6. 完整覆盖、分层与 B 去重

实际阶段和准备逐条保留。阶段、preparation、candidate scalar log 和 final 训练身份必须与摘要的全部 stage keys 完全一致；每种 gradient/trial/step 条数与完整摘要 training objectives 相符。没有抽样、挑选最好 stage 或遗漏零更新/退化分支。Nnew=0 精确复用实际 B，不伪造 C stage。

完整阶段分层为：mode×train K；mode×scope×parent K×new count；mode×cohort×receiver×scenario×K×new；mode×model seed×cohort×K×new；mode×row×scope×K×new。保留 actual/information/updated/zero-update counts、rank、stop/rejection、初末/差值损失和真实资源，CSV 中嵌套统计使用 JSON 文本。已验证资源分层另行原样导出，不从训练均值反推外层成绩。

B 的物理绑定按 row、scope、parent K、train K、注册旧类列表和精确训练物理 ID 序列识别，不能按 split_id、新类数或 U norm 近似去重。全部实际执行照常计费，重复 B 上下文数量单列；去重后的训练描述单列 `deduplicated_B_statistics`。去重不取消实际费用、不创建独立样本，也不假设不同 seed/cohort/物理 ID 可共享状态。

## 7. 成本、缓存与真实缺项

preparation 费用只计一次；stage 只计 student/final head 和 CE 伴随。FINAL 事件先复制 preparation 再覆盖 stage counter，因此 `counts` 排除 preparation-only 字段，`all_event_counters` 仍保留它们；PREPARATION 与 STAGE 中共享的 raw/reference/kernel 工作字段各自在所属阶段保留。当前 scalar log 的全部整数计数亦保留，人口/坐标计数不当成求解工作。

实际工作包括 R0 原解与 EDF 额外三角求解、B/C prior preparation 原解和 prior score、student initial/trial/final 原解与 Cholesky、缓存 CE 伴随、adapter/dictionary、距离和核。reference pair 是 raw distance work 的子集，不再加一次。STEP 重复 accepted TRIAL，GRADIENT/FINAL 展示复用已接受 cache 的 head forward 时间；不能按曲线行逐项累计。阶段和准备最终计数及已验证 summary resources 是费用口径。

AJLR_FINAL 在独立 outer inference 之前产生；`fit_stages` 的 compactor 又不保留嵌套 `score_workload`。每阶段 outer prior/residual work 因而为 null；**完整摘要的 `resources` 与 `resource_statistics` 已单独记录 outer residual/prior inference 的实际工作、时间与分层**，采集器安全读取这些资源字段，不打开 outer scores，不从头尺寸推算未知费用。后续不能把缺项补成 0。

`persistent_state_bytes` 是返回对象实际保留的唯一数值 buffers，包括训练坐标、几何和完整 head cache 以及实际 B prior；它不是最小部署常驻。`deployment_numeric_state_bytes` 才是当前实现单列的推理所需数值 arrays，仍不同于完整部署包、Python 对象和峰值 RSS。两者分别对应摘要 `candidate_fit_maximum_persistent_state_bytes` 与 `candidate_fit_maximum_deployment_numeric_state_bytes`，不得互相替代。prepared/cache、adapter、prior/head bytes 可能共享或重叠，不与最大值相加。NPZ 压缩文件字节是训练证据存储，不是部署状态或新增通信量。

当前未测部署包、增量传输、GPU/星载时间显存、独立 kernel 时间等保持 null/N/A，继承已验证的测量原因。CPU/BLAS 线程、run wall 与 lane/stage work sums 使用实际资源；并行 lane work sum 不是 run wall。冻结 encoder、原始 support 特征缓存或文件写入费用不能由少量训练坐标数推断为低端到端开销。新地面 payload/statistics 的实测 0 保留，不能把其他未测项也写成 0。

## 8. 输出 schema 与合成检查

| 输出 | 内容 |
|---|---|
| `summary.json` | 完成状态、来源、coverage、全部实际计数、资源/资源分层、原始归档引用和字节、完整分层、B 去重与边界 |
| `stages.jsonl/csv` | 全部实际 B/C stage；初末/差值、坐标/谱、成本、停止/拒因、C 注册与后续 adapter 的分别描述、final head |
| `curves.jsonl/csv` | INITIAL、GRADIENT、全部 TRIAL、STEP、FINAL；监督 CE/class means、梯度、接受条件、缓存与 head 机制 |
| `preparations.jsonl/csv` | 逐准备实际计数、谱/白化、prior folds 与 final problem；原始坐标引用 |
| `prior_heads.jsonl/csv` | C 冻结实际 U_B 的每折旧 prior head、引用、已测机制与合法训练 score 描述 |
| `curve_statistics.jsonl/csv` | 按 mode/train K/事件种类/iteration/trial/acceptance 的全部可用标量统计 |
| `by_*.csv` | 五类完整训练分层 |
| `resources_*.csv` | 完整摘要已有资源分层，含 outer prior/residual 成本 |
| `B_binding_groups.csv` | 精确 B 物理绑定的代表上下文与重复数；原始费用保留 |
| `archive_by_phase.csv` | 摘要已核实的原始 NPZ phase 数量、压缩/数值 bytes 和归档时间 |
| `report.md` | 简明训练目标表和读数边界，不报告 outer/query 性能 |

JSON 中缺项为 null，CSV 中为 `N/A`。statistic 包含 known count、missing count、mean/min/max/sum；均值表只描述其标明的阶段范围，不伪造独立 parent 估计。全坐标和大核矩阵以原始引用保存，输出没有 teacher 文件。

合成检查覆盖完整四 row 生命周期、动态 rank 与空坐标、未来计数保留和 preparation/stage 分账、重复物理 B 去重、C prior→Z0→final 的不同监督变化、旧/新 class RMS、实测角度补回、`g_Z−Z` 与无 `/N` 边界、精确 τ=0/合法 null、12 次拒绝与接受 cache、不完整摘要提前拒绝、JSON 外层字段跳过、训练事件缺失、NPZ 目录逃逸、完整 CSV/JSONL 与禁止覆盖、远端双阶段 stdin 接口。fixture 不宣称数学核通过、目标域性能或完整独立验证；这些由核心/摘要原有测试与完整实数据摘要负责。
