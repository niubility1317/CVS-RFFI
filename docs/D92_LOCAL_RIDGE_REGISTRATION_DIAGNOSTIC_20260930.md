# LocalRidge 注册机制 support 诊断

本诊断不改冻结 LocalRidge，也不继续 Residual8。它只回答：旧类 support 与旧新类 support 分别从头拟合时，旧类准确率变化中，有多少对应旧类内部排序变化，有多少对应新增类别的额外竞争。它不测试真实地面 DG，不证明因果，也不设置性能晋级门槛。当前健康 LocalMargin 运行保持原样。

## 输入和固定 pilot

沿用两个已有 `residual-noeq-*` capsule 及原冻结 Phase1 的合法 support 缓存。实际 manifest 契约为 `p2_min_v1 / VALIDATED_ONCE`，`channel={route: residual, mode: post_sync, equalization_enabled: false, fs_hz: 25000000}`；完整生产场景为 `practical_high / practical_mid / practical_low_urban`。这是实际身份绑定检查，不重建或重新验证数据。没有历史 D11 数据、权重或适应状态输入；不读取 query、source 样本、逐样本 source 特征或任何已有评分。

准备工具只读安全原 support 配置与两个 `support_splits.json` 身份清单。先验证完整生产矩阵、模型/checkpoint/capsule 绑定，再确定性选取两个最小 model seed、各 cohort 排序前两个 `(receiver, scenario)` 键、两个完整矩阵的共同最小 support seed。每个所选键保留 `K=1/5/10/20 × Nnew=0/2/5/10/20`。

当前已核实的选择为：model seed `2026092701 / 2026092702`；support seed `2026092711`；rx3 的 `(19-1, practical_high)`、`(19-1, practical_low_urban)`；rx1 的 `(20-19, practical_high)`、`(20-19, practical_low_urban)`。每个 model/cohort row 为 40 parent，共 160。旧类固定 6 类，新类分别为 0、2、5、10、20 类；全注册类别数分别为 6、8、11、16、26。**pilot 不包含 practical_mid 或其余 receiver**，不能代表这些未测场景。

配置保留原完整 `producer_matrix`（rx3 900 / rx1 300 splits），另存显式 `selection.splits` 的 40 个真实 split ID 及 receiver、scenario、K、support seed、类别顺序和 new_count。运行时先按完整生产矩阵加载原缓存，再核对并取出明确列出的 splits。不存在组合就报错，不按指标换组合、不隐式选择、不扫参数。准备 CLI 接受 `--identity-manifests rx3=path rx1=path --commit <已知提交>`；实现交付不等于已生成配置或启动。

## 两次拟合与一次离线限制

每个物理 OOF fold 或 proxy anchor 先构造 C0 的旧新类训练/held 分割，再按旧类身份过滤得到 B0 的训练与同一旧类 held 行。B0/C0 旧类训练 ID、标签和原特征必须完全相同，旧类 held ID、特征、标签也完全相同，禁止从两个独立分割碰巧配对。

| 记号 | 状态 | 输出 |
|---|---|---|
| B0 | 原冻结 LocalRidge，仅当前训练折旧类 support 拟合 | 在旧类候选集上逐样本预测 |
| C0 | 原冻结 LocalRidge，当前训练折旧类与全部注册新类 support 联合拟合 | 对每个 held 样本面对全部注册类别预测 |
| C_old | 原封不动的 C0 分数，仅离线取其旧类列 | 旧类内部排序诊断；零额外拟合 |

两次拟合均调用原 `fit_branch_local_ridge(..., arm='local_ridge')`。带宽、中心化、迹匹配、标签中心化与核的原规则不变，并且各自只使用训练折。C0 的核尺度和分类头可能与 B0 不同，因此不得拼接 B0 旧类分数与 C0 新类分数，不比较跨拟合原始分数差作为机制证据。`C_old` 是有旧类候选限制的离线分析，不是部署预测器，也不把 held 样本的真实 old/new 角色输入 C0。

Nnew=0 时只拟合一次 B0，并将状态和分数复用为 C0；C_old 与 C0 相同，两个分解项和总差严格为零，新类准确率及新旧 H 为 null。不同 Nnew parent 不共享拟合状态；即使旧类物理映射一致，也保留明确的 parent 身份和成本。

True K1 不拟合、不造 held 样本，只记录输入/数值身份检查。K1 不能由视图变成多个物理样本。来自 K5/10/20 的单 anchor proxy 单独报告，不充当真实 K1 泛化证据。

## 可辨识分解与六种转换

在同一个旧类 held 样本上，令 `b/r/c` 分别为 B0、C_old、C0 的正确性（0 或 1）：

`b - c = (b - r) + (r - c)`。

所有预测使用一致的物理类别 ID 字典序打破精确分数并列，列置换不改变预测。C0 使用 C_old 相同旧列和同一 tie 顺序，因此 `c <= r`。第一项是有符号的旧类内部排序改变；第二项是新类加入候选集带来的额外正确性损失。第二项非负。两项并非独立可操控原因；第一项也包含注册引起的核/头联合变化，不能进一步拆为唯一因果贡献。

| `(b,r,c)` | `b-r` | `r-c` | `b-c` | 含义 |
|---|---:|---:|---:|---|
| 111 | 0 | 0 | 0 | 三者正确 |
| 110 | 0 | 1 | 1 | 旧排序正确，新增类夺走正确预测 |
| 100 | 1 | 0 | 1 | C0 旧排序已错误 |
| 011 | -1 | 0 | -1 | C0 旧排序修正了 B0 |
| 010 | -1 | 1 | 0 | 旧排序改善与新增竞争损失抵消 |
| 000 | 0 | 0 | 0 | 三者错误；新类胜出也不构成额外正确性损失 |

因此“C0 预测新类”的比例不能直接当作新增类造成的旧类准确率损失。必须同时保存转换计数；否则 010 抵消会被总差掩盖。若保存竞争 margin，只能取同一个 C0 内的旧类最大分数与新类最大分数差，描述具体预测边界，不与 B0 分数尺度混算。

## 聚合与资源

OOF 按每类物理 ID 排序后位置模 3 分割。先汇总同 parent 全部 held 行的正确数和样本数，再算每类及 old/new 宏平均。K5 每类三个 held 折为 2、2、1，不能直接平均三个折准确率。六种转换、两项分解和总差使用同一批旧类 held 行及相同权重，聚合后继续满足恒等式。

Proxy 遍历当前 parent 每类所有排序 anchor 位置，先在 parent 内平均，再等权汇总 parent。不同 anchor 会重复使用物理样本，不视为独立数据或独立重复实验。OOF 与 proxy 分开，按 model、cohort、receiver/scenario、parent K、Nnew 完整报告；单列 old-only Nnew0。完成全部 160 parent 后再汇总，不从部分结果选择子集。

总计 40 个 true K1 parent、120 个三折 OOF parent（360 路径）、1,400 个 proxy anchor，共 1,760 条 B0/C0 评估路径。其中 352 条 Nnew0 路径只调用一次头拟合，其余 1,408 条调用两次，总计 3,168 次头拟合；每 row 为 440 路径、792 次头拟合。实际 factorization_count 单独记录，退化核路径可能低于头拟合次数。adapter 更新和 optimizer_steps 均为 0，C_old 额外拟合为 0。记录头拟合耗时、数值状态成本、分解统计与实际总耗时，不承诺资源或性能优势。

调度仅自有 2 CPU lanes，每 lane BLAS 2，CUDA 设备空。使用原私有 CPU transport，逐文件发布必要 import 闭包，不包含未验证 Residual 草稿，不加载 encoder 或 checkpoint。输出与原缓存/capsule 隔离，已有输出拒绝覆盖；技术失败只标记所属 lane，其他健康 lane 完成，保留证据且不自动重试。root 唯一发布与启动 owner。

## 正确性验证范围

合成测试覆盖六种可行转换及不可行 `c>r`、错误新类归因、不同 B/C 尺度、精确 tie、类别列置换、B/C 旧训练/held 身份错位、Nnew0 严格复用、K1 零拟合、不等折样本加权，以及原核退化的有限分数路径。编排测试覆盖确定性身份选择、完整矩阵缺项/重复、实际类别数量、metadata-only preflight、两 lane、CPU 发布闭包、marker 行绑定、失败保留和输出不覆盖。测试由 root 统一执行；本文件不声明尚未取得的通过结果。
