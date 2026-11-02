# Phase2 方法命名与后续对比设计

日期：2026-11-02。本文固定既有 R42 的方法名称，并记录后续实验设计建议；不改变既有模型、运行或实验结果，也不构成新实验启动记录。

## 方法名称

将 Phase2 域适应与新类注册的既有 R42 冻结方法命名为 **SCALR**，英文全称 **Support-Calibrated Adaptive Local Ridge**，中文为“支持集校准自适应局部岭方法”。完整系统称 **PAMF＋SCALR**：PAMF 指现行 Tri-SIG-TIME SSL 基座，SCALR 指原 R42 的域适应与注册读出。SCALR 不包含 DURA；接入 DURA 的系统需明确 support 和 query 路径。

SCALR 的身份是原 R42 固定实现及源端祖先，并非将原始 LocalRidge、历史 D92 或其他相同名称的 R42 实验改名。已完成 Phase2 Group1 使用的四个合法 Tri-SIG-TIME SSL、源端映射/残差、局部核及其 support 选择、旧类头先验及固定融合规则保持原配置。K=1 使用原先冻结的源先验规则；K>1 使用当前合法 support 内的物理留出规则选择既定候选。query 不参与选择或拟合。源训练的额外状态、目标 support 拟合成本和 DURA 的成本应分别披露；零目标神经更新不等于零拟合成本。

B 阶段使用 6K 个目标旧类有标签 support；C 阶段使用 6K 个旧类和 NK 个新类 support 联合重新拟合。相同输入与支持集下，A→C 和 A→B→C 的最终 C 按原实现等价；B 的核系数不作为 C 的初始化。已搁置的两个零目标旧类 support 派生方案不是 SCALR 的替代品，不自动恢复。历史运行和文件名保留 R42；后续呈现使用“SCALR（原 R42）”以便追溯。

## 第二组：适应后接入原生 C 阶段方法

用户指定第二组不使用给域适应方法增加通用注册头的路线，而以原生 CIL 方法自身的 C 阶段为比较对象。目标为 A→B→C 完整系统：各方法使用各自源模型，在相同 6K 目标旧类 support 下实施明确的 B 适配，随后传递该方法真正需要的权重、原型、旧类权重或合法记忆，使用 NK 新类 support 执行其原生 C 核心。

候选为 PAMF＋SCALR、PAMF＋原版 LocalRidge、MoPC、Orthogonal、LoRa。MoPC 保留 Group1 作者 attentionResNet、输入适配与实际 C 配方；Orthogonal 保留其正交状态与增量权重机制；LoRa 保留其特征、合法 gallery 和 kNN 判决。每条外部方法的 B 如何使用旧目标 support、如何构造与 B 编码器一致的 C 初始化状态、源数据访问、历史记忆、优化步和 session 预算须在实施前具体绑定；无法合法衔接则标 N/A，不以泛化头或重置源模型充数。新增 B 是项目的适应扩展，应展示为“方法＋B适配→原生C”，不能称整条链路为论文原始方法。

SCALR 的局部核状态不能直接加载进外部网络。保持各方法自己的表示和状态，不暗换成 PAMF 特征；原生 A→C 在 Group1 已完成，作为 0 旧目标标签参照。SCALR A→C 仍需要 6K 旧目标标签；A→C 不代表零旧类标签。用户不要求外部 CIL 的 DURA 实验，因此外部方法保留 raw/raw，PAMF 内部严格区分 S0/S1 与 Q0/Q1，Q2 同记录融合另列。不得将跨输入差写成同输入算法配对差。

如第二组需比较不同 DA 的 B 阶段接各原生 CIL 注册核心，另固定 DA×CIL 链路矩阵，例如 DADDA-SDA／MRIOR-SDA 的 B 分别接 MoPC／Orthogonal／LoRa 的 C。必须逐对核实 B 编码器、特征空间、旧类原型或权重、教师和历史状态是否可供 C 使用；在不同网络上移植 C 核心或添加必要接口时明确命名“DA→CIL 核心移植”，披露变更与预算，不能称作者网络原版，也不能用空状态或源权重重置冒充 B→C。未经合法接口绑定的组合记 N/A。具体 3×3 可行性及最终矩阵待实现核查，不能预称已支持。

优先主点 K5/N20：3 池×7 RX×8 MIX×4 seed＝672 个配对单元，复用 Group1 已完成的 SCALR、LocalRidge 与原生 CIL 结果；先逐项核实可继承的源模型和 B 状态。随后覆盖 K=1/5/10/20、N=2/5/10/20，至少包含原生外部结果已有的 MIX01 84 单元曲线，并建议主要链路扩至全部 8 MIX；单场景作为补充。一次预测前冻结本轮所有对比及预算，预测闭合后独立 truth-last。上述是设计范围，不声称已经实施。

主表分 A0 旧类、B 旧类、C 旧类/新类/宏 H/F1/类别最低召回；列出实际旧/新目标标签、源与目标更新量、历史状态及计时。并列原生 A→C、适应后 A→B→C 和 PAMF 纯路径；原生方法的缺少阶段为 N/A，非零。新旧 query 共同面对 6+N 类，禁止按真身份路由；自然 query 和 common-K 交集分开。

## 后续分组建议

1. **同一 PAMF 表示读出对比**：在相同冻结 F、相同 support 和路径下比较原型、kNN、线性 Ridge、原版 LocalRidge、SCALR 及固定预算线性 CE。共享表示的简化对照不冒称完整 ProtoNet 或 LoRa。先固定小范围源侧或协议合法诊断，再按预定矩阵扩展；不得依据目标 query 成绩选择性报告。
2. **SCALR 机制消融**：对照冻结源映射、残差强度、头先验、视图/核、support 内选择等，只有可独立改变的实现才称单因素；每种配置提前固定，并重点报告双侧 DURA 的 K=1 和困难 LU。不能将历史 MILKR 的方法机制和结果直接归给 SCALR。
3. **连续注册与成本**：最终 20 新类下比较一次20、10+10、5×4、2×10，每阶段报告旧类、历史新类、本轮新类、宏 H、内存、增量和累计耗时；SCALR 每轮完整重拟合应如实计费，CIL 保持原生 session 状态和预算。分解旧类内部决策变化及新类竞争，不直接将旧类总下降等同神经遗忘。
4. **独立确认**：现有 MIX 已用于开发和比较，单场景在 MIX 公开后完成；后续使用此前未参与方法选择且数据权限合规的新设备、接收机、日期或划分，固定方法后验证，不把既有开发结果改称盲测。

历史第一组入口：[总报告](../automation_reports/CV-SincNet/20261102-phase2-r42-scenes-m4-r08/results_group1/report.md)、[MIX 全 K/N](../automation_reports/CV-SincNet/20261101-phase2-r42-mix16-mgpu-m4-r07/results_mix_kn/report.md)、[外部方法](../automation_reports/CV-SincNet/20261102-phase2-r42-scenes-m4-r08/results_group1/external_report.md)。ISSL 按用户要求保持停止，Tweak M10 和 CSIL 作者版本未合法绑定保持 N/A，Hu 各版本身份不能混淆。
