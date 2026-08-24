# Phase1挑战条件化PA算子辨识需求追踪表

日期：2026-08-24
设计代号：`CCOI-PA-V1`（Challenge-Conditioned Operator Identification with PA）
配套报告：`docs/CVS_PHASE1_CCOI_PA_V1_DESIGN_20260824.md`
当前交付状态：设计与代码落点审计完成；新增实现、训练和N607实验均未开始。

## 1. 状态定义

- `verified`：现有代码、配置或正式结果已提供可复用能力，且本轮完成只读核对。
- `pending`：V1必须实现，但本轮仅完成设计。
- `deferred`：有研究价值，但不进入V1，避免同时改变多个机制。
- `rejected`：与物理假设、Phase1协议或最小可证伪原则冲突。
- `blocked`：设计已明确，但缺少不能由模型代理替代的真实证据。

## 2. 需求—实现—验证追踪

| ID | 来源 | 可检验需求 | 目标文件/模块 | 状态 | 验证方式 | 说明 |
|---|---|---|---|---|---|---|
| T01 | `项目.md` Phase1边界 | 训练、校准和模型选择只使用源域划分，不读取目标域或query | `code/SSDG/train_ssdg.py`、数据加载与配置 | verified | 核对现有Phase1训练入口和协议说明 | CCOI不得扩大数据权限 |
| T02 | `项目.md` Phase1默认路线 | 保留`ADV3B02_CORE90_SOFT_E200`的clean+satellite concat、卫星CE和E1–200课程 | 现有Phase1配置与launcher | verified | 核对技术报告、周报及训练入口 | 新侧路不是新基线 |
| T03 | 粘贴文本“稳定核与状态子空间” | 继续使用身份分支`z_id`和域分支`z_dom`的解耦结构 | `code/model_dual_cvsincnet.py` | verified | 代码只读核对 | CCOI输出进入身份侧，不删除域侧 |
| T04 | 粘贴文本“先做PA/包络机制” | 复用现有记忆多项式PA路径和包络门控，不复制第二套PA骨干 | `code/model.py` | verified | 核对`MemoryPolynomialLift`、`EnvelopeGate1d` | V1只需暴露池化前时序特征 |
| T05 | 粘贴文本“跨四元组结构” | 批次中具备至少两个TX×两个domain的矩形单元 | `code/cvsrffi/balanced_tx_rx_sampler.py` | verified | 核对`BalancedTxDomainBatchSampler`与`tx_rx_rectangles` | 可直接支撑DiD采样 |
| T06 | `项目.md` Phase1完成定义 | 最终评估包含clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak` | 现有评估器与报告链路 | verified | 核对历史正式报告的分场景输出 | 不允许仅报LEO均值 |
| T07 | `项目.md` Phase1交付 | 保留prototype/radius/energy/tail等部署知识导出能力 | 现有deployment bundle构建链 | verified | 核对现有部署报告与代码入口 | 新算子原型只能追加，不能替换既有输出 |
| T08 | 现有模型兼容性 | 默认关闭CCOI时保持旧checkpoint严格加载与原模型输出 | `code/model_dual_cvsincnet.py`、checkpoint加载器 | verified | 核对当前严格加载路径；V1设计采用外部wrapper | 真正实现后仍需回归测试 |
| T09 | 粘贴文本“双视图” | 同一固定接收IQ产生强标准化内容视图与弱处理指纹视图 | 新建`code/cvsrffi/ccoi_pa.py`，训练数据视图函数 | pending | 单元测试两视图同源、形状一致、无目标数据 | 内容视图不得做逐token幅度归一化而抹除PA激励 |
| T10 | 粘贴文本“挑战编码” | 以长度64、步长16切分256点IQ，输出32维token挑战码 | `code/cvsrffi/ccoi_pa.py::PAChallengeEncoder` | pending | 形状、边界、梯度和确定性测试 | 13个重叠token是工程起点，不是科学结论 |
| T11 | 粘贴文本“挑战类型覆盖” | 维护48类源域挑战码本及软分配、熵和覆盖率 | `PAChallengeEncoder`、配置 | pending | 码本占用、熵、塌缩测试 | 码本只由源域拟合并冻结 |
| T12 | 粘贴文本“编码器不能携带设备/域” | 记录TX/RX/day探针，训练时用源域泄漏约束而非目标域校正 | trainer、`code/SSDG/losses.py`、分析脚本 | pending | 与随机、shuffle、RX-code、time-code对照 | 探针接近机会水平的阈值按类别数归一化报告 |
| T13 | 粘贴文本“局部条件响应” | 从现有PA分支暴露池化前`B×64×64`时序图 | `code/model.py` | pending | 旧输出不变、特征图形状、checkpoint兼容测试 | 仅在显式flag下返回中间图 |
| T14 | 粘贴文本“FiLM/条件归一化” | 由挑战码生成逐token缩放/平移，得到条件PA响应`r_t` | `ccoi_pa.py::PAConditionalResponseHead` | pending | q置乱、q常量、梯度路径及数值稳定测试 | 避免q与响应分支端到端串谋 |
| T15 | 粘贴文本“同挑战公平比较” | clean/同物理样本卫星视图作为已知同挑战锚点；跨样本匹配使用冻结q与置信权重 | trainer、匹配器 | pending | 已知锚点召回、代理匹配覆盖率、置信校准 | 代理最近邻不能冒充真实语义匹配 |
| T16 | 粘贴文本“四元组与DiD” | 同TX同挑战跨domain拉近；异TX同挑战同domain推远；跨domain保持TX相对差 | trainer、`losses.py` | pending | 合成矩形批次符号测试、无有效四元组安全跳过 | 复用T05采样器，不另建全局重排器 |
| T17 | 粘贴文本“集合级系统辨识” | 对条件响应做带置信度的DeepSets/attention池化，输出64维`theta_pa` | `ccoi_pa.py::OperatorPool` | pending | 排列不变性、空掩码、单token与多token测试 | V1采用线性复杂度集合池化 |
| T18 | 粘贴文本“留出挑战预测” | 从挑战子集估计`theta_pa`，预测同一样本/集合内未见挑战响应统计 | `ccoi_pa.py::HeldoutChallengePredictor`、trainer | pending | 固定holdout mask、NMSE/R²、对照头比较 | 不跨目标query聚合，不使用query truth |
| T19 | 粘贴文本“可辨识性门控” | 输出coverage/entropy/有效挑战数和证据充足度，不足时仅标记低置信 | `ccoi_pa.py::ObservabilityGate`、scorer | pending | 覆盖率—准确率曲线、全部拒绝防护测试 | Phase1已知类评估中低置信样本仍计错，不得删样本 |
| T20 | 粘贴文本“晚期证据融合” | 基线CosFace logits与算子logits独立，源域固定融合系数 | wrapper、配置、评估器 | pending | `alpha=0`严格复现基线；alpha扫描仅用`V_cal/V_select` | 禁止早期拼接把算子侧变成普通增维 |
| T21 | 粘贴文本“对照与归因” | 完成C0–C4同row单seed矩阵，逐项加入条件化、DiD、留出预测 | 配置、launcher、正式报告 | pending | 同checkpoint/split/seed/budget逐row核对 | 先小矩阵证伪，达门槛后再扩多seed |
| T22 | 粘贴文本“指标体系” | 报告ID/RX/day probe、margin retention、NMSE/R²、coverage/entropy、单包/多包及四场景性能 | 分析脚本、正式报告 | pending | 指标单测、字段完整性和同row检查 | 所有指标均按源域选择规则冻结 |
| T23 | 粘贴文本“情况B” | Soft-DTW用于顺序相同但时间偏移的挑战对齐 | 后续`ccoi_alignment.py` | deferred | 仅当V1证明q条件化有效且局部错位成为主误差时立项 | Soft-DTW时间和空间复杂度为二次量级 |
| T24 | 粘贴文本“情况C” | partial OT与dustbin处理局部重叠和不可匹配token | 后续匹配器 | deferred | 需要真实内容对应子集评估选择偏差 | 防止只挑“容易token”造成虚假提升 |
| T25 | 粘贴文本“多机制算子” | 新增PA记忆、差分谱、I/Q不平衡等多分支算子 | 后续V2 | deferred | PA-V1通过后逐机制消融 | V1只识别PA/包络响应 |
| T26 | 粘贴文本“神经算子/超网络” | 用生成器重建完整波形或局部响应 | 后续V3 | deferred | 需独立的可辨识性与生成质量证据 | 当前样本量下风险高于收益 |
| T27 | 粘贴文本“设备状态” | 将设备分解为稳定核和多状态子空间 | 后续状态审计 | deferred | 需跨上电/重载/时间的真实状态标签 | 现有domain标签不能替代设备状态 |
| T28 | 粘贴文本开篇校正 | 把内容相似直接解释为信道/接收机已消除 | 无 | rejected | 物理因果审查 | 内容对齐只控制激励，不消除传输与接收链 |
| T29 | 粘贴文本负例 | 无条件把同TX所有样本压到单点 | 无 | rejected | 几何审查 | 会抹除条件响应并制造表示塌缩 |
| T30 | 既有消融证据 | 直接把RF32/FFT96等大特征早期拼接到身份头 | 无 | rejected | 历史同row消融与归因审查 | 改为独立算子证据晚融合 |
| T31 | `项目.md`数据权限 | 用目标域、query、query role或query truth训练/校准挑战编码器和门控 | 无 | rejected | 协议负测 | 属于硬协议失败 |
| T32 | `项目.md`独立query决策 | 跨目标query联合估计一个设备算子或按batch配额重分配 | 无 | rejected | 独立决策负测 | Phase2只能在合法support内部聚合，query逐条独立 |
| T33 | 粘贴文本“匹配精度/召回/覆盖” | 给出真实语义挑战匹配的precision/recall，而非只报码本代理一致性 | 数据元数据或人工核验子集 | blocked | 需要先确认WiSig记录中是否存在可靠重复payload/前导字段或建立人工核验集 | 这是当前最高风险缺口，不能用模型置信度自证 |

## 3. 状态统计

| 状态 | 数量 |
|---|---:|
| verified | 8 |
| pending | 14 |
| deferred | 5 |
| rejected | 5 |
| blocked | 1 |
| 合计 | 33 |

## 4. 进入实现前的最小闭环

1. 先核实T33：数据中是否存在可用于真实挑战匹配评估的重复前导、payload标识或受控子集。没有时，明确把训练匹配称为“代理挑战匹配”，不宣称语义内容已对齐。
2. 仅实现T09–T20需要的侧路、loss和配置；T23–T27不进入V1代码。
3. 先运行旧checkpoint兼容、协议负测、真实checkpoint无query smoke，再完成一次P0/P1正确性审查。
4. 按C0–C4单seed最小矩阵运行；低性能只触发分析和下一候选，不作为进程中途停止条件。

## 5. 追踪结论

本追踪表实现的是“设计级严格覆盖”：粘贴文本中的核心物理主张、V1机制、可证伪实验和协议边界均已映射到现有能力或明确代码落点。它不是“实现级完成”；T09–T22仍为`pending`，T33仍为`blocked`，因此当前不得声称CCOI-PA已训练、有效或优于现有Phase1。
