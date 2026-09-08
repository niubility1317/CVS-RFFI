# CVS项目场景与数据协议

版本：2026-08-24
协议模式：`p2_min_v1`

## 文件职责

本文件只定义CVS-RFFI/CV-SincNet的科学场景、数据集合、数据生成、Phase1/Phase2边界、Stage2-A/B/C权限和可声明范围。活动性能目标、当前候选方法、实验矩阵、seed、资源上限、N607操作、Git流程和实验结论均由独立文档管理。

一次性Phase2数据builder/validator实现边界见[PHASE2_DATA_VALIDATION_APPENDIX.md](PHASE2_DATA_VALIDATION_APPENDIX.md)。

## 项目场景与集合

CVS研究天基RFFI中的弱标注跨接收机域泛化，以及部署到目标卫星接收机域后的少样本域适应与新类注册。WiSig/ManySig是地面代理数据，LEO弱信道叠加是物理启发的部署压力代理，不等价于真实在轨验证。

### N607实验承载角色

N607是大规模训练、Phase2方法实验、125稳定性screen、独立确认矩阵和资源审计的主要计算与证据承载面。它不是`R_s`或`R_t`中的接收机，不是source/target数据来源，不是卫星实体，也不产生任何Phase2协议例外。代码与协议先在本地Git承载面修改和验证，SSH、环境、GPU、launcher、日志与报告操作由`AGENTS.md`管理。

```text
x = R_d(H_d * T_y(s)) + n
R_t ∩ R_s = ∅
Y_old ∩ Y_new = ∅
Y_unknown ∩ (Y_old ∪ Y_new) = ∅
```

`T_y`是身份来源，`H_d`是传播/星地信道扰动，`R_d`是接收机响应。Phase1出现过的TX在Phase2只能是旧类。

## Phase1地面数据

Phase1是weak-label/semi-supervised source-domain DG：

```text
L_s = {(x_i,y_i,d_i): receiver(x_i) ∈ R_s}
U_s = {(x_j,d_j): receiver(x_j) ∈ R_s, y_j hidden or unavailable}
rho_label ≤ 0.1
```

当前统一划分语义为相对source全池`L_s/U_s/V=0.07/0.63/0.30`：有TX标签训练集、无TX标签训练集和单一source validation。三个角色均不得包含`R_t`，物理样本ID两两不交。`V`可用于source侧校准、阈值冻结和checkpoint选择，但不得反向传播、更新EMA、prototype、normalization或其他持久状态；不得再把`V`拆成`V_cal/V_select`等方法角色。Phase1可使用source clean与卫星增强训练，但这不授予Phase2访问这些样本或派生状态的权限。

## Phase2最小数据协议

### 单物理样本单LEO接收观测

每个clean/raw物理IQ在进入Phase2前只允许叠加一次随机LEO弱信道：

```text
received_i = H(c_i,seed_i)(clean_i)
c_i ∈ {leo_clear_weak,leo_low_elev_weak,leo_rain_weak}
```

一个稳定`physical_sample_id`只能绑定一个场景、一个随机信道实现和一份固定接收IQ。禁止由同一物理样本生成多场景、多随机实现或多LEO状态副本。三场景物理ID两两不交；单场景support/query物理ID不交。

`K-shot`表示每类K个互不重复的物理support。由固定接收IQ计算的均衡、FFT、归一化或其他数学表征不增加K，不得调用LEO模拟器、恢复clean或生成第二份LEO观测。support计算view可参与状态更新；query计算view只能用于当前逐样本推理，不能更新任何状态或参与选择。

### 允许输入与禁止输入

Phase2运行时采用穷尽式白名单，只允许读取：`p2_min_v1`、`VALIDATED_ONCE`固定目标域LEO received IQ及匹配的`capsule_id/split_id`；当前row合法target support标签、必要注册类别表和无query真值/角色split；地面预先计算、随checkpoint上传并保持不可变的类原型及必要类别映射；冻结checkpoint和预登记算法配置。

除此之外，不得读取、构造或恢复任何地面source/clean样本、源域数据加载器、source replay/cache、源域特征、样本级embedding、源域统计量、源域BatchNorm状态、伪源域样本、生成式源数据、可还原源样本的中间信息或其他外部source状态。不得读取query真值、query角色、真实query batch类别集合/数量或其他query反馈。

地面类原型只能作为不可训练的类别锚点和冻结判决依据；不得反向更新、在线重估、追加源域信息，或扩展为D92式协方差、LDA、持久分类头、样本级记忆或类条件源域统计。缺少合规原型时不得回读地面样本重建。

### query只测试

每个query独立面对全部已注册类别。禁止query真值/角色、真实query batch类别集合或数量、每类配额、标签排序/分块以及Hungarian、optimal transport或其他跨query全局重排。预测artifact先冻结，独立scorer之后才连接真值；评分结果不得回流。

### 一次验证、跨方法复用

数据builder完成唯一观测、物理ID、集合和禁止成员检查后输出最小句柄：

```text
protocol_schema = p2_min_v1
phase2_data_status = VALIDATED_ONCE
capsule_id = <content identity>
split_id = <receiver/TX/scenario/K/support-query identity>
single_leo_observation = true
clean_source_runtime_access = false
query_fit_access = false
query_decision_policy = per_sample_all_registered_classes
```

只有固定接收IQ字节、物理ID、receiver/TX集合、scenario、K、support/query划分或协议schema变化才重验。候选、adapter、超参数、epoch、原型规则、method lock、模型状态和资源预算变化不使data capsule失效。hash、签名、allowlist、访问账本和pre-open检查由builder/validator一次性自动完成，见[PHASE2_DATA_VALIDATION_APPENDIX.md](PHASE2_DATA_VALIDATION_APPENDIX.md)。

## Stage2权限

本节及前述Phase2最小数据协议只约束Stage2主方法及其内部候选。CSIL、MoPC-HR等外部论文对比方法可按原论文完整流程使用base/source数据、历史统计、训练批次和评估流程，也不受主方法资源预算约束。正式对比默认要求全部新类注册及新类评测样本叠加并记录LEO星地信道；经用户显式要求，可另跑仅用于归因的matched无LEO新类诊断，保持方法、物理样本ID、support/query划分、K-shot、seed和旧类评测条件不变，只替换同一新类物理记录的未叠加IQ。该诊断必须标为`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`并与正式LEO结果隔离，不能用于CVS卫星场景性能声明、方法晋级或Stage2协议有效性证明。对比方法须披露权限差异，其结果不能作为主方法协议合规证据。

| 阶段 | target信息 | 任务 |
|---|---|---|
| Stage2-A | 无target TX标签 | zero-label target reference/diagnostic |
| Stage2-B | `Y_old`的K-shot support标签 | 旧类目标域适应与校准 |
| Stage2-C | `Y_old∪Y_new`的K-shot support标签 | 同一目标域内旧类适应与新类注册 |
| Phase3 | 未注册类仅作独立评估 | unknown rejection备用扩展 |

Stage2-C中旧类适应和新类注册同等重要；注册前/后旧类比较必须来自同一row、同一query和同一推理规则。具体性能门槛、确认矩阵和资源限制属于独立目标文档。

floor评价覆盖全部实际注册类，不使用预选难类清单。算法、loss、校准与更新规则必须对类标签置换保持同一形式；禁止按具体TX/class ID设置白名单、专属分支、专属权重、专属阈值或专属超参数。可以从每类合法support估计半径、不确定度或权重，但全部类别必须采用同一公式和同一超参数生成规则。

## 声明边界

可以声明CVS研究弱标注跨接收机DG、LEO压力下旧类少样本适应与新类注册。不能把WiSig称为真实卫星数据，不能把LEO模拟称为真实在轨验证，不能把source-only DG称为few-shot适应，也不能把旧类提升、unknown拒识或协议无效结果当作新类注册成功。

## Checkpoint训练数据一致性与继承污染禁令

本节约束正式CVS主方法的完整模型来源，覆盖Phase1初始化、恢复训练、教师蒸馏、特征提取、模型融合及Phase2/Phase3所用基座。数据权限约束整个产生流程，不能只检查本轮训练配置。旧报告中的“成熟基座”“固定基线”、默认checkpoint路径或历史可加载记录均不授予复用权限。

1. **禁止跨训练划分继承。**用于初始化或续训的checkpoint，其产生流程必须与本次预登记的Phase1数据契约一致：数据集及版本、raw/equalized与物理记录映射、source/target接收机与日期、TX集合及标签映射、L_s/U_s/V比例和各角色实际物理样本ID、划分算法及影响划分的seed。仅数据集文件名、split名称、样本数或比例相同，不构成一致。任一不一致即禁止用于该正式run；无污染但训练集不同，也不能作为匹配初始化或严格同条件基线。
2. **物理样本身份优先。**复制文件、改路径、重编号、换seed、重新归一化、切片或叠加新的LEO信道，不会消除底层物理IQ的历史归属。衍生视图必须追溯原始物理记录；不得把同一记录的不同视图放入互斥角色。只改变模型随机初始化seed且实际数据契约不变，不算数据划分改变。
3. **禁令追溯所有继承来源。**核对直接checkpoint及其全部上游预训练、resume、warm-start、teacher/EMA、蒸馏、集成与融合来源。部分加载骨干、丢弃分类头、冻结参数、只取embedding、重新训练adapter或重置optimizer，均不能解除既有污染。原型、归一化统计、阈值和其他随模型继承的持久状态同样受限；未知祖先不能默认视为从零训练。
4. **禁止历史目标暴露与测试选模。**任一上游使用当前target的物理样本（有标签或无标签）、统计量、伪标签，或其指标参与训练、校准、早停、checkpoint选择、候选排序、超参数调整及选择性重跑，该来源及其后代均不得用于当前干净泛化结论。仅检查“本轮不反向传播目标标签”不够。历史源域V若转为当前target，同样构成污染。任何测试集选模（包括含测试指标的joint_safe）均不满足source-only选模要求；即使旧测试集与当前target不同也不得作为合规来源。模型完全冻结后的独立最终评分本身不等于训练污染，但其结果不得再参与选择或反馈研发。
5. **缺证据即禁止使用。**加载前依据实际生效配置、数据索引/可复现划分、训练和选模记录核对来源，禁止只相信launcher默认值、文件名或口头声明。当前数据契约、每个继承来源及选模依据、差异和结论写入现有run报告即可，无需另建签名、哈希链或审批系统。无法确认实际物理ID或祖先来源时记为CHECKPOINT_PROVENANCE_UNVERIFIED；不一致记为CHECKPOINT_DATA_CONTRACT_MISMATCH；已证实目标参与训练或选择记为CHECKPOINT_TARGET_CONTAMINATED。三者均禁止该来源进入正式run，不得先跑后补证据。
6. **恢复路径与比较边界。**无合规checkpoint时，仅可在已有训练授权内按本次数据契约从零训练，并只以单一source V或预登记固定轮次选模；不能私自改变训练集、目标集或借用旧权重节省预算。共享初始化的消融使用同一合规checkpoint；从零训练的对比固定数据划分，披露训练随机性。公平比较还须披露预训练/续训总预算，不能把历史额外训练隐去。改变合法数据契约后须重新判断checkpoint适用性。
7. **不得追认或洗白污染结果。**发现问题后保留原checkpoint、日志和结果，在受影响报告中标明来源及后代范围；污染结果记为PROTOCOL_INVALID_FOR_CLEAN_GENERALIZATION，来源不明结果记为PROTOCOL_VALIDITY_UNVERIFIED，不进入正式有效结果汇总、晋级或干净泛化对比。产物完整不代表协议有效；共享污染源也不能抵消污染。既有运行的停机、删除、覆盖或重跑仍遵守原授权，不由本条自动授权。
8. **阶段权限不互相替代。**Phase2/Phase3继承的地面基座必须匹配预登记Phase1契约，而非要求其训练集等于target support；合法Stage2 support适配保持原权限，query始终不得参与拟合或选择。外部论文方法的显式权限例外不能用于给CVS主方法洗白，也不自动豁免训练/测试污染披露。只有用户明确另立跨数据预训练研究时，才可作为单独任务披露权限差异，不能标为本契约下的匹配source-only实验。

此项是现有输入权限检查的一部分。VALIDATED_ONCE只证明数据capsule自身已验证，不证明任意checkpoint均适用；换checkpoint只需核对其与现有契约及历史来源的兼容性，不触发未改变数据capsule的重复builder验证。相同来源与契约已有可复用核对结论时直接引用；来源、契约或新污染证据改变才重查受影响部分。
