# CVS项目场景与数据协议

版本：2026-08-07
协议模式：`p2_min_v1`

## 文件职责

本文件只定义CVS-RFFI/CV-SincNet的科学场景、数据集合、数据生成、Phase1/Phase2/Phase3边界、Stage2-A/B/C权限、Phase3部署期协同输入和可声明范围。活动性能目标、当前候选方法、实验矩阵、seed、资源上限、N607操作、Git流程和实验结论均由独立文档管理。

一次性Phase2数据builder/validator实现边界见[PHASE2_DATA_VALIDATION_APPENDIX.md](PHASE2_DATA_VALIDATION_APPENDIX.md)。

## 项目场景与集合

CVS研究天基RFFI中的地面弱标注跨接收机域泛化、目标接收机域少样本适应与新类注册，以及部署阶段多接收节点协同的unknown拒识、anonymous entity关联和可信确权。WiSig/ManySig是地面代理数据，LEO弱信道叠加是物理启发的部署压力代理，不等价于真实在轨验证。阶段定义不代表Phase3已实现或已达到性能目标。

### N607实验承载角色

N607是大规模训练、Phase2方法实验、125稳定性screen、独立确认矩阵和资源审计的主要计算与证据承载面。它不是`R_s`或`R_t`中的接收机，不是source/target数据来源，不是卫星实体，也不产生任何Phase2协议例外。代码与协议先在本地Git承载面修改和验证，SSH、环境、GPU、launcher、日志与报告操作由`AGENTS.md`管理。

```text
x = R_d(H_d * T_y(s)) + n
R_t ∩ R_s = ∅
Y_old ∩ Y_new = ∅
Y_unknown ∩ (Y_old ∪ Y_new) = ∅
```

`T_y`是身份来源，`H_d`是传播/星地信道扰动，`R_d`是接收机响应。Phase1出现过的TX在Phase2只能是旧类。

## Phase1地面开放世界就绪表征

Phase1是weak-label/semi-supervised source-domain DG：

```text
L_s = {(x_i,y_i,d_i): receiver(x_i) ∈ R_s}
U_s = {(x_j,d_j): receiver(x_j) ∈ R_s, y_j hidden or unavailable}
rho_label ≤ 0.1
```

当前统一划分语义为相对source全池`0.07/0.63/0.30`：有TX标签训练集、无TX标签训练集、互斥source validation。三部分均不得包含`R_t`。Phase1可使用source clean与卫星增强训练，但这不授予Phase2或Phase3部署推理访问这些样本或样本级派生状态的权限。

Phase1只在地面训练开放世界就绪的底层特征提取器，可优化前端、卷积层、`z_id/z_dom`、normalization、projection、fusion以及prototype、radius、energy和不确定性输出。最终bundle可包含已注册类基础几何、半径/能量/尾部分布先验和接收质量/域不确定性；这些只为部署期本地证据提供底层输入，不等于完成真实unknown拒识、多节点协同或运营身份确权。

Phase1开放世界研发必须使用TX互斥的`source_known_train_tx`、`source_known_validation_tx`和`source_proxy_unknown_tx`。proxy unknown的全部物理样本排除训练；同一已知TX的不同receiver、channel、day或SNR view不能伪装成unknown。Phase1不得读取target query真值、回流Phase3确认unknown、执行多节点消息或anonymous track，也不得把source proxy unknown指标写成Phase3真实unknown结果。

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

Phase2只允许读取不可变Phase1 bundle、一次验证的固定接收IQ capsule、support标签/注册类别表/无query真值split及算法配置。禁止clean/raw/source样本、样本级source feature、source cache/replay、clean派生信号和外部source状态。

唯一例外是在target访问前由Phase1多样本聚合、与checkpoint共同封存、只读且不可更新的int8域×类模型知识及量化尺度。它不得包含raw IQ、单样本feature、全精度exemplar、source cache、可逆索引或可独立替换sidecar。

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
| Phase3 | 多节点冻结本地证据、未注册类观测和合法确权上下文 | 部署期unknown拒识、anonymous entity关联、可信确权和注册授权 |

Stage2-C中旧类适应和新类注册同等重要；注册前/后旧类比较必须来自同一row、同一query和同一推理规则。具体性能门槛、确认矩阵和资源限制属于独立目标文档。

floor评价覆盖全部实际注册类，不使用预选难类清单。算法、loss、校准与更新规则必须对类标签置换保持同一形式；禁止按具体TX/class ID设置白名单、专属分支、专属权重、专属阈值或专属超参数。可以从每类合法support估计半径、不确定度或权重，但全部类别必须采用同一公式和同一超参数生成规则。

## Phase3部署期多接收节点协同

每个节点部署同一冻结或合法适配后的Phase1特征提取器，只根据当前已接收IQ形成`z_id`、`z_dom`、质量`q`、类距离`d_class`、本地未知分数`e_unknown`和本地已注册身份预测`p_local`。本地证据必须先冻结为不可变artifact，节点决策只能是registered、unknown或defer。predictor不得读取query真值、真实old/new/unknown角色、真实batch构成、类别配额或独立scorer结果。

Phase3可以融合冻结本地证据以及合法绑定的卫星可见性、时间窗、频率/波束、轨迹、位置和历史anonymous track。正式方法必须处理接收机/信道差异、节点缺失或延迟、本地冲突、同一事件的多节点观测、跨过境关联及相关证据去重；平均、投票和最高置信节点只能作为基线。任何registered query被reject或defer都按身份错误计数。

```text
一个emission_event_id
多个satellite_reception_id
仍然只计为一个shot
```

只有可证明来自同一物理发射事件的记录才能作为严格same-event协同证据。非同步多接收机数据必须标为“多接收节点代理协同”，不得声称真实在轨同步多星验证。

unknown观测只能先形成`anonymous_entity_id`。可信确权输出候选物理身份、证据来源与独立性、冲突、置信度、有效期及`registration_authorized`。授权后必须重新采集K个独立物理事件作为新support；历史unknown query不得追溯改成support。新support生成新的`split_id`并按`p2_min_v1`验证后，才交给Stage2-C注册。

同时研究底层表征和协同时，必须保留A原基座+单节点、B新Phase1+单节点、C原基座+协同、D新Phase1+协同的同输入消融，分别报告`B-A`、`C-A`和`D-B-C+A`。不得只比较A与D后把全部收益归因于协同。

## 声明边界

可以声明CVS研究地面弱标注跨接收机DG、LEO压力下旧类少样本适应与新类注册，以及部署阶段多接收节点代理协同的Phase3方法。不能把WiSig称为真实卫星数据，不能把LEO模拟称为真实在轨验证，不能把非同步多接收机代理称为真实同步多星验证，不能把source-only DG称为few-shot适应，也不能把source proxy unknown、旧类提升、unknown拒识或协议无效结果当作新类注册成功。
