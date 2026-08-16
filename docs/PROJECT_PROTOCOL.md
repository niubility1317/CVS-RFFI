# CVS项目场景与数据协议

版本：2026-08-17
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

当前统一划分语义为相对source全池`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`：有TX标签训练集、无TX标签训练集、校准validation与选模validation，其中`V_s=V_cal∪V_select`。四个角色均不得包含`R_t`。`L_s`、`U_s`与`V_s`可以共享source已知TX身份，但物理样本ID在所有角色间必须两两不交。Phase1可使用source clean与卫星增强训练，但这不授予Phase2访问这些样本或派生状态的权限。

### Source-only proxy unknown研发边界

`proxy_train`只由`L_s`生成。它是相对当前episode注册类别表的source代理角色，而非真实未见TX；训练proxy可参与拒识相关反向传播。`U_s`不生成proxy，因为训练过程不可读取其TX真值。

`P_cal`只由`V_cal`生成且只用于校准与阈值冻结；`P_select`只由`V_select`生成且只用于source侧模型选择。validation proxy不得反向传播，不得更新EMA、prototype、normalization或其他持久状态。source proxy指标只能写作代理未知研发性能，不能替代真实target unknown性能。

target unknown TX身份与source训练/validation TX身份必须互斥。任何target角色，包括target-known与target unknown，均不得用于训练、校准、选模、候选重排或触发选择性重跑。模型、几何与阈值在target访问前冻结；预测artifact先封存，独立scorer之后才能连接truth。真实unknown结论只来自这一次性、role/truth-blind的target评估，不反馈研发。

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
| Phase3 | 未注册类仅作独立评估 | unknown rejection备用扩展 |

Stage2-C中旧类适应和新类注册同等重要；注册前/后旧类比较必须来自同一row、同一query和同一推理规则。具体性能门槛、确认矩阵和资源限制属于独立目标文档。

floor评价覆盖全部实际注册类，不使用预选难类清单。算法、loss、校准与更新规则必须对类标签置换保持同一形式；禁止按具体TX/class ID设置白名单、专属分支、专属权重、专属阈值或专属超参数。可以从每类合法support估计半径、不确定度或权重，但全部类别必须采用同一公式和同一超参数生成规则。

## 声明边界

可以声明CVS研究弱标注跨接收机DG、LEO压力下旧类少样本适应与新类注册。不能把WiSig称为真实卫星数据，不能把LEO模拟称为真实在轨验证，不能把source-only DG称为few-shot适应，也不能把旧类提升、unknown拒识或协议无效结果当作新类注册成功。
