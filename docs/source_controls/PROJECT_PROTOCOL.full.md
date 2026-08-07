# CVS项目场景与数据协议

版本：2026-08-07
协议模式：`p2_min_v1`

## 1. 文件职责

本文件只定义CVS-RFFI/CV-SincNet的科学场景、数据集合、数据生成、Phase1/Phase2/Phase3边界、Stage2-A/B/C权限、Phase3部署期协同输入及可声明范围。

本文件不保存活动性能目标、当前候选方法、实验矩阵、seed清单、epoch/参数/显存上限、优化路线、N607操作、Git流程或某次实验结论。这些内容分别属于独立目标文档、方法设计、实验报告和`AGENTS.md`。

协议的Git公开正文见[PROJECT_PROTOCOL.md](github_publish/CVS-RFFI-repo/docs/PROJECT_PROTOCOL.md)；一次性Phase2数据builder/validator实现边界见[PHASE2_DATA_VALIDATION_APPENDIX.md](github_publish/CVS-RFFI-repo/docs/PHASE2_DATA_VALIDATION_APPENDIX.md)。本文件定义语义，后两份文档分别承担Git承载和实现说明，不额外创造数据权限。

## 2. 项目场景

CVS的主场景是：

> 天基射频指纹识别中的地面弱标注跨接收机域泛化、目标接收机域少样本适应与新类注册，以及部署阶段多接收节点协同的未知拒识、匿名实体关联和可信确权。

项目采用“地面训练、星上部署”。Phase1在地面学习开放世界就绪的发射机身份表征并封存deployment bundle；Phase2只使用目标接收机已经接收到的LEO弱信道IQ、合法support标签和该bundle，完成旧类域适应与新类注册；Phase3发生在部署阶段，每个接收节点先形成冻结的本地身份、域、质量和拒识证据，再由协同推理完成unknown拒识、anonymous entity关联和可信确权。Phase3不替代Phase2，也不得把unknown query直接转成Stage2-C support。

Phase3是正式研究阶段，不是Phase2的同义改名。项目当前是否已经实现Phase3、是否完成真实在轨多星验证以及是否达到具体性能门槛，只能由对应Git实现、实验报告和完整artifact证明；阶段定义本身不构成完成声明。

WiSig/ManySig是地面代理数据，不是真实卫星数据；LEO弱信道叠加是物理启发的部署压力代理，不等价于真实在轨验证。

### 2.1 N607实验承载角色

N607是CVS-RFFI/CV-SincNet大规模训练、Phase2方法实验、125稳定性screen、独立确认矩阵和资源审计的主要计算与证据承载面。N607不是`R_s`或`R_t`中的接收机，不是source/target数据来源，不是卫星实体，也不构成clean/source访问、query拟合或其他Phase2协议的例外。所有代码与协议修改先在本地Git承载面完成并验证，再按`AGENTS.md`规定同步到N607；具体SSH、环境、GPU、launcher、日志和报告操作只由`AGENTS.md`及实验报告管理。

## 3. 观测模型与集合

接收IQ抽象为：

```text
x = R_d(H_d * T_y(s)) + n
```

- `T_y`：发射机硬件非理想性，是身份来源。
- `H_d`：传播或星地信道，是域扰动。
- `R_d`：接收机链路响应，是跨接收机偏移。
- `n`：噪声。

接收机集合：

```text
R_s = source training receivers
R_t = target receiver domain
R_t ∩ R_s = ∅
```

发射机集合：

```text
Y_old = Phase1已见发射机
Y_new = Phase1未见、Phase2注册发射机
Y_unknown = 未注册发射机
Y_old ∩ Y_new = ∅
Y_unknown ∩ (Y_old ∪ Y_new) = ∅
```

在Phase1出现过的TX在Phase2只能是旧类，不能重命名为新类。

## 4. Phase1地面开放世界就绪表征协议

Phase1是weak-label/semi-supervised source-domain DG，不是部署few-shot：

```text
L_s = {(x_i,y_i,d_i): receiver(x_i) ∈ R_s}
U_s = {(x_j,d_j): receiver(x_j) ∈ R_s, y_j hidden or unavailable}
rho_label = |L_s| / (|L_s| + |U_s|) ≤ 0.1
```

当前统一划分语义为相对source全池`0.07/0.63/0.30`：有TX标签训练集、无TX标签训练集、互斥source validation。三部分均不得包含`R_t`。需要研究标注率时，可使用`rho_label∈{0.005,0.01,0.02,0.05,0.1}`，但不得改变集合含义。

Phase1可以使用source数据的clean与卫星信道增强视图训练；这不授予Phase2或Phase3部署期推理读取这些样本、样本级派生状态或训练期scorer结果的权限。Phase1具体模型、loss、训练轮数和选择规则属于方法/实验文档，不写入本协议。

### 4.1 Phase1职责与输出

Phase1只在地面训练，不执行多接收节点消息传递、anonymous track维护或真实运营身份确权。它优化底层射频特征提取器，使身份表征具备TX可分性、跨接收机稳定性、LEO弱信道鲁棒性、类内紧致和类间margin，并降低未见TX上的过度置信，为后续距离、能量、尾部分布或不确定性拒识提供可用几何。

Phase1可以重新训练Sinc或时域前端、频域及PA分支、卷积层、`z_id`身份表征、`z_dom`域扰动表征、normalization、projection、fusion以及prototype、radius、energy和不确定性输出。具体启用哪些组件由独立方法设计冻结，本协议不要求一个候选同时堆叠全部机制。

Phase1最终交付：

```text
开放世界就绪的特征提取器
已注册类基础几何
类别半径／能量／尾部分布先验
接收质量与域不确定性输出
不可变deployment bundle
```

这些输出只是Phase3本地证据提取的底层输入，不等于已经完成真实unknown拒识、多节点协同或可信确权。

### 4.2 Source-only proxy unknown研发边界

Phase1开放世界研发必须在source-only范围内建立TX身份互斥的开发集合：

```text
source_known_train_tx
source_known_validation_tx
source_proxy_unknown_tx
```

`source_proxy_unknown_tx`的全部物理样本必须排除在任何训练池、训练view和模型选择回流之外。同一个已知TX的不同接收机、不同信道、不同日期或低信噪比view不能伪装成unknown。该TX级开发划分是现有source数据角色之上的开放世界研发约束；具体TX清单、seed和样本数属于独立目标或实验文档。

Phase1不得读取目标接收机query真值，不得使用Phase3后续确认的unknown作为训练数据，不得把source proxy unknown指标写成Phase3真实unknown结果，也不得输出真实运营身份或`registration_authorized`。候选晋级和性能门槛属于活动目标；协议层只要求所有晋级证据保持source-only、TX互斥和真实checkpoint可导出。

## 5. Phase2最小数据协议

### 5.1 单物理样本单LEO接收观测

每个clean/raw物理IQ记录在进入Phase2前只允许进行一次随机LEO弱信道叠加：

```text
received_i = H(c_i, seed_i)(clean_i)
c_i ∈ {leo_clear_weak, leo_low_elev_weak, leo_rain_weak}
```

对每个稳定`physical_sample_id`，只能绑定一个`c_i`、一个随机`seed_i`和一份固定`received_i`。禁止把同一clean/raw物理样本复制后分别叠加多种场景、多个随机信道实现或多个LEO状态，再作为多份support/query进入Phase2。

三种LEO场景用于评估不同接收条件；同一matched实验切片中，三个场景的物理样本ID集合必须两两不交。单场景内support与query的物理样本ID也必须不交。

### 5.2 K-shot与接收后计算view

`K-shot`表示每个已注册类别有K个互不重复的物理support样本。由同一固定接收IQ计算的均衡、裁剪、相位/幅度归一化、FFT或其他确定性/随机数学表征仍属于同一个物理样本，不增加K。

接收后计算view必须满足：

- 输入只能是固定、已封存的`received_i`；
- 不得调用LEO信道模拟器，不得恢复clean参考，不得产生第二个LEO观测；
- support view可以参与适配、注册和状态更新；
- query view只用于该query的逐样本推理，不得更新模型、原型、阈值、温度、门控、候选选择、早停或回滚状态。

### 5.3 Phase2允许输入

Phase2运行时只允许读取：

1. 不可变Phase1 deployment bundle；
2. 已一次验证并封存的Phase2固定接收IQ capsule；
3. 当前row的support标签、注册类别表和不含query真值的split；
4. 与数据无关的算法配置。

Phase2不得读取clean/raw/source样本、样本级source feature、source cache、source replay、clean派生logit/prototype/statistics、dataset构建路径或能够影响决策的外部source状态。

唯一例外是：在任何target访问前，由Phase1多样本聚合并与checkpoint共同封存的只读int8域×类模型知识及量化尺度。该组件不得包含raw IQ、单样本feature、全精度exemplar、source cache、可逆样本索引或可独立替换sidecar；Phase2不得更新它。具体量化布局和半径/偏移编码属于bundle实现规范，不属于数据场景协议。

#### 5.3.1 可选地面旧类压缩多原型组件

当前研发默认继续使用原有Phase1 bundle输入面。地面旧类压缩多原型是默认关闭的未来候选项，不因本协议列出而自动向现有方法、bundle或历史实验开放。只有新的Phase1 bundle与method lock同时显式声明`ground_old_multiprototype_enabled=true`时，Phase2才可读取该组件。

启用后，bundle可为每个`Y_old`类别携带不超过3个int8压缩原型及其量化尺度、粗粒度混合权重和聚合半径：

```text
1 <= M_c <= 3, c in Y_old
ground_old_prototype[c,m] = Aggregate({z_i})
physical_sample_count({z_i}) >= 2
```

每个原型必须由至少2个互不重复的Phase1地面物理样本聚合，不能是单个clean样本、单样本feature或exemplar的改名形式。原型数量、聚合算法、量化位宽、尺度、权重和半径规则必须在任何target访问前仅由Phase1数据固定并与checkpoint共同封存；不得使用target support、target query、125结果或其它Phase2反馈选择该格式。

组件只能覆盖`Y_old`，不得预置`Y_new`或`Y_unknown`知识；不得包含raw/clean IQ、成员ID、样本索引、精确成员清单、可逆聚类归属、样本级logit/cache或独立可替换sidecar。不同类别必须采用标签置换等价的同一构造和打分聚合规则；当各类原型数不同时，类别分数必须进行类内数量归一化，不能因原型更多获得裸max或累加优势。

这些地面原型是只读Phase1模型知识，不是target接收观测，不增加K，不生成第二个LEO view，也不能替代Stage2-B/C的合法target support。启用、停用或更换该组件只改变`bundle_id`；只要固定接收IQ、物理ID、receiver/TX集合、场景、K、support/query划分和schema不变，就不得触发`capsule_id/split_id`数据重验。缺少合规组件时必须回退到原有路线，不能在Phase2运行时回读地面样本重建它。

### 5.4 query只用于测试

Phase2 predictor的状态只能由Phase1 bundle和注册support决定。每个query必须独立面对全部已注册类别；禁止访问或利用：

- query真值以及真实old/new/unknown角色；
- 真实query batch类别集合或各类数量；
- 每类query配额、标签排序或分块；
- Hungarian、optimal transport、global quota或其他跨query全局重排。

预测必须先形成不可变预测artifact；之后独立scorer才可按opaque query ID连接真值并计算指标。scorer输出不得回流到适配、注册、阈值、选择、排序或重跑决策。

### 5.5 一次验证、跨方法复用

Phase2数据builder完成唯一观测、物理ID互斥、support/query互斥、接收机/TX集合和禁止成员检查后，输出：

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

研发和评估流程只核对以上最小句柄并直接运行。只有以下数据事实变化时才重做数据验证：固定接收IQ字节、物理ID、receiver/TX集合、scenario分配、K、support/query划分或协议schema。

候选方法、adapter、超参数、epoch、原型更新规则、method lock、checkpoint推理状态、资源预算或报告格式变化，不得使数据capsule失效，也不得触发重新追溯clean/source构建过程。若验证失败，只修复直接失败的数据项；其他`VALIDATED_ONCE`切片继续实验。

hash、签名、allowlist、访问账本和pre-open检查由数据builder/validator一次性自动完成，详见Git承载面的`docs/PHASE2_DATA_VALIDATION_APPENDIX.md`。它们是实现证据，不在每个方法目标或实验报告中重复展开。

## 6. Stage2阶段权限

| 阶段 | 可用target信息 | 任务 | 不可声明 |
|---|---|---|---|
| Stage2-A | 无target TX标签；可有无标签LEO接收IQ | zero-label target-domain reference/diagnostic | 旧类few-shot适应、新类identity accuracy |
| Stage2-B | `Y_old`的K-shot target support标签 | 旧类目标域适应与校准 | 新类注册性能 |
| Stage2-C | `Y_old∪Y_new`的K-shot target support标签 | 在同一目标域同时完成旧类适应和新类注册 | 缺少任一侧时的完整Stage2-C成功 |
| Phase3 | 多接收节点冻结本地证据、未注册类观测及合法外部确权上下文 | 部署期unknown拒识、anonymous entity关联、可信确权与注册授权 | 直接用unknown query训练/更新Phase2，或以unknown结果替代旧类/新类结果 |

Stage2-C中的旧类适应与新类注册是同等任务；旧类注册前/后必须来自同一row、同一旧类query和同一推理规则。具体指标门槛与确认矩阵属于独立目标文档。

以上Phase2数据权限、query访问和逐样本决策约束只用于Stage2主方法及其内部候选的合法性与晋级判定，不限制为论文复现和性能比较而运行的外部对比方法。CSIL、MoPC-HR等对比方法可按原论文完整训练/增量流程访问其所需的base/source数据、历史统计、训练批次和评估流程，也不受Stage2主方法资源预算约束。正式对比方法默认必须继承的项目数据条件是：凡作为新类注册或新类评测输入的样本均须叠加并明确记录LEO星地信道；其他`p2_min_v1`限制不作为对比方法结果有效性的门禁。经用户显式要求，可以另跑仅用于归因的matched无LEO新类诊断：保持方法、物理样本ID、support/query划分、K-shot、seed和旧类评测条件不变，只把新类support/query换成同一物理记录的未叠加IQ。该结果必须标为`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`，与正式LEO结果隔离，不能用于CVS卫星场景性能声明、方法晋级或Stage2协议有效性证明。对比结果必须标明其数据与训练权限，不能反向用作Stage2主方法满足协议的证据。

floor是面向全部实际注册类的通用下界评价，不是预选难类清单。任何算法、loss、校准或更新规则必须对类标签置换保持同一形式；禁止按具体TX/class ID设置白名单、专属分支、专属权重、专属阈值或专属超参数。允许从各类合法support自适应估计半径、不确定度或权重，但所有类必须使用同一公式和同一超参数生成规则。

## 7. Phase3部署期多接收节点协同协议

### 7.1 单节点本地证据

每个接收节点部署同一冻结或合法适配后的Phase1特征提取器。节点`m`只根据当前已接收IQ形成：

```text
z_id^(m)       发射机身份表征
z_dom^(m)      接收机／信道域表征
q^(m)          观测质量与可靠性
d_class^(m)    到各注册身份的距离
e_unknown^(m)  本地未知能量或拒识分数
p_local^(m)    本地已注册身份预测
```

本地证据必须先成为不可变artifact，之后才能进入协同推理。单节点决策只能是`registered identity`、`unknown`或`defer`。本地预测不得读取query真值、真实old/new/unknown角色、其他query的真实类别构成、类别配额或独立scorer结果。

### 7.2 协同输入与决策边界

Phase3可以融合各节点冻结的本地证据，以及与当前观测合法绑定的卫星可见性、时间窗、频率、波束、轨迹、位置约束和历史anonymous track。融合位置可以在星间链路、星座协同节点或地面网关，但必须在报告中明确。

正式协同方法必须显式处理接收机响应差异、传播/SNR差异、节点缺失或延迟、本地预测冲突、同一发射事件的多节点观测、跨过境匿名实体关联以及同源或高度相关证据的重复计权。简单平均、多数投票和选择最高置信节点只能作为基线，不能单独代表完整Phase3方法。

任何registered query被reject或defer，都必须在对应已注册类准确率中按错误计数。Phase3不得通过全部拒绝规避身份识别责任。

### 7.3 事件、接收与shot计数

Phase3区分物理发射事件与节点接收记录：

```text
一个emission_event_id
多个satellite_reception_id
仍然只计为一个shot
```

只有能够证明来自同一物理发射事件的多节点接收记录，才能作为严格same-event协同证据。不同时间、不同物理发射或无法建立事件绑定的记录不得拼接成一个K-shot样本。若现有数据不是同步多接收机采集，必须标为“多接收节点代理协同”，不得声称真实在轨同步多星验证。

### 7.4 Anonymous entity、可信确权与Stage2-C交接

被拒识的观测只能先关联为`anonymous_entity_id`，该ID表示多个观测可能来自同一物理射频链，不是最终语义身份。可信确权至少输出候选物理身份、证据来源、证据独立性、冲突标志、标签置信度、有效期和`registration_authorized`。

只有`registration_authorized=true`后，系统才能为获批身份重新采集K个独立物理发射事件作为新类support。历史unknown query保持为不可变检测证据，不得追溯改成support。新support按`p2_min_v1`形成新的`split_id`并完成对应数据验证后，交由Stage2-C执行旧类适应、新类注册、全部已注册类统一竞争以及遗忘/floor评价。

### 7.5 多节点数量与因果归因

活动目标可以在独立目标文档中冻结`N_sat`、receiver、seed、K和新类规模。协议要求至少保留`N_sat=1`单节点基线，并在任何协同收益声明中报告具体节点数、节点子集选择规则、缺失节点处理和证据相关性控制。

同时改变Phase1底层表征和Phase3协同方法时，必须使用同一输入口径分别评价：

```text
A：原Phase1基座+单节点
B：新Phase1特征提取器+单节点
C：原Phase1基座+多节点协同
D：新Phase1特征提取器+多节点协同
```

由`B-A`估计底层表征贡献，由`C-A`估计协同推理贡献，由`D-B-C+A`估计交互贡献。不得只比较A和D后把全部收益归因于协同推理。

## 8. 数据集角色与声明边界

- WiSig/ManySig可作为terrestrial proxy benchmark、source receiver family或目标接收机代理域。
- `R_t`必须与`R_s`不相交；当现有合法目标接收机覆盖不足时，可使用未进入Phase1的其他接收机/数据子集，但仍须满足集合与唯一观测协议。
- target-old与target-new必须来自同一个已定义目标接收机域和同一LEO弱信道场景口径。
- 具体文件路径、样本数量、receiver/TX清单和hash属于数据资产登记表，不写入场景协议。

可以声明CVS研究的是地面弱标注跨接收机DG、LEO压力下少样本旧类适应与新类注册，以及部署阶段多接收节点代理协同的Phase3方法。不能把WiSig称为真实卫星数据，不能把LEO模拟称为真实在轨验证，不能把非同步多接收机代理称为真实同步多星验证，不能把source-only DG称为few-shot适应，也不能把source proxy unknown、旧类提升、unknown拒识或协议无效结果当作新类注册成功。
