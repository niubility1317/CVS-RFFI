# Phase2统一物理样本池与最大query设计

## 目标与边界

本设计将WiSig的`ManySig.pkl`、`ManyTx.pkl`、`ManyRx.pkl`和`SingleDay.pkl`从四个固定实验入口重构为一个去重的物理样本引用面，使Phase1采用不同source接收机组合时，Phase2能够按合法目标接收机补集生成可复用的数据capsule，并在不违反`p2_min_v1`的前提下最大化query数量。

设计不修改当前科学协议。每个物理IQ记录在Phase2前仍只随机绑定一个`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`观测；三scene物理ID两两不交；K-shot按每类独立support物理样本计数；query只用于逐样本推理并独立面对全部注册类。checkpoint、adapter、超参数、epoch和方法规则变化不得触发数据重建或重验。

本设计只定义builder、split和capsule的目标结构。当前样本数来自2026-08-27对N607现有四个数据文件的只读清点；跨文件去重、扩展receiver完整覆盖和最终query数仍需由后续只读审计确认。文中的`373,280`是`CAPACITY_UPPER_BOUND`，不是已验证split规模或实验结果。

## 已知数据资产

统一按`equalized=1`统计；`equalized=0/1`若来自同一采集记录，不得计为两个独立物理样本。

| 子集 | TX数 | RX数 | 天数 | 单元容量 | 当前总数 | 主要角色 |
|---|---:|---:|---:|---:|---:|---|
| `ManySig` | 6 | 12 | 4 | 每个TX×RX×day固定1000 | 288,000 | Phase1旧类训练、Phase2高密度旧类 |
| `ManyTx` | 150 | 18 | 4 | 通常每个TX×RX×day最多50，部分稀疏 | 509,128 | 多天新类、unknown和扩展receiver |
| `ManyRx` | 10 | 32 | 4 | 通常每个TX×RX×day最多200，部分稀疏 | 247,684 | receiver扩展和已有类补充 |
| `SingleDay` | 28 | 10 | 1 | 每个TX×RX固定800 | 224,000 | 第4天高密度旧类和新类 |

四个总数不能直接相加。不同文件中的TX、RX和日期存在交叉；相同物理记录只能进入canonical pool一次。

## 总体架构

数据面分为四层：

```text
WiSig原始文件／完整原始索引
    ↓
canonical physical pool
    ↓
receiver and class eligibility registry
    ↓
MAXQ与BAL4D split registry
    ↓
按需runtime capsule
```

### Canonical physical pool

每条记录包含：

```text
physical_sample_id
tx_label
receiver_label
capture_date
capture_session
original_signal_index_or_offset
source_asset_refs
equalization_role
fixed_scene
fixed_overlay_seed
received_iq_ref
```

`source_asset_refs`允许一条canonical记录指向多个来源文件，但不增加物理样本计数。若完整WiSig原始索引可用，优先以原始索引生成`physical_sample_id`；否则使用官方元数据与一次性builder摘要识别重复记录。身份关系无法确认时，不能把可疑记录相加。

### Eligibility registry

registry按`source_profile_id`、`class_profile_id`、K和scene计算receiver资格。对注册类集合`C=Y_old∪Y_new`，receiver进入一个K-shot split的必要条件为：

```text
min_{class∈C,scene∈S} N(class,receiver,scene) >= K
```

资格判断只读取物理样本可用性，不读取模型预测、准确率、query真值或角色。

### Split registry

每个split记录：

```text
protocol_schema
source_profile_id
target_profile_id
class_profile_id
query_policy
K
scene
support_ids
query_ids
capsule_id
split_id
phase2_data_status
```

不同K或不同support/query划分使用不同`split_id`。同一split在通过`VALIDATED_ONCE`后跨checkpoint和方法直接复用。

### Runtime capsule

runtime capsule只包含：

- 当前row合法support received IQ与标签；
- query received IQ与opaque ID；
- 已注册类别表；
- `p2_min_v1`、`capsule_id`、`split_id`和`VALIDATED_ONCE`最小句柄。

capsule不包含clean/raw路径、source数据入口、query truth、query角色、真实batch类别数、类别配额或跨query重排信息。

## Source与target接收机设计

### 最大容量source profile

推荐新增：

```text
SRC5_MAXP2 = {
  1-19,
  18-2,
  19-2,
  2-19,
  3-19
}
```

这5个接收机不在SingleDay的10个receiver中，因此可以保留全部SingleDay receiver作为Phase2目标域。ManySig上该profile含120,000条Phase1物理样本，继续按`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`划分为8,400/75,600/18,000/18,000条。

所有target profile都显式满足`R_t∩R_s=∅`。同一个receiver可以在另一个source profile中承担不同角色，但不得在同一profile中同时属于source和target。

### 目标receiver分层

第一层`R_dense7`同时具备ManySig高密度旧类和SingleDay高密度第4天数据：

```text
R_dense7 = {
  1-1,
  14-7,
  2-1,
  20-1,
  7-14,
  7-7,
  8-8
}
```

第二层`R_sd3`只由SingleDay提供完整高密度注册类候选：

```text
R_sd3 = {
  13-13,
  2-20,
  8-13
}
```

第三层`R_mtx7`由ManyTx提供多天低密度扩展：

```text
R_mtx7 = {
  1-20,
  13-7,
  18-19,
  19-1,
  20-19,
  8-14,
  8-7
}
```

三层最多形成17个候选目标接收机。最终receiver数由完整覆盖审计决定；任何receiver缺少某个注册类的三scene K-shot最低样本量时，只从对应class/K profile中移除，不影响其他已验证split。

历史source组合与当前5个目标receiver继续保留为兼容profile，不因本设计失效。

## 类别设计

`Y_old`保持：

```text
Y_old = {
  14-10,
  14-7,
  20-15,
  20-19,
  6-15,
  8-20
}
```

SingleDay除去6个旧类后有22个新类候选：

```text
1-11,10-11,10-7,11-1,11-17,11-4,11-7,
13-3,15-1,16-16,2-19,20-12,20-7,
3-13,3-18,4-11,5-5,6-1,7-10,7-11,8-18,8-3
```

上述条目均为TX标签；例如候选TX`2-19`与source receiver`2-19`属于不同字段，不表示同一实体或集合重叠。

后续审计只按数据覆盖选择嵌套集合`Y_new5⊂Y_new10⊂Y_new20`。排序键依次为：

1. 所有候选目标receiver上三scene的Kmax可行性；
2. 前三天ManyTx覆盖；
3. 去重后的物理query总数；
4. receiver、day和scene最小单元样本数；
5. TX标签字典序，作为完全并列时的确定性tie-break。

禁止依据模型预测、准确率、类难度、query truth或晋级结果选择TX。`Y_unknown`与`Y_old∪Y_new`保持互斥，并使用相同scene生成与query推理口径。

## Scene分配

builder在连接support/query角色前，按`TX×receiver×day`分层随机分配唯一scene和overlay seed。每个物理样本只生成一份received IQ。

对1000条单元采用334/333/333并跨day旋转余数；对800条单元采用267/267/266；对50条单元采用17/17/16。实际去重后不足标准容量的单元采用相同确定性分层随机规则近似均分。

scene分配一旦形成`capsule_id`便不可因方法或checkpoint改变。三scene结果只能在独立评分后汇总，不能在预测期间通过真实类别数、quota或全局重排耦合。

## 双query策略

### MAXQ_ALL_UNIQUE

每个K拥有独立support/query split。每个scene、每类抽K个support，其余所有合法唯一物理样本进入query：

```text
N_query = N_unique - 3 * K * |C| * |R_t|
```

其中`|R_t|`是通过该class/K profile资格检查的receiver数。该split用于最大化有效样本量、降低总体置信区间并分析长尾class/receiver。

独立scorer必须同时输出sample-micro、class-macro、receiver-macro、day-macro和scene-macro指标。方法运行时不得读取各类、各day或各receiver的真实query数量。

### BALANCED_4DAY_CORE

固定`R_dense7`或最终审计得到的四天完整核心receiver；对每个TX×receiver×day×scene按共同最小容量抽取query。K=1/5/10/20使用嵌套support，并从同一Kmax保留池外取固定query。

该split用于K-shot消融、方法主表和严格matched比较。它不追求绝对样本最大，但防止`2021_03_23`和高密度receiver主导结果。

`MAXQ_ALL_UNIQUE`与`BALANCED_4DAY_CORE`使用不同`split_id`，两者结果不得合并成同一实验row。

## 容量上限

以下估算假定`Y_new20`在所有候选receiver上满足K=20的三scene覆盖，并且SingleDay与ManyTx重复已正确去除。

### `R_dense7`

每个receiver的旧类上限为`6×4000=24,000`。每个新类最多由SingleDay第4天800条与ManyTx前三天各50条组成，共950条；20类为19,000条。总计43,000条，扣除`26类×20×3=1,560`条support，得到41,440条query。7个receiver共290,080条query。

### `R_sd3`

每个receiver为`26×800=20,800`条，扣除1,560条support得到19,240条query。3个receiver共57,720条query。

### `R_mtx7`

若每类四天均为200条，每个receiver为`26×200=5,200`条，扣除1,560条support得到3,640条query。7个receiver共25,480条query。

三层条件上限为：

```text
receiver = 17
support = 26,520
query = 373,280
```

当前5 receiver、20新类、K=20均衡方案为132,200条query，因此条件上限约为其2.82倍。去重审计完成前，该数字必须标为`CAPACITY_UPPER_BOUND`；正式数量只来自最终canonical inventory和split manifest。

## ManyRx与原始WiSig索引

ManyRx不能单独支撑6旧类+20新类的完整Stage2-C receiver，但可以为重叠TX/receiver/day补充唯一物理记录。只有canonical ID证明其记录未被其他子集包含时才增加query。

如果完整WiSig原始索引可访问，builder应直接从原始索引创建canonical pool，四个`.pkl`只承担兼容引用与交叉检查。若只能使用四个`.pkl`，则采用保守去重：确认重复的记录合并；无法证明独立的记录不相加。

## 失败处理与可复用性

- 单个TX×receiver×day覆盖不足：只影响包含该单元的class/receiver profile；
- receiver无法满足三scene K-shot：从该K/profile移除receiver，不降低K或复制样本；
- 跨文件物理身份不确定：保守去重并记录`identity_resolution=conservative`；
- SingleDay主导MAXQ：保留全部query，但由day-macro指标揭示偏置；主表使用BAL4D；
- 某个split验证失败：只修复该split，其他`VALIDATED_ONCE`切片继续复用；
- checkpoint、候选方法或资源配置变化：不改变data capsule状态。

## 后续实现范围

设计批准后的实现分为四个独立工作包：

1. 只读inventory与跨子集重复审计，生成canonical coverage表；
2. canonical ID与source-asset crosswalk builder；
3. receiver/class eligibility与MAXQ/BAL4D split builder；
4. capsule materializer、聚焦协议负测和真实checkpoint无query smoke。

实现不得增加协议外gate、seal、receipt链、逐实验重复hash或全量125前置矩阵。第一次真实实验默认先使用单seed关键Target5/Target25或更小同row矩阵，达到预登记科学门槛后再扩展。

## 验收标准

- 同一物理记录跨四个子集只出现一次；
- `equalized=0/1`不会被错误计为两个shot；
- 每个physical ID只有一个scene、seed和received IQ；
- 三scene物理ID互斥，support/query物理ID互斥；
- 每个receiver/class/scene的support数严格等于K；
- `MAXQ_ALL_UNIQUE`包含除support外的全部合法唯一物理样本；
- `BALANCED_4DAY_CORE`在receiver、class、day和scene上使用冻结的共同容量；
- runtime capsule不含clean/source/query truth和quota信息；
- 同一split可跨不同checkpoint和方法复用；
- inventory、split manifest和独立统计给出一致的最终样本数。
