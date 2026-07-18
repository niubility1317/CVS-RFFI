# Stage2轻型域适应与新类注册研发目标

版本：2026-07-18
状态：active goal定义
数据协议引用：`protocol_schema=p2_min_v1`

## 1. 单一总目标

基于ADV3B02 final checkpoint，研发并验证可逐样本部署的极轻型Phase2方法，使Stage2-B旧类目标域适应与Stage2-C新类注册同等有效，并同时解决多新类条件下的旧类遗忘、old/new混淆和持续floor类。开发阶段只用注册support选择方法与统一超参数；query等同测试集，只用于最终评分。

数据合法性只通过`p2_min_v1`的`VALIDATED_ONCE capsule_id/split_id`引用，不在本目标重复建设准入、hash或authority系统。固定接收IQ、物理ID和split未变化时，直接进行方法研发与实验。

## 2. 成功判据

### 2.1 K10主门槛

在锁定候选的独立确认矩阵上同时达到：

- target-old注册后总体准确率`old_acc_after_increment ≥ 92%`；
- 每个旧类确认集准确率`min_old_class_acc ≥ 88%`；
- `seen_new_acc ≥ 92%`，当真实seen-new TX数为5；
- `seen_new_acc ≥ 90%`，当真实seen-new TX数为10；
- `seen_new_acc ≥ 86%`，当真实seen-new TX数为20；
- 同row报告`old_acc_before_increment`、`old_acc_after_increment`、`seen_new_acc`、`H_old_new`和逐类混淆，不能跨row拼接最好值。

真实seen-new TX数2也必须评估和报告，但不另造用户未指定的绝对门槛；它用于小规模注册机制诊断。

### 2.2 K5与K1/K20

- K5在每种新类规模和每个核心指标上相对matched K10下降不超过3pp；
- K1、K5、K10、K20全部执行同row注册前/后遗忘评估；
- K1总体与每个receiver的`old_adaptation_gain=old_acc_after_increment-old_acc_before_increment ≥ 0`；
- K1在相同旧类query上明显优于直接ADV3B02旧类头：总体paired差值至少+2pp、matched paired 95% CI下界大于0、每receiver不为负；
- K5/K10/K20的平均遗忘不得高于matched identity-only单qKNN；
- K20用于检查support增加后是否饱和或反向遗忘，不能参与开发选参。

### 2.3 floor与混淆

必须同时优化总体均值和最低类，禁止用均值掩盖floor。优先监控历史持续floor：旧类`14-7`、`20-19`、`6-15`，新类`09f8`、`f608`，同时报告所有实际类别而不是只报这些已知难类。

候选必须降低两类失败：新类不可达和新类侵入旧类。不得再以support拟合100%、support侵入为0或LOO安全单独证明held query安全。

## 3. 开发与确认设计

### 3.1 开发锁定

- 只在预登记development seed、K10工作点，使用注册support内部预登记的leave-one-physical-sample-out或nested support-held代理选择一套候选、表征组合、adapter结构、loss、epoch和所有超参数；
- support内部开发代理同时覆盖Stage2-B旧类、模拟Stage2-C后的旧类/新类、调和均值、最低类和遗忘风险；真实development query与confirmation query都必须在候选和超参数完全锁定后只预测、评分一次，不得据结果继续调参；
- K1/K5/K20及独立确认seed不能反向调参；
- target query标签、角色、类别数量或指标不能参与拟合、早停、rollback、路由或候选选择。

### 3.2 独立确认矩阵

锁定后覆盖：

```text
target receivers: 5
confirmation seeds: at least 5, independent from development
LEO scenes: leo_clear_weak, leo_low_elev_weak, leo_rain_weak
K: 1, 5, 10, 20
real seen-new TX counts: 2, 5, 10, 20
```

若合法目标接收机或真实新TX覆盖不足，可使用未进入Phase1的其他WiSig/ManySig接收机或TX子集；仍须引用合法`p2_min_v1` capsule，且不得使用clean样本。

历史“125任务”保留为候选稳定性screen：运行`5 receivers × 5 independent confirmation seeds × 5 evaluation slices = 125 jobs`。五个slice固定为`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`和`K1/new20`；每个job内部都评估三个LEO场景。比较方法作为同一job/同一数据上的matched候选列或配套结果，不构成125的第五轴。125 screen不能替代完整K1/5/10/20×new2/5/10/20正式矩阵。

## 4. 研发路线优先级

### 4.1 第一优先：先提高Stage2-B旧域头

当前可复核最强比较器B3约为注册前old86.67%、注册后old73.33%、new73.33%，说明域适应和注册均未解决。下一路线必须先在support-held old proxy和旧类floor代理上超过B3，再扩展注册机制；冻结后才在真实held query评估，不再以继续叠加hard visibility gate替代旧域适应研发。

优先尝试同一固定接收IQ上的规范化拼接表征：`z_id160 + FFT96 + RF32`，并做分块L2归一化、能量/温度可学习缩放和matched ablation。维度增加不是目的；只有support-held代理性能与proxy Pareto改善才进入冻结query评估，真实held query只用于screen/晋升判断，不再调参或修改候选。

### 4.2 第二优先：连续联合适配与注册

在一个全注册类空间内联合训练target-old和target-new support：

- ground old int8聚合知识只作只读身份先验、正则或不确定度参考，不直接覆盖target原型；
- target-old原型负责域校正，target-new原型独立注册；最终部署的target-old和target-new原型均须量化，优先int8；FP16/FP32只作为matched精度/速度/状态ablation，按Pareto证明最终格式；
- 采用class-specific cosine head、正值对角度量加极低秩残差或其他低参数连续变换；
- loss同时包含old support分类、new support分类、ground-anchor弱正则、类内半径收缩、类间`margin > radius_i + radius_j`、old/new collision惩罚及adapter幅度正则；
- 只允许support梯度，使用快速闭式或少步梯度更新；不用binary visibility/hard release作为主学习机制。

### 4.3 遗忘保护

- 注册前冻结一份旧类决策状态作为teacher/anchor，仅用于support侧蒸馏与参数位移约束；
- 对每个旧类按support不确定度决定target校正与ground int8先验的融合强度；
- 新类追加必须是append-only class state，不重写ground int8组件；
- 用support-held/leave-one-physical-sample-out风险约束连续margin，但其只作开发代理，不能替代query确认；
- 若发生“新类不可达”或“旧类侵入”，优先修正连续几何、校准和loss，不再增加多层hard gate。

## 5. 资源与部署约束

首选Pareto目标：adapter可训练参数不超过50k、适配不超过20epoch、持久化增量状态尽量低于256KB。

正式硬上限：

```text
trainable adapter parameters ≤ 80,000
adaptation epochs ≤ 30
optimizer steps ≤ 50
persistent incremental state ≤ 256 KB
dense query graph = false
query-dependent batch optimization = false
```

为机制探索允许单独使用150%档：不超过120k参数、45epoch、75 optimizer steps和384KB。探索档不能进入正式确认或部署Pareto；正路线必须压缩回正式档再重验。

相对identity-only单qKNN报告增量MAC、平均/P95时延、峰值显存、backbone/FFT前向次数和状态字节。三种强制matched对比方法固定为`identity-only single-qKNN`、`ProtoNet CDA`和最强合法target-support-only轻适应基线；direct ADV3B02另作0-support性能/资源锚。目标是在性能达标后，使新增adapter/注册状态和适配计算低于三种对比方法；不虚构“比direct ADV3B02的0-support状态更少”的要求。

## 6. 执行节奏与止损

1. 直接复用`VALIDATED_ONCE`数据，先做最窄matched开发实验；不因算法变化重做数据封装。
2. 每个新机制必须回答它修复的是旧域不足、floor、新类不可达、旧类侵入还是量化/资源；没有机制假设不启动大矩阵。
3. 连续三个完成的探索轮后做一次技术复盘，审查完整日志、逐类/逐receiver结果和同row注册前后性能，再决定第四轮。
4. 只在开发seed显著优于B3并改善floor后启动125 screen；通过screen后才进入完整独立确认矩阵。
5. 兼容性、loader或报告问题只做最小修复，不把外围工程包装成研发完成。

## 7. 完成证据

完成必须同时包含：

- 合法TX/receiver/scenario/K/support-query清单及`capsule_id/split_id`；
- 每个run的锁定candidate、seed、配置、checkpoint/bundle ID和完整训练/闭式求解日志；
- 同row注册前/后old、seen-new、`H_old_new`、遗忘、所有逐类和逐receiver结果；
- 注册前Stage2-B对target-new只报告`not-yet-enrolled reference`，例如被旧类吸收率或score margin，不得称为`seen_new_acc`；注册后Stage2-C才报告`seen_new_acc`，并同时保留before-reference与after结果；
- 5 receivers×至少5 seeds×3 scenes的独立确认矩阵，覆盖K1/5/10/20及真实新类2/5/10/20；
- target-old与target-new原型量化格式、量化误差、状态字节和append-only生命周期；
- adapter参数、epoch/step、MAC、平均/P95时延、峰值显存、前向次数、持久状态的资源审计；
- 相对上述三种强制matched对比方法及direct ADV3B02锚的同row Pareto表；
- 自动化报告、异常/失败说明、可复现命令和Git提交。

只有技术证据齐全且上述性能门槛全部通过，才能标记目标完成；完成实验矩阵但性能未达标，应明确记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不能改写为成功。
