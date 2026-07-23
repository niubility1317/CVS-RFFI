# CSIL与MoPC-HR官方仓库执行锁

日期：2026-07-23

状态：`METHOD_LOCKED_IMPLEMENTATION_PENDING`

## 1.权威顺序

正式对比运行采用以下顺序：

1.官方GitHub仓库中可执行的主方法路径；
2.官方论文用于解释方法意图和补充代码未说明的定义；
3.论文公式与公开代码冲突时，主运行采用`OFFICIAL_CODE_EXECUTION_SEMANTICS`，冲突逐项披露；
4.CVS只允许替换数据入口、输入长度、ADV3B02特征接口、类别数和小样本无法执行时的最小兼容处理。

旧运行`adv3b02_paperfull_ci_20260722_v7`及
`adv3b02_paperfull_newclass_no_leo_20260723_v1`采用混合移植语义，永久标记为
`LEGACY_ADAPTATION_CHANNEL_DIAGNOSTIC_NON_OFFICIAL`，不得代表官方方法性能。

官方版本：

- CSIL：`pcwhy/CSIL@8ce8637daf4dc60eeb1c56bff64c050c5b2353e9`
- MoPC-HR：`xmuLdz/MoPC-HR@ae6554316ad1a2175920e330133a2f103408bf78`

## 2.CSIL执行锁

主增量内核锁定
`ContinualLearning/WorkStage/CSIL.m`。仓库还包含多个实验变体和调度脚本，
其中部分活动行调用`noCSI`或棋盘掩码变体；这些不混入CSIL主运行。

### 2.1网络与扩展

- ADV3B02仅替换原始卷积特征提取器。
- 正式头为`z_id(160)→fc_bf_fp(C)→zeroBias Fingerprints(C)`。
- 新增`m`类时，`fc_bf_fp`从`C`行扩到`C+m`行，旧行保持，新行按
  `1e-4*Uniform[0,1)`初始化。
- Fingerprint矩阵从`C×C`扩到`(C+m)×(C+m)`；旧块保留，交叉块为0，
  新块来自新类support的`fc_bf_fp`响应均值归一化。
- 零偏置输出为`5*cosine+5`，数值稳定项为`(1e-9)^2`。

官方初始化从扩展前`fc_bf_fp`响应截取最后`m`个坐标，仅在`m≤C`时有定义。
本矩阵新2/新5严格采用该路径；新10/新20超过旧类宽度6，使用扩展后的新增坐标，
并标记`CLASS_CARDINALITY_INITIALIZATION_ADAPTATION`。

### 2.2增量训练

- 3epoch，batch20，初始学习率0.01，时间衰减0.01，momentum0.9。
- 官方更新为
  `v=0.9v+lr*(grad+2*0.05*w)`，随后`w=w-v*mask`。
- 冻结ADV3B02旧特征提取器、旧`fc_bf_fp`行和旧Fingerprint块；只更新新增
  `fc_bf_fp`行、bias和新增Fingerprint块。
- 官方全局随机60%训练切分和floor丢尾在K-shot小样本上可能产生0步。
  正式复现保留该行为，不补样本、不缩batch，并逐cell记录
  `official_zero_step_due_to_drop_last`。

### 2.3损失与Fisher

- Fisher不是CE梯度。官方对旧模型输出计算
  `mean(log(p-min(p_all)+1e-5))`的梯度，再逐元素取`exp(g^2)`。
- EWC为旧参数重叠区域的`ΣF(θ-θ_old)^2/2`，不按元素数归一化。
- KD为旧Fingerprint响应与新模型旧类响应的平方和除以固定32，再乘0.2。
- 总损失为`CE+EWC+0.2KD`。

### 2.4base状态

80条旧状态无效。正式状态必须由8400条source池重建。MoPC-HR按其trainer使用
全部8400条base训练；CSIL按`makeDataTensor.m`的精确下标将同一已打乱池切成
5879条base train和2521条互斥`cX` Fisher validation，并包含：

- CSIL专属训练后ADV3B02状态；
- base `fc_bf_fp`权重与bias；
- base Fingerprint矩阵；
- 上述旧模型全部参数对应的官方Fisher；
- source样本计数和构建参数。

## 3.MoPC-HR执行锁

主执行路径锁定`MoPC_HR_trainer.py`。

### 3.1网络与阶段

- ADV3B02替换官方ResNet1D特征提取器，特征维数从256改为160。
- 采用预分配全类宽度的`nn.Linear(160,total_classes)`，保留bias。
- base阶段只用base类source训练；增量阶段只用当前新增类support训练。
- 每阶段20epoch，batch16，SGD，lr0.01，momentum0.9，
  weight decay`2e-4`。
- ADV3B02在base和增量阶段均参与optimizer；上一阶段模型只作reference。

### 3.2真实优化目标

- CE使用当前阶段样本，但logit范围覆盖全部已注册类。
- HR按`list(model.parameters())`的逐参数tensor顺序设置
  `λ_j=1-(j-1)/J`，使用`λ_j||θ_old-θ_new||_2`，不是平方L2。
- protoAug按每个minibatch生成同数目的旧类伪特征；旧类均匀有放回采样，
  加`Normal(0,0.05)`，分类logit除以2后做CE。
- KD会被计算和记录，但官方总损失未包含KD。
- 实际总损失为`CE+protoAug+HR`，HR等效系数为1。

### 3.3原型校正与推理

- 校正相似度为`old_proto @ new_previous.T`，随后按行softmax。
- `delta=new_current-new_previous`。
- `old=0.97*old+0.03*(softmax@delta)`。
- 校正原型只供后续增量阶段protoAug使用；当前阶段query始终使用模型线性
  分类器的全部已注册logit argmax。

官方main展示的增量间隔为25、10、5、3。CVS矩阵的新5/新10直接对应公开
间隔；新2/新20仍走同一参数化trainer，但标记`CLASS_SCHEDULE_ADAPTATION`。

### 3.4论文与代码冲突

| 项目 | 论文 | 官方代码主运行 |
|---|---|---|
| MPC相似度 | cosine | raw dot+softmax |
| HR粒度 | 网络layer | parameter tensor |
| HR形式 | 平方L2 | 非平方L2 norm |
| HR系数 | β未给出数值 | 等效1 |
| protoAug温度 | 未写 | logit除以2 |
| KD | 描述蒸馏 | 计算但不进入总损失 |

因此正式声明只能是
`official-github-execution-aligned with ADV3B02 and CVS data-interface adaptations`，
不能声明为逐论文公式复现。

## 4.必要接口适配

允许：

- ADS-B/AIS路径替换为WiSig/ManySig数据入口；
- 输入长度替换为256；
- 原编码器替换为ADV3B02的160维`z_id`；
- 类别数和标签映射；
- 正式运行的新类support/query叠加固定LEO弱信道；
- 修复MoPC-HR中不存在的`self.args.batch_size`和硬编码256维；
- query延迟到模型状态锁定后打开。

不允许：

- 将CSIL换成固定32维附加投影；
- 将MoPC-HR换成原型最近邻query；
- 混用论文cosine与代码HR；
- 用80条样本或target-old fallback代替完整source base状态；
- 让query参与训练、选择、早停或重跑。

两个方法的原始编码器均由预训练ADV3B02替换，并在8400条source上执行各自base
训练。该项固定标记为`BASE_MODEL_INITIALIZATION_ADAPTATION`；它是用户指定
ADV3B02接口的必要变化，不得描述为原论文编码器逐结构复现。

## 5.实现前parity门禁

1.CSIL Fisher、EWC、固定`/32`KD、SGDM衰减更新逐值fixture。
2.CSIL新增行/块初始化和所有旧参数hash不变。
3.MoPC CE、`/2`protoAug、逐parameter非平方HR及total逐值fixture。
4.MoPC KD对current gradient贡献为0。
5.MoPC raw dot+softmax校正逐值fixture，且与论文cosine结果明确不同。
6.同阶段query不读取校正原型，后续阶段protoAug读取上一阶段校正状态。
7.K1适配后非0步；样本数不小于batch时仍等于floor批数。
8.正式base样本数必须为8400，旧80条状态必须被schema拒绝。
