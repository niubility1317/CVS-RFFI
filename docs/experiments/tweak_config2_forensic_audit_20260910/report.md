# Tweak论文数值失配：端到端法证审计计划

- 审计对象：已完成的`v6`官方LoRa Config2→Config1—4复现实验；只读核验，不改变代码、数据、远端release或启动V7。
- 目标：从网络、样本、数据流、目标函数、训练调度和评分链逐项确认实现是否逻辑正确，或者准确定位论文未公开而足以改变数值的条件。
- 证据等级：`VERIFIED`=可独立复现实证；`DEFECT`=实现或逻辑错误；`UNDERSPECIFIED`=论文未公开且无法由现有产物唯一判定；`PENDING`=尚未检查。

| ID | 链路与论文要求 | 核验动作 | 通过条件 | 初始状态 |
|---|---|---|---|---|
| R1 | 论文原文与图13b/14 | 重提取Eq.(1)、IV-A、VI-A、VI-C及图中同row基准 | 逐条映射，无臆测补充 | PENDING |
| R2 | 网络图8 | 层序、核尺寸、通道、池化、BN、LeakyReLU、12-D输出、参数量的代码/运行时双核验 | 与论文逐层一致 | PENDING |
| R3 | 共享网络语义 | 验证三元组损失对共享同一encoder的三个输入产生梯度；检查BN训练/推理行为 | 无分支权重或BN状态误用 | PENDING |
| D1 | 官方原始样本 | 全80个`.dat`/SigMF元数据、cf32字节数、每条156,250帧、设备与配置命名 | 输入完整且没有dtype/截断错误 | PENDING |
| D2 | IQ读取与帧边界 | 独立numpy基准对比I/Q顺序、128点边界、memmap索引、连续性和浮点值 | 读取张量逐元素相同 | PENDING |
| D3 | 75/25与校准N | 检查每record物理索引双射、train/test不交、校准为训练集10%，以及是否存在训练/测试泄漏 | 角色映射与论文一致或明确标注未公开 | PENDING |
| D4 | 10个发射机 | 审计设备1—10选择与完整25设备母集；检索论文/公开材料是否给出ID | 若无ID则标UNDERSPECIFIED，不把默认当作者设置 | PENDING |
| B1 | mini-batch构成 | 审计64样本的类数、每类数、轮换、每epoch使用覆盖与遗漏 | 实现和方法元数据一致；论文未公开部分隔离 | PENDING |
| B2 | 样本时间与M=10 | 检查物理帧随机化、训练序列、测试10帧聚合是否意外跨越或破坏采样语义 | 无索引越界/交错/标签漂移 | PENDING |
| L1 | Eq.(1) | 独立数值/自动求导核验平方L2、margin=.1、ReLU、符号和均值归约 | 手算、实现和梯度一致 | PENDING |
| L2 | hard mining | 枚举batch有效有向triplet、严格`dAN<dAP`mask、活跃比例、损失/梯度的对称性 | 不漏/多选；明确全量平均是否合理 | PENDING |
| L3 | 塌缩机制 | 对V6 checkpoint与可控batch追踪类内半径、类间距、参数/梯度范数；比较train/eval | 定位塌缩发生层与目标关系 | PENDING |
| T1 | LR搜索与重置 | 审计五个probe是否从同一初始权重开始、optimizer动量是否重置、选择规则是否使用测试 | 无状态串扰或test选择 | PENDING |
| T2 | epoch/ckpt调度 | 审计18310batch/epoch来源、seed、best epoch保存条件、日志—checkpoint一致性 | 可从原始日志和checkpoint读回 | PENDING |
| T3 | 数值稳定性 | 审计V6全训练NaN/Inf、梯度与参数尺度，并重现batch-hard对照NaN条件 | 错误与方法性失败分开 | PENDING |
| E1 | Algorithm1校准 | 独立复算每类centroid与平均L2 radius；检查N=11718和标签排列 | 逐元素/误差阈值一致 | PENDING |
| E2 | Algorithm2与M=10评分 | 独立truth-last复算4×4和联合校准；检查包含半径、最小A/B规则与分组 | 分数精确复现且truth未进入训练/校准 | PENDING |
| X1 | 版本和远端链 | 核验V6源码commit/archive、N607环境、完整日志、results与checkpoint哈希 | 本地/远端版本闭合 | PENDING |
| X2 | 反向逻辑审计 | 从最终分数反追每个函数入口、参数和产物；寻找未调用、错误默认、双重归约、训练/评估模式残留 | 无死路径或不合理连接 | PENDING |

## 执行规则

1. 先完成R/D/B/L/T/E/X读审计和最小临时诊断；没有确认根因不改生产代码。
2. 每项附精确文件、函数、运行时证据和限制；`UNDERSPECIFIED`不是“通过”。
3. 若出现`DEFECT`，只提出一个可测试的修复假设，先写失败测试，再决定是否需要新的不可覆盖run根。
4. V7只在审计收敛为单一、可验证的根因且用户确认设计后启动；不得以低分或空闲GPU作为启动理由。

## 已完成证据与判定（2026-09-10）

| ID | 判定 | 证据与结论 |
|---|---|---|
| R1 | VERIFIED | 对论文方法文字、Eq.(1)、图8、训练段和配置可移植性实验逐项复核。论文明确：原始IQ为`2×128`、margin=`.1`、SGD momentum=`.9`、batch=`64`、训练`100`epoch、N=`10%`、M=`10`，并只规定从mini-batch中选取`d(A,N)<d(A,P)`的triplet；未公开采样器、设备ID、物理frame split、离散LR判据和best-checkpoint判据。 |
| R2 | VERIFIED | `TweakEncoder`运行时形状为`2×128→128×122→128×118→pool→256×53→256×49→pool→6144→256→12`，2,218,508参数；卷积核、通道、pool、BN、LeakyReLU和12维输出与图8相符。图8没有最终输出激活层。 |
| R3 | VERIFIED | V6真实官方batch的前向/反向全部有限，16组参数梯度均存在；训练入口在一个共享encoder上对完整mini-batch前向后挖掘，校准/评分均显式`eval()`，没有分支权重或BN状态被错误复用的证据。 |
| D1 | VERIFIED | 四个Config各有25个`IQ_1…25.dat`，每条20,000,000个`cf32`复样本、160,000,000字节，即156,250个128点frame；正式V6消费Config1…4的1…10号40条记录。 |
| D2 | VERIFIED | 对Config2/IQ_1的frame`0,1,17,1024,117186,117187,156249`以独立NumPy memmap逐元素复算，I/Q、frame边界和float值完全相同，最大绝对误差0。 |
| D3 | VERIFIED / UNDERSPECIFIED | 实现的156,250个物理frame经双射仿射置换后分为train=117,187、calibration=11,718（train子集）、test=39,063；train/test不交且并集完整。论文只给75/25和N=10%，并未给物理frame选取/顺序，故不能把本实现的`stride=104789,offset=104658`当作作者设置。 |
| D4 | UNDERSPECIFIED | 完整官方母集确有1…25号设备，V6使用1…10；论文称10个设备但没有公开其具体ID，不能证明这十个就是作者使用的十个。 |
| B1 | VERIFIED / UNDERSPECIFIED | 每batch为8类×8样本=64；每epoch18,310个batch，每类使用117,184个source frame，仅因整batch约束遗漏3个。无标签漂移、越界或重复frame。论文未公开每batch的类别/样本组成。 |
| B2 | UNDERSPECIFIED | 每个M=10组只聚合test逻辑序列相邻10个frame；仿射映射使它们在物理record上间隔104,789个frame，既无重叠也非时间连续。论文未说明M=10是连续、随机还是何种事件分组；这是影响数值的未公开条件。 |
| L1 | VERIFIED | Eq.(1)按平方L2、`.1`margin、ReLU和均值实现；独立四点手算得到9.6，自动求导有限。V5前的raw-L2已被排除，V6正式入口使用平方距离。 |
| L2 | DEFECT | V6把batch内全部25,088个有效有向triplet中11,784个严格hard样本直接平均。可重现实证显示初始batch损失随uniform embedding scale从1降到`.01`而从`.58281285`降到`.10004829`，径向梯度内积为`+0.9656255`，梯度下降必然缩小全局尺度。故“全量严格hard均值”是内在退化的实现解释，不能作为论文唯一复现；论文只给filter而未给该平均规则。 |
| L3 | DEFECT | 保存的V6模型同一真实batch仍有46.97%严格hard triplet，但loss=`.10000002`；embedding点积径向梯度已接近零，类内半径`6.088e-5`大于平均类中心距`2.893e-5`。这是上述目标诱发的表示塌缩，不是NaN/OOM或仅BN eval状态。 |
| T1 | VERIFIED / UNDERSPECIFIED | 五个probe均由同一初始state加载、各自新建SGD+momentum并只读Config2 source；选中的`.01`满足first/last window下降。论文仅说在`1e-2…1e-6`中确保loss下降，未给grid、probe时长或tie-break；本规则不是作者可验证设置。 |
| T2 | VERIFIED | 全日志106条=5 probe+100 epoch+1 complete；每个epoch均18,310 active batch。最低epoch loss为第95轮`.10000000524104195`，与checkpoint/results metadata完全一致。 |
| T3 | VERIFIED | 完整日志无Traceback、error、exception、NaN、Inf、OOM或Killed，checkpoint全参数有限。V6失败是优化目标几何退化，不是运行时数值异常。 |
| E1 | VERIFIED | 每个Config校准均为117,180个12维embedding（10设备×11,718）；centroid和平均L2 radius采用与release相同的Algorithm1计算，无训练权重更新。 |
| E2 | VERIFIED | 在N607原release、原数据、原checkpoint和Torch 2.1.0+cu121上，以独立一次性scorer重新计算4×4单配置和4条联合校准，20/20行的accuracy和correct decisions均逐位等于`results.json`。truth只在最终比较prediction时进入。 |
| X1 | VERIFIED | V6源码提交`a234e6e6`、预登记`1fe76128`、release archive SHA256=`73c00d…63c0f14`、完整日志/结果/checkpoint读回SHA都已闭合；N607实际环境Torch 2.1.0+cu121、CUDA。 |
| X2 | DEFECT（非V6正式入口） | `training.py:shared_triplet_loss`仍使用raw L2，与Eq.(1)平方L2矛盾；README仍写旧`batch-hard`而V6正式runner为全量strict-hard。这两处不会进入V6正式命令，但会误导后续调用，必须在任何新设计前统一。另：释放版predict用`torch.cdist`，手写`vector_norm`在已塌缩embedding上改变20行决策；不是release评分闭环错误，但证明其数值排序已不稳定。 |

## 审计结论与边界

1. 输入、IQ读取、网络拓扑、训练/评估模式、完整训练调度和最终评分闭环均已核验，未发现能够单独解释低分的数据错位、泄漏、score truth提前接入或远端版本错配。
2. 已确认的实质性问题是V6对论文未定义的hard-triplet集合采取“全量严格样本均值”，其目标本身存在可证明的缩尺度塌缩方向；V6结果因此不能被用于判断Tweak原论文算法无效。
3. 尚不能把任何单一替代采样器称为“原作者算法”。精确设备10选取、75/25的frame规则、M=10分组、hard miner、初始化/BN细节、LR选择与best epoch均仍是影响数值的未公开条件。
4. 没有修改生产代码、数据、N607 release或启动V7。本轮只读审计完成后，若要继续，必须先由用户确认一个单变量、可证伪的V7设计；同时应先修正`training.py`和README的内部一致性缺陷并加回归测试。
