# PAMF Phase2 第一比较组：域适应、新类注册与资源开销

日期：2026-10-30。状态：**DESIGN_ONLY，训练继续暂停**。本文替代旧第一组的比较逻辑，不撤销旧停止标记，不覆盖历史产物，不构成新的启动指令。机器可读设计见 [配置](../configs/pamf_phase2_group1_design_20261030.json)。

用户已确定：A 为各方法自己的源训练，B 为旧类目标适应，C 为新类注册；A 保留各自完整监督方法并于中期加入普通 PL；纯度量方法以距离 softmax 最大概率≥0.95 筛选伪标签。DURA 使用 support/query 四格，比较识别性能和实际计算资源，并覆盖不同目标接收机。本文新增的温度、主工作点及分层矩阵属于具体设计建议，不冒充历史已执行设置。

## 1. 比较问题与阶段

| 编号 | 比较对象 | 路径 | 回答的问题 |
|---|---|---|---|
| G1-B | PAMF 与域适应方法 | A→B | 跨接收机、星地 received 条件下，少量旧类 support 能带来多少收益、耗费多少资源？ |
| G1-C | PAMF 与原生注册方法 | A→C | 跳过 B，注册新类后的旧类、新类识别和注册成本如何？ |
| G1-BC | PAMF 部署过程 | A→B→C | 先服务旧类再注册新类时，状态与增量成本如何？ |
| G1-G | PAMF 的 DURA 因子实验 | 上述三条路径的四格 | support 恢复、query 恢复及其交互分别影响什么？ |

A 是训练，不是测试。源训练冻结后、B 前的旧类只读评估记为 **A0**。A0 与 B 在相同旧类 query 上配对；C 中所有 query 一律面对 6+N 个已注册类，不按真实旧/新身份路由。

第一组不加入“DA 方法＋NCM/FT/JointCE 注册”“CIL 方法＋B10 通用微调”等组合。原生阶段不存在就记 N/A。后续若研究组合，另立实验组，不能填入本组原生行。

## 2. 方法范围、版本与逐阶段继承

限定为现有清单中的 **8 个域适应家族、5 个注册家族**。同论文的代码路线不重复计数。RA-Collab 的单节点 FT 是此前补充候选，本组不自动加入；RadioNet FT/Triplet、旧 Feature Separation、MoPC public-code/fixed 和 ISSL fixed 等版本不自动变成额外主方法。

“原生”指保留论文的任务机制，不表示 WiSig、256 点 IQ 和半监督源训练已经逐位复现原论文。各行均注明输入移植和本组 A-PL 扩展；有标签 SDA 另作明确标记。

### 2.1 G1-B：域适应表

| 方法行 | A：源训练要求 | B：原有适应/判决机制 | C | B 继承及 source 权限 | 身份与当前限制 |
|---|---|---|---|---|---|
| PAMF-PL＋MILKR | 三视图 SIG-TIME；完整 PAMF-Sup＋普通 PL | F 冻结；用旧类 support 拟合 local_ridge | 另在 G1-C | A 的 F；只用当前 support，不读源 replay/原型 | PAMF 本组普通 PL 变体；不冒称历史完整 REPL |
| ProtoNet CDA/PTN | 自己的 episode 编码器和原型负对数似然 | 冻结编码器，用目标旧类 support 建原型、距离判决 | N/A | A 编码器；A0 原型仅由 L_s 形成，B 用当前目标原型 | 原生 CDA 机制＋WiSig/PL 适配；不共享 PAMF 骨干 |
| MRIOR-SDA | 原接收机影响抑制方法的源模型及监督目标 | 保留 GAD、DV-KL 等已绑定机制；目标 support 监督按现有 SDA 版本 | N/A | A 模型/估计器，允许原机制所需 L_s replay | **原论文 UDA 的项目 SDA 扩展**，不可写成原生 SDA |
| DADDA-SDA | 原 DADDA 源模型及源监督目标 | 动态分布对齐/LMMD，使用 support 真标签的既有 SDA 版本 | N/A | A 编码器、分类器/对齐状态，原机制所需 L_s replay | **闭集 UDA 的项目 SDA 扩展**；不扩头注册 |
| Two-stage UDA＋FT | 独立的两阶段方法源模型 | 原顺序域对抗→条件对齐→support 微调 | N/A | A 状态；原机制所需 L_s replay；B 内三步连续继承 | 原生两阶段机制＋当前 support-only 任务适配，不与 MRIOR 混同 |
| Feature Separation / Hu2024 | 原 TX/RX 分类、分离与熵目标的完整源网络 | 原 few-shot 适应；按原规则冻结 target RX 分支 | N/A | A 的共享编码器及 TX/RX 分支；只开放合法 B 输入 | 同 DOI 一家族；不使用 legacy 普通 CNN 代理 |
| POSTER Fine-Tuning | 原 POSTER/Homegrown 网络和监督分类 | 原 dense-layer reset 与微调范围 | N/A | A 特征层；按作者规则重置指定层 | 原生 FT 机制＋数据/PL 移植 |
| RadioNet ADA＋kNN | 原 ADA 路线的源监督模型 | 完整 ADA 对齐后冻结特征，target support 建 kNN | N/A | A 网络及域判别器；合法 L_s replay；保留实际 kNN | 不能只用 DF 骨干或普通 FT 代替 ADA；当前 k=K 为任务绑定，须披露 |
| Tweak | 原度量编码器、batch-hard 损失 | 原中心与半径校准、多观测判决 | N/A | A 编码器和合法源状态；B 当前旧 support | **原生 M=10 尚缺合法观测组输入**；M=1 是明确输入协议扩展，不能冒名进入原生主表 |

本组沿用此前已点名的 MRIOR-SDA、DADDA-SDA，不同时新加原生 UDA 行扩大矩阵。其他原生 UDA 环节只可读取当前 support 的 IQ；query 不能充当额外无标签适应池。所有 B 参数在目标访问前固定，B 不按 query 或 support 重代入训练准确率早停/选择超参。

### 2.2 G1-C：新类注册表

| 方法行 | A 必须形成的状态 | C 操作 | C 从何开始 | 目标旧/新 support 的实际使用 | 身份与当前限制 |
|---|---|---|---|---|---|
| PAMF-PL＋MILKR | 冻结 F，无需源 enrollment | 旧、新 support 联合完整重拟合 local_ridge | 直接从 A 的 F | 必须 6K 旧＋NK 新 | 原有 local_ridge 直接 C，不加新头或源原型 |
| CSIL | 原通道分离网络、source-only 教师与 Fisher 等原有状态 | 原通道扩展、隔离/mask、KD/EWC | A 状态，**无 B10** | 新 support；旧类按原合法源/历史机制，不自动加入旧目标 replay | 原 MATLAB/Fisher 路径当前未合法绑定；empirical-Fisher corefix 只能明确列修复版，不能冒称作者原版 |
| MoPC-HR | 原源编码器、旧类原型及原生历史状态 | 动量原型校正、原型增强、层次正则；按原 session 递进 | A 状态，**无 B10** | 新 support；旧信息来自 A 的原生原型/教师 | 本组选论文方程语义复现一版；public code 的损失差异单列附注，不算第二方法 |
| Orthogonal Incremental SEI | 原基类几何训练、正交/分类状态 | 原冻结规则、新类权重初始化和增量校准 | A 状态，**无 B10** | 新 support；旧类状态继承 A | 保留原空间约束机制；不能只用 NCM 代理 |
| ISSL | 原监督源网络及其原生历史/教师状态 | 原自监督与知识迁移注册过程 | A 状态，**无 B10** | 按原输入要求使用当前注册 support；旧信息来自 A 的合法历史状态 | 作者行为与 repaired-v1 不混用；原始 detach/梯度累积/queue 行为须在实现绑定时逐项注明 |
| scalable/channel-robust LoRa RFFI | 原 triplet 编码器和 L_s enrollment gallery | 冻结特征，追加新设备 enrollment，原 kNN 判决 | A 编码器和旧 gallery | 新 support；旧 gallery 只来自 L_s；不自动替换为目标旧样本 | 原生 enrollment 机制；WiSig短 IQ/STFT/Torch 移植要注明，不等同原 LoRa 数据复现 |

这五种外部方法不需要共同先做 B；各方法 A 的完整原有监督机制必须保留。当前 stopped runner 中“统一骨干＋统一 B10”的代码不能直接作为本组执行器。

### 2.3 共同可用标签与实际使用标签

主工作点 K=5、N=20：B 可用 30 个旧类目标标签；C 可用 30 个旧类＋100 个新类目标标签。A→C 只是跳过域适应优化，**不表示取消旧类 support 标注**。同一数据角色对所有方法可用，算法只消费其原机制需要的部分。

PAMF C 确实使用旧、新目标标签；若某 CIL 只使用新目标标签，则表中必须写成“旧目标标签实际使用 0，新目标标签 100”，并同时列源历史内存。**相同可用上限不等于相同实际用量**。不能仅凭该主表声称严格等使用标签量下的算法优越性。必要时另附“仅新标签”制度表：原生 CIL 可运行，当前 PAMF local_ridge 记不适用；不通过暗加源原型补齐 PAMF。

## 3. local_ridge 能否直接注册与 B→C 的真实含义

代码依据：`experiments/pamf_phase2_practical/milkr/d92_branch_local_ridge.py::fit_branch_local_ridge` 和 `d92_branch_interaction.py::_prepare`。拟合接口只有当前特征、support 标签/ID、类别及 arm，不接收 B state；准备函数要求每个已注册类均有相同正 K。

因此：

1. **可以 A→C**：F 不变，对当前 6+N 类 support 提特征并拟合。
2. C 不能只给新类 support 却继续按现有实现保留旧六类。
3. 相同 F、support、顺序规范化和处理路径下，AC 与 ABC 的 C 核系数及分数相同。旧类 B 的核带宽、中心化和系数不会成为 C 的初始化。
4. ABC 的价值是阶段 B 已能提供适应后的旧类服务；注册发生时再完整拟合当前所有类。其终态不构成第三种更强算法。
5. B 和 C 在一次任务内可复用冻结 F 的旧 support 特征；C 的带宽和整个核系统重新计算。不得把完整重拟合称作秩更新。

资源计费应为：`ABC在线成本 = B成本 + C新增到达后的增量成本`。如保留旧 support 特征，C 增量只新增新 support 的 G/F 前向，但仍完整拟合所有类。`AC直接C成本` 包含全部旧、新 support 提特征。不能把 B 成本与已含旧特征成本的完整 AC 成本直接相加造成双计。

## 4. A：完整监督方法＋中期普通 PL

### 4.1 源数据与方法身份

所有 A 使用相同 source RX、source day 和旧六类；L_s=6300，U_s=56700，V=27000。U_s 的 TX 真值不导出。使用同一合法源数据合同，不代表强制使用同一骨干或同一监督损失。

`L = L_supervised_native(L_s) + 1 × L_PL(U_s)`。监督项保留原模型的分类、triplet/episode、分离、正交等完整必要组成。PL 是本组统一增加的项目半监督适配，故整行展示为“方法名＋普通 PL”，不能声称原论文原来就有这项半监督机制。

PAMF 使用当前三视图 SIG-TIME 结构，六类完整 F 为 232993 参数。本组 A 使用 PAMF-Sup 的完整监督骨架＋普通 PL，保持自然时序；不加入 REPL 的时序重排、样式扰动、梯度投影、U 对比或接收机外推。其名称写 **PAMF-PL（本组中期开启）＋MILKR**，历史 Tri-SIG-TIME SSL/REPL 的结果保持原名。

源训练保留原生、已绑定的 source-only 输入表示及有标签增强。PAMF-Sup 的原有合法 As 监督项等也保持；不为 LoRa/Tweak 新造星地 CE 头。不在本组额外追加共享目标信道增强、目标恢复或目标统计。各方法的源端增强差异及开销需列明，不能将“相同源池”描述为“完全相同训练视图”。

### 4.2 分类网络的普通 PL

用当前 student 在 eval/no_grad 下生成硬伪标签，不更新 BN 等持久统计，不另建 EMA。若原分类器已有 softmax，则直接采用原 logits 的概率。接受 `max p ≥ 0.95`；伪标签 detach 后进入原监督分类目标的普通 CE，伪标签 label smoothing=0。U 项以整个 U batch 大小归一化，空接受集合为 0。原 L_s 监督项的 label smoothing 不因 PL 改变。

每个原生监督更新最多配一个 U batch，建议 U batch=2×原生 L batch；不以 U loader 长度改变每 epoch 原有 L 更新次数。U 使用独立随机循环，记录唯一曝光和重复曝光。原生 episode 方法按原 episode 支持/查询结构构造 U 项，不能改为任意 CE 骨干。

### 4.3 纯度量网络：距离 softmax，阈值 0.95

用户已选择此路线，不增加可训练 softmax 分类头。对没有原生分类概率的 LoRa/Tweak，**仅伪标签筛选器**采用以下非参数规则：

\[
\bar z(x)=f_\theta(x)/\|f_\theta(x)\|_2,\quad
\mu_c=\operatorname{unit}\left(\operatorname{mean}_{(x,y)\in L_s,y=c}\bar z(x)\right),
\]
\[
d_c(u)=\|\bar z(u)-\mu_c\|_2^2,\quad
p_c(u)=\frac{\exp[-d_c(u)/T]}{\sum_{j=1}^{6}\exp[-d_j(u)/T]},\quad
\hat y=\arg\max_c p_c,\quad m=\mathbf1[\max_c p_c\ge0.95].
\]

具体建议固定 **T=0.1**，目标访问前锁定，不按接受率或目标分数调温度。单位化特征的平方距离在 [0,4]，若 T=1，六类最大概率的上界仅 `exp(4)/(exp(4)+5)≈0.9161`，无法达到 0.95。T=0.1 是距离尺度约定，不是准确率校准结论。

每个 PL epoch 开始，用当前编码器 eval/no_grad 对 **L_s** 生成六类中心；该 epoch 固定中心，下一 epoch 刷新。中心不使用 V、U 真标签、目标样本或 query；计算开销归 A。零向量/非法特征作为数值失败显式记录，不静默换成随机中心。

接受的 U 作为伪标注 anchor，与当前 L_s 中同类正例和异类负例按原规则组成 triplet/batch-hard 损失；正负参考来自真实 L_s，U 不互相担保标签。U 度量项按 U anchor 全 batch 归一化，未接受 anchor 权重为 0。保留原 margin、挖掘方式及距离；上式归一化与 T 只用于筛选器，不能静默改原监督度量损失和部署 kNN。ProtoNet 已有原生距离概率和 episode NLL，保留其原尺度，并固定 0.95 筛选；不额外强行统一成 LoRa/Tweak 的筛选器。

距离 softmax 的基本形式见 [Prototypical Networks](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)。0.95 只是模型分数，不能保证伪标签正确率 95%，也不能据此断言各模型置信度已经校准；参见 [Guo 等的校准研究](https://proceedings.mlr.press/v70/guo17a.html)。本组不额外加入 VAT、图传播、EMA 或复杂门控。

### 4.4 日程、预算与冻结

本组提出固定完整上限，前一半只监督，后一半监督＋PL；1-based 启用轮次为 `floor(E/2)+1`。为保证每个正式行确实完成中期 PL，**本组取消训练提前终止，冻结最终轮**；原生 optimizer/LR 调整保留并记录。此为明确的公共日程适配，不冒称保留了作者全部早停行为。V 仅监测/原生 source-only LR 调节与事先声明的 Tweak LR 选择，不读取目标选模。

| 方法 | 本组 A 完整轮次建议 | PL 起始轮 | 预算说明 |
|---|---:|---:|---|
| PAMF-PL | 50 | 26 | 保留对应完整监督更新表；若沿现有 222 步/轮则累计 11100 步，不因 U 改步数 |
| ProtoNet CDA | 200 | 101 | 现有项目 episode 预算；须记录 episodes/轮与 optimizer steps，不能宣称作者原数据轮数 |
| MRIOR-SDA | 200 | 101 | 现有 source 匹配上限；保留原源阶段组件 |
| DADDA-SDA | 200 | 101 | 同上 |
| Two-stage UDA＋FT | 200 | 101 | A 不提前执行目标 UDA；两阶段属于 B |
| Feature Separation/Hu2024 | 200 | 101 | 原匹配上限；不再因原 patience=20 提前结束 |
| POSTER | 200 | 101 | 现有 Adam 0.001、L batch128 为待绑定原配置的起点 |
| RadioNet ADA | 200 | 101 | A 只有 source；10000 次目标 ADA 属于 B |
| Tweak | 每 LR 100 | 51 | 原 source-only 五 LR 路线 0.01/0.001/0.0001/0.00001/0.000001；若保留搜索，五条均完成普通 PL 并计费 |
| CSIL | 200 | 101 | 原编码器与 Fisher 数值路径须先绑定；不能直接使用通用六层骨干代替 |
| MoPC-HR | 200 | 101 | 原基类监督和原型状态完整保留 |
| Orthogonal Incremental SEI | 200 | 101 | 原基类空间几何训练完整保留 |
| ISSL | 300 | 151 | 现有源预算；Adam 0.03、L batch128 属于已定位移植配置，不等同已证明作者数值复现 |
| LoRa triplet | 1000 | 501 | 原匹配最大预算；RMSprop 0.001、L batch32；本组固定上限会比原早停更贵，须如实披露 |

上述是可审查的训练预算设计，不是新 launcher 的 resolved config。尚未核实的作者 backbone、optimizer/margin/episode 细节在实现时按原入口绑定，不从通用 runner 猜默认值。原版与移植存在差异时写明；不能把该差异抹成“全部原生且已可运行”。

### 4.5 既有 Phase1 能否复用

仅当实际 L/U/V 物理角色、RX/day/TX、输入表示、架构、完整监督项、PL 损失/权重/中期日程、预算/冻结规则及全部祖先均匹配，才直接复用。同名网络、同 seed、相同六类不能代替检查。

纯监督 final-E200、晚期 PL、REPL 或只匹配数据但训练机制不同的 checkpoint，不能充作本组普通 PL 中期训练。也不能在一个最终监督模型后追加 PL 并改名“中期开启”。可以复用同契约合法 A-Sup 诊断或 source 数据预处理，但主行未匹配的 A 必须另行训练；本次设计不开始补训。DURA 固定权重及合法 received/DURA 缓存可以复用。

## 5. B/C 训练预算绑定

本组先保留已有、公开注明的目标阶段预算，不为公平表面上“统一 10 epochs”而删算法核心。以下为项目匹配预算，不代表原论文的原数据完整复现；实际 batch 数和每个子模块更新数必须输出。

| 方法 | B 预算设计 | C 预算设计 | 状态规则 |
|---|---|---|---|
| PAMF MILKR | 1 次旧类完整拟合，0 反传 | 1 次全已注册类完整拟合，0 反传 | C 不继承 B 核；可复用未改变的 F support 特征 |
| ProtoNet CDA | support 原型计算，0 反传 | N/A | 每个任务由 A 重新开始 |
| MRIOR-SDA | 100 outer updates；每 outer 7 次估计器更新；已定位 LR0.0006 | N/A | 不能把 100 outer 记成仅 100 optimizer steps |
| DADDA-SDA | 100 对齐更新，support 真标签条件对齐 | N/A | source/target 输入仅合法池 |
| Two-stage | 100 DANN＋100 LMMD＋100 support FT 更新 | N/A | B 内按原顺序继承 |
| Feature Separation | 100 epochs；已定位 Adam0.005、batch256 | N/A | 完整 attention/分离机制，不套 legacy 一轮一步 |
| POSTER | 10 epochs，Adam0.001，batch256 | N/A | 原指定层重置/微调 |
| RadioNet ADA＋kNN | 10000 ADA iterations，然后 kNN 建库 | N/A | 原域对齐和最终 kNN 都执行，不另加 FT30 主行 |
| Tweak | 原校准，0 梯度更新 | N/A | M10 数据绑定未完成 |
| CSIL | N/A | 已定位移植预算 3 epochs、masked SGDM0.01 | 原 Fisher 路径未绑定前不宣称原版可运行 |
| MoPC-HR | N/A | 每 5 新类 session 20 epochs、SGD0.01 | N20 为 4 sessions；N2 为最后不足 5 的短 session，明确任务扩展 |
| Orthogonal | N/A | 50 epochs、SGD0.08 | 保留原冻结范围/几何校准 |
| ISSL | N/A | SSL2000 epochs＋transfer300 epochs | 不缩成 B10 或一次 NCM；不能把固定版和作者行为混用 |
| LoRa | N/A | enrollment 与原 kNN(k=15)，0 梯度更新 | 保留 L_s 旧 gallery；K1 时 ties/退化须报告，不按目标成绩改 k |

不同 N 的正式结果均从同一 A 独立构造，不把 N2、N5、N10、N20 四个独立任务的成本累加成一个 N20。原生方法在单个 N20 内的递进 sessions 则必须连续继承并累加成本。新类顺序固定，结果不回流重排。

## 6. DURA 四格及共享计算

G 固定 DURA v1.0（R69 native_map_prior），266951 参数；FP32/TF32off，既有 `InferenceEngine(reuse=True, graph=True, native=False)` 路径；不在 B/C 更新 G。G 的源训练独立列成本，其原 F-free/TX 无标签权限不因本组改变。

| 用户编号 | 记号 (support,query) | B/C support | query | 需要的目标拟合状态 |
|---|---|---|---|---|
| 1.1 | (0,0) | 原星地 received | 同一 received | Raw-support 状态 |
| 1.2 | (0,1) | 原星地 received | 同一 received 经 G | 与 1.1 完全相同 |
| 2.1 | (1,0) | 同一 received 经 G | 原星地 received | G-support 状态 |
| 2.2 | (1,1) | 同一 received 经 G | 同一 received 经 G | 与 2.1 完全相同 |

每个 B/C 工作点只有 **2 次 support 拟合、4 路测试**。支持路径标志在 B 和 C 保持一致；本组不再增加 B raw/C restored 等八格组合。每格内 AC/ABC 终态等价，先用无 query 合成检查/实现接口确认，再复用 C 终态；B 服务性能与全流程成本仍各自报告。

四格全部报告，不按目标最高分挑选“PAMF 最终版本”。跨方法主表固定 PAMF(0,0) 为本组基础方法；DURA 四格单独主表展示，必要时并列 PAMF＋G(1,1)，但明确它增加 G，不能把它与外部方法的裸网络成本混在一起。

因子效应按配对数据计算：support 效应 Y10−Y00，query 效应 Y01−Y00，交互 Y11−Y10−Y01+Y00；分别对 B、C_old、C_new、H 与成本计算。恢复造成失配的两格也必须保留。

## 7. 三池数据、接收机、support/query 与信道

复用 `configs/pamf_phase2_threepool_episodes_r10_20261029.json` 的 VALIDATED_ONCE 数据。方法与 PL 改变不触发数据重造、逐文件哈希或重新审核数据。原始目标记录共 554823；P01/P02/P03 分别 184936/184944/184943，互斥且并集完整。每 RX/TX 最小池 196，K20 时最小 query=176≥150。

| 维度 | 固定设置 |
|---|---|
| source RX | 1-19、18-2、19-2、2-19、3-19 |
| target RX | 1-1、14-7、2-1、20-1、7-14、7-7、8-8，七个分别适应与报告 |
| target 日期 | 2021_03_01、2021_03_08、2021_03_15、2021_03_23 |
| 旧类 | 14-10、14-7、20-15、20-19、6-15、8-20，共 6 类 |
| 新类 | 原 MILKR 固定 20 类次序，取前 N；完整名单从 r10 配置继承，不重新按容量/成绩选类 |
| K | 1、5、10、20；同 episode 内 support 为同一排列前缀 |
| N | 2、5、10、20；B 的 N=0 |
| 每池综合 | MIX01–MIX08，8 组 |
| 每池历史单景 | LS01–05：low 郊区；MU01–05：mid 城市；LU01–05：low 城市 |
| source model seeds | 2026092701/02/03/04；不改变数据抽样或信道 |

每个 pool/episode/RX/TX 抽 K 个 support，其余**全部**为 query；不用固定 150 截断。池内不同 episode 可复用原始记录、改变 support/query 角色，各 episode 已有独立信道；同 episode 跨方法、K、seed、DURA 路径共享 received。跨 episode、K、RX、pool、source seed 的目标适应状态清空，不把其他任务学到的原型、BN、标签或参数带入。

A 训练器不接收目标路径；B 只开放旧类 support 和冻结后的旧类测试入口，未来新类 IQ 不提前进入算法；C 再开放当前注册前缀的新 support/query。离线 builder/scorer 的完整类别元数据不等于算法可以提前读取整 bank。

三个纯场景是三个独立实验条件：一个条件内所有样本统一该场景类型，但每条记录的衰落/噪声实现仍不同；不把三景混合当作历史单景复现。

星地参数：practical residual/post_sync/no additional EQ；600 km、25 MHz、2.45 GHz；原 FP64 IQ 前 256 点转 FP32，信道前一次 unit RMS，信道后不逐条重新归一化；receiver seed2027，session=RX/day。输入为固定星地 received，P0 不是 clean。

综合概率继续使用已制备的几何参考配置，不因本组重设：

| 环境 | high | mid | low |
|---|---:|---:|---:|
| 郊区 | 4.369141% | 7.317985% | 38.312874% |
| 城市 | 4.369141% | 7.317985% | 38.312874% |

实际运算使用原配置未舍入值。综合 low/mid/high 区间为 10°–30°/30°–45°/45°–80°；mid 城市单景保留原 20°–45°。该分布是单星被动监听宽视场的空间几何参考，不是已测实际活动频率，也不宣称中高仰角调度。权重问题不通过识别成绩反推。

## 8. 主表、附表与评价

### 8.1 论文主表布局

| 表 | 行 | 关键列 |
|---|---|---|
| 主表 1：旧类域适应 | PAMF(0,0)＋8 DA 家族；未绑定原版写 N/A | A0、B 类宏准确率、ΔB、Macro-F1、最差 RX、实际目标标签数、B 拟合时间、峰值显存/CPU内存、B 更新参数/步数、部署状态大小 |
| 主表 2：直接新类注册 | PAMF(0,0)＋5 注册家族 | C_old、C_new、H、类宏准确率/Macro-F1、最差 RX、实际旧/新目标标签数、源历史内存、C 时间/显存/状态 |
| 主表 3：PAMF DURA 四格 | (0,0)/(0,1)/(1,0)/(1,1) | B、C_old、C_new、H、support G/F/拟合成本、query G/F/判决时延、F/G/头存储 |
| 主表 4：PAMF 过程成本 | A→B、A→C、A→B→C | B 时刻性能、C 终态、B成本、C增量、累计成本、旧support缓存；明确 AC=ABC 终态 |

所有主表主要工作点为 **MIX、K=5、N=20、7 RX、3 池×8 episode×4模型 seed**。域适应表只含旧六类。注册表不以新类单独闭集准确率替代旧新统一竞争。

### 8.2 附表及曲线

1. B 的 K=1/5/10/20 曲线与 C 的 K×N=4×4 曲线；首版范围固定 MIX01，覆盖全部三池、七 RX、四 seed，主点与主表精确复用。
2. 低郊区、中城市、低城市三个纯景，各 5 episode，主 K/N，全部三池/RX/seed。六单景全展开不属于本次已制备三景矩阵，不以名称“六景”增加未建条件。
3. 每 RX、每 TX、每 day 分解；七个 RX 单列，跨 RX 等权宏平均、最差 RX、离散程度。目标端不能把七 RX 混池共同训练后仍声称逐接收机适应。
4. A 完整监督损失、PL 接受率/类分布、U 唯一曝光、训练曲线、实际 source cost 与 source 参数；不增加目标真值在线 PL 精度监控。
5. 原生/移植/数值修复/未绑定输入表；当前 CSIL author、Tweak M10 的 N/A 理由保留，不用代理结果填空。
6. 资源全量表：逐阶段及组件成本、并发条件、缓存条件、失败率和实际完成样本数。

第一组不强制再增加 MILKR branch_ridge/interaction_ridge 消融、额外 REPL 对比、14 套监督重训或泛化方法。这些属于其他科学问题；已有同契约源监督对照可作标明来源的附加背景。

### 8.3 指标与统计口径

B 主指标是六类宏准确率与 Macro-F1，配对增益 B−A0。C 分别计算旧类宏准确率 A_old、新类宏准确率 A_new，`H=2 A_old A_new/(A_old+A_new)`，并报告 6+N 类宏准确率、micro 准确率及全部分母。C 的旧类下降也包含新增竞争类的影响；冻结 F 的 MILKR 不应统称编码器灾难性遗忘。

先在每 pool/RX/model seed 内对 episode 求均值，再对三池、RX 等权汇总，最后报告四个模型 seed 的均值±SD。另报告池差异、同池 episode 波动和 RX 分布。重复 episode 是条件重复，不把 8×3 个仿真任务当作独立 RX/TX 总体样本；不把全部 query 当作独立实验重复计算极小标准误。比较使用相同池/RX/seed/episode 的配对差；不从四格里事后选最优再给单一显著性。

K 曲线仍以每个 K 的全补集为正式指标，另用已冻结预测在共同 query 交集（嵌套 support 时等于 K20 的 query）上计算配对差，避免测试样本变化混入 K 收益。此项只在评分端求交，不读取 query 拟合模型。目标四日包含较早采集日，不把该表称作严格未来时间泛化；历史已暴露基准也不改称新的独立盲测。

## 9. 资源记录：每次真实拟合必须同时完成

### 9.1 各阶段边界

| 成本 | 计时范围与必须单列的组件 |
|---|---|
| A | 数据读取/预处理、完整监督前半程、监督＋PL后半程、伪标签前向、中心刷新、反传、优化器、原生额外状态构建、保存；Tweak LR 搜索全部计费 |
| B | 当前旧 support 可用→数据传输→可选 G→F→原生适应/核拟合/建库→可部署 B 状态；query 不计入训练 |
| C-direct | A 已冻结、当前旧/新 support 可用→必要 G/F→原生 C→可部署 C 状态 |
| C-after-B | B 状态已有→新 support 到达→必要新数据 G/F→C；旧缓存/教师/原型是否复用逐项注明 |
| 推理 | 单个 query received→可选 G→F→全部已注册类判决；batch1 和固定 batch 吞吐分别报告 |

必须同时有端到端 wall time 与组件时间；CPU 核求解与 CUDA 时间分别记录，计时前后 CUDA 同步。CUDA event 不等于全流程时间。首次加载/编译/graph capture、预热、稳定重复分开；缓存路径不把省掉的 G/F 成本写成算法天然零成本。

### 9.2 每次拟合的记录字段

- 标识：方法/版本、native/移植/修复、commit、数据句柄、source seed、RX/pool/episode/K/N、support/query flag、物理 GPU、设备型号、精度、batch、并发、缓存状态。
- 时间：stage wall、数据/传输、G、F、伪标签、原型/核、forward/backward/optimizer、保存；开始/结束时间及完成状态。
- 更新：各子模块实际 epochs/batches/optimizer steps、完整模型参数、实际更新参数、梯度参数、额外教师/判别器/queue/optimizer state；MILKR 的系数/核状态不能计为 0 参数或 0 存储。
- 样本：source L/U 曝光、PL 接受数量/比例/预测类分布、旧/新 target 标签可用与实际使用数量、原生 replay 曝光、唯一物理样本数与重复使用次数。
- 内存：CUDA allocated 基线/峰值/增量、reserved、CPU RSS 基线/采样峰值与采样间隔、持续部署 F/G/分类器/gallery/历史状态字节数。整个进程生命周期的 high-water 不能冒充 C 独立峰值。
- 计算：F/G 神经网络 MACs/FLOPs 的约定与覆盖；CPU 核距离/求解另报复杂度和实测时间。参数个数不是 FLOPs。GPU worker-hours 与物理 GPU-hours 分开。
- 稳定性：NaN/OOM/数值失败、重试、未完成阶段与实际曝光；保留失败原产物，不自动换算法或缩预算。

功耗仅在可归因的控制测量中积分报告；共享 GPU 的整卡功率不能分配给某个 worker 后称该算法能耗。无法测得写 N/A，不用估算值冒充实测。

### 9.3 并发实验与控制资源测量分开

将来经用户重新授权启动后，精度矩阵维持本轮 **每 GPU 两个本任务 worker，其他 owner 另计**，启动前检查剩余显存，绝不停止其他任务。实际并发时间有争用，只能称运行时间。

另设固定代表性工作点的控制资源测量：同硬件、同精度、同输入、同缓存条件，单个测量 worker，无其他占用才标为隔离测量；完整 B/C 流程独立重复至少 3 次。无需为资源重跑整张精度矩阵。每次正式拟合仍全程记录，控制测量用于可比时延，二者分栏。此次仅制定规则，不进行 GPU 测量。

### 9.4 PAMF 成本的可检查量

F=232993、G=266951；使用 G 时两网络合计 499944 参数，另加 MILKR 任务状态。不预先宣称 PAMF 总网络比所有基线小；某些基线本身更小。

MILKR 的 S=(6+N)K，显式输入维数 736，交互特征隐式计算。一个 FP64 Gram 为 8S² 字节，系数为 8S(6+N) 字节；当前持久数组及标量近似 `8S[736+(6+N)+2]+32` 字节。最大 K20/N20 时 S=520：单 Gram 2163200 字节（约2.063 MiB），持久头状态3178272字节（约3.031 MiB），均不包含 Python/audit 及拟合临时峰值。不能以此理论值代替实际 RSS/显存。

核几何和求解的示意复杂度为 O(S²D+S³)，查询约 O(SD) 每样本并随类别系数计算增加；无梯度并不等于零计算。第一组实际要检验的是**目标阶段更新负担、状态量及性能之间的关系**，不能预写性能不降或资源必胜。

ProtoNet 的原型更新、LoRa 的 enrollment 也可没有目标反传。因此“零梯度更新”不是 PAMF 独有优势。要同时给出准确率—拟合时间—内存—推理时延的比较，允许其他方法在某项资源指标更优；也不能只拿最重的 ISSL 证明全部方法都不轻。

## 10. 分层矩阵与预算

以下按完整 8 DA＋5 CIL 家族计算上限；未绑定行不会冒名执行。B/C 的“一次拟合任务”不等于一次 optimizer step，ISSL 和原生 session 的全部内部预算包含在内。

| 层 | 固定范围 | 基础数据单元 | 唯一目标阶段拟合数 |
|---|---|---:|---:|
| 主比较 | MIX8，三池，7 RX，4 source seed；K5/N20 | 3×8×7×4=672 | 672×(9 B＋6 C＋2 PAMF恢复support)=**11424** |
| K/N 曲线增量 | MIX01，三池，7 RX，4 seed；B额外3K，C额外15个K/N点 | 3×1×7×4=84 | 84×[(9＋1)×3＋(6＋1)×15]=**11340** |
| 三纯景附表 | 3景×5 episode，三池，7 RX，4 seed；K5/N20 | 3×5×3×7×4=1260 | 1260×17=**21420** |
| 总目标拟合上限 | 上述三层并集，主点不重复 | — | **44184** |

主比较含 PAMF B/直接C raw-support 各一次，恢复 support 各额外一次；同一状态两种 query 只增加预测。ABC 不额外产生不同的 C 拟合。当前若 Tweak 原生 M10 与 CSIL 原版继续 N/A，主比较可执行项上限为 11424−672−672=10080；完整三层减去各自任务后为38808。表中保留 N/A，不把缺项当失败分数 0。

A 上限为 14 家族×4 seed＝56 份选定源 bundle。若 Tweak 保留五 LR 搜索，实际为其他13×4＋Tweak5×4＝**72 条源训练轨迹**、56份选定 bundle，不能只报56次训练。完全匹配的旧 A 可抵扣实际新增训练数；现在尚未给出逐 checkpoint 最终复用清单，不虚报“可复用44个”。DURA 不重新训练。

目标预测也要独立计费。按每个数据单元统计“某阶段、某已拟合状态、某 query 路径的一整池预测”，主比较为672×[(8个外部B＋4个PAMF格)＋(5个外部C＋4个PAMF格)]=14112个预测单元；曲线新增14364，三纯景26460，共54936。它们不是物理样本数，也不包括 A0、源验证和必要控制测量。每个单元的实际 query 条数依 r10 补集不同，最终按真实分母汇总 G/F/判决时间与总前向次数，不能只用拟合数估计全部GPU预算。

ISSL 主点单独就有672×(2000＋300)=1545600个本地C epoch；epoch大小取决于support/native loader，不等同源epoch。不得为赶预算静默缩成20轮。GPU小时/完成日期现在未知，须在未来首个固定单seed/P01/MIX01/RX1-1工作点完整运行后，用实测的逐方法A/B/C及query时间投影：

`总worker小时 = Σ(方法×工作点数量×实测完整stage秒数)/3600 + 实际新增A与必要G/F生产成本`。

不以一小步耗时乘epochs冒充已经测得完整训练时间。先完成固定小矩阵的技术与科学核验，再按届时授权展开；当前是设计，不启动该小矩阵。

## 11. 当前实现差距与交付边界

1. 已确认 local_ridge 的直接 C 条件及 AC/ABC 终态等价；无需发明新头。
2. 三池数据与既有 frozen DURA 可按原合法句柄复用，不重新建数。
3. 当前 stopped 外部 runner 的统一 B10/组合 C 路线不符合本组，不能直接恢复；A 中期 PL 与度量伪标签采样仍需真正实现并核对各原生入口。
4. CSIL 原作者 Fisher 数值路径和 Tweak M10 合法多观测输入仍是具体缺项。固定样本窗口不能按真实 TX 拼 query 轨迹；M1/corefix 只能独立注明扩展/修复，不能在原生行替补。
5. 本文不包含新训练/预测/评分结果，不确认任何方法已经达到性能或资源目标。后续执行另按本次设计形成最小预登记和精确代码配置，不以设计文件当运行成功证据。

## 12. 可追溯依据

- [科学协议](PROJECT_PROTOCOL.md)：阶段权限、三池、received、query 与历史边界。
- [方法总清单](CVS_COMPARISON_METHOD_CATALOG_20260930.md)：原6项、扩展家族及原生/项目扩展区分。
- [旧外部执行登记](PAMF_PHASE2_EXTERNAL_THREEPOOL_EXECUTION_20261029.md)：只用于定位预算和版本差异，其组合路线不继承为本组方法。
- [资源审计](PAMF_PHASE2_RESOURCE_AUDIT_20261029.md)：旧记录缺项，不以补填估计代替实际资源采集。
- [当前 PAMF 定义](PAMF_METHOD_DEFINITION_20261027.md)、[DURA v1](DURA_V1.md)：本组普通 PL 变体与全局 REPL/DURA 历史身份区分。
