# 两篇RFFI论文严格方法一致性设计

## 目标

在不绑定论文原始数据集的前提下，使Tweak和Feature Separation的本地实现对论文已经公开的方法要求逐项一致、可执行且可测试。任何论文没有公开、且没有作者源码或书面确认支持的算法细节不得用本地默认值冒充“一比一复现”。

## 不变边界

- 两个实现保留在`paper_reproduction/gaskin_tweak_2023`和`paper_reproduction/hu_feature_separation_2024`，不调用或改写`paper_reproduction/cvs_aligned`、`feature_separation_crossrx`或任何CVS运行器。
- 数据文件、下载、原始设备采集和论文样本内容属于`DATA_EXTERNAL`，不作为本次实现范围；不过数据进入模型后的切段、预处理、划分语义、随机采样和评测规则仍属方法，必须实现或明确锁定。
- 严格模式只有在所有`PAPER_DISCLOSED`要求已验证、所有`AUTHOR_REQUIRED`字段已从可追溯作者来源填入时，才能返回`STRICT_METHOD_PARITY_READY`。否则只能返回`PAPER_INFORMED_NOT_STRICT`，并列出缺项。

## 三类可追溯要求

|类别|含义|运行行为|
|---|---|---|
|`PAPER_DISCLOSED`|论文明确给出的公式、结构、流程或参数|必须实现、测试并通过严格配置校验。|
|`AUTHOR_REQUIRED`|论文没有给出足以唯一实现的数值或算法细节|严格训练、校准或评测拒绝启动，直到`author_source`、取值和来源定位被完整登记。|
|`DATA_EXTERNAL`|原始数据资产及其传输方式|由外部适配器提供；不得影响方法状态的严格判定。|

## Tweak方法设计

1. 保持论文的`2×128`编码器拓扑、12维嵌入、欧氏triplet损失和`margin=0.1`。
2. 新增共享encoder三元组训练入口。它必须在同一模型实例上完成anchor、positive和negative前向，并把挖掘策略、batch构造、优化器、学习率候选、100epoch训练、best-checkpoint选择写入可执行配置。
3. 新增校准集构建与校验：每类固定N，默认训练样本的10%；支持同一设备在多个目标域生成独立的centroid/radius状态。
4. 新增M输入聚合，默认`M=10`，先平均每个输入的12维embedding，再调用closed/open判定。closed规则严格为“先在`A-Distance<0`的类中取最小A；若不存在则取最小A-Distance”；open规则严格为任一`A<=Distance`即Admit。
5. 新增闭集场景、已知/未知均衡5次随机试验、最小质心距离AUROC分数、TPR、FPR和已接受样本分类准确率的通用评测器。
6. LeakyReLU负斜率、BN超参数、bias、初始化、硬挖掘的完整筛选规则和学习率搜索的精确离散集合均登记为`AUTHOR_REQUIRED`，除非后续获得可引用作者来源。

## Hu方法设计

1. 输入适配器接收`2×256`IQ，经论文定义的预处理后拼接`1×256`Welch-PSD，形成`3×256`。同步、前导提取、均衡和归一化必须有可执行的阶段接口；具体未公开算法和Welch窗、分段、重叠登记为`AUTHOR_REQUIRED`。
2. 主干只允许由经来源确认的图6/ResNet18拓扑构建。现有“5个双残差阶段且每块插入attention”的版本不能作为严格实现继续使用。主干、attention位置、分类器BN/激活层级必须可由结构manifest审计。
3. 保持论文`L_FS=L_CE+λ1L_Sim+λ2L_CLFEtx+λ3L_CLFErx`的正号、Frobenius相似度和两项熵损失。严格模式将CE的reduction、各λ和梯度尺度作为可验证配置项。
4. 把训练期噪声、多径和Doppler增强实现为每epoch可复现的变换；路径数、最大时延、每路径时延、衰减、相位和Doppler遵照论文已给范围。未公开的采样分布、应用概率与顺序登记为`AUTHOR_REQUIRED`。
5. 新增双接收机训练、Adam(lr=.005)、batch=256、验证选模、checkpoint、scheduler和loss停滞停止的训练入口。scheduler、epoch上限/耐心值及精确定义因原文未公开而由严格配置锁定。
6. 新增无新接收机、跨接收机、跨日、通道增强、损失消融和每TX25样本微调的通用评测矩阵。微调只读TX标签；encoder/TX/RX哪些参数和BN状态可变必须由作者来源锁定。

## 配置与拒绝机制

每个论文目录新增独立`strict_method.json`。其中`author_required`条目必须包括`value`、`author_source`和`source_locator`；任何一个缺失时，`validate_strict_method_config`返回机器可读的未闭合项，训练与评测入口抛出`StrictMethodParityError`。数据适配器传入的IQ和标签不属于该拒绝条件。

## 测试与验收

- 每条`PAPER_DISCLOSED`要求有行为测试；每条`AUTHOR_REQUIRED`要求有严格配置拒绝测试和来源填充后的通过测试。
- 测试覆盖共享encoder梯度、M聚合边界、closed/open边界、校准域隔离、Tweak指标；Hu的PSD/预处理接口、结构manifest、loss缩放、增强可复现性、训练选模、微调冻结和各评测矩阵生成。
- 严格模式测试不得访问真实论文数据，不得产生CVS输出。

## 完成定义

当且仅当两目录的所有`PAPER_DISCLOSED`条目均为`verified`，且所有`AUTHOR_REQUIRED`条目都有作者来源且严格验证通过时，报告`STRICT_METHOD_PARITY_READY`。在作者来源缺失期间，实现仍可完成论文已披露部分，但最终状态必须是`BLOCKED_BY_UNDISCLOSED_AUTHOR_DETAILS`，不是“一比一复现”。
