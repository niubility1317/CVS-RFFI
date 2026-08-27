# 两篇RFFI论文严格方法一致性设计

## 目标

在不绑定论文原始数据集的前提下，使Tweak和Feature Separation的本地实现对论文已经公开的方法要求逐项一致、可执行且可测试。论文没有公开、且尚未定位到作者源码或书面确认支持的算法细节，统一采用明确、可复现的常用默认设置，并在结果中保留其未公开来源属性。

## 不变边界

- 两个实现保留在`paper_reproduction/gaskin_tweak_2023`和`paper_reproduction/hu_feature_separation_2024`，不调用或改写`paper_reproduction/cvs_aligned`、`feature_separation_crossrx`或任何CVS运行器。
- 数据文件、下载、原始设备采集和论文样本内容属于`DATA_EXTERNAL`，不作为本次实现范围；不过数据进入模型后的切段、预处理、划分语义、随机采样和评测规则仍属方法，必须实现或明确锁定。
- 严格模式只有在所有`PAPER_DISCLOSED`要求已验证、所有`AUTHOR_REQUIRED`字段已从可追溯作者来源填入时，才能返回`STRICT_METHOD_PARITY_READY`。否则只能返回`PAPER_INFORMED_NOT_STRICT`，并列出缺项。

## 三类可追溯要求

|类别|含义|运行行为|
|---|---|---|
|`PAPER_DISCLOSED`|论文明确给出的公式、结构、流程或参数|必须实现、测试并通过严格配置校验。|
|`UNPUBLISHED_DEFAULT`|论文没有给出足以唯一实现的数值或算法细节|使用文档化的通用默认值运行；结果必须带`unpublished_defaults`清单，不能将该值归因于作者设定。|
|`DATA_EXTERNAL`|原始数据资产及其传输方式|由外部适配器提供；不得影响方法状态的严格判定。|

## Tweak方法设计

1. 保持论文的`2×128`编码器拓扑、12维嵌入、欧氏triplet损失和`margin=0.1`。
2. 新增共享encoder三元组训练入口。它必须在同一模型实例上完成anchor、positive和negative前向，并把挖掘策略、batch构造、优化器、学习率候选、100epoch训练、best-checkpoint选择写入可执行配置。
3. 新增校准集构建与校验：每类固定N，默认训练样本的10%；支持同一设备在多个目标域生成独立的centroid/radius状态。
4. 新增M输入聚合，默认`M=10`，先平均每个输入的12维embedding，再调用closed/open判定。closed规则严格为“先在`A-Distance<0`的类中取最小A；若不存在则取最小A-Distance”；open规则严格为任一`A<=Distance`即Admit。
5. 新增闭集场景、已知/未知均衡5次随机试验、最小质心距离AUROC分数、TPR、FPR和已接受样本分类准确率的通用评测器。
6. LeakyReLU负斜率设为PyTorch默认`0.01`，BatchNorm使用`eps=1e-5,momentum=0.1,affine=true,track_running_stats=true`，卷积/线性层使用PyTorch默认bias和Kaiming-uniform初始化；hard mining使用batch-hard（最远正例、最近负例）；学习率候选固定为`[1e-2,1e-3,1e-4,1e-5,1e-6]`。这些值均登记为`UNPUBLISHED_DEFAULT`。

## Hu方法设计

1. 输入适配器接收`2×256`IQ，经论文定义的预处理后拼接`1×256`Welch-PSD，形成`3×256`。同步、前导提取、均衡和归一化必须有可执行的阶段接口；未公开实现采用“可选preamble相关同步、最小二乘复标量均衡、每段零均值单位RMS归一化”。Welch采用Hann窗、`nperseg=256`、`noverlap=128`、常数去趋势、双边density PSD和`fs=1.0`。
2. 主干采用图6优先的常用解释：`16→32→64→128→256`五个两卷积残差阶段、最后全局平均池化和`256→512`投影，仅在512维共享特征处应用图示的attention。现有“5个双残差阶段且每块插入attention”的版本不能作为论文方法实现继续使用。图6与“ResNet18”文字不完全一致的处理登记为`UNPUBLISHED_DEFAULT`。
3. 保持论文`L_FS=L_CE+λ1L_Sim+λ2L_CLFEtx+λ3L_CLFErx`的正号、Frobenius相似度和两项熵损失。CE采用常用的batch mean；`λ1=λ2=λ3=1.0`，均登记为`UNPUBLISHED_DEFAULT`。
4. 把训练期噪声、多径和Doppler增强实现为每epoch可复现的变换；路径数、最大时延、每路径时延、衰减、相位和Doppler遵照论文已给范围。未公开的采样分布采用均匀分布，三种增强均每样本应用一次，顺序为多径、Doppler、噪声。
5. 新增双接收机训练、Adam(lr=.005)、batch=256、验证选模、checkpoint、scheduler和loss停滞停止的训练入口。未公开部分固定为200epoch上限、`ReduceLROnPlateau(factor=.1,patience=10)`和20个epoch早停耐心值。
6. 新增无新接收机、跨接收机、跨日、通道增强、损失消融和每TX25样本微调的通用评测矩阵。微调只读TX标签；默认更新encoder、TX分支和TX分类器，并令RX分支及其BatchNorm保持eval和无梯度。

## 配置与拒绝机制

每个论文目录新增独立`strict_method.json`。其中每个未公开条目必须包括`value`、`rationale`和`status="UNPUBLISHED_DEFAULT"`；`validate_method_config`返回机器可读的论文公开项、默认项和数据外部项。训练与评测入口把`unpublished_defaults`写入结果元数据。数据适配器传入的IQ和标签不属于该配置校验条件。

## 测试与验收

- 每条`PAPER_DISCLOSED`要求有行为测试；每条`UNPUBLISHED_DEFAULT`要求有配置内容、可复现行为和结果元数据测试。
- 测试覆盖共享encoder梯度、M聚合边界、closed/open边界、校准域隔离、Tweak指标；Hu的PSD/预处理接口、结构manifest、loss缩放、增强可复现性、训练选模、微调冻结和各评测矩阵生成。
- 严格模式测试不得访问真实论文数据，不得产生CVS输出。

## 完成定义

当两目录的所有`PAPER_DISCLOSED`条目均为`verified`时，报告`PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS`，并列出全部默认项。只有作者源码或书面确认使全部默认项可归因于作者时，才可进一步报告`STRICT_METHOD_PARITY_READY`。
