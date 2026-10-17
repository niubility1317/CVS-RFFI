# 接收机泛化 IQ 增强：本次原始文献核查的适用边界

2026-10-16。当前 F＋跨 RX 同 TX、固定 E20 的 additive As 三组正在运行。本说明只记录后续 A_g 的证据边界，不改变该实验、晋级条件、预算或目标权限，不是新增发布门槛。

1. Ma 等的 [Towards Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification](https://arxiv.org/abs/2512.12070) 采用谱图、对比预训练和 Siamese 分类训练，并在 LoRa 数据上评估。它支持分别考虑信道和接收机差异的研究动机；其模型、表示和训练阶段不同于当前 CVS，论文成绩不能作为本项目 A_g 或 practical 收益，也不能与现有 RDrop/SupCon 的增益相加。

2. Shen 等的 [Towards Receiver-Agnostic and Collaborative Radio Frequency Fingerprint Identification](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/535502020/TMC_2023_Receiver_Agnostic_RFFI.pdf) 使用真实多接收机 LoRa 数据、对抗训练及协同推理，并另有微调。其接收机变化证据涉及真实硬件；不能由此将已接收 IQ 的数字变换描述为替换真实 RX。本项目不引入论文的附加判别器、目标微调或多接收机部署推理。

3. [Domain Generalization for Cross-Receiver Radio Frequency Fingerprint Identification](https://arxiv.org/abs/2411.03636) 的摘要以可分离条件和发射机／接收机表示分解为依据。该条件不是“任意 IQ 扰动都身份保持”的结论。本次仅核对摘要，没有把它作为某个具体增强算子的有效性证据。

4. [Towards Distance-Invariant Radio Frequency Fingerprinting via Augmented Unsupervised Learning](https://ojs.aaai.org/index.php/AAAI/article/download/37005/40967) 第 3 页列出缩放、片段重排、窗口裁剪及时间／幅度／窗口扭曲，且包含对比／原型对齐及后续无标签目标适应。以下是本项目判断：文中的传播动机不能直接证明这些变换对固定参考窗口和当前身份特征安全。片段重排会改变时序连接，窗口伸缩会改变参考定位；本项目不据此新增上述变换、重开已关闭的缩放／时移配方或使用目标数据适应。

本次核查范围有限，未取得能直接支持全通相位滤波、采样时钟扭曲等候选在当前数据上身份保持且跨 RX 有效的证据。它们尚未实现、登记或启动；物理动机不是实证结论，也不是宣称不存在相关文献。后续若提出新 A_g，仍需先有适用于本数据的独立依据，再固定一项最小源域反证；不把健康、输入有变化或论文命名作为识别收益。

当前唯一在检验的假设仍是：保留完整原始 raw/ordinary 绝对监督系数后，追加完整 practical(raw) 是否同时保护 clean、改善 urban，并超过等尺度重复 raw 监督。现有三组固定 source RX1-held/seed01/scratchE20/8 L 前向，最终源读出前不改变方案。完整联合目标、A_g 证明、B00/B10/B01/B11 交互项、跨 seed 和正式 target clean＋六 practical 仍未完成。


## 采样时钟变化的定向补查（2026-10-16）

本次只做文献证据核对，没有登记、实现或启动新的 IQ 增强。当前唯一在检验的候选仍为原配方 seed02 additive As 确认。使用 nature-academic-search 的来源核对方法；本环境没有该 skill 列出的 arXiv／CrossRef MCP，改用网页工具读取 arXiv 正文和作者机构原稿。只采用原始研究内容，不采用搜索摘要、商业术语页面或综述作为增强有效性的依据。

| 原始研究与已核对位置 | 可支持的事实 | 对本项目的限制 |
|---|---|---|
| Cekic、Gopalakrishnan、Madhow，[Wireless Fingerprinting via Deep Learning: The Impact of Confounding Factors](https://arxiv.org/html/2002.10791)，v3，2021-03-09，IV-A 至 IV-D、表 I 至 VI | 研究 CFO／信道混杂、补偿与增强；文中的 clock drift 实验落实为 CFO，比较了共享跨设备的扰动集合和独立随机扰动。 | 这不是 ADC 采样时钟重采样的身份保持证据。结果还涉及多日模拟与测试时多视图，不能移植到本项目 E20／默认单次预测，不能为已失败频移路线救跑。 |
| Meneghello、Rossi、Restuccia，[DeepCSI](https://www.research.unipd.it/retrieve/c6e521ee-850a-416d-9dfb-22ffa96cd163/DeepCSI_Rethinking_Wi-Fi_Radio_Fingerprinting_Through_MU-MIMO_CSI_Feedback_Deep_Learning.pdf)，ICDCS 2022，DOI 10.1109/ICDCS54860.2022.00106，PDF 第 9 至 10 页、式 (9)、图 16 | 作者把 SFO 列入相位偏差模型；在其 MU-MIMO 反馈表示上，offset cleaning 降低识别表现，说明被移除成分可能携带 TX 证据。 | 它不是 CVS 原始 IQ 的重采样增强实验；也不是单独隔离 SFO 的因果结果。不能把采样时钟变化一概视为接收机无关 nuisance，更不能由校正失败推出随机扰动有效。 |
| Wei、Li、Zhang、Yang，[Division-based Receiver-agnostic RFF Identification in WiFi Systems](https://arxiv.org/html/2511.15458)，III、V-A | 依赖特定 WiFi 前导的频域相除、参考设备／训练字段结构，并采用对应表示与分类流程。 | 不是当前固定 CVS 的标签保持 IQ 增强对照，不引入其参考访问、替代表示或多个分类网络。 |

明确区分两种算子：CFO 是样本上的复指数相乘，采样频率偏差是时间坐标重采样。即使均用 ppm 表示，前者不能给后者提供幅度依据。按定义推导，采样比例误差 epsilon 在样本 n 的位移约为 epsilon×n；本轮实际 args 为 equalized=1、wisig_out_len=256、sample_rate_hz=25000000；例如 20 ppm 在该窗口 n=255 的端点仅约 0.00510 个样本。这只是单位核对，不是推荐幅度、有效性或身份保持证明；有限插值误差仍可能超过所模拟的变化。

定向检索使用 RF fingerprint／receiver／sampling clock／sampling frequency offset／augmentation 等组合，核对范围有限。当前结论为“未获得足够直接依据”，而非断言所有文献均不存在该证据。没有因此选择 sampling-clock、SFO、time-warp 或 all-pass 候选，也没有重开已关闭的 timing／frequency／spectral 路线。后续若考虑时钟类观测变化，需要先在合法源域区分跨 TX／day 一致的 RX 成分与 TX 成分，并证明数值实现没有以插值伪迹替代预期扰动；仍须以匹配 E20 跨 RX clean 识别收益判断，不能用文献标题、扰动很小或健康检查代替。


## E30 后的公开 IQ 增强实现核查（2026-10-17）

本轮唯一核查假设为：公开的非对称增强实现能否为当前 256 点 CVS 提供一项可直接检验、区别于已关闭路线的 A_g。最小动作是读取公开代码的实际 forward、算子和数据归一化；没有执行第三方代码、下载第三方权重、读取真实 IQ、加载 checkpoint、增加 NN 或训练更新。这是来源适用性判断，不是新实验或发布门槛。最新预算仍为 E30，完整源 RX1/3/4/6/8、L6300/U56700（TX 隐藏）/V27000 与 F＋跨 RX 同 TX 不变，无训练域留出。已有 E20＋10 与 scratch E30 分开披露。

[AA-FAL 公开代码](https://github.com/SONGzyyyyyy/AA-FAL/tree/f623ebcce20ab2f1956e44a2c6a9b6f566b69308)的 utils.py 中，Data_augment_time 使用 1200 点局部区间，forward 无条件追加局部样本替换；其随机分支还可能分别改 I 或 Q。该窗口的 randint 上界在当前长度下为 256−1200＝−944，不能原样应用；分别缩放 I/Q 也会改变共轭镜像分量，未取得 TX 身份保持依据。频域路径及 trainer 另涉及频域表示和独立网络，不能把其整体效果归为本项目的轻度 IQ 观测增强。由此只否定“原样直接迁移”的假设，不证明所有非对称增强无效；不缩短窗口、减小强度或另搭网络救跑。

题名 Green Physical Layer Authentication Based on Few-Shot RFF via Asymmetric Augmentation and Frequency-Assisted Learning、2026 年、五位作者和 [DOI 10.1109/TGCN.2026.3669577](https://doi.org/10.1109/TGCN.2026.3669577)已由 Crossref 注册元数据核对；IEEE 正文入口要求浏览器验证，本轮没有取得全文，也没有独立认证该代码与论文的对应。上述代码事实仅归于已读仓库，不推断论文所有实验设置，不引用其成绩。GitHub raw 入口连接重置后改用公开 GitHub API 读取同一版本 blob；来源文本仅存本地私有工件，不复制第三方代码进项目。

[ASA 作者仓库说明](https://github.com/Edith-xx/Cross-Domain-RFFI/blob/main/README.md)明确操作中间特征的均值和标准差，并采用谱图分解及多尺度网络；它不能补足当前 IQ 算子的依据。[Chillet 等的 ICC 2024 机构原始摘要](https://researchportal.tuni.fi/en/publications/how-to-design-a-channel-resilient-database-for-radio-frequency-fi/)研究虚拟数据库和传播多样性；本轮全文未取得，其摘要也不能支持新的 RX 数字替换或区别于已测试 practical 的 A_g。没有把这些论文的准确率当成 CVS 收益。

另一个曾考虑的常量偏置入口，检索后确认已有[完整 DC 源负结果](../automation_reports/CV-SincNet/20261016-diagnostic-F-DC-source-crossfit-manysig-m1-r01/report.md)：跨 TX/day 风险改善为负，固定路线关闭。本轮没有重测或换窗。原来提到的 antithetic practical 噪声也在历史登记中明确属于关联对话方向，本轮没有进入该路线。

判定：NO_NEW_COMPATIBLE_IQ_CANDIDATE；直接迁移的弃用条件已触发，当前没有待发布的新 A_g/A_s 或 E30 训练。范围有限的核查不证明现有数据不存在有效增强，也不缩减完整目标。后续只有形成不同于已关闭路线、可由合法源数据反证的具体假设时才固定相应最小 E30 比较；目标分数不得参与候选或幅度选择。无需为本次文献记录新增实验登记、独立审查或远端发布。
