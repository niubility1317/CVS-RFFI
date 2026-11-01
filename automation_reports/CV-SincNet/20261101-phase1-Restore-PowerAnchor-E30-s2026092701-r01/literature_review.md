# 恢复性能研究：外部依据与下一步机制假设

**主验收保持为原地面完整 Tri-SIG-TIME SSL/default head/Main split：六个 practical 场景分别距同一 raw clean≤3 个百分点，并保持 clean。** 训练效率为支线。本说明在 PowerAnchor 完成正式预测前形成，不改变该版配方、30 轮预算、final30 冻结或七景测试。

## 当前证据给出的判断

源端困难与跨 RX 泛化需要分别检查。已完成源 V 诊断中，几种新恢复器的 F(clean) 判决保持优于 DURA；它们在 Main 的真实 TX 准确率仍可能更低。该结果弱化了单纯训练记录记忆这一解释，尚未证明接收机变化是唯一原因。增大模型、降低源 IQ 误差或增大特征回归权重均不能作为性能成功依据。

同一批源缓存的低档城区记录中，失锁占645/1436；FeatureCond 的162条 F(clean) 判决不一致中，131条来自失锁子组。真作用加简单总体高斯先验的128步恢复在该子组仍明显弱于神经先验；传入额外 clean 特征则提高判决一致率。这些是有限源端、非部署反事实，既不是目标 TX 准确率，也不是最优恢复上限。[缓存困难分析](../20261101-diagnostic-Restore-FeatureCond-source-paths-s2026092701-r01/source_difficulty.md)

末轮源梯度几何诊断没有发现普遍的主损失/latent 梯度冲突或 globalclip 激活，因此目前不据此引入 PCGrad、分组裁剪或更多优化器机制。PowerAnchor 只检验投影前未受监督的幅度自由度。[源梯度诊断](../20261101-diagnostic-Restore-source-gradient-geometry-s2026092701-r01/analysis.md)

## 已核对的原始研究及适用边界

| 原始研究 | 可借鉴内容 | 与本轮条件的差别 |
|---|---|---|
| Yin 等，2025，Noise-Robust RFFI Using Denoise Diffusion Model | 无类别条件的噪声预测、按噪声水平选择恢复步骤 | 使用静态信道、人工加噪，并在恢复信号上训练下游分类器；不能直接继承到冻结 F0 的跨 RX 星地复合失真。[arXiv 正文](https://arxiv.org/html/2503.05514v1) |
| He 等，2026，Cross-Receiver RFFI Under Varying Channels | 将信道抑制与接收机校准分开处理 | 部署校准需要源/目标接收机的有线 challenge-response 配对；本任务没有该数据权限，也不能换成其 DSQ 分类器。[arXiv 正文，III-D](https://arxiv.org/html/2603.08402v1) |
| Shen 等，Towards Receiver-Agnostic and Collaborative RFFI | 显式处理接收机影响 | 训练识别特征并另有多接收机融合/微调；本轮 F0 冻结，单条 IQ 接口不允许静默继承这些条件。[作者预印本](https://arxiv.org/pdf/2207.02999)，[期刊记录](https://livrepository.liverpool.ac.uk/3176924/) |
| Xie 等，2025，Channel2Channel | 在对数谱上学习压制信道干扰的 RFF 表征 | 输出为特征提取路径，并有 SVM/多帧融合；不能将其准确率视为同一 IQ 恢复器或本轮冻结 F 的对照。这里只核对到大学接受稿记录及摘要，未完整读取正文。[作者大学记录](https://livrepository.liverpool.ac.uk/3192953/) |
| Liu 等，ICML 2023，I²SB | 用配对退化/clean 端点学习扩散桥，训练可直接采样中间状态 | 证据来自图像恢复；迁移到复 IQ、固定识别器和未知设备，需要独立实验。[PMLR 正式论文](https://proceedings.mlr.press/v202/liu23ai.html) |
| Delbracio、Milanfar，TMLR 2023，InDI | 从退化观测逐步恢复，研究单步回归的均值化问题 | 图像感知细节改善不等于发射机身份正确；不能将“更真实”当作 TX 线索被保留。[作者机构论文页](https://research.google/pubs/inversion-by-direct-iteration-an-alternative-to-denoising-diffusion-for-image-restoration/) |

检索日期为2026-11-01，属于围绕当前机制问题的定向检索，不声称穷尽文献。专用 arXiv/CrossRef MCP 检索工具未提供，改用网页检索并在 arXiv、会议论文集、作者和大学原始记录核对。部分大学PDF返回406/超时，未把未读全文的内容写成已核实方法细节；无外部权重或代码被引入运行。

## 可以继续检验的假设

一种值得独立检验的结构方向，是保留逐记录 received IQ 为条件，用单阶段训练学习逐步恢复过程，而不是只回归一次 clean 波形。其动机是源失锁子组的特征不确定性和有限高斯先验的不足；扩散桥/直接迭代提供了可实现的配对学习形式。这里是研究推断，尚无本项目收益证据，也未宣称存在可恢复的特定比例。

若实施，仍只用原 L_s/U_s 的模拟 received/clean 配对；中间训练状态可以由合法源配对构造，部署初态和条件只能来自当前 received IQ。F0 仅以已授权的冻结连续特征作训练约束，不使用 TX/logits/CE/六类原型或新身份头；OE 仅冻结诊断，V/目标不训练或选模。新 G 依然从零单阶段30轮，恢复的多次推理调用不得冒称多阶段训练。迭代数、训练曝光、F 知识和完整计算成本需要明确报告。

这一方向的实质风险是：生成先验可能补出与观测不一致的细节，输出更像 clean 并不等于真实身份保真；有限步数还可能产生分布误差。应以同一原 Main 七景逐条净纠错、两个总体的 clean 保持和冻结后的 OE 诊断检查。所有新版本均先固定方案、完整预测后独立评分，不依据 Main 错误挑超参数或选择性重跑。

另一个可检验方向是源端接收机变化的稳健性，但当前没有证据证明简单删去 RX 均值或强制相位不变有效。原 F0 对源 clean 的90°相位变换本身会改变部分判决，因此任何源端风格扰动都需要定义对应 clean 参考，不能默认变换后身份表征应完全不变。[配对风格诊断](../20261101-diagnostic-Restore-source-style-equivariance-s2026092701-r01/analysis.md)

## “突出发射机特质”怎样验收

恢复器可以改变线索的表达，使固定 F 更容易使用观测已有的身份信息。它不必逐点复制某一 clean 才可能改善识别，但也不能由生成细节宣称产生了新的独立发射机证据。

本轮报告因此增加与同记录 raw clean 判决的配对分析：分别统计修复 clean 原错误、损伤 clean 原正确，以及 F(clean) 判决一致率；同时保持真实 TX 准确率为主指标。该分析复用冻结预测，不新增头、不反馈训练。即便准确率超过 raw clean，也只说明固定系统的任务收益；关于波形/信息恢复极限，仍需条件不确定性或可辨识性界，现有试验不能提供可信的“最多八成或九成”。[理论与指标说明](../20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01/theoretical_limits.md)
