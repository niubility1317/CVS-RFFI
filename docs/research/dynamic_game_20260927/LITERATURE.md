# 文献依据与创新性边界

检索日期：2026-09-27。本页只把能够查到原始论文页或作者机构记录的论述作为已核实依据；没有把搜索摘要或历史对话当成论文全文。

## 已核实的直接前作

1. **Ganin等，Domain-Adversarial Training of Neural Networks，JMLR 17(59)，2016。**域判别与特征提取通过梯度反转形成相反的训练目标，是DANN/GRL的经典来源。因此，域对抗本身不构成本项目的新颖性。[JMLR原文与PDF入口](https://jmlr.org/papers/v17/15-239.html)
2. **Acuna等，Domain Adversarial Training: A Game Perspective，ICLR 2022。**论文明确将域对抗写成博弈，研究连续动力学及高阶离散求解，并讨论与局部均衡有关的条件。因此，“首次从博弈动力学理解DANN”“首次用高阶求解改善域对抗”都不是可支持的创新声明。其理论条件和视觉任务结果不能直接迁移为本项目AdamW、裁剪、多辅助目标和RFFI数据上的收敛保证。[作者论文预印本](https://arxiv.org/abs/2202.05352)
3. **Shen等，Towards Receiver-Agnostic and Collaborative Radio Frequency Fingerprint Identification。**该工作已在RFFI中使用对抗学习获取接收机无关表征。预印本发布于2022年，DOI在线记录为2023年，TMC卷期为2024年23(7):7618–7634；这几个年份代表不同出版阶段，不应互相当成矛盾。因此，“首次将域对抗用于跨接收机RFFI”也不可支持。[作者机构记录](https://livrepository.liverpool.ac.uk/3176924/)、[预印本](https://arxiv.org/abs/2207.02999)、[正式论文DOI](https://doi.org/10.1109/TMC.2023.3340039)

## 未完成核验的邻近文献

检索命中2026年Ad Hoc Networks的“Plug-and-play”域泛化RFFI条目，但此次出版商全文入口读取失败。此前对话将其与对抗学习和MAML联系起来，这一具体方法组合在本次未完成原文核验，不用作已确认事实，也不据此宣布本项目与它等价或优先。[待核对的出版商条目](https://www.sciencedirect.com/science/article/pii/S157087052600065X)

检索组合覆盖RFFI、RF fingerprinting、specific emitter identification与extragradient、Runge–Kutta、game dynamics等词。本次没有找到足以直接判定本项目全部响应设计已被同场景复现的文献，但检索并非系统综述；“未检出”不等于“文献不存在”，不构成首创证明。

## 本项目目前可以怎样表述

可以描述为：在固定弱标注跨接收机数据契约下，对DAOT/FastTrust、域对抗场及若干离散响应求解器进行实现与消融；通过冻结每主步教师、路由、随机状态及归一化语境，降低求解器比较中的额外变化；在历史三seed评估中观察到完整EG相对SIM的正增量。

CF-EG、TR-EG、XT-DANN和DRIC-Lite的贡献主张应逐项对照原文、计算图和同预算实验。当前代码实现了这些局部构造，但性能并未支持它们全部优于完整EG。特别是TR的小幅一致增量、DRIC局部约束以及四格正负不一的交互，都不宜扩大为统计显著、全局收敛、稳定协同或跨数据集普适结论。代码与数值证据见[实现考证](chapters/code_findings.md)和[结果分析](RESULTS.md)。
