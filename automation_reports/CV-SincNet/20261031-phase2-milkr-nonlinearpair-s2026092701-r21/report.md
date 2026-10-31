# MILKR 有界非线性配对编码先导

R22 保持完整 Tri-SIG-TIME SSL、DURA v1.0、原神经分类头冻结。R20/R21失败不扩展、不重新选择其其他源配置。query仍严格Q0 raw、Q1仅G输出，不依赖待回复的双路query融合口径。

新方法只在source训练连续特征编码器，不训练新的分类头。C4同记录moment输入和pairedCCA rank64固定；八个block各有raw/G两个2d→64GELU→64Tanh小网络，输出层零初始化，h=unit(h0+0.25·net(x)/sqrt64)。残差范数最多0.25、归一化前范数至少0.75，初始状态等价原CCA。网络342016个FP32参数，CCA278720个FP64参数，约3.43MiB额外参数。训练只用合法sourceL的raw/G同记录配对，loss=配对MSE+0.1原latent锚定；无TX标签/分类输出/F/G入网络损失。AdamW学习率0.001、WD0.0001、600步、batch256，seed2026103101+fold0..4，最终+5。

sourceL6300物理记录、既有ep91/92共12600raw/G对。每个五RX/TX-held折中，CCA与NN均排除heldRX及两TX全部RX/两epoch。原CCA零残差控制同折、同数据，每个new/control独立120个K1核规则以源四路宏H选择，平分取更大ridge/固定序；K20用原R20 nested physicalsupport规则。source gate：整体H较control至少+0.5点，两S Q0均值不低于control−0.25点，每K1/20、S均Q1≥Q0，raw旧/新均值不低于control−1点。源F仍已见过六TX，adapter-held不等于完整F未知类，源三景不同于目标六景，不称新盲测。

source gate失败直接科学停止、target0；通过后才全合法sourceL重拟合最终CCA/NN一次，保存单npz的冻结bias/projection/MLP权重。没有source样本、特征库、类均值/协方差/optimizer进入target。F/G/原神经头更新0，target只用当前K support拟合MILKR，物理K不增加。真实checkpoint noquery smoke为launcher第一步。

sourcePASS后firstRX1-1/P01/MIX01/seed2026092701四角点K1/20×N2/20，B8+C16=24行、严格S0/S1、对应supportstate共享Q0/Q1、全已注册类别竞争。24预测固定后独立truth-last，原版同case复用。每corner两S Q0≥原版+5点且Q1>Q0、旧新均值guard≥原版−1点才同模型补12KN；最终仍需16KN全部过后六RX确认。源训练门槛不代替用户目标。

sourceGPU4、targetGPU0，启动前实测剩余显存，不动其他owner；技术错误保留产物不盲重试。完整commit/命令/环境/CWD、checkpoint来源/data句柄、路径/停止条件见experiment.json。

唯一独立P0/P1审查PASS。零初始化等价、残差幅度、保存重载/逐包独立、合成配对训练误差下降、源门槛逐K query保护、旧orbit/K1/源隔离/非法输入共14项相关检查通过。真实source分类收益尚未验证；实际checkpoint noquery smoke为launcher第一步，PASS后继续。
