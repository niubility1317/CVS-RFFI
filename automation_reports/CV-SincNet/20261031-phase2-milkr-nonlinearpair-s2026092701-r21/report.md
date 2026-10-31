# MILKR 有界非线性配对编码先导

R22 保持完整 Tri-SIG-TIME SSL、DURA v1.0、原神经分类头冻结。R20/R21失败不扩展、不重新选择其其他源配置。query仍严格Q0 raw、Q1仅G输出，不依赖待回复的双路query融合口径。

新方法只在source训练连续特征编码器，不训练新的分类头。C4同记录moment输入和pairedCCA rank64固定；八个block各有raw/G两个2d→64GELU→64Tanh小网络，输出层零初始化，h=unit(h0+0.25·net(x)/sqrt64)。残差范数最多0.25、归一化前范数至少0.75，初始状态等价原CCA。网络342016个FP32参数，CCA278720个FP64参数，约3.43MiB额外参数。训练只用合法sourceL的raw/G同记录配对，loss=配对MSE+0.1原latent锚定；无TX标签/分类输出/F/G入网络损失。AdamW学习率0.001、WD0.0001、600步、batch256，seed2026103101+fold0..4，最终+5。

sourceL6300物理记录、既有ep91/92共12600raw/G对。每个五RX/TX-held折中，CCA与NN均排除heldRX及两TX全部RX/两epoch。原CCA零残差控制同折、同数据，每个new/control独立120个K1核规则以源四路宏H选择，平分取更大ridge/固定序；K20用原R20 nested physicalsupport规则。source gate：整体H较control至少+0.5点，两S Q0均值不低于control−0.25点，每K1/20、S均Q1≥Q0，raw旧/新均值不低于control−1点。源F仍已见过六TX，adapter-held不等于完整F未知类，源三景不同于目标六景，不称新盲测。

source gate失败直接科学停止、target0；通过后才全合法sourceL重拟合最终CCA/NN一次，保存单npz的冻结bias/projection/MLP权重。没有source样本、特征库、类均值/协方差/optimizer进入target。F/G/原神经头更新0，target只用当前K support拟合MILKR，物理K不增加。真实checkpoint noquery smoke为launcher第一步。

sourcePASS后firstRX1-1/P01/MIX01/seed2026092701四角点K1/20×N2/20，B8+C16=24行、严格S0/S1、对应supportstate共享Q0/Q1、全已注册类别竞争。24预测固定后独立truth-last，原版同case复用。每corner两S Q0≥原版+5点且Q1>Q0、旧新均值guard≥原版−1点才同模型补12KN；最终仍需16KN全部过后六RX确认。源训练门槛不代替用户目标。

sourceGPU4、targetGPU0，启动前实测剩余显存，不动其他owner；技术错误保留产物不盲重试。完整commit/命令/环境/CWD、checkpoint来源/data句柄、路径/停止条件见experiment.json。

唯一独立P0/P1审查PASS。零初始化等价、残差幅度、保存重载/逐包独立、合成配对训练误差下降、源门槛逐K query保护、旧orbit/K1/源隔离/非法输入共14项相关检查通过。真实source分类收益尚未验证；实际checkpoint noquery smoke为launcher第一步，PASS后继续。

## 实际结果

**VERIFIED**：真实F/G noquery smoke PASS，archive传输/remotecompile、PID/CWD/argv/GPU与日志增长核实。代码commit `e75ef87fcc07a2db679a8c3f89dcb2168fb780dd`。五折CCA/MLP均source-only拟合，NN每折600步、256batch，无TX标签入NN损失；source heldout分类仍用合法L标签。源流程173.48秒。

source gate：False，四路H相对控制+0.2640pp，两S raw均值差[-0.2505750728619749, 0.8588213382823673]pp，每K/S Q1−Q0 [13.457695380382706, 13.677987373580681, 9.243849537428305, 19.50873777824469]pp，raw old/new差[0.23437499999999778, -0.78125, -0.07812500000000111, 1.4062499999999978]pp。

| 源 RX | 训练前配对误差 | 训练后配对误差 | NN 更新步数 |
|---:|---:|---:|---:|
| 1 | 0.778044 | 0.717523 | 600 |
| 3 | 0.814774 | 0.750761 | 600 |
| 4 | 0.800158 | 0.738295 | 600 |
| 6 | 0.778291 | 0.717656 | 600 |
| 8 | 0.818277 | 0.754751 | 600 |

| 源 K | S | Q | 控制宏H% | 非线性宏H% | 差pp |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 59.0054 | 58.7646 | -0.2408 |
| 1 | 0 | 1 | 71.4545 | 72.2223 | +0.7678 |
| 1 | 1 | 0 | 70.6856 | 71.0678 | +0.3822 |
| 1 | 1 | 1 | 84.7029 | 84.7458 | +0.0429 |
| 20 | 0 | 0 | 80.9217 | 80.6613 | -0.2603 |
| 20 | 0 | 1 | 90.0692 | 89.9052 | -0.1640 |
| 20 | 1 | 0 | 72.0052 | 73.3406 | +1.3354 |
| 20 | 1 | 1 | 92.6006 | 92.8494 | +0.2488 |

源科学门槛未通过，按预登记停止；target预测/评分均为0，未生成最终全source编码checkpoint，不扩大实验。配对训练误差下降不能替代识别宏H收益，不把未测试说成目标负结果。全部所属进程退出。

基座Tri-SIG-TIME SSL、DURA和默认神经头始终冻结；Q1严格仅G输出。source F见源六TX、源proxy与目标未知类/六景有差异；仅开发证据，不称新盲测。最终16KN两S每格+5/Q1>Q0尚未达成，goal ACTIVE。
