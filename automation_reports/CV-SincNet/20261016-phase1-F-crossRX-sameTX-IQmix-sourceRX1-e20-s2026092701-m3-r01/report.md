# F＋跨 RX 同 TX：同身份实测 IQ 混合的源域 E20

状态：PREREGISTERED，未发布、未训练。联合目标未完成。执行代码身份以本次 Git commit 和 release_commit.txt 为准，发布后填入 execution_summary.json。

当前唯一假设：在同 TX/day 的不同实测包之间，固定 10% 的轻度、相干对齐 IQ 混合能带来留出 RX 的身份识别收益，且收益超过同 RX 平均本身。实测 IQ 的数字组合不能称为更换真实接收机。

## 依据与已知边界

G0 两个预定冻结默认头对实际训练 L5040 的 50,400 项预测完成独立复算。跨 RX 相对 raw 的 accuracy 变化为 −0.105820 和 −0.191799 个百分点；两个模型均通过原固定身份保持界限。相对 IQ 改变量中位数为 7.81%，最大为 11.13%。这只支持进入识别收益检验，不能证明跨 RX clean 泛化。G0 另计 50,402 次 NN 包前向、0 次训练更新。

旧 normalized 和 additive As 配方的跨种子 clean 失败保留且关闭。本实验是独立 A_g 路线，不以修改它们的系数、信道或种子救回旧配方。原始 WiSig 文献说明 256 点前导及 25 Msps 观察：[WiSig，IV-B](https://arxiv.org/pdf/2112.15363)。[mixup 原论文](https://arxiv.org/abs/1710.09412)仅支持混合的数学形式，不是 RFFI 标签保持或域泛化的证据。

## 数据、基座与三个固定行

run_id：20261016-phase1-F-crossRX-sameTX-IQmix-sourceRX1-e20-s2026092701-m3-r01。model seed 为 2026092701，split 为 392005。

基座为 F＋跨 RX 同 TX 对比学习：三路 SIG_TIME、统一 Sinc、joint160、默认 cosine 头与骨干联合训练，NoExplicitCFO 和必要内部补偿保留。CROSS_RX 仍只作用于原普通增强的浅层/中层，不扩展至新视图，不新增部署模块、对齐损失或分类头。

原验证载体 L6300/U56700/V27000 保留；RX1 在模型、教师、训练和校准前排除。实际训练契约为 RX3/4/6/8、day1/2/3、6 TX 的 L5040/U45360/V21600；RX1 原 L1260 仅作冻结后源域留出读出。此契约不能冒称完整 L6300 或 L_train5580/L_dev720。U 的 TX 真值隐藏，原 13L＋7U、旧伪标签规则和自身 EMA 保持。

| 行 | 第四 IQ 对输入 | 权重与控制 |
|---|---|---|
| B00 | 原始 raw | .25；匹配未新增 A_g 控制 |
| C_same | 同 TX/day、同 RX 的混合 | .25；控制平均作用与 3 个供体变体 |
| B10 | 同 TX/day、其他训练 RX 的混合 | .25；唯一新增 A_g 候选 |

三行完整保留原 raw .75＋ordinary .25 的监督和 SupCon，再加 .25 R(第四视图)。R 是既有 smoothing .01 CE 与同一 IQ 的 RDrop1 KL。每个物理包 4 对、8 次 L 前向，pair 权重总和均为 1.25；每对按物理包均值归一化，不再除以新增视图数。绝对 raw 监督保留，不保证 clean 必然受保护。所有行普通增强相同；B00 不是完全无增强。

新视图仅用于可见 L。一次生成 IQ 后，两次随机子网络前向复用同一个 IQ；RDrop 是训练配方而非 IQ 泛化机制。A_s 训练关闭，不增加 U/教师视图。

## 固定 IQ 算子与训练银行

仅用实际过滤后的原生 train_loader.dataset 可见 L5040 建立一次临时银行，三行各读取同样 5040 个包。按现有 opaque ID 在每个 TX/RX/day 的 70 包单元排序。跨 RX 在三个其他训练 RX 各取 next-rank＋1；同 RX 控制取 rank＋1/＋2/＋3，以匹配三个不同变体。按每个 anchor 的累计访问次数 mod3 轮转，不依质量、置信度或目标结果筛选。G0 的同 RX 控制只检查了＋1；此处在训练前明确扩为三个，跨 RX 候选不变。

算子完全复用 G0：供体 RMS 对齐 anchor；用 lag20 估计相对频率、在 80:160 窗对齐公共相位；0.9 anchor＋0.1 donor；恢复 anchor RMS，全 256 点。不循环移位、裁切、制造接缝或叠加噪声/PA/完整信道。保留 anchor 频率和现有内部补偿，不把 CFO 作为身份。固定 .1 与相对 RMS 上限 .25 为先验工程边界，不是目标或源分数搜索。

银行只存在训练模块全局，Observer 持有审计引用；不进入 model、teacher、buffer、checkpoint、导出特征或部署状态。U/V/RX1 不进入银行。每次查找核对实际 metadata 与标签权限，不丢弃 anchor。

## 预算、初始化与源读出

每行从零 E20、13L＋7U、222 step/epoch，共 4,440 更新；lr .0002、weight decay .0001、batch128、原 cosine/Sinc 物理单位优化保留。总 60 model-epochs、13,320 更新。无父权重、无续训。旧构造配置的 nominal F50 和 legacy E200 字段是架构/历史元数据，实际参数、预算断言和 native 训练路径均为 E20；不得据此报告 F50/E200 成本。

每行 LNN 4,546,560，含新增对 1,136,640；普通增强生成 568,320。C_same/B10 各生成 568,320 个混合包及同数 donor gathers；B00 无混合，但执行同一银行选择计数。每行银行直接读取 5040；U/自身 EMA/V/最终源读出、额外初始化检查、实际耗时另计。dispatcher 仅一次匹配自身原 B00 scratch E20 checkpoint 的无 query 烟测，2 个合成包；checkpoint 不作为训练初始化。

每行 final20 冻结后立即自动完成源 RX1 clean＋六种既定 practical，共 8820 预测；三行 26,460。预测固定后按源标签独立复算。formal target 不开放，本次明确为先行源域反证。逐场景 accuracy、MacroF1、TX/day 分层、双方控制的修复/新增错误、suburban 绝对值与 own-clean gap 全部报告。最终全训练契约的 clean＋六 practical、B00/B10/B01/B11、交互项和跨种子仍未完成。

## 固定判定与停止

B10 必须同时相对 B00、C_same 达到：留出 clean accuracy 至少＋1 个百分点；clean MacroF1 不下降；每种 practical 不低于 −0.5 个百分点；seen V 不低于 −0.5 个百分点。预算、银行权限、默认头/教师隔离与原共同 L 目标 ≤0.5 的健康条件另核实，健康本身不代表识别成功。

任一科学条件失败即关闭此固定 IQmix 配方，不通过 coefficient、alignment、window、donor、RX、seed、预算重新搜索。通过也只支持独立源种子确认，不宣称 A_g/DG 或联合目标达成。联合 low urban >60% 与 suburban 各场景距自身 clean ≤2 个百分点的首轮探索目标保持，不能靠降低 clean 达成。

系统性协议、角色、初始化、预算、输出冲突、非有限、无法启动或非法预测失败时保留产物、取消未启动行，不自动重试；已健康运行任务保持。无管理员操作、无健康旧任务干预。

## 执行与已完成验证

环境：普通 N607 CVS-RFFI，FP32；CPU dispatcher 2 threads，loader0。GPU 由既有共享容量规则分配，每 GPU 最多 3 个所有 owner 总实验进程，空闲显存至少 12000 MiB。

CWD：PROJECT/releases/cvs_F_sameTX_IQmix_sourceRX1_e20_20261016_r01。启动命令：CVS-RFFI/bin/python -u experiments/cvs_same_tx_iq20/runner.py。唯一 owner：codex/root/F-sameTX-IQmix-E20-20261016。不可变输出：PROJECT/runs/上述 run_id，每行 source 与 logs。实际绝对路径、命令和配置见 experiment.json。

已完成本地 35 个协议负例；3 个真实 F 结构合成联合优化步骤；B00 与既有 additive D_raw 的 loss 和默认头梯度精确匹配；三行初始化及 RNG 状态匹配、8 前向与同 IQ RDrop、joint160 导出、银行不进状态均 PASS。本地工程成本 4114 synthetic NN、3 synthetic updates、5040 synthetic bank、1024 IQmix、128 ordinary，0 real IQ、0 target、0 另训头。一次独立 P0/P1 执行审查 PASS。N607 一次普通账号/资源只读预检完成。

预期产物：真实 checkpoint 烟测、原生 resolved args/source contract/init/frozen、全部 4440 步/行、银行与访问审计、每行七视图预测/评分、独立完整源分析、实际成本、同 row 报告及登记。仅发布包作一次本地/远端 SHA 比较和远端 compile；不增加签名、数据重验或审批层。
