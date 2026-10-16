# F＋跨 RX 同 TX：同身份实测 IQ 混合的源域 E20

状态：ANALYZED／VERIFIED，三组 E20 和全部源视图已完成。固定源规则：FAIL_FIXED_SAMETX_IQMIX_CLOSED。联合目标未完成。执行代码身份以本次 Git commit 和 release_commit.txt 为准，发布后填入 execution_summary.json。

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


## 实际发布与启动读回

执行 commit 192e6a84d72ca89d29489ce2750a20eb36bc765d 已推送，远端 OID 与本地 HEAD 独立比对一致。一次归档 SHA 比较、remote compile VERIFIED；匹配真实 checkpoint 无 query 烟测 PASS。dispatcher 与三个 worker 的 PID/CWD/argv/CUDA 环境、实际 GPU 进程和逐步日志增长独立核实 VERIFIED。三行实际参数为 seed2026092701、20/13/7、scratch、无 baseline/teacher 父权重。resolved 契约为 L5040/U45360/V21600、8 次 L 前向、保留 raw.75/ordinary.25 并追加 .25，权重总和1.25，As 训练关闭。

- B00-RX1-s2026092701：GPU 0，已观察 step 177/4440，仅为运行证据，无最终识别结论。

- C_same-RX1-s2026092701：GPU 1，已观察 step 133/4440，仅为运行证据，无最终识别结论。

- B10-RX1-s2026092701：GPU 2，已观察 step 146/4440，仅为运行证据，无最终识别结论。


## 与并行对话的方向重合

在当前三组已启动后，重新读取[开发 CVS Phase1 星地增强训练](thread://01a13a98-fbec-7ba3-86b9-4cd56b8992c0?hostId=local) 的最新可用轮次。对话也已转向同 TX/day 的跨 RX 实测 IQ 插值，并加入同 RX 控制。其描述采用参考窗复标量对齐与整包变化上限；本路采用相对频率及公共相位对齐后固定 .1 组合。算子细节不同，但属于同身份双观测插值这一类，不能宣称两条不同增强方向或把结果拼成同一配方的跨 seed 证明。

对方最后可见记录仍处于实现及审查阶段；本次未检查其远端作业，因此不据对话状态断言其作业已启动或停止，也未发送消息或修改其任务。本次已运行三组的预算、判据、自动七视图源评分和产物保留全部不变。后续新候选避开同身份插值类；此处仅校正研究分工，不改变科学判定，不追加审批或重跑。


## 完整执行与独立源结果

三 worker 和 dispatcher 自然结束，无技术失败。完整复算 3×20 轮、13320 步 native/view/Sinc 记录、全部 L/U 物理序列、每个供体与 visit-slot，以及 26460 个冻结硬预测。三组实际 L/U 物理序列相同；native 参数仅 candidate_id/output_dir 不同。所有结果均为源 RX1-held／seed2026092701，不是正式 target 或跨 seed 证明。

| 场景 | B00 accuracy % | C_same accuracy % | B10 accuracy % | B10−B00 pp | B10−C_same pp |
|---|---:|---:|---:|---:|---:|
| clean | 80.396825 | 81.349206 | 80.873016 | +0.476190 | -0.476190 |
| practical_high | 39.761905 | 40.079365 | 39.841270 | +0.079365 | -0.238095 |
| practical_mid | 39.523810 | 41.031746 | 39.841270 | +0.317460 | -1.190476 |
| practical_low_suburban | 36.269841 | 36.666667 | 36.507937 | +0.238095 | -0.158730 |
| practical_high_urban | 38.968254 | 40.079365 | 39.603175 | +0.634921 | -0.476190 |
| practical_mid_urban | 32.777778 | 32.063492 | 32.063492 | -0.714286 | +0.000000 |
| practical_low_urban | 27.857143 | 27.936508 | 28.095238 | +0.238095 | +0.158730 |

每场景为同一批 1260 个源物理包，完整 MacroF1、混淆矩阵、双方控制修复／新错及全部 TX/day 分层见 source_analysis.json。

| 对照 | clean F1 差 pp | clean 修复／新错 | low urban 修复／新错 | seen V 差 pp |
|---|---:|---|---|---:|
| B10−B00 | +0.353285 | 19/13 | 25/22 | +0.050926 |
| B10−C_same | -0.546640 | 5/11 | 23/21 | +0.032407 |

| 固定条件 | 结果 |
|---|---|
| B00_clean | FAIL |
| B00_clean_macroF1 | PASS |
| B00_each_practical | FAIL |
| B00_seen_V | PASS |
| C_same_clean | FAIL |
| C_same_clean_macroF1 | FAIL |
| C_same_each_practical | FAIL |
| C_same_seen_V | PASS |
| healthy | PASS |

| 行 | high suburban 差距 pp | mid suburban 差距 pp | low suburban 差距 pp |
|---|---:|---:|---:|
| B00 | 40.634921 | 40.873016 | 44.126984 |
| C_same | 41.269841 | 40.317460 | 44.682540 |
| B10 | 41.031746 | 41.031746 | 44.365079 |

上述差距相对各行自身 clean，不能通过 clean 下降解释为成功；探索目标仍为各 suburban 绝对差距 ≤2pp。固定预测的四类 clean/practical 错误转移见 error_transitions.md/json，仅作描述，不删样本、不改变门槛。

### 完整成本与训练轨迹

| 行 | native 第一 ordinary 平滑 CE | 共同 base 目标 | extra 对目标 | 实际 NN 包前向 | 训练秒 |
|---|---:|---:|---:|---:|---:|
| B00 | 0.26397779 | 0.31081180 | 0.09727651 | 5823318 | 1374.175 |
| C_same | 0.26520306 | 0.31115793 | 0.09988539 | 5823318 | 1560.172 |
| B10 | 0.26448714 | 0.31098458 | 0.10085881 | 5823318 | 1425.460 |

每行 4440 更新、4546560 L 前向（含 extra1136640）、568320 次 ordinary 生成。每行直接银行读取 5040，三行共 15120；C_same/B10 各 568320 次 IQmix 及 donor gathers，B00 为 0。三行总 60 model-epochs／13320 更新，实测总 NN 17469956，含 dispatcher 匹配真实 checkpoint 的合成 smoke2。

学生 train/eval 计数 14828784／1425606；teacher train/eval 仪表计数 0／1189110，扣除继承 offset 6；held 读出 26460。U／自身 EMA／seen V 均单列，teacher IQmix 和 As 训练为 0。原 G0 为 50402 NN／0 更新；本轮本地工程 4114 合成 NN／3 合成更新另计，不混入真实训练。

损失为最后 222 步均值。native 第一 ordinary CE 不是完整目标；全部 base／extra CE、RDrop KL、SupCon 系数和总目标已逐步重算。硬预测不含 logits，读出 CE 保留运行时报告，未冒称独立复算。source_contract.counts 保留原载体计数，实际角色以 role_ids 的 L5040/U45360/V21600 为准，银行逐 ID 与实际 L 角色集合相同且不与 U/V 相交。

完整 20 轮训练曲线、七视图绝对结果／own-clean gap、全部 TX/day 图位于本地私有 local_artifacts/F_sameTX_IQmix_E20_20261016/final/，不提交原始 IQ、权重、样本 ID、日志或生成输出。

### 结论与后续边界

此固定配方未通过预登记条件，关闭当前 IQmix 配方，不以 coefficient、alignment、window、donor、RX、seed 或追加预算救跑。G0 身份保持不等于跨 RX 身份提取收益，健康通过也不能补偿识别条件失败。

本轮 A_s 训练关闭；目标 IQ/SNR/信道、困难样本和其他健康任务未改。尚缺有效 A_g 的跨 seed 与正式 target clean＋六 practical 证据、同契约 B00/B10/B01/B11 与交互项和完整联合验证。low urban>60%、suburban 接近自身 clean 和 clean 净收益仍必须共同证明。后续新候选避开同身份插值类；E20 是用户指定且历史测试基准已暴露的压缩探索预算，不声称新的独立盲测。


## 负结果的具体含义与下一假设

B10 clean 相对 B00 修复 19／新增错误 13，净增 6 个正确样本；相对 C_same 修复 5／新增错误 11，净少 6 个正确样本。low urban 相对 B00 修复 25／新增错误 22，净增 3 个。局部小幅净收益不能证明跨 RX 变化超过同 RX 平均的价值。

完整训练曲线显示三组整体轨迹接近；第 6 轮开始原 SupCon，随后按原日程增大，第 14 轮开始原 U。总目标在 SupCon 启用后上升不能单独当作 CE 崩溃。三组 source V 与共同 base 健康通过，但这没有补偿留出识别的失败。已逐张查看全部训练曲线、七视图结果／自身 clean 差距与全部 TX/day 图。

TX/day 的描述性差异必须保留：B10 相对 B00 的 TX2 clean −3.333333pp、TX5 clean +4.285714pp，day2 clean −1.666667pp；相对 C_same 的 TX2 clean −4.285714pp、TX3 mid suburban −8.095238pp。它们来自固定同一批样本，不作新候选选择或损伤机制因果解释。B10 三个 suburban 距其自身 clean 为 41.031746／41.031746／44.365079pp，远未接近 2pp。G0 的源 fit 身份保持只证明当时的有限默认头保护界限，并未证明源留出泛化增强。失败的是这个固定配方，不据此宣称所有 IQ 混合或 A_g 均不可能。

当前唯一下一研究假设（尚未登记／实现／启动）：practical 随机实现的有限采样可能导致更新不稳定；在相同 expected objective、相同 IQ 前向数和原监督权重下，对两个独立完整 practical 视图分别监督，可能比重复同一视图更好地保留 clean。旧 As 跨 seed 的 clean 不稳定只提供问题动机，尚未证明是采样方差导致，不能预告有效。

最小新矩阵拟保持 F＋CROSS_RX、scratchE20、原 L/U/V 与默认头：原完整 raw.75／ordinary.25＋SupCon 共同目标不变，再加 .25 的两对视图均值。三个控制为两对 raw、同一个完整 As IQ 的两对重复、两个独立完整 As IQ 各一对。均 10 次 L 前向、pair 权重和1.25；每一对 RDrop 复用同一个已生成 IQ。只平均各视图损失，不平均接收 IQ；不提高 SNR、减弱信道或过滤困难样本。原 PairedChannel 的 namespace 控制场景与传播随机流，receiver_for_session 保留固定 receiver seed 和 RX/day/eq session，不使用 TX 标签定义硬件。训练只加可见 L，U/教师规则保持。

拟定放弃标准将在首个新真实运行前冻结：双独立 As 相对重复 As 的 clean 至少 +1pp，clean F1 不下降、每 practical 不低于 −.5pp；相对同前向 raw 控制 clean/F1 不下降、low urban 至少 +5pp、urban 均值至少 +3pp，suburban 与 seenV 保护。任一失败即关闭固定 MC2 组织，不扩 M、不改系数／RX／seed／预算救跑。这是下一方案草案，未冒称本实验的新结果；正式新实验仍需原最小发布流程。当前失败 IQmix 不续训、不扩种子，完整联合目标保持。
