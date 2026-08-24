# FastTrust-QB3有界域混淆正式E200实验报告

## 当前状态

- run_id：`phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1`
- 状态：`ANALYZED`
- 实现提交：`37de27cc83943283a54d8d19b93dc4f89159f5f6`
- release/report提交：`a4fbb465cc9c94e3b1af41efe73ccc32ce9ee750`
- Git发布：`VERIFIED`，本地HEAD与`origin/work/cvs-active`OID一致。
- 性能状态：5行均完成200/200epoch、final checkpoint、Clean与三种LEO weak评测，结构化artifact已完整解析，实验性能闭合。

## 设计依据与实际调整

QB0/QB1在E104–E106终止前，`train/loss_adv_labeled`分别已升至约29.7和303.7。故本轮没有只修U侧对抗项，而是把有标签与U两条`z_id→domain`路径统一改成有界混淆。域判别器在`z_id.detach()`上最小化CE；身份编码器在冻结域判别头时最小化`KL(p(domain|z_id)||Uniform)`；`z_dom`域CE保持不变。两个域判别头前向和损失均在内部强制float32，外层AMP不能先把logits溢出。

P-set与P-conditional已拆成独立开关、系数、尾段退火和梯度遥测。P的95%APS覆盖与N的99%排除目标解耦；QB3固定全局APS，避免按真类拟合、按预测类使用的错配。风险阈值同时满足总体与最差source receiver精度，H/P分别使用0.05/0.10有效权重质量预算，并施加class×receiver直接质量上限。

C4使用冻结Core90对U clean view生成的`z_id`作为目标，约束student strong view的归一化`z_id`。该行不读取U的TX truth，用于判断P是否主要提供一般身份稳定梯度。设计报告提出的模型状态hash没有实现为额外发布门；精确恢复改由每epoch覆盖的完整finite checkpoint、E90固定恢复点、RNG/optimizer/scaler/EMA状态和首次异常包承担，符合当前项目的最小发布规则。

## 设计要求→代码位置→验证方式→状态

| 设计要求 | 代码位置 | 验证方式 | 状态 |
|---|---|---|---|
| 有界域混淆与梯度隔离 | `code/cvsrffi/bounded_domain_confusion.py`、`code/SSDG/train_ssdg.py` | 饱和logits、AMP head前向、判别器/编码器梯度归属测试；真实checkpoint前向反向 | `VERIFIED` |
| 有标签与U路径统一修复 | `code/SSDG/train_ssdg.py` | 训练集成回归、真实checkpoint smoke、独立P0/P1审查 | `VERIFIED` |
| P/N APS解耦与fit/use一致 | `code/cvsrffi/muse_ssdg.py` | 95%/99%解耦测试、全局阈值路由测试 | `VERIFIED` |
| P集合特征进入安全校准 | `code/cvsrffi/muse_ssdg.py` | 12维P特征与权重维度测试 | `VERIFIED` |
| 最差source receiver风险 | `code/cvsrffi/muse_ssdg.py` | 单receiver不安全时拒绝阈值的聚焦测试 | `VERIFIED` |
| H/P独立预算与cell cap | `code/cvsrffi/muse_ssdg.py` | H≤0.05、P≤0.10质量测试 | `VERIFIED` |
| P-set/P-cond因果分离 | `code/SSDG/train_ssdg.py` | 禁用P-cond时总损失和梯度归零的RED→GREEN测试 | `VERIFIED` |
| E181–E200分项退火 | `code/cvsrffi/muse_ssdg.py` | E180/E200边界测试 | `VERIFIED` |
| 梯度遥测 | `code/SSDG/train_ssdg.py` | parser/集成测试；正式运行读回`g_L/g_H/g_Pset/g_Pcond/g_adv` | 正式运行`FAILED`：`batch_idx==0`不可达 |
| finite恢复与异常包 | `code/SSDG/train_ssdg.py` | launcher传播、代码编译；正式运行检查artifact | `VERIFIED`：5行artifact完整 |
| 训练加速 | matrix与QB3 launcher | E200/U256不变；一GPU一行、eval batch1024、source-heavy每10epoch、尾20epoch逐epoch | `VERIFIED_NO_GAIN`：墙钟未实质缩短 |
| clean与三LEO最终闭合 | 既有Phase1 final evaluator | 最终检查四个metrics JSON、日志与`final_ssdg.pth` | `VERIFIED` |

## 正式矩阵

| 候选 | GPU | 有界混淆 | H | P-set | P-cond | U特征锚点 |
|---|---:|---:|---:|---:|---:|---:|
| `E200_C0_BC_NO_U_ID` | 0 | 开 | 关 | 关 | 关 | 关 |
| `E200_C1_BC_H` | 1 | 开 | 开 | 关 | 关 | 关 |
| `E200_C2_BC_H_PSET` | 2 | 开 | 开 | 开 | 关 | 关 |
| `E200_C3_BC_H_PSET_PCOND` | 3 | 开 | 开 | 开 | 开 | 关 |
| `E200_C4_BC_U_FEATURE_ANCHOR` | 4 | 开 | 关 | 关 | 关 | 开，系数0.04 |

全行固定：Core90初始化、seed392002、E200、U batch256、`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`、相同split、相同训练步数和Core90 LEO_WEAK拼接增强。N关闭。H/P有效质量预算分别为0.05/0.10，P-cond系数0.02。域权重为`z_dom=0.16`、discriminator=0.08、confusion=0.08。

## 本地验证证据

- 聚焦RED已观察：缺失有界模块、缺失tail函数、P-cond开关误接线、AMP head未强制float32和非有限logits未拒绝。
- 修复后相邻回归：158项通过。
- Python编译：`bounded_domain_confusion.py`、`muse_ssdg.py`、`train_ssdg.py`通过。
- 参数dry-run：E200解析成功。
- 真实checkpoint无query smoke：加载本地真实E194 checkpoint，严格重建无missing/unexpected key；42条合成source形状输入完成三教师校准、路由、student前向和反向；P特征维度12，H质量为2.10/42=0.05，所有loss和梯度有限，query输入数0。
- 独立P0/P1审查：首次发现2个P1并完成唯一一次定点复审。P-cond关闭失效与AMP head前向溢出均已修复；复审结论`PASS`。
- 本地Git Bash通道：`FAILED`。桌面执行层把指定Git for Windows错误路由为`/bin/bash`，因`pwd -W`不支持而停止；没有执行launcher，也没有用WSL绕行。N607原生Bash语法检查将作为发布前远端编译的一部分。

## 发布参数

- 本地工作区：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 远端代码根：`/home/szu2070436088/2510044040/CV-SincNet`
- launcher：`code/scripts/launch_phase1_adv3b02_fasttrust_qb3_bc_hps_e200_20260824.sh`
- matrix：`configs/phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824.json`
- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1`
- 日志：每个候选目录的`train.log`、`metrics_epoch.jsonl`及dispatcher日志。
- 预期训练artifact：每行`final_ssdg.pth`、`latest_finite_ssdg.pth`、`recovery_e90_ssdg.pth`、完整epoch指标；发生首个异常时额外产生`first_rc4_anomaly.pt`。
- 预期最终评测artifact：`metrics_clean.json`、`metrics_joint.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`、`metrics_leo_rain_weak.json`及对应日志。

## 停止规则与结论边界

只在协议/query越权、输出覆盖、错误checkout、无prediction闭合、launcher级故障或至少两行出现相同确定性预评测异常时停止本run的后续dispatch，并且只能处理本run绑定的进程树。中间准确率低、C2/C3负收益或P覆盖不足均不得终止训练。

本轮预计墙钟约7.5–9小时。估计以QB2完整E200的9小时8分42秒为基线：source-heavy评测从每5epoch降至每10epoch，评测batch从512增至1024，五行在五张GPU上一GPU一进程并行；每epochfinite checkpoint和服务器I/O可能抵消部分收益。该时间是工程估计，不是已验证吞吐。

只有五行各自完成final checkpoint、clean和三种LEO weak场景评测后，状态才能从`RUNNING`进入`ARTIFACTS_COMPLETE`。单seed只支持同row机制判断；target结果不得用于改参、候选重排或补跑。

## N607发布与启动证据

- 启动确认时间：`2026-08-24T12:23:31+08:00`。
- 普通账号只读preflight：`VERIFIED`。用户为`szu2070436088`，主机为`dell-DSS8440`，项目根可见；启动前GPU0–7均为1MiB、0%利用率，且无compute process。
- release归档：`E:\type10-7\release_artifacts\phase1_fasttrust_qb3_a4fbb465.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_fasttrust_qb3_a4fbb465.tar.gz`；按规则仅进行一次本地/远端SHA核对，双方均为`4ac8035b5b4ff2529e445b398e5783817ded1fb2228c1d635480d4d1b48c5890`。
- 不可变release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_qb3_a4fbb465`。
- 远端发布前验证：两个launcher的`bash -n`、三个Python生产文件编译、完整5行`--dry-run`均通过；干跑后正式run root仍不存在。
- 正式dispatcher：PID`2163660`，父PID为1，命令行为不可变release中的QB3 launcher；CWD为`/home/szu2070436088`，代码、matrix和worker均通过显式`CODE_ROOT/MATRIX/WORKER`绑定到上述release。
- 启动进程/GPU：5个主训练进程分别绑定GPU0–4，显存约1746–1880MiB，读回利用率为19%–42%；GPU5–7仍为1MiB、0%。未超过每GPU两个训练进程限制。
- 日志：5个候选的`train.log`均已创建，大小为6895–7015字节；8秒首epoch窗口内未追加epoch行，GPU计算仍活跃，符合按epoch落盘的初始化阶段表现。外层dispatcher日志已记录正式`rows=5,dry_run=0,epochs=200,U=256`。
- 异常扫描：未发现Traceback、CUDA OOM或非有限异常指纹。
- 启动阶段最高证据状态曾为`RUNNING`；终态已由下文完整artifact读回推进至`ANALYZED`。

## 终态结论

本轮最重要的结论是：风险校准的`H+P-set+P-conditional`完整机制在冻结的同row矩阵中取得了方向一致的增益。`E200_C3_BC_H_PSET_PCOND`相对`E200_C0_BC_NO_U_ID`，Clean提高0.802个百分点，三种LEO均值提高0.308个百分点，最差LEO场景提高0.337个百分点，最差receiver×LEO单元提高0.375个百分点。它不是只提高平均值而牺牲最差单元，因此机制假设获得了初步支持。

但是，当前证据不足以把C3直接提升为正式默认方法。原因有三：第一，只有seed392002一个seed，0.3个百分点量级的LEO增益尚未得到重复性验证；第二，最差receiver×LEO准确率仍只有57.617%，跨接收机弱点没有根治；第三，所有候选都出现了低频非有限梯度跳步，梯度分项遥测又因条件错误没有真正启用。正确结论是“C3通过单seed同row可证伪验证，值得进入冻结的多seed复验”，而不是“已经证明普遍显著优于历史方法”。

## Artifact与评测完整性

5个候选均具备完整的`status.txt`、200行`metrics_epoch.jsonl`、完整`metrics_epoch.csv`与`train.log`、`final_ssdg.pth`、`latest_finite_ssdg.pth`、`recovery_e90_ssdg.pth`和`first_rc4_anomaly.pt`。每行均保存`metrics_clean.json`、`metrics_joint.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`、`metrics_leo_rain_weak.json`及对应评测日志。

四个最终评测全部加载epoch200 checkpoint，每个场景评测60000条样本、5个未见target receivers。所有行均为strict checkpoint reconstruction：`strict_requested=true`、`checkpoint_load_strict=true`，没有missing key、unexpected key、shape mismatch或fallback。完整日志中没有Traceback、CUDA OOM、Killed或RuntimeError。因此，训练与最终评测artifact状态为`ARTIFACTS_COMPLETE`，本报告完成解释后状态为`ANALYZED`。

## 最终结果总表

| 候选 | Clean | LEO clear | LEO low-elev | LEO rain | LEO均值 | LEO场景floor | receiver×LEO floor | ΔClean对C0 | ΔLEO均值对C0 | Δreceiver floor对C0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0：NO_U_ID | 84.4000 | 75.4883 | 72.9800 | 72.2083 | 73.5589 | 72.2083 | 57.2417 | 0 | 0 | 0 |
| C1：H | 84.6417 | 75.3667 | 72.7867 | 72.1450 | 73.4328 | 72.1450 | 57.4333 | +0.2417 | -0.1261 | +0.1917 |
| C2：H+P-set | 84.8867 | 75.6667 | 73.0567 | 72.4333 | 73.7189 | 72.4333 | 57.0750 | +0.4867 | +0.1600 | -0.1667 |
| C3：H+P-set+P-cond | **85.2017** | **75.8517** | **73.2033** | **72.5450** | **73.8667** | **72.5450** | 57.6167 | **+0.8017** | **+0.3078** | +0.3750 |
| C4：U feature anchor | 84.3150 | 75.3967 | 72.9050 | 72.1483 | 73.4833 | 72.1483 | **57.9167** | -0.0850 | -0.0756 | **+0.6750** |

这里的百分点差值来自同一seed、同一split、同一Core90初始化、相同U batch256和相同训练步数的冻结矩阵。C3在Clean、三个LEO场景及其均值上均为全矩阵最高；C4只在最差receiver×LEO单元上最高，但平均性能没有提高。

## 单因素因果分解

| 同row比较 | ΔClean | ΔLEO均值 | ΔLEO场景floor | Δreceiver×LEO floor | 解释 |
|---|---:|---:|---:|---:|---|
| C1−C0：加入严格H | +0.2417 | -0.1261 | -0.0633 | +0.1917 | H提供少量Clean与最差单元收益，但单独使用不足以改善LEO平均泛化 |
| C2−C1：加入P-set | +0.2450 | +0.2861 | +0.2883 | -0.3583 | 集合监督是LEO平均收益的主要来源之一，但会扩大局部receiver风险 |
| C3−C2：加入P-cond | +0.3150 | +0.1478 | +0.1117 | +0.5417 | 条件分布监督补回P-set的局部floor损失，并进一步提高Clean和LEO均值 |
| C4−C0：仅U特征锚定 | -0.0850 | -0.0756 | -0.0600 | +0.6750 | 一般表征稳定只改善最坏单元，不能代替伪标签身份信息对均值的贡献 |

该分解支持“提高U数据利用率不能等价为让所有U样本承担同一种身份损失”。C2尾段约88.3%的U样本进入H/P原始路由，但有效加权coverage只有12.92%；C3主动减少原始P路由，使约72.85%的U进入H/P，同时有效coverage维持12.87%，反而取得更好的最终指标。这说明关键变量是校准后的有效梯度质量和损失形态，而不是名义覆盖率最大化。

## 逐接收机详细结果

receiver映射为`rx7=20-1`、`rx8=3-19`、`rx9=7-14`、`rx10=7-7`、`rx11=8-8`。

### Clean准确率

| 候选 | rx7 | rx8 | rx9 | rx10 | rx11 |
|---|---:|---:|---:|---:|---:|
| C0 | 81.4667 | 78.6417 | 97.2167 | 93.6000 | 71.0750 |
| C1 | 82.6083 | 79.2083 | 96.7250 | 94.1250 | 70.5417 |
| C2 | 82.4833 | 79.1000 | 97.4250 | 94.8667 | 70.5583 |
| C3 | **82.8333** | **79.6583** | 97.3917 | **94.9250** | **71.2000** |
| C4 | 82.5750 | 78.7417 | 97.4000 | 92.8917 | 69.9667 |

C3相对C0在5个Clean receiver上全部提高，差值依次为+1.367、+1.017、+0.175、+1.325和+0.125个百分点，说明总体Clean增益并非由单个接收机独占。

### LEO clear准确率

| 候选 | rx7 | rx8 | rx9 | rx10 | rx11 |
|---|---:|---:|---:|---:|---:|
| C0 | 72.0667 | 60.9583 | 92.2083 | 81.8500 | 70.3583 |
| C1 | 71.0583 | 60.6250 | 92.4750 | 82.1833 | 70.4917 |
| C2 | 71.9083 | 60.6417 | 92.7667 | 82.6250 | 70.3917 |
| C3 | 71.9583 | **61.2333** | **92.8167** | **82.6833** | **70.5667** |
| C4 | 71.2833 | 61.2833 | 92.7500 | 81.6167 | 70.0500 |

### LEO low-elev准确率

| 候选 | rx7 | rx8 | rx9 | rx10 | rx11 |
|---|---:|---:|---:|---:|---:|
| C0 | 68.8417 | 58.4333 | 89.6750 | 79.4167 | 68.5333 |
| C1 | 67.8500 | 58.2167 | 89.6250 | 79.4250 | 68.8167 |
| C2 | 68.0833 | 58.3000 | **90.1000** | 80.0750 | 68.7250 |
| C3 | 68.2667 | 58.7917 | 90.0000 | **80.1333** | **68.8250** |
| C4 | 67.9833 | **58.8000** | 90.2250 | 78.9333 | 68.5833 |

### LEO rain准确率

| 候选 | rx7 | rx8 | rx9 | rx10 | rx11 |
|---|---:|---:|---:|---:|---:|
| C0 | 69.9083 | 57.2417 | 88.4583 | 77.6583 | 67.7750 |
| C1 | 68.9333 | 57.4333 | 88.6167 | 77.9333 | 67.8083 |
| C2 | **69.6500** | 57.0750 | **88.9917** | 78.4333 | 68.0167 |
| C3 | 69.6083 | 57.6167 | 88.8333 | **78.5750** | **68.0917** |
| C4 | 69.1250 | **57.9167** | 88.6417 | 77.2000 | 67.8583 |

C3相对C0改善了15个receiver×LEO单元中的12个。唯一一致退化的是rx7：clear、low-elev、rain分别下降0.108、0.575和0.300个百分点。最差单元仍为rx8-rain，但从57.2417提高到57.6167。因而C3提升了广度，却还没有解决rx7在LEO域的适应问题，也没有消除rx8-rain的绝对低性能。

## 伪标签利用率与梯度质量

以下统计使用E181–E200完整结构化记录，平均每batch有效U样本数约255.652。

| 候选 | H平均数 | P平均数 | representation平均数 | H有效coverage | P有效coverage | 总有效coverage | H+P原始利用率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0 | 0 | 255.652 | 0 | 0 | 0 | 0 |
| C1 | 11.236 | 0 | 244.416 | 3.3618% | 0 | 3.3618% | 4.40% |
| C2 | 10.572 | 215.151 | 29.930 | 3.2397% | 9.6841% | **12.9238%** | **88.29%** |
| C3 | 13.774 | 172.442 | 69.436 | **4.8529%** | 8.0195% | 12.8723% | 72.85% |
| C4 | 0 | 0 | 255.652 | 0 | 0 | 0 | 0 |

C2的分段总有效coverage为：E1–10为10.923%，E11–40为10.015%，E41–90为10.073%，E91–160为14.760%，E161–180为13.306%，E181–200为12.924%。C3对应为11.497%、10.002%、12.054%、10.340%、12.842%和12.872%。两者都没有在E91后发生覆盖率坍塌。

尾段C2的平均`p_correct=0.988529`、`p_set_safe=0.999512`；C3分别为0.988844和0.999512。C3尾段原始P-set loss约1.06×10^-5，原始P-cond loss约8.37×10^-4，说明P-cond虽覆盖权重较小，但提供了比纯集合排除更明确的类内分布方向。C2日志中也会计算非零的原始P-cond诊断值，但其开关关闭后总损失贡献和梯度已经由聚焦测试验证为零，不能把原始诊断量误认为C2实际使用了P-cond。

尾段退火在正式训练中确实生效：C3在E181的H/P-set/P-cond缩放均为1.0；E190分别为0.810526/0.621053/0.526316；E200分别为0.600000/0.200000/0。它避免在训练末端继续用同等强度固化伪标签误差。

## 200epoch收敛与稳定性

| 候选 | E100 LEO mean/floor | E160 LEO mean/floor | E180 LEO mean/floor | E200 LEO mean/floor | E181–200均值/floor |
|---|---:|---:|---:|---:|---:|
| C0 | 92.299/90.746 | 93.683/92.349 | 94.056/92.738 | 94.119/92.833 | 94.069/92.785 |
| C1 | 92.413/90.786 | 93.865/92.587 | 94.108/92.857 | 94.169/92.976 | **94.132/92.931** |
| C2 | 91.590/89.929 | **94.003/92.857** | **94.153/92.952** | 94.140/92.937 | 94.111/**92.935** |
| C3 | 91.847/90.238 | 93.616/92.349 | 94.087/92.802 | 94.122/92.913 | 94.078/92.857 |
| C4 | 92.561/90.937 | 93.868/92.595 | 94.090/92.762 | 94.106/92.849 | 94.078/92.803 |

表中是合法source-validation LEO proxy，不是target测试。所有行从E100到E180持续提高，E181–E200维持稳定，没有重现QB0/QB1在E104–E106附近的数值崩溃。尾段Clean source-val准确率均约98.64%–98.67%。值得注意的是，source-val上C1/C2略高于C3，而最终target上C3最好，说明source proxy对候选间0.1个百分点量级差异不够敏感。后续不能用本次target结果继续细调或重排候选，只能预先冻结C3复验。

## 有界域混淆的工程收益

5行共1000个epoch记录中，`train_rc4_components_finite`始终为1。以C3为例：有标签域混淆损失范围0.027468–0.081206、均值0.064218；有标签域判别损失范围2.547119–2.607190、均值2.574774；U域混淆损失范围0.025447–0.071543、均值0.055618；U域判别损失范围2.530955–2.607039、均值2.556123。

此前QB0/QB1无界对抗损失分别上升到约29.7和303.7，并在E104–E106技术失败。本轮所有控制与候选均完整跑到E200，域相关损失保持在有限窄区间，证明“把身份编码器目标改为有界均匀分布混淆，并把域头内部前向固定为float32”解决了原有的系统性发散。由于本轮还同时改动了P机制和训练配置，该结论限于数值稳定性，不能把最终准确率差值全部归因于有界混淆。

## 异常、日志问题与证据边界

### 低频非有限梯度仍然存在

所有行的总loss均有限，`train_skipped_nonfinite_loss=0`，但仍出现共享的非有限梯度跳步：C0为36批/29个epoch，C1为36批/29个epoch，C2为35批/30个epoch，C3为40批/30个epoch，C4为38批/30个epoch。每行约41400个训练step，对应0.0845%–0.0966%的step被跳过。

所有行首次异常都发生在E1 batch1，总loss约1.05645且有限，但clip前总梯度为NaN。首次异常时C0、C4也没有H/P身份损失，因此该问题不可能只由P-set或P-cond引起，更可能位于所有行共享的反向路径。当前检查函数只返回整体布尔值，异常包没有保存首个非有限参数名及分项梯度，故现有证据不能进一步把问题归因到具体层。保护逻辑按epoch重置计数，单epoch跳步数远低于8批和5%的停止阈值，所以没有破坏训练闭合；但下一轮应在第一次异常时记录具体参数名、loss分项和AMP scaler状态。

### 原始日志中的NaN多数是无效占位符

每个`train.log`有9002行，文本扫描可命中约800处`nan/inf`，但主要来自每epoch固定输出的4类禁用诊断：`DM-ACCEPT active=0`中的`p95=nandeg/proxy_vaccept=nan`、训练期不运行target测试产生的`overall_tx=nan% (0/0)`、受保护target joint metric关闭后的NaN，以及`lambda_sat_cons=0`时的`sat_cos=nan`。结构化记录的实际活动数值均可正常解析，完整日志也没有运行异常。

这暴露的是可观测性污染：`nonfinite_train_metric_count=4`和`nonfinite_val/test_metric_count=1`在每个epoch都会被这些禁用字段触发，使告警无法区分“功能未启用的N/A”和“活动计算出现非有限值”。建议把未启用字段记录为`null`并增加`active`标记，非有限计数只统计活动路径。

### 梯度分项遥测没有实际启用

配置要求在E1、E41、E91、E161、E181、E200采集`g_L/g_H/g_Pset/g_Pcond/g_adv`，但1000个epoch记录的`train_rc4_gradient_telemetry_active`全部为0，相关字段没有落盘。代码训练batch使用`enumerate(...,start=1)`，而遥测条件要求`batch_idx==0`，条件永远不可达。这不影响已完成模型的前向结果，但使本轮无法直接证明各伪标签分项的梯度比例，是需要在下一次正式训练前修复的P1可观测性问题。

## 训练耗时与加速效果

| 候选 | 总训练墙钟 | 平均epoch | 峰值allocated | 结论 |
|---|---:|---:|---:|---|
| C0 | 9小时7分6.7秒 | 163.979秒 | 约6.08GiB | 与历史QB2接近 |
| C1 | 9小时5分32.4秒 | 163.508秒 | 约6.08GiB | 与历史QB2接近 |
| C2 | 9小时16分49.9秒 | 166.895秒 | 约6.08GiB | 本矩阵最慢 |
| C3 | 9小时5分57.5秒 | 163.633秒 | 约6.08GiB | 比历史QB2快约0.50% |
| C4 | 9小时1分59.8秒 | 162.446秒 | 约6.11GiB | 本矩阵最快 |

完整矩阵墙钟由最慢C2决定，为9小时16分49.9秒。历史QB2为9小时8分42秒、平均epoch164.61秒、峰值allocated约3.20GiB。当前C2比历史QB2慢约8分8秒，即1.48%；C3仅快约2分45秒，即0.50%。因此，本轮`eval batch=1024`、source-heavy每10epoch等设置没有形成可观测的端到端加速，原先7.5–9小时ETA偏乐观。

峰值allocated从历史约3.20GiB升至约6.08–6.11GiB，接近翻倍，而墙钟基本不变。最可能的解释是更大的评测batch提高了瞬时显存，但每epochfinite checkpoint、float32域头和P路由计算抵消了稀疏评测的节省。由于没有只改变单一加速参数的A/B行，这些只能视为机制推断，不能作严格因果结论。

下一轮加速应优先做不改变优化轨迹的工程A/B：分别测量`eval batch512/1024`、减少非必要每epoch大checkpoint写入、缓存或合并teacher/calibration前向，并把训练step、source-heavy评测和checkpoint I/O分段计时。正式E200科学矩阵仍应保持200epoch，不用50/100epoch替代；可以先用短跑验证吞吐和技术正确性，但不能据此宣称性能闭合。

## 与历史结果的谨慎比较

历史QB2的Clean/LEO均值/floor为85.168/73.592/58.017；历史R1 U256为85.152/73.656/58.525；历史R4 full U256为84.540/74.463/60.383。当前C3分别为85.202/73.867/57.617。

- 对历史QB2：C3为+0.034/+0.275/-0.400个百分点。
- 对历史R1：C3为+0.050/+0.211/-0.908个百分点。
- 对历史R4：C3为+0.662/-0.596/-2.766个百分点。

这些历史行不具备本轮严格同row因果条件，只能用于定位量级。C3虽是本轮冻结矩阵的完整机制优胜者，但尚未超过历史R4的LEO均值和floor，说明“伪标签有效利用率提高”已经取得正向证据，却仍未恢复历史强方法在最差LEO接收机上的鲁棒性。

## 统计解释与下一步

每个最终场景有60000条样本，每个receiver有12000条，但本轮只保留聚合指标，没有候选间逐样本配对预测，因此无法做配对McNemar检验。按独立二项近似，两个73%准确率的60000样本结果差异，其95%不确定范围约为正负0.5个百分点；真实配对不确定度可能更小，但当前artifact无法计算。C3相对C0的LEO均值+0.308个百分点属于“方向一致但需要复验”的证据，不能单凭单seed宣称统计稳健。

推荐的下一步是：

1. 冻结C3与C0，增加预注册多seed复验，不再根据本次target结果微调阈值或重排候选；若资源允许，可保留C2作为P-cond单因素确认。
2. 在复验前只修复两项可观测性问题：梯度遥测`batch_idx`条件和首次非有限梯度的参数级定位；不改变伪标签算法，以免破坏同机制复现。
3. 将rx7的三种LEO退化和rx8-rain绝对低floor作为下一轮source-only诊断对象，检查receiver条件风险预算是否在这些域上失配，但禁止读取target truth调参。
4. 单独开展不改变训练数学定义的加速A/B，重点量化checkpoint I/O、评测batch和teacher前向；默认正式预算继续为E200。

## 最终交付状态

| 交付项 | 状态 | 最高证据层级 | 依据 |
|---|---|---|---|
| 5行训练 | `VERIFIED` | `ARTIFACTS_COMPLETE` | 每行200条epoch记录、final/recovery checkpoint完整 |
| 四场景评测 | `VERIFIED` | `ARTIFACTS_COMPLETE` | 每行Clean和三LEO均完成60000样本strict评测 |
| 数值稳定性 | `VERIFIED_WITH_RESIDUAL_RISK` | E200完整记录 | 域损失全程有限；仍有0.08%–0.10%梯度跳步 |
| 性能解释 | `VERIFIED` | `ANALYZED` | 完整结构化记录、stdout日志、逐场景与逐接收机同row分析 |
| 方法晋级 | `PARTIAL` | 单seed候选通过 | C3主指标全正，但需冻结多seed复验且未超历史R4 floor |

本轮最高交付状态为`ANALYZED`。科学结论为：C3值得进入下一阶段复验，但尚未达到可替换默认方法的证据强度。
