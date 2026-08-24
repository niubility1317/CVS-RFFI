# FastTrust-QB3有界域混淆正式E200实验报告

## 当前状态

- run_id：`phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1`
- 状态：`LOCAL_VERIFIED`
- 实现提交：`37de27cc83943283a54d8d19b93dc4f89159f5f6`
- Git发布：`VERIFIED`，本地HEAD与`origin/work/cvs-active`OID一致。
- 性能状态：尚未启动，不能声称性能完成。

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
| 梯度遥测 | `code/SSDG/train_ssdg.py` | parser/集成测试；正式运行读回`g_L/g_H/g_Pset/g_Pcond/g_adv` | 本地`VERIFIED`，远端待运行 |
| finite恢复与异常包 | `code/SSDG/train_ssdg.py` | launcher传播、代码编译；正式运行检查artifact | 本地`VERIFIED`，远端待运行 |
| 训练加速 | matrix与QB3 launcher | E200/U256不变；一GPU一行、eval batch1024、source-heavy每10epoch、尾20epoch逐epoch | 配置`VERIFIED`，吞吐待运行 |
| clean与三LEO最终闭合 | 既有Phase1 final evaluator | 最终检查四个metrics JSON、日志与`final_ssdg.pth` | 待运行 |

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

