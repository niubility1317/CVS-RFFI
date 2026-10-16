# 原普通增强按物理 RX 分组：最小源域对照

状态：PREREGISTERED，尚未训练；本轮只检验训练组织候选，不宣称新的物理增强、Ag 有效或联合目标完成。

起点提交 f68a9f9642a7975a34792beafc7b278c0251d238。实际代码／配置身份为本次发布的 release_commit.txt；后续分析提交另记。

## 已有证据和唯一假设

固定基座 F＋跨 RX 同 TX SupCon（SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、默认 cosine scale30），网络及头按原流程联合训练。此前 ordinary As 可改善 urban，但跨 RX clean 代价不稳定；coherentmix、timegrid、跨 RX donor、固定课程、LA16 等既定候选未通过各自预登记条件。最新窄带谱线诊断的 RX/bin 跨折稳定性仅 14/18，低于 15/18；不能据此制造 RX 去谱线增强。上述候选保持关闭，不用改名重试。

原普通增强部分触发开关以及 taper／多径核由整批共享。当前只检验：让物理 RX 之间独立抽取同一套原有条件，是否改善跨 RX SupCon 的 clean 泛化。既有单包算子、概率、范围、顺序、边界及归一化保持，实际事件数不要求相等。此假设没有识别收益保证。

## 固定矩阵与预算

| row | 唯一区别 | 预算 |
|---|---|---|
| BATCH-RX1-s2026092701 | 原整批共享条件；分组结果只作 shadow | scratch E20／4440 updates |
| RXGROUP-RX1-s2026092701 | anti-shortcut／general-channel 按实际 RX 分组；整批结果只作 shadow | scratch E20／4440 updates |

两组均为原 .25 RD(ordinary)＋.75 RD(raw) 六次 L 前向，原两次 ordinary 前向复用同一个已生成 IQ，SupCon 只复用这两次；四次 raw 前向、原损失系数、AdamW／cosine、13L＋7U／旧门控／独立 EMA 不改。总计 40 model-epochs／8880 updates，绝非单模型 E40。每组 L NN 3,409,920；原 U／教师／V 及 FINAL 8820 留出样本前向按实际完整日志计数，预计每组 4,686,678 NN packets；真实 checkpoint 兼容 smoke 额外 2 个合成 NN。新增 stage shadow 只产生 IQ，不增加网络前向。

normal PA／DAC 原调用保持在分组阶段之外，class signature 关闭；不添加新的 PA／DAC，不声称反演或替换 TX／RX 硬件。只有显式真实 L 上下文的 normal view 分组；U 包括非负伪标签和所有 defect 路径均直通原实现。private RNG 按 step／stage／物理 RX 刷新，seed20261017831，不使用 TX／day／质量／损失，恢复原 CPU／Python／CUDA 随机流。

## 数据、模型与评价边界

复用原 VALIDATED_ONCE source_contract／split392005，无重划分或新增标签。fit 为 RX3/4/6/8：L5040、U45360（TX 真值隐藏）、V21600。source RX1 L1260 仅固定 FINAL20 评价；原 carrier 加载和 metadata 校验保留，原角色不改变。两个学生／EMA／优化器均从零，无父权重；已核实旧 B00 同有效契约 E20 默认头只严格加载作 2 个合成输入兼容 smoke，随后丢弃，绝不初始化或教师。

IQ2×256／25 MHz／原 RMS 位置不改。FINAL 默认头自动预测 clean 和六种原 practical：high、mid、low_suburban、high_urban、mid_urban、low_urban。保持原 residual／post_sync／EQ off（G=1）、reference_snr_db30 和 reflect 边界、传播／接收处理与 eval 随机流；不弱化测试。这里只是合法源域留出读数，不接触正式 target、query 或 truth，不能代替正式 truth-last 测试。

每 row 固定20 epoch 后冻结，逐 row 自动保存七视图预测、margin/30、完整物理状态与物理 ID。七视图完成后独立离线重算 accuracy／macro F1／状态分层／固定 BATCH clean 正确集合的卫星保持率、repair／break、训练曲线、门控和覆盖。源成绩不改变另一健康 row 或预算。

## 预先固定的判断与停止规则

RXGROUP 相对本次新 BATCH：held clean 至少＋1 pp，clean macro F1 不下降，seen-V 下降不超过 0.5 pp，最终 L CE≤0.5；均需20 epoch／4440 updates／3,409,920 L NN／SupCon3330 steps／4440 coupling calls／8880 stages／568320 L view packets／完整72 TX×RX×day cells，finite、合法、无新增 U／As／head fit／目标访问。全部六种 practical 必须报告，但不作为本次独立 clean 候选晋级门槛。

FAIL 关闭此固定 RX 分组候选，不改变算子子集、分组、RX、seed、阈值或预算救援。PASS 也只允许另行预登记第二 seed 源确认，不构成已验证 Ag、联合突破或 low urban>60%。

系统技术失败仅限本 run 的角色／来源／非有限值／错误预算／不可启动／输出碰撞；保留局部产物，无自动重试，不停止或热改任何健康旧任务。性能差属于科学负结果。

## 执行和最小验证

环境：普通 N607 CVS-RFFI Python，原 full FP32；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_rxgroup_sourceRX1_E20_20261017_r01。单一 owner codex/root/FcrossRX-rxgroup-20261017；容量锁按全部 owner 每 GPU≤3、可用≥12000 MiB，CPU threads2；不使用管理员账号。

命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_rxgroup_sourceRX1_E20_20261017_r01/experiments/cvs_f_rxgroup20/runner.py

run root：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-rxgroup-sourceRX1-e20-s2026092701-m2-r01

日志：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-rxgroup-sourceRX1-e20-s2026092701-m2-r01/logs/<row>.stdout.log

实际模块路径、native args／resolved config／augmentation scalar 与 channel 配置由 run 保存。源码依赖固定 backend 7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b；新增代码位于 experiments/cvs_f_rxgroup20。

本地合成验证：20 epoch native parity、12 项协议负例、U 非负伪标签直通、TX 标签置换不影响生成 IQ、原 CPU／Python 流及 PA 流保持均 PASS（8448 合成 IQ call packets，0 NN／真实 IQ）。真实 CVS 合成训练检查 PASS：206 state 相同；BATCH 与原完整 L loss／全部梯度／RNG 逐位相同；RXGROUP 使68个参数梯度变化；六次 L 前向／默认头联合更新／export160 确认；10 项模型负例；2322 合成 NN／3 合成 optimizer updates。梯度变化不能替代识别提升。

一次独立限定 P0/P1 静态 review PASS，无直接发布阻断；未读真实 IQ 或 checkpoint。一次 N607 只读 preflight VERIFIED。发布只使用一次 archive SHA 本地／远端比较及一次远端 compile，smoke PASS 立即继续训练，不增加签名／receipt／额外审批门槛。

预期产物：完整20 epoch及4440 step logs、L/U物理序列、coupling trace及72cell覆盖、原门控及教师计数、七视图预测与状态、独立逐 row 指标及全曲线／分层图、真实 NN／墙钟时间、失败项和最终提交。尚无本轮成绩。

发布状态：VERIFIED，code commit 82975411f152e2be7d80ebc79b7caa850d2ee858；真实 checkpoint 合成2 NN smoke PASS，dispatcher98932／BATCH99020 GPU0／RXGROUP99095 GPU1，独立 PID／CWD／argv／GPU／首 epoch 日志增长核对通过。两组健康运行，尚无最终成绩。
