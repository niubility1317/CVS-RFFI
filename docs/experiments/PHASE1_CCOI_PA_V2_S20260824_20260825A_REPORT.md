# PHASE1_CCOI_PA_V2_S20260824_20260825A实验报告

## 预登记

- 状态：`ANALYZED`；工程闭环`VERIFIED`，科学结论`SCIENTIFIC_FAILURE_NO_PROMOTION`。
- 候选：`CCOI-PA-V2`，单seed最小矩阵`C0/C1/C2/C3/C4`。
- 科学对照：冻结`ADV3B02_CORE90_SOFT_E200`；沿用V1的同split、同seed、同训练/评估预算和四场景，C1–C4保持同容量。
- 修复范围：原始`meta.rx_i`接收机导出；算子独立分类与源域`V_cal`尺度对齐凸融合；有界码本有效数/集中度正则。Core90、source roles、场景和目标/query边界不变。
- Git实现提交：`8a959d00da768d1134ce859bd366052f4ea9c109`，分支`codex/phase1-ccoi-pa-v1-20260824`，远端OID已独立核对一致。
- 主要文件：`code/train_phase1_ccoi_pa.py`、`code/score_phase1_ccoi_pa.py`、`code/cvsrffi/ccoi_pa.py`、V2 launcher/config、聚焦测试及V2设计/追踪报告。
- 本地验证：`ssr-gpu`中35项CCOI聚焦测试通过；三个生产Python文件语法编译通过；C0–C4 dry-run通过；一次定点P0/P1检查闭合。
- 本机Git Bash：既有路由证据为`FAILED`，未在错误Bash通道执行launcher；发布后在N607运行`bash -n`。
- 源域协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，`rho_label≤0.1`；目标域、query、query role和query truth不进入训练、校准或选择。
- 场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别输出。
- seed：训练和卫星扰动均固定为`20260824`，用于与V1同row比较。
- N607环境/CWD：普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；release目录`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00`。
- 输入checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 输入数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- GPU：`0`；2026-08-25 01:05直连预检显示8张RTX 3090利用率均为0、显存占用1MiB，无compute app和`train_phase1_ccoi_pa.py`进程。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A`；smoke使用同名`_REAL_CKPT_NO_QUERY_SMOKE`独立不可覆盖根。两者预检均不存在。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A.out`；launcher监督日志使用同run ID独立文件。
- release归档：`E:\type10-7\release_archives\phase1_ccoi_pa_v2_8a959d00.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00.tar.gz`；本地与远端SHA256均为`976bfe2919f4632e5b5b277b915ec418c7754866c9d2a058859439429eab5628`，传输状态`VERIFIED`。
- 精确命令：`cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00 && ROOT=$PWD CHECKPOINT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_20260825 RUN_ID=PHASE1_CCOI_PA_V2_S20260824_20260825A GPU=0 bash code/scripts/launch_phase1_ccoi_pa_v2_20260825.sh`。
- 预期artifact：`protocol_and_smoke.json`、挑战预训练历史、每row校准、sidecar、challenge audit、`prediction.jsonl`、后置`truth.jsonl`、`metrics.json`、matrix manifest和完整日志。
- 直接技术停止规则：仅在协议/query泄漏、错误checkout/CWD/run-root/GPU、输出碰撞、无法启动、prediction无法闭合，或至少两个row出现相同确定性预prediction异常时停止；不因低准确率停止。只处理该run绑定的进程树并保留全部局部artifact。
- 科学门槛：C2或更高row相对C1的LEO均值和receiver-floor分别至少提升0.30个百分点；clean下降不超过0.50个百分点；C4 holdout NMSE相对C1下降至少5%且R²大于0。未过线记负结果，不中止健康运行。
- 新run授权：本报告仅授权唯一run ID `PHASE1_CCOI_PA_V2_S20260824_20260825A`；不得重复启动或覆盖旧run。

## 运行更新

- 2026-08-25 01:07：release归档已同步；远端SHA256读回为`976bfe2919f4632e5b5b277b915ec418c7754866c9d2a058859439429eab5628`，与本地一致，传输状态`VERIFIED`。
- release目录已新建且未覆盖旧目录；三个V2生产Python文件远端编译通过，三个对应`.pyc`均完成独立读回；launcher远端`bash -n`通过。
- 启动前资源再次确认：无`train_phase1_ccoi_pa.py`进程、无NVIDIA compute app，目标run和smoke根均不存在。
- 2026-08-25 01:09：唯一launcher PID`2500917`已启动，PPID为1，CWD为release目录；正式训练PID`2501324`，完整cmdline、run-root、seed、GPU0和日志路径均与预登记一致。
- smoke已先行通过并由`protocol_and_smoke.json`独立读回：`Phase1_source_only`、source roles为5,880/52,920/12,600/12,600，比例`0.07/0.63/0.15/0.15`，`rho_label=0.1`；源/目标receiver交集为0；checkpoint严格加载`missing/unexpected/mismatch=0/0/0`，195个state tensor；PA图`[64,64,64]`、logits`[64,6]`且有限；`target_or_query_access=false`。
- launcher日志已依次出现`REAL CHECKPOINT NO-QUERY SMOKE`、`[CCOI-SMOKE] PASS`、`[CCOI-V2-SMOKE] PASS`和`FULL MATRIX`，正式矩阵已健康进入训练。
- 启动后GPU0出现另一条先于本run启动的无关meta-adapter训练PID`2498587`，占约486MiB；本run占约620MiB，两条训练进程未超过每GPU允许上限。该无关进程不属于本run，不做任何干预。

## 完成状态与完整性

- 2026-08-25 02:22：唯一launcher自然退出；监督日志依次出现`[CCOI-PREDICTIONS] COMPLETE`、`[CCOI-SCORE] ANALYZED`和`[CCOI-V2-LAUNCH] ANALYZED`，没有重复启动、强制停止或输出覆盖。
- C0–C4每行均有1,632,000条prediction和1,632,000条后置truth；远端`wc -l`独立读回共16,320,000行。五个`metrics.json`均为`ANALYZED`，且`truth_joined_after_prediction=true`。
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`四场景全部闭合；receiver键为真实`0–11`，不再出现V1的`-1`聚合，receiver导出修复得到真实实验验证。
- 主日志与监督日志未发现Traceback、OOM、NaN/Inf、exception、killed、invalid receiver、target/query越权等异常指纹。运行健康性结论为`VERIFIED`。

## 同row结果

所有数值均为百分比；`LEO-floor均值`是三个LEO场景receiver-floor的算术均值。

| row | clean | clear | low-elev | rain | LEO均值 | LEO-floor均值 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 90.1402 | 78.4564 | 75.5686 | 75.1892 | 76.4047 | 56.8639 |
| C1 | 90.1235 | 78.4456 | 75.5338 | 75.1770 | 76.3855 | 56.8736 |
| C2 | 90.1255 | 78.4544 | 75.5490 | 75.1804 | 76.3946 | 56.9097 |
| C3 | 90.1201 | 78.4397 | 75.5422 | 75.1804 | 76.3874 | 56.8722 |
| C4 | 90.1250 | 78.4583 | 75.5431 | 75.1833 | 76.3949 | 56.8986 |

相对C1：C2的clean、LEO均值、LEO-floor均值分别变化`+0.0020`、`+0.0092`、`+0.0361`个百分点；C3分别为`-0.0034`、`+0.0020`、`-0.0014`个百分点；C4分别为`+0.0015`、`+0.0095`、`+0.0250`个百分点。clean保护全部通过，但LEO均值和receiver-floor的`+0.30`个百分点门槛全部未通过。

## 机制诊断

- q预训练10个epoch完整结束；总损失由`0.4697`降至`0.3113`，masked/temporal/variance损失分别降至约`0.00566/0.00553/0.00876`，未出现数值异常。
- V2软码本约束表面达标：soft effective codes为`35.216/48`，最大soft均值概率为`0.0994`。但硬argmax只使用`4/48`个码，计数为`30/11/8886/3673`，最大单码占`70.52%`。这说明当前正则只摊平了软概率，没有解决离散挑战码塌缩；该诊断比仅看soft统计更可信。
- C4真实holdout NMSE为`0.12593`，相对C1的`1.67857`下降`92.50%`，且按`R²=1-NMSE`得到`0.87407`，通过预登记的机制拟合门槛。但C4真实配对相对自身shuffle对照仅改善`2.815%`（shuffle NMSE=`0.12957`），表明可辨认的样本特异性条件信息仍然偏弱。
- C1–C4的`V_cal`自动选择均为`alpha=0.1`，融合尺度约`1.416–1.502`；尺度修复使sidecar不再因量纲失配而数值失活，但最终分类增益仍只有约`0.01`个百分点，因此“融合失活”不是唯一瓶颈。

## 科学判定与后续路线

- 工程判定：`VERIFIED`。V2真实checkpoint smoke、receiver修复、源域尺度校准、四场景prediction/truth和独立评分均完整闭合。
- 科学判定：`SCIENTIFIC_FAILURE_NO_PROMOTION`。虽然clean约束和C4 holdout拟合门槛通过，但C2/C3/C4对C1的LEO均值与receiver-floor增益均远低于`+0.30`个百分点，不晋级多seed或完整确认。
- 否定的解释：不能再把失败归因于receiver缺失或单纯融合尺度过小；两项已修复且真实验证。当前数据更支持“条件码离散塌缩、条件信息对分类决策增量极弱”。
- 下一候选必须直接约束锐化后的近硬分配并报告hard occupancy，例如对低温softmax分布施加有效码数/最大占用约束，同时保持有界目标，且先做单seed关键row最小可证伪实验。不得直接扩大当前V2矩阵或把软码本统计当作条件系统辨识成功证据。
