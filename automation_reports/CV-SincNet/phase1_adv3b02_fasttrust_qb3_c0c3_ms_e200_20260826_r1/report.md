# FastTrust-QB3冻结多seed复验与训练加速报告

## 当前状态

- run_id：`phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826_r1`
- 状态：`ANALYZED`
- 科学边界：Phase1 source-only；固定`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；target结果不反馈阈值、候选或重训。
- 目标：冻结上一轮C0/C3训练数学定义，以两个新增seed补足三seed证据；同时修复可观测性，并用不改变优化轨迹的工程A/B选择恢复checkpoint间隔。

## 设计追踪

| ID | 来源章节 | 要求 | 目标文件 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|---|
| `QB3-MS-01` | 设计报告第20节；上一轮终态建议1 | 冻结C0与C3，不用target结果调参；新增两个seed，与seed392002组成三seed复验 | 新matrix、新launcher、本报告 | `analyzed` | 3个seed的C0/C3均完成E200、Clean和三LEO同row评分 | C2仅有seed392002单因素证据；未把target结果反馈为新参数 |
| `QB3-OBS-01` | 设计报告第18节；上一轮报告第224–226行 | 修复首batch梯度遥测不可达条件 | `code/SSDG/train_ssdg.py`、聚焦测试 | `local_verified` | RED→GREEN；真实checkpoint无query smoke读回待远端执行 | `enumerate(...,start=1)`，遥测改为在`batch_idx==1`触发 |
| `QB3-OBS-02` | 设计报告第18节；上一轮报告第212–216行 | 首次非有限梯度在裁剪前记录首个参数名、非有限元素数和分项loss | `code/SSDG/train_ssdg.py`、聚焦测试 | `local_verified` | 人工NaN/Inf梯度RED→GREEN；源代码顺序回归验证先定位后裁剪 | 不改变有限梯度优化轨迹；异常梯度原本也不会执行optimizer step |
| `QB3-SPD-01` | 设计报告第16节；上一轮报告第228–242行 | 分段记录训练batch、基础validation、heavy source validation和checkpoint I/O耗时 | `code/SSDG/train_ssdg.py`、聚焦测试 | `local_verified` | 手算计时字段测试通过；短跑artifact读回待远端执行 | 用于定位墙钟，不改变loss或数据 |
| `QB3-SPD-02` | 上一轮报告第238–242行 | 同seed短跑A/B比较恢复checkpoint每1epoch与每5epoch，同时比较eval batch 512/1024；正式E200采用A/B胜出的非数学配置 | 技术A/B matrix与正式matrix | `profile_preregistered` | 相同训练step的2×2分段墙钟读回 | E200/U256与原始阶段边界不变 |
| `QB3-PROTO-01` | `项目.md`第4节、第4.3节 | 保持source-only、Core90 LEO_WEAK增强、Clean与三种LEO weak终评 | matrix、launcher、scorer | `local_verified` | 协议负测通过；真实checkpoint无query smoke、终态artifact待远端执行 | 不访问Phase2 support/query或target truth |
| `QB3-REL-01` | `AGENTS.md`八项最小流程 | Git提交、唯一release归档、N607预检/编译/启动核验、独立scorer | 本报告与release | `verified` | commit/OID、归档SHA、远端编译、PID/CWD/cmdline/GPU/log、最终artifact | 白名单外事项记为`REJECTED_EXTRA_GATE`且不阻断 |

## 预登记骨架

- 候选：冻结`C0=bounded confusion+NO_U_ID`与`C3=bounded confusion+H+P-set+P-cond`。
- seed：新增`713101`、`713102`；与已完成`392002`共同解释三seed结果。
- epoch/U batch：`200/256`，不降低正式训练步数。
- 预期artifact：每行`final_ssdg.pth`、结构化epoch指标、Clean及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`独立评测。
- 技术停止：仅协议/query越权、错误stage/seed/split、输出覆盖、错误checkout、确定性重复异常、无prediction闭合或run归属不清。
- 性能结论边界：启动与`RUNNING`不是性能结果；新增四行全部评测闭合并由独立scorer同row评分后才可进入`ANALYZED`。

## 本地实现与验证

- 代码：修复one-based loader下的首batch梯度遥测；在梯度裁剪前定位首个非有限梯度参数；增加训练batch、基础validation、heavy source validation、checkpoint I/O和other五段耗时。
- 启动器：原worker接受行级seed和实际`TOTAL_EPOCHS`；新通用matrix launcher显式传递冻结的`rc4_lambda_hard`及其余QB3参数，并拒绝覆盖既有run root。
- 速度剖析配置：`configs/phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826.json`。
- 聚焦验证：31项通过；FastTrust/QB3相邻完整回归：164项通过；`train_ssdg.py`语法编译通过；`git diff --check`通过。
- 唯一P0/P1审查：发现并修复“异常梯度在裁剪后定位”和“matrix未显式传递hard loss权重”两项P1；定点修复后未发现遗留P0/P1。P2及白名单外事项不阻断。
- 上一轮五行E200复核：每行均为200条epoch记录和9002行训练日志，fatal fingerprint扫描未见Traceback、OOM、Killed或RuntimeError；旧遥测字段全部未激活，与已修复的`batch_idx==0`不可达原因一致。各行存在低频非有限梯度跳步，但均完成最终artifact；因此本轮只增强定位，不据此改科学参数。

## 速度剖析矩阵预登记

- profile run_id：`phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826_r2`。
- 候选：同一冻结C3、同一seed392002、同一21epoch前缀、同一U batch256；仅做`eval_batch_size∈{512,1024}`×`recovery_checkpoint_interval∈{1,5}`的2×2工程A/B。阶段边界为17/18/19/20/21，前16epoch与正式E200的阶段定义完全一致，之后每个阶段至少进入一次；profile不用于性能判断。
- GPU：四行依次使用GPU4、GPU5、GPU6、GPU7；每GPU slot limit为1，且不超过每GPU两个训练进程的上限。
- 本地环境/CWD：`ssr-gpu`，`E:\type10-7\github_publish\CVS-RFFI-repo`。
- N607环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，不可变release根目录。
- 输入：WiSig `ManySig.pkl`和冻结`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；只使用Phase1源域角色。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826_r2`；各行日志位于其候选目录与`dispatcher_logs/`。
- 精确启动命令：`MATRIX=<release>/configs/phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826.json bash <release>/code/scripts/launch_phase1_adv3b02_fasttrust_qb3_matrix_20260826.sh`。
- 预期artifact：四行`metrics_epoch.jsonl`、`train.log`、分段耗时字段、checkpoint和源域最终评测。该profile只判断工程速度，不产生性能晋级结论。
- profile选择规则：优先选择训练数学不变且完整artifact闭合的组合；若墙钟差小于3%，保留恢复更密集的间隔1；若eval batch1024没有至少3%验证耗时收益，则正式矩阵采用512以降低显存。
- 技术停止：协议/query越权、错误seed或checkpoint、输出覆盖、错误checkout、两行重复确定性异常、无artifact闭合或run归属不清；不得因短跑准确率停止。

## Profile r1系统技术失败与处置

- r1使用2epoch但错误保留正式E200阶段边界17/41/69/161/181，四行均在训练状态初始化时以同一`ValueError: MUSE schedule boundaries must be strictly increasing`失败。
- 失败发生在第一个训练batch之前；四行状态均为`TRAIN_FAILED`，无性能结果，分类为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- r1主dispatcher已退出，GPU读回的活跃Python进程属于另一个`phase1_jmrs01_20260826`任务，不属于本run；未终止或修改任何进程。r1目录和日志完整保留，不覆盖、不删除、不原地重启。
- 修复仅新增合法的profile阶段配置和新run_id r2，不改变正式C0/C3的E200科学矩阵。r2发布前重新执行本地测试、Git提交、release归档、N607编译和干跑。

## Profile r2发布与启动证据

- 冻结代码提交：`20c2311933b123804518a3d07dc30bbe9cdd9ea0`；GitHub远端`work/cvs-active`独立OID读回一致。
- release归档：`E:\type10-7\release_artifacts\phase1_fasttrust_qb3_20c23119.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_fasttrust_qb3_20c23119.tar.gz`；按规则只比较一次本地/远端SHA，双方均为`f82e0110e7b42e9cb8ae3bef06842a1aadbb797c0fe43db371bedfee0cd24b8a`。
- 不可变release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_qb3_20c23119`；远端Python编译、两个Bash语法检查及四行完整`--dry-run`通过。
- 启动时间：N607本地时间2026-08-26 00:48；主PID3201670，launcher子PID3201672；四个训练Python分别绑定GPU4–7，cmdline均指向r2 run root和不可变release。
- 日志增长：主dispatcher日志已写入`QB3-MATRIX-RUN`，四份候选`train.log`均增长到约7KB并通过数据划分、真实checkpoint、阶段配置和telemetry初始化，跨过r1失败点。
- 资源边界：GPU0已有独立`phase1_jmrs01_20260826`进程；本run不使用GPU0且未对其进行任何操作。GPU4–7启动时各一行训练，不超过每GPU两个训练进程。
- 当前证据等级：`RUNNING`。这只证明合法启动与日志增长，不是速度结论或科学性能结果。

## 正式多seed矩阵冻结

- `REJECTED_EXTRA_GATE`：速度profile是NONBLOCKING工程测量，不得延迟正式科学矩阵发布；其结果不回写本run参数。
- matrix：`configs/phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826.json`；run_id与本报告一致。
- 四行：seed713101的C0/C3、seed713102的C0/C3。C0关闭H/P-set/P-cond，C3开启H/P-set/P-cond；其余QB3预算、bounded confusion、source角色、Core90增强和E200阶段定义与seed392002冻结行一致。
- GPU映射：`MS_S713101_C0_BC_NO_U_ID`→GPU0，`MS_S713101_C3_BC_H_PSET_PCOND`→GPU1，`MS_S713102_C0_BC_NO_U_ID`→GPU2，`MS_S713102_C3_BC_H_PSET_PCOND`→GPU3。
- 工程配置：`eval_batch_size=512`、`recovery_checkpoint_interval=1`。前者依据上一轮1024没有可见提速但显存约翻倍的真实证据，后者保留最密恢复；两者不改变loss、数据或候选定义。
- 环境/CWD：N607 Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为本次不可变release根；输入为WiSig `ManySig.pkl`及冻结Core90 checkpoint。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826_r1`，不可覆盖；日志位于每个候选目录与`dispatcher_logs/`。
- 精确命令：`MATRIX=<release>/configs/phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826.json bash <release>/code/scripts/launch_phase1_adv3b02_fasttrust_qb3_matrix_20260826.sh`。
- 预期artifact：每行200条结构化epoch记录、`final_ssdg.pth`、恢复checkpoint、Clean和三种LEO weak的独立metrics/log；四行prediction完整后才由独立scorer连接truth并作同row三seed分析。
- 技术停止：仅协议/query越权、错误stage/seed/split/checkpoint、输出覆盖、错误checkout、同一确定性异常至少两行、无prediction闭合或run归属不清。低性能不停止。
- 本地验证：新增正式matrix测试先RED后GREEN；与profile、速度测试联合18项通过；包含新增matrix测试的完整相邻回归165项通过。

## 正式矩阵发布与启动证据

- 冻结提交：`99da6a0f95afa11b99e738edfb93ad87121f2f0f`；GitHub远端`work/cvs-active`独立OID读回一致。
- release归档：`E:\type10-7\release_artifacts\phase1_fasttrust_qb3_99da6a0f.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_fasttrust_qb3_99da6a0f.tar.gz`；按规则只比较一次本地/远端SHA，双方均为`b028bcfc1145d46fec2edbe42915a1de22e0750c3d9f2add8fc91c328c5d3752`。
- 不可变release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_qb3_99da6a0f`；远端Python编译、两个Bash语法检查和正式四行`--dry-run`通过。
- 启动时间：N607本地时间2026-08-26 00:54；主PID3207460，launcher子PID3207462；四个主训练Python分别为PID3207506、3207513、3207515、3207518，并绑定GPU0–3、正式run root和不可变release。
- 启动读回：四份`train.log`均增长到约7KB，已完成源域划分、真实checkpoint和telemetry初始化；fatal fingerprint扫描未见Traceback、OOM、Killed、RuntimeError或阶段边界错误。
- 资源读回：GPU0因独立JMRS任务与本run合计两个训练进程；GPU1–3各一本run训练进程，均未超过每GPU两个训练进程。GPU4–7继续运行NONBLOCKING速度profile。
- 当前证据等级：`RUNNING`。正式四行未到`ARTIFACTS_COMPLETE`，尚无Clean/三LEO完整结果，也没有三seed晋级结论。

## 最终执行闭合

2026-08-26 11:34（N607本地时间）完成最终只读核对：四个正式候选目录均为`ARTIFACTS_COMPLETE`，每行包含连续E1–E200的200条结构化epoch记录、epoch200的`final_ssdg.pth`、Clean及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`独立评测和联合receiver×scenario明细。dispatcher输出`QB3-MATRIX-COMPLETE`，本run没有残留`train_ssdg.py`进程。

所有最终评测均满足：

- checkpoint epoch为200；
- `strict_requested=true`且`checkpoint_load_strict=true`；
- missing key、unexpected key和shape mismatch均为0；
- `fallback_used=false`；
- 每个场景评测60000条样本，覆盖5个未见接收机，每个接收机12000条；
- 全量训练日志和全部评测日志未出现Traceback、CUDA OOM、Killed或RuntimeError。

因此，本run的最高交付状态从`RUNNING`推进为`ARTIFACTS_COMPLETE`，完成本报告分析后为`ANALYZED`。这一状态证明训练和评分闭合，不自动等于方法晋级。

## 优化修改的完整说明

### 1. 伪标签机制与数值稳定性

本轮冻结复验的C3是上一轮真实实验中的扩展机制：冻结Core90 anchor与EMA teacher共同提供证据，风险校准后将U样本路由到H、P-set或R；C3另开启小权重P-conditional。H和P分别受有效质量预算及class×receiver上限约束，不允许固定比例回填。所有未获得安全身份方向的样本仍可参与合法的域与表征学习，但不能生成部署期prototype、radius、tail或proxy unknown。

最关键的稳定性修改是将身份编码器上的无界GRL反转交叉熵改为有界均匀域混淆，同时把域判别器和混淆路径固定为float32。上一轮五候选和本轮四个正式行共1800个E200 epoch均完整闭合，没有重现旧QB0/QB1在E104–E106附近的对抗损失爆炸。这里能够确认的是“当前实现下系统性数值发散被消除”；由于伪标签、域目标和工程配置同时变化，不能把最终准确率差异全部归因于单个改动。

### 2. P-set与P-conditional的因果拆分

P-set与P-conditional现在具有独立开关、权重和尾段退火。P-set只要求student概率留在安全候选集合，不强制候选集合内选出唯一类别；P-conditional才传递集合内部相对分布。P的95%集合覆盖目标与关闭的N路由彻底解耦，并固定全局APS，避免“按真类拟合、按预测类使用”的条件错配。

seed392002的五行同row分解显示：C2相对C1的LEO均值提高0.2861个百分点，但receiver×LEO floor下降0.3583个百分点；C3相对C2的LEO均值再提高0.1478个百分点，并把receiver×LEO floor提高0.5417个百分点。它说明P-set主要贡献平均鲁棒性，P-conditional在该seed中补回局部floor，但这仍是单seed单因素证据，不能越过设计报告中“P-conditional默认关闭”的source-only准入要求。

### 3. 异常诊断从“知道跳步”推进到“知道哪一层先坏”

此前异常包只知道总梯度非有限，且检查发生在梯度裁剪之后。本轮改为在裁剪前记录首个非有限参数、NaN/Inf元素数、loss分项、学习率和AMP上下文。四个正式行的第一次异常高度一致：

- 时间：E1 batch1；
- 总loss仍有限；
- 首个非有限参数：`id_backbone.sinc.low_hz_`；
- 非有限元素：24个，全部为NaN，没有正负Inf；
- C0和C3、seed713101和713102均得到同一指纹。

这一证据排除了H、P-set或P-conditional是首次异常的必要条件，因为C0没有U身份损失且同样出现该指纹。当前最可信根因已收缩到共享Sinc前端反向路径或其AMP数值尺度。保护逻辑只跳过该批次并保存恢复点，未出现非有限loss，也没有形成连续确定性故障。

### 4. 梯度遥测修复后的真实边界

旧代码使用one-based batch编号却检查`batch_idx==0`，导致遥测永远不触发。本轮改为首batch触发，四个正式行均在E1、E41、E91、E161、E181和E200记录`gradient_telemetry_active=1`，证明调度条件已经修复。

但是，H、P-set和P-conditional的分项梯度范数在所有采样点仍为0，而相同记录中的有效身份coverage非零。代码读回显示，遥测对`out_s["z_id"]`求梯度，但融合student路径的伪身份loss由另一计算图中的分类logits产生，观测张量没有绑定到实际loss图。因而本轮不能使用这些0值声称“伪身份梯度不存在”或“梯度占比已达标”。这项可观测性从“未触发”推进到“触发但绑定错误”，仍未完全闭合；下一版应对实际伪身份logits或共享可训练参数集合求分项梯度。

### 5. 训练速度与资源可观测性

每个epoch新增五段计时：训练batch、基础validation、heavy source validation、checkpoint I/O和other；同时记录U样本吞吐和峰值显存。worker接受行级seed和实际epoch数，通用launcher显式传递全部冻结QB3参数并拒绝覆盖已有run root。这些修改不改变loss、数据、训练步数或checkpoint选择。

## 速度Profile实验结果

Profile r2固定同一C3、seed392002、21epoch前缀和U batch256，仅比较`eval_batch_size∈{512,1024}`与`recovery_checkpoint_interval∈{1,5}`。四行均为`ARTIFACTS_COMPLETE`；短跑准确率受并行随机性影响，只用于工程计时，不进入科学晋级。

| 配置 | 总墙钟 | 训练batch | 基础验证 | heavy验证 | checkpoint I/O | 峰值显存 | U吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CK1/E1024 | 35分53.1秒 | 2087.81秒 | 16.83秒 | 20.33秒 | 1.80秒 | 5.834GiB | 558.78样本/秒 |
| CK1/E512 | 35分53.3秒 | 2083.77秒 | 18.78秒 | 20.93秒 | 2.04秒 | 2.959GiB | 557.19样本/秒 |
| CK5/E1024 | 35分19.5秒 | 2056.18秒 | 17.44秒 | 18.93秒 | 0.36秒 | 5.836GiB | 562.77样本/秒 |
| CK5/E512 | 35分56.7秒 | 2089.67秒 | 18.90秒 | 20.60秒 | 0.29秒 | 2.957GiB | 555.39样本/秒 |

工程结论如下：

1. CK5相对CK1在E1024下缩短1.56%，在E512下反而增加0.16%，均未达到预登记的3%门槛。即使把checkpoint写入从约2秒压到约0.3秒，其占总墙钟也不足0.1%，不能产生实质加速。因此正式E200保留每epoch恢复点。
2. E1024相对E512没有稳定的3%验证或总墙钟收益，却把峰值显存从约2.96GiB提高到约5.84GiB，增加约97%。正式矩阵采用E512，把峰值显存相对上一轮约6.08GiB降低到约3.20GiB，降幅约47%，为每GPU并发留出明显余量。
3. 训练batch占profile内计时约97%–98%，在正式E200中约98.5%。评测batch和checkpoint间隔都不是当前速度瓶颈。下一轮真正值得做的速度A/B是冻结anchor clean logits预缓存、确保EMA双弱视图单次拼接前向，以及针对student/anchor重复前向的计算图剖析；不能再用减少评测或降低恢复密度冒充训练加速。
4. 正式四行墙钟为9.42–10.63小时，没有形成比seed392002旧矩阵约9.1小时更快的端到端证据。seed713101的GPU0还与独立任务共驻，不能把其10.63小时归因于候选本身。当前已验证的速度收益是“显存和并发容量改善”，不是“单行训练时间缩短”。

## seed392002五候选同row结果

| 候选 | Clean | LEO clear | LEO low-elev | LEO rain | LEO均值 | 场景floor | receiver×LEO floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0：无U身份监督 | 84.4000 | 75.4883 | 72.9800 | 72.2083 | 73.5589 | 72.2083 | 57.2417 |
| C1：H | 84.6417 | 75.3667 | 72.7867 | 72.1450 | 73.4328 | 72.1450 | 57.4333 |
| C2：H+P-set | 84.8867 | 75.6667 | 73.0567 | 72.4333 | 73.7189 | 72.4333 | 57.0750 |
| C3：H+P-set+P-cond | **85.2017** | **75.8517** | **73.2033** | **72.5450** | **73.8667** | **72.5450** | 57.6167 |
| C4：U特征锚点 | 84.3150 | 75.3967 | 72.9050 | 72.1483 | 73.4833 | 72.1483 | **57.9167** |

这一矩阵证明C3在单seed上同时改善Clean、三类LEO均值和局部floor，但也证明“最优均值”和“最优最差单元”并不由同一机制取得：C4的receiver×LEO floor最高，却没有平均性能收益。

## 三seed冻结C0/C3实验数据

| seed | 候选 | Clean | Clean receiver floor | LEO clear | LEO low-elev | LEO rain | LEO均值 | 场景floor | receiver×LEO floor |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 392002 | C0 | 84.4000 | 71.0750 | 75.4883 | 72.9800 | 72.2083 | 73.5589 | 72.2083 | 57.2417 |
| 392002 | C3 | 85.2017 | 71.2000 | 75.8517 | 73.2033 | 72.5450 | 73.8667 | 72.5450 | 57.6167 |
| 713101 | C0 | 84.7050 | 70.8667 | 75.2900 | 72.7783 | 72.7333 | 73.6006 | 72.7333 | **58.4000** |
| 713101 | C3 | 85.7067 | **75.6917** | 75.5150 | 73.0817 | 72.9267 | 73.8411 | 72.9267 | 58.2750 |
| 713102 | C0 | 84.0067 | 70.3083 | 74.5433 | 72.6267 | 72.2100 | 73.1267 | 72.2100 | 57.7833 |
| 713102 | C3 | 84.6383 | 72.6750 | 74.9233 | 73.0217 | 72.6717 | 73.5389 | 72.6717 | **58.0250** |

### 配对差值

| seed | ΔClean | ΔLEO均值 | ΔLEO场景floor | Δreceiver×LEO floor |
|---:|---:|---:|---:|---:|
| 392002 | +0.8017 | +0.3078 | +0.3367 | +0.3750 |
| 713101 | +1.0017 | +0.2406 | +0.1933 | -0.1250 |
| 713102 | +0.6317 | +0.4122 | +0.4617 | +0.2417 |
| 三seed均值 | **+0.8117** | **+0.3202** | **+0.3306** | +0.1639 |
| seed间样本标准差 | 0.1852 | 0.0865 | 0.1343 | 0.2589 |

### 三seed均值±样本标准差

| 指标 | C0 | C3 | C3−C0配对均值 |
|---|---:|---:|---:|
| Clean | 84.3706±0.3501 | **85.1822±0.5344** | **+0.8117** |
| LEO clear | 75.1072±0.4983 | **75.4300±0.4700** | +0.3228 |
| LEO low-elev | 72.7950±0.1773 | **73.1022±0.0926** | +0.3072 |
| LEO rain | 72.3839±0.3026 | **72.7144±0.1944** | +0.3306 |
| LEO均值 | 73.4287±0.2624 | **73.7489±0.1823** | **+0.3202** |
| LEO场景floor | 72.3839±0.3026 | **72.7144±0.1944** | **+0.3306** |
| receiver×LEO floor | 57.8083±0.5796 | **57.9722±0.3323** | +0.1639 |

三seed最稳健的事实是：C3的Clean、LEO clear、LEO low-elev、LEO rain、LEO均值和LEO场景floor在3/3个seed上均高于各自C0。Clean配对增益范围为+0.632至+1.002个百分点；LEO均值配对增益范围为+0.241至+0.412个百分点。该结果把上一轮的“单seed方向信号”提升为“跨三个冻结seed重复的平均性能信号”。

极端receiver×LEO floor没有达到同样强度：seed392002和713102改善，seed713101下降0.125个百分点。其三seed平均增益只有0.164个百分点，seed间标准差0.259个百分点。正确结论是C3改善了总体和场景级鲁棒性，但没有稳定解决最差接收机单元。

## 训练稳定性和异常统计

四个正式行共800个epoch，每epoch推断为207个训练batch，共约165600个step。非有限loss跳步为0；非有限梯度跳步分别为39、39、35和38批，合计151批，占约0.0912%。所有行都完成optimizer、checkpoint和最终评测闭合。

| 候选 | 墙钟 | 峰值显存 | 非有限梯度跳步 | 跳步率 | 遥测触发epoch |
|---|---:|---:|---:|---:|---|
| seed713101 C0 | 10.63小时 | 3.199GiB | 39 | 0.0942% | 1/41/91/161/181/200 |
| seed713101 C3 | 9.97小时 | 3.203GiB | 39 | 0.0942% | 1/41/91/161/181/200 |
| seed713102 C0 | 9.42小时 | 3.201GiB | 35 | 0.0845% | 1/41/91/161/181/200 |
| seed713102 C3 | 10.02小时 | 3.205GiB | 38 | 0.0918% | 1/41/91/161/181/200 |

低频跳步不是C3特有，不能用来否定伪标签机制；但首次一致定位到Sinc频率参数后，下一版应把Sinc参数化、采样率/截止频率尺度和AMP反向作为独立技术修复对象。修复必须保持C0/C3同row，不能通过target结果调科学参数。

## 与设计报告的符合度及未闭合项

当前C3复验用于确认上一轮冻结信号，并不等同于附件设计报告定义的“默认FastTrust-QB3”。两者的关键差异如下：

| 项目 | 设计报告默认值 | 本轮冻结C3 | 判定 |
|---|---|---|---|
| H/P/R路由 | 开 | 开 | 符合 |
| 接收机分组风险和class×receiver上限 | 开 | 开 | 符合 |
| 有界域混淆 | 开 | 开 | 机制符合 |
| H/P有效质量预算 | 0.04/0.06，总上限0.10 | 0.05/0.10，总上限0.15 | 不符合最终默认值 |
| 候选集合大小 | 默认最多2 | 最多3 | 不符合最终默认值 |
| P-conditional | 默认关闭 | C3开启，系数0.02 | 属于扩展消融，不是默认候选 |
| confusion权重 | 起点0.02 | 0.08 | 不符合起始值 |
| 全U clean trust-region | 系数0.03 | 本轮特征锚点为0 | 尚未实现为本轮默认路径 |
| `V_select-as-U`盲评估 | H precision/AURC、P set coverage/size | 未生成独立truth-blind伪标签artifact和scorer结果 | 科学晋级证据缺失 |
| 实际H/P梯度占比 | 必须可观测 | 遥测触发但绑定错误 | 可观测性缺失 |

因此，不能把本轮C3三seed正收益写成“附件所定义最终QB3已验证”。本轮准确说法是：QB3扩展机制的平均性能收益得到多seed支持；设计报告推荐的默认C2形态仍需一次严格source-only、设计值对齐的同row验证。

## 科学结论与晋级判定

### 已证明

1. 有界域混淆与float32域头路径使所有C0/C1/C2/C3/C4及新增多seed行稳定完成E200，旧无界GRL崩溃未复现。
2. C3相对C0在3/3个seed上提高Clean和三种LEO场景的平均准确率；三seed配对均值为Clean+0.8117个百分点、LEO均值+0.3202个百分点、LEO场景floor+0.3306个百分点。
3. 正式E512把峰值显存从上一轮约6.08GiB降至约3.20GiB，没有减少训练step或E200阶段，显著改善并发承载能力。
4. 首次非有限梯度已定位到共享`id_backbone.sinc.low_hz_`，不再停留在“某处出现NaN”的模糊诊断。

### 尚未证明

1. receiver×LEO极端floor没有3/3seed一致改善，C3没有根治rx级最坏单元。
2. n=3只能支持重复方向和效应量描述；没有逐样本配对prediction，不能补做严格配对显著性检验。
3. 当前C3不是设计报告默认C2，且没有独立`V_select-as-U`伪标签质量scorer，因此不能晋级为正式默认方法。
4. 单行训练墙钟没有改善；当前速度成果是显存降低和瓶颈定位，不是端到端加速。
5. 作为非同row历史背景，历史R4的LEO均值/floor约74.463/60.383，仍高于当前C3三seed均值73.749/57.972；该比较不能作因果排序，但说明极端鲁棒性仍有明显距离。

### 最终判定

- 实验交付：`ANALYZED`。
- 科学信号：`MULTI_SEED_POSITIVE_SIGNAL`。
- 方法晋级：`NO_PROMOTION_TO_DEFAULT`。
- 推荐主线：保持“提高单位U梯度的可靠身份信息，而不是扩大唯一伪标签数量”；下一候选回到设计报告默认C2，即H+P-set、P-conditional关闭、H/P预算0.04/0.06、候选集合最多2、confusion起点0.02，并补齐clean trust-region。

## 下一轮最小实验建议

1. 先修复梯度遥测绑定：对实际伪身份logits或共享参数集合测量`g_H/g_Pset/g_L`，同时保留当前Sinc首异常定位。该修复只改变观测，不改变训练数学。
2. 单seed同row运行设计对齐的C0与C2；先完成`V_select-as-U`独立伪标签artifact和scorer，检查H总体/最差receiver precision、P总体/最差receiver set coverage、平均集合大小及AURC。只有source-only门槛通过才允许作target确认。
3. 速度只做一个单因素A/B：缓存冻结anchor的clean logits，保持student、EMA、batch、epoch和所有loss不变。当前计时已经证明训练batch是唯一值得优先优化的主项。
4. 若设计对齐C2通过，再做3seed确认；不得根据本轮target的receiver差值修改阈值、receiver预算或类别参数。

## 可复算证据

- 完整分析脚本：`analyze_results.py`。
- 机器可读摘要：`analysis_summary.json`。
- 输入边界：本地保存了4个profile行、4个正式行及seed392002五候选的全部`metrics_epoch.jsonl`、`metrics_epoch.csv`、训练日志、Clean/三LEO JSON和评测日志；正式报告不把大体积原始日志推入Git。
- 全量扫描范围：所有结构化epoch记录逐行解析，所有训练和评测日志全文扫描，不以tail或抽样替代最终结论。
