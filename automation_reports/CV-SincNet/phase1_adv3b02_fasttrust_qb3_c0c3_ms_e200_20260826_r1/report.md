# FastTrust-QB3冻结多seed复验与训练加速报告

## 当前状态

- run_id：`phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826_r1`
- 状态：`FORMAL_RUNNING / PROFILE_R2_RUNNING`
- 科学边界：Phase1 source-only；固定`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；target结果不反馈阈值、候选或重训。
- 目标：冻结上一轮C0/C3训练数学定义，以两个新增seed补足三seed证据；同时修复可观测性，并用不改变优化轨迹的工程A/B选择恢复checkpoint间隔。

## 设计追踪

| ID | 来源章节 | 要求 | 目标文件 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|---|
| `QB3-MS-01` | 设计报告第20节；上一轮终态建议1 | 冻结C0与C3，不用target结果调参；新增两个seed，与seed392002组成三seed复验 | 新matrix、新launcher、本报告 | `pending` | matrix语义测试、dry-run、最终同row评分 | C2仅在资源允许时作为单因素确认，不作为主矩阵必需行 |
| `QB3-OBS-01` | 设计报告第18节；上一轮报告第224–226行 | 修复首batch梯度遥测不可达条件 | `code/SSDG/train_ssdg.py`、聚焦测试 | `local_verified` | RED→GREEN；真实checkpoint无query smoke读回待远端执行 | `enumerate(...,start=1)`，遥测改为在`batch_idx==1`触发 |
| `QB3-OBS-02` | 设计报告第18节；上一轮报告第212–216行 | 首次非有限梯度在裁剪前记录首个参数名、非有限元素数和分项loss | `code/SSDG/train_ssdg.py`、聚焦测试 | `local_verified` | 人工NaN/Inf梯度RED→GREEN；源代码顺序回归验证先定位后裁剪 | 不改变有限梯度优化轨迹；异常梯度原本也不会执行optimizer step |
| `QB3-SPD-01` | 设计报告第16节；上一轮报告第228–242行 | 分段记录训练batch、基础validation、heavy source validation和checkpoint I/O耗时 | `code/SSDG/train_ssdg.py`、聚焦测试 | `local_verified` | 手算计时字段测试通过；短跑artifact读回待远端执行 | 用于定位墙钟，不改变loss或数据 |
| `QB3-SPD-02` | 上一轮报告第238–242行 | 同seed短跑A/B比较恢复checkpoint每1epoch与每5epoch，同时比较eval batch 512/1024；正式E200采用A/B胜出的非数学配置 | 技术A/B matrix与正式matrix | `profile_preregistered` | 相同训练step的2×2分段墙钟读回 | E200/U256与原始阶段边界不变 |
| `QB3-PROTO-01` | `项目.md`第4节、第4.3节 | 保持source-only、Core90 LEO_WEAK增强、Clean与三种LEO weak终评 | matrix、launcher、scorer | `local_verified` | 协议负测通过；真实checkpoint无query smoke、终态artifact待远端执行 | 不访问Phase2 support/query或target truth |
| `QB3-REL-01` | `AGENTS.md`八项最小流程 | Git提交、唯一release归档、N607预检/编译/启动核验、独立scorer | 本报告与release | `pending` | commit/OID、归档SHA、远端编译、PID/CWD/cmdline/GPU/log | 白名单外事项记为`REJECTED_EXTRA_GATE`且不阻断 |

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
