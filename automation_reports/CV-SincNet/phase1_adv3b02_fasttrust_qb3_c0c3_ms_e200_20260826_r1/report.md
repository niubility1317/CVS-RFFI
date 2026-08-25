# FastTrust-QB3冻结多seed复验与训练加速报告

## 当前状态

- run_id：`phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826_r1`
- 状态：`LOCAL_VERIFIED_PROFILE_PREREGISTERED`
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

- profile run_id：`phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826_r1`。
- 候选：同一冻结C3、同一seed392002、同一2epoch前缀、同一U batch256；仅做`eval_batch_size∈{512,1024}`×`recovery_checkpoint_interval∈{1,5}`的2×2工程A/B。
- GPU：四行依次使用GPU0、GPU1、GPU2、GPU3；每GPU slot limit为1，且不超过每GPU两个训练进程的上限。
- 本地环境/CWD：`ssr-gpu`，`E:\type10-7\github_publish\CVS-RFFI-repo`。
- N607环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，不可变release根目录。
- 输入：WiSig `ManySig.pkl`和冻结`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；只使用Phase1源域角色。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826_r1`；各行日志位于其候选目录与`dispatcher_logs/`。
- 精确启动命令：`MATRIX=<release>/configs/phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826.json bash <release>/code/scripts/launch_phase1_adv3b02_fasttrust_qb3_matrix_20260826.sh`。
- 预期artifact：四行`metrics_epoch.jsonl`、`train.log`、分段耗时字段、checkpoint和源域最终评测。该profile只判断工程速度，不产生性能晋级结论。
- profile选择规则：优先选择训练数学不变且完整artifact闭合的组合；若墙钟差小于3%，保留恢复更密集的间隔1；若eval batch1024没有至少3%验证耗时收益，则正式矩阵采用512以降低显存。
- 技术停止：协议/query越权、错误seed或checkpoint、输出覆盖、错误checkout、两行重复确定性异常、无artifact闭合或run归属不清；不得因短跑准确率停止。
