# SOFESA Stage2-B轻型快速目标域适应实验记录

- run ID：`sofesa_stage2b_c123_t5t25_s713101_20260824`
- 当前状态：`LOCAL_VERIFIED_PENDING_COMMIT`
- 硬停止：香港时间2026-08-24 05:00；到点后不再启动、派发、扩展或切换新工作，不终止届时仍正常运行的N607实验。
- 基线checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- Git分支：`work/cvs-active`
- Git commit：`PENDING`

## 方法

SOFESA（Support-Only Frozen-head Encoder Sparse Adaptation）在冻结的ADV3B02编码器内部选择极少量非分类参数，用合法target support标签计算原checkpoint分类头的交叉熵，并对support特征相对冻结基座的漂移和参数漂移施加约束。原分类头只提供固定判决，不训练、不替换，也不新增协方差、LDA或持久分类头。梯度只流入预登记的原编码器参数；适配结束后冻结全部参数，query逐样本只读推理且不得更新状态。

训练目标为：

`L = L_support_CE + 0.05 * L_feature_anchor + 1e-4 * L_parameter_anchor`

候选按以下顺序推进。只有前一候选完成同row评分且未达到门槛，才允许切换后一候选。

| 候选 | 精确训练面 | 预期训练参数 | 占ADV3B02总参数 |
|---|---|---:|---:|
| C1 `c1_norm_affine` | `t1/t2/t3/f1/f2/f3/pa_b1/pa_b2/pa_b3`的Norm affine | 1,040 | 0.0991% |
| C2 `c2_norm_gates` | C1加`freq_gate.conv`与`pa_gate.net` | 1,205 | 0.1148% |
| C3 `c3_norm_gates_fproj` | C2加`f_proj` | 6,485 | 0.6178% |

三者均冻结`cls_head/classifier/adv_head/dom_head/identity_capacity/sat_anchor_identity_adapter`等分类、域判决和外挂参数。

## 协议与输入边界

正式实验只接受匹配的`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id`和`split_id`。适配阶段只打开固定LEO received IQ与合法support标签；不读取地面source/clean样本、source cache、query真值、query角色或query反馈。适配API没有query参数，query推理API没有truth/role参数。固定数据不因候选或超参数变化而重新验证。

本地D18包只用于真实checkpoint无query工程smoke；它不构成当前正式同row性能证据。正式矩阵必须绑定当前VALIDATED_ONCE row的原始received-IQ包。

## 最小预登记矩阵

- seed：`713101`
- 状态比较：同一冻结输入和判决规则下的`DA0_REG0`与`DA1_REG0`
- 首轮行：单seed最小`Target5`、`Target25`
- 首候选：C1
- 更新预算：20步，Adam，学习率`1e-3`，梯度裁剪`1.0`；硬上限40步
- 资源预算：训练参数≤总参数1%
- 晋级：`DA1_REG0-DA0_REG0`旧类均值≥+1.0pp且旧类floor≥+0.5pp，并且无协议泄漏、预算合规
- 未达标：记录`SCIENTIFIC_FAILURE_NO_PROMOTION`及同row指标，再推进下一候选
- query：适配状态冻结后独立推理，不参与选层、早停、回滚或状态更新

## 环境、路径与预期artifact

- 本地CWD：`E:\\type10-7\\github_publish\\CVS-RFFI-repo`
- 本地Python：`C:\\Users\\lh594\\.conda\\envs\\ssr-gpu\\python.exe`
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- N607 checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 正式输入包：`PENDING_STRICT_SAME_ROW_BINDING`
- GPU：`PENDING_PRELAUNCH_OCCUPANCY_READBACK`
- 不可覆盖输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/sofesa_stage2b_c123_t5t25_s713101_20260824`
- 预期artifact：稀疏适配state、audit、truth-blind prediction、scorer结果和本报告增量
- 正式命令：在严格同row输入路径与GPU回读后写入，不使用占位路径启动

系统技术失败停止条件仅限：协议/权限越界、错误row或split、输出覆盖、错误checkout、无法产生prediction、scorer连接错误，或至少两行出现相同的确定性pre-prediction异常。不得因中间性能低而停止。

## 当前证据

- RED：聚焦测试最初因核心模块缺失失败。
- GREEN（修正候选白名单前）：核心行为3项和enrollment加载2项分别通过。
- GREEN：审查定点修复后的聚焦测试10/10通过。
- 邻近回归：本次相关的support adapter与predictor bundle两文件fresh运行38/38通过。此前三文件合并运行47项中45项通过，另2项失败均来自未改动的旧`adv3b02_supervised_da_runner`测试fixture缺少当前Phase2 contract字段。
- 真实checkpoint无query smoke：旧D18包曾完成精确C1工程反传（1,040/1,049,665，0.09908%，1步，分类头变化0），但独立审查发现该包manifest未绑定四个VALIDATED_ONCE句柄，因此该证据降级为`INVALID_FOR_PROTOCOL_CLOSURE`；修复后的合规smoke为`PENDING_MATCHED_MANIFEST`。
- 一次P0/P1审查：初审发现P0×1、P1×2；三项已定点修复，10/10聚焦测试通过，定点复核确认三项全部`CLOSED`且未新增审查项。
- Git commit/push/远端OID：`PENDING`
- 正式Target5/Target25矩阵：`BLOCKED_MISSING_MATCHED_MANIFEST`。本地所有`enrollment_only` manifest均不含`phase2_data_status`；N607已知support-only NPZ manifest同样缺少四个句柄，修复后的runner按设计拒绝该输入。
- 性能结论：`UNKNOWN`
- 语法验证：核心模块与runner执行`py_compile`，退出码0。

## 协议异常

2026-08-24路径定位子任务在对M23运行根执行定点搜索时误命中已连接truth的score文件。子任务没有向主流程回传任何指标或类别信息，C1→C2→C3候选及超参数在该事件前已经冻结；但该访问仍违反路径定位任务的truth-blind约束。主流程已立即中止该子任务，不使用其后续发现，也不把该M23行作为“无协议泄漏”的正式证据。正式矩阵必须改用未被该子任务触及且能严格绑定的VALIDATED_ONCE原始IQ行，否则保持`FAILED_PROTOCOL_BOUNDARY/UNKNOWN_PERFORMANCE`。

03:15重新执行N607直连预检并VERIFIED：普通账号、项目根和8张RTX3090可见；GPU2有既有负载，GPU0/1/3–7空闲。本目标尚未同步release、尚未启动任何N607训练或预测进程。
