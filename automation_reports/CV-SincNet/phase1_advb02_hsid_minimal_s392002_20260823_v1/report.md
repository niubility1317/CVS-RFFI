# ADV3B02 CORE90 CVS-HSID最小实验报告

状态：`LOCAL_VERIFIED / P0P1_REVIEW_PASS / N607_PREFLIGHT_PASS / RELEASE_PENDING`

## 1.目标与边界

- run ID：`phase1_advb02_hsid_minimal_s392002_20260823_v1`
- 基线提交：`8c5cae87a91e567ef06c680e5f9a372ebee8e166`
- 基线checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 科学目标：把SID从160维嵌入残差升级为频率坐标正确、具有RX/day/LEO分层可辨识性、独立原型证据和Raw主导安全融合的CVS-HSID。
- 数据边界：仅使用Phase1 source侧`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；`R_s∩R_t=∅`；训练、选模和mask统计均不读取target/query。
- Core90边界：Raw CORE90完全冻结；默认LEO_WEAK场景为`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；Core90卫星CE仍从E80起按0.68权重生效。
- seed：`392002`。
- 结果声明边界：本报告在prediction与独立评分完成前只声明实现、验证、发布和运行状态，不声明性能提升。

## 2.需求追踪表

| ID | 来源章节 | 验收要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| T01 | 1.3、18/P0 | `post_stage_common.build_baseline_model()`传递`sid_max_residual_ratio`，构建后与checkpoint重建均保持配置 | `code/post_stage_common.py`、SID模型测试 | completed | 聚焦测试通过 | 仅闭合原预登记缺口，不作为主创新 |
| T02 | 3.4、7.1 | 所有局部幅相差分先在完整FFT坐标计算，再按mask和连续segment聚合 | `code/cvsrffi/spectral_identifiability.py` | completed | 非连续mask反例通过 | 禁止压缩频点后伪邻接 |
| T03 | 3.5、7.4 | 使用显式`f↔-f`索引，仅对双侧有效bin计算镜像耦合 | 同上 | completed | 精确镜像索引测试通过 | 不再使用压缩序列`flip` |
| T04 | 3.6、7.2 | 幅度身份残差扣除加权二阶平滑趋势 | 同上 | completed | 二次趋势fixture通过 | 保留局部纹波和带边残差 |
| T05 | 3.7、7.3、7.5 | 相位使用复数增量的实部/虚部；输出真实能量、fade、相干、趋势、clip、DC和SNR代理质量 | 同上 | completed | 低能量有限梯度与7维质量测试通过 | 复数谱运算强制FP32 |
| T06 | 3.1、3.2、6 | 分别估计TX×RX、TX×day、TX×LEO和纯样本残差；默认权重`2.0/1.0/1.5`；支持worst/CVaR统计和common/nonlinear/domain角色mask | 同上、`code/scripts/audit_phase1_spectral_identifiability.py` | completed | 随机效应与角色mask测试通过 | 不重复计算域方差，不主要惩罚RX公共偏移 |
| T07 | 4、8 | 新建32～64维独立SID空间与独立归一化prototype logits，不再修改Raw嵌入 | 同上、`code/model_dual_cvsincnet.py` | completed | Raw嵌入逐位不变测试通过 | 当前维度48 |
| T08 | 8.3、13.1 | 最终logit为Raw主导的有界质量门融合；保存raw/spec/fused预测、margin、gate和质量；harmed惩罚高于rescue | 模型、训练与评测路径 | completed | 融合上界、零初始化回退和margin保护测试通过 | `alpha_max=0.20` |
| T09 | 9、13、15 | SID空间加入cross-RX SupCon、receiver-CVaR与TX×RX可加性交互损失；source-only选模包含source RX floor和RX×day floor | `code/SSDG/train_ssdg.py`、`code/cvsrffi/eval.py` | completed | 损失、选模与source分组下界测试通过 | 不使用强GRL推动Raw |
| T10 | 14、16、18 | 发布单seed最小可证伪矩阵：`S0_CORE90`、`R3_SPEC_PROTO`、`X0_HIER_PROTO`、`F0_HIER_FUSION`、`X2_RX_ROBUST` | 新launcher | completed | `bash -n`、五行`--dry-run`通过 | 发布本身不扩成多seed |
| T11 | 17 | 逐样本保存`y/raw/spec/fused`预测、margin、gate、RX/day/TX/scenario、质量变量，并输出paired rescue/harm统计所需字段 | 评测/诊断路径 | completed | same-row artifact schema测试通过 | 独立scorer连接truth后才作结果判断 |
| T12 | 项目协议 | checkpoint smoke必须`query_input_count=0`、`target_input_count=0`且Raw成熟参数无漂移 | smoke与报告 | completed | 真实checkpoint无query smoke为`VERIFIED` | Phase1 source-only硬边界 |

## 3.最小实验矩阵

| row | 方法 | 目的 |
|---|---|---|
| `S0_CORE90` | 冻结ADV3B02 CORE90 | 同命令、同评测基线 |
| `R3_SPEC_PROTO` | 频率坐标修复+独立SID prototype | 判断描述数学错误与融合位置的影响 |
| `X0_HIER_PROTO` | R3+RX/day/LEO分层mask | 判断receiver-aware可辨识性是否修复跨RX退化 |
| `F0_HIER_FUSION` | X0+Raw主导质量门融合 | 判断rescued/harmed是否转为净正 |
| `X2_RX_ROBUST` | F0+cross-RX SupCon+receiver-CVaR+TX×RX交互 | 直接优化source receiver最差风险 |

所有训练行固定200epoch；低性能只触发分析，不触发技术停止。训练完成后必须保留clean和三种LEO_WEAK逐场景评测。

## 4.本地状态与验证记录

- 隔离工作树：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/advb02-ntrs-leo-weak-20260820`
- 工作分支：`codex/advb02-hsid-20260823`
- 基线已跟踪状态：clean。
- 基线测试：16项通过；仅有2条既有AMP API弃用警告。
- 变更后相关回归：`68 passed`；仅有3条既有AMP API弃用警告。
- 静态检查：8个Python入口`py_compile`通过；launcher`bash -n`和五行`--dry-run`通过；`git diff --check`通过。
- 真实checkpoint无query smoke：`E:/type10-7/local_artifacts/phase1_advb02_hsid_smoke_20260823/smoke.json`为`VERIFIED`；`batch_role=L_s`、`query_input_count=0`、`target_input_count=0`、Raw可训练参数0、HSID可训练参数14,570、`primary_raw_logit_max_abs=0.0`、输出全部有限。
- 小批量P0实跑：64次物理样本cluster bootstrap完成，`bootstrap_selection_probability`范围为0～1且不再是全1占位；common/nonlinear/domain三个role的中心DC band均为0。
- 独立P0/P1初审：0个P0、5个P1；问题为独立评测重建参数缺失、融合系数负区间死锁、分层mask未统一排除DC、bootstrap稳定度占位和launcher缺少首步smoke。五项均已定点修复。
- 唯一一次定点复审：原5项`5/5 PASS`，剩余P0/P1为无；未扩大审查范围。
- 变更文件与用途：
  - `code/cvsrffi/spectral_identifiability.py`：分层统计、正确复数谱描述、独立原型与融合。
  - `code/model_dual_cvsincnet.py`：接入独立SID证据，不改Raw嵌入。
  - `code/post_stage_common.py`：闭合checkpoint模型重建参数传递。
  - `code/SSDG/train_ssdg.py`：HSID损失与receiver-aware source选模。
  - `code/scripts/audit_phase1_spectral_identifiability.py`：生成分层角色mask。
  - `code/scripts/launch_phase1_advb02_hsid_20260823.sh`：不可覆盖矩阵launcher。
  - `code/cvsrffi/eval.py`、`tools/eval_cvs_checkpoint_sat_channel.py`、`code/eval_feature_diagnosis.py`：source receiver下界与same-row HSID prediction artifact。
  - 聚焦测试与真实checkpoint smoke：证明可达性、source-only和Raw冻结。

## 5.发布预登记

- 实现提交：`445a966b2f53fadcc9a807c625a776d295e93590`；已推送到`origin/codex/advb02-hsid-20260823`并独立核对远端OID一致。
- 环境：本地`ssr-gpu`；远端Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- CWD：本地为上述隔离工作树；远端为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_hsid_minimal_s392002_20260823_v1-<release-commit8>`。
- 输入：上述ADV3B02 CORE90 checkpoint与Phase1 source划分。
- 输出：`runs/phase1_advb02_hsid_minimal_s392002_20260823_v1/`。
- 日志：`logs/phase1_advb02_hsid_minimal_s392002_20260823_v1/`。
- GPU：GPU0/GPU1。2026-08-23 02:55 CST直接N607只读preflight通过；两卡当时各有1个既有训练进程，本实验每卡再发布1个并保持上限2。其余既有进程不干预。
- 技术停止规则：仅在协议/query泄漏、错误checkpoint/checkout/row、输出冲突、无prediction闭合、确定性重复异常、OOM/NaN或进程归属不清时停止精确run-owned进程树；不得因低性能停止。
- 预期artifact：分层mask与统计、每row checkpoint和训练日志、clean与三种LEO_WEAK结果、逐样本raw/spec/fused prediction、paired rescue/harm字段、terminal status。
- 资源与路径：项目盘剩余7.3TB；ManySig、基线checkpoint和远端Python均存在；本run的`runs/`、`logs/`和release目标均不存在。
- P0准备命令：`cd <release> && RUN_ID=phase1_advb02_hsid_minimal_s392002_20260823_v1 GPU_0=0 GPU_1=1 bash code/scripts/launch_phase1_advb02_hsid_20260823.sh --prepare-p0 --only=R3,X0,F0,X2`。
- 正式启动命令：`cd <release> && nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet RUN_ID=phase1_advb02_hsid_minimal_s392002_20260823_v1 GPU_0=0 GPU_1=1 MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_advb02_hsid_20260823.sh --only=S0,R3,X0,F0,X2 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_hsid_minimal_s392002_20260823_v1/driver.out 2>&1 < /dev/null &`。
- release同步：对一个Git release归档执行一次本地/远端SHA-256比较，远端解压后只执行一次Python编译和launcher语法检查；不增加成员hash、seal或receipt。

## 6.额外gate处理

除项目八项白名单外不增加任何审核、seal、receipt或逐文件哈希。若旧文件要求额外gate，记录`REJECTED_EXTRA_GATE`并继续最小流程。
