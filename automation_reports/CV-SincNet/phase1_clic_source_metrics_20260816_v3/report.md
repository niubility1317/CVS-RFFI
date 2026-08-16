# Phase1CLIC源域指标补全v3预注册报告

## 状态与边界

- 实验ID：`phase1_clic_source_metrics_20260816_v3`。
- 当前状态：`LOCAL_VERIFIED / INDEPENDENT_REVIEW_ALLOW / P0=0 / P1=0 / SMOKE_INVOCATION=0 / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT`。
- 目的：仅修复v2真实技术失败中的held source-V日轴合同；正式数据为`4TX×7RX×2day×300=16800`，每个`TX/RX`单元仍为600条。保留原有三LEO弱场景单观测、场景/类/RX/day正分母、C/G共享cache、zero-fit、source-only与非选择性审计边界。
- 禁止：不重试、恢复、覆盖或重标v2；不读取target/query/truth/性能；不修改训练、checkpoint、阈值、channel、seed、scene、metrics或正式矩阵。

## v2封存事实与v3差异

| 项目 | v2封存事实 | v3冻结处理 |
|---|---|---|
| v2结构smoke | 唯一smoke在cache阶段以`CLICSourceVLeoCacheError: C clean-v4 V day axis drifted`自然退出 | 仅新run ID，不恢复v2 |
| v2formal | `FORMAL_INVOCATION=0`，formal根/日志/outer均未启动 | smoke技术闭合后才允许唯一formal入口 |
| v2性能 | `NO_PERFORMANCE_RESULT`，未读取性能 | v3技术阶段同样不读取性能 |
| 根因 | builder把held V误设为3日，真实clean-v4 V物理日轴固定为`2021_03_01/2021_03_08` | 冻结该精确日集合和`FROZEN_SOURCE_DAY_COUNT=2`；仍要求16800总行、7RX、每TX/RX600、每TX/RX/day300 |
| 重试 | `retry=NO` | `retry=NO`；v3若技术失败只能再申请全新非覆盖run ID |

## 冻结实现合同

| 项目 | 冻结值 |
|---|---|
| cache run ID | `phase1_clic_source_metrics_20260816_v3` |
| formal运行根 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_metrics_20260816_v3` |
| formal日志根 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_source_metrics_20260816_v3` |
| F1smoke根 | `/home/szu2070436088/2510044040/CV-SincNet/runs/.smoke_phase1_clic_source_metrics_20260816_v3_F1` |
| F1smoke日志根 | `/home/szu2070436088/2510044040/CV-SincNet/logs/.smoke_phase1_clic_source_metrics_20260816_v3_F1` |
| 原始输入 | `runs/phase1_clic12_20260812_v5/F1{C,G}_CLIC12`训练/terminal、`runs/phase1_clic_postfreeze_20260812_v4/F1{C,G}_CLIC12/source_clean_proxy.npz`、PAIR-v3和ManySig；禁止镜像、hardlink替代路径或target/query输入 |
| root安全 | smoke保持冻结canonical project root；formal/smoke根使用exact`mkdir`认领；PID/log以`noclobber`独占；碰撞退出3且保留攻击者marker |
| smoke | 仅F1，共享cache后按`F1C→F1G`串行forward；无scorer、无性能读取、无formal根创建 |
| formal | `6cache→12forward→6pair score→1aggregate`；dry-run精确25行 |
| GPU | formal cache/forward固定GPU0..5且每GPU最多两forward；score/aggregate仅CPU；F1smoke仅GPU0、C/G串行 |

## 本地TDD与发布门

- RED行为：真实`2021_03_01/2021_03_08`且`2day×300`held-V表必须通过；旧`3day×200`即使仍为16800行和每TX/RX600也必须拒绝；任意两个替代日期、单TX/RX单元599/1、缺日/多日、manifest day-axis漂移与C/G physical binding漂移必须在输出前拒绝。
- GREEN最小修复：只改cache run identity、冻结真实2日集合、`FROZEN_SOURCE_DAY_COUNT=2`和每TX/RX/day300验证、相应错误文案和v3脚本/测试身份；不改channel、seed、scene、metrics、数据字节或训练输入。
- 实现席串行验证：单一`ssr-gpu`wrapper完成相关`py_compile`并执行3个测试文件，结果`68 collected / 68 passed / 0 failed`、exit0；完成后`conda.exe/python.exe/pythonw.exe`均为0。
- 独立复审：静态与无需Conda的`bash -n`、formal/smoke dry-run=`25/3`、tracked/untracked diff-check均通过，结论`P0=0/P1=0/ALLOW`。复审席另一次pytest调用虽显示68个进度点和`[100%]`，但因session/exit回执丢失严格记为`TEST_COMPLETION_RECEIPT=UNKNOWN`，不作为通过证据；发布门采用实现席可审计exit0回执。
- N607交接前必须具备：主控授权的串行`ssr-gpu`RED→GREEN、受影响回归、`py_compile`、CLI`--help`、两份`bash -n`、formal/smoke dry-run=`25/3`、`git diff --check`、独立`P0=0/P1=0`复审、Git提交和不可覆盖archive。

## 预注册运行与停止规则

- 唯一runner在本地门、版本化、报告更新和远端预检均闭合后，先执行一次F1 smoke；只有完整技术artifact、cache/feature reopen、hash/physical/scene/source-only/zero-fit检查闭合后，才执行一次formal。
- 只因协议/访问/哈希/覆盖/错误checkout、确定性技术异常或至少两不同row同一pre-receipt异常停止；绝不因准确率、AUROC、floor或任何性能值停止。
- 预期formal artifact：6个shared cache+receipt、12个feature+binding、6个pair metrics、1个aggregate。技术完成不等于任何七门通过或候选晋级。
- 任何N607落地、启动、监控、artifact读取与报告终态由唯一runner和主控完成；本预注册文件不声明已发布、已运行或存在性能结果。

## Runner发布与技术终态（2026-08-16）

- 本run唯一runner按主控冻结commit`a98819f5fa1252d7d7864b0003dc5d613c70b825`执行；SCP恰1次。远端归档为`35865177`bytes，SHA256=`eac3575dffb3d9045b3eb3dda6be9bc42e898959b2d7b19604d133184134e29b`，保留在`releases/.phase1_clic_source_metrics_20260816_v3_a98819f5.tar.gz`。
- Windows`core.autocrlf=true`造成归档中Python/report物理CRLF；只读审计确认stage与归档5112项清单完全一致、无外来文件、无lone CR，CRLF归一化后六个冻结SHA全部闭合。仅两份Bash launcher在exact stage内用remote Python临时文件原子替换并保留模式，随后atomic rename至`releases/phase1_clic_source_metrics_20260816_v3_a98819f5`；未二次SCP、未删除tar/stage证据。

| 文件 | 远端release物理SHA256 | CRLF归一化SHA256 | 冻结SHA匹配 |
|---|---|---|---|
| `build_phase1_clic_source_v_leo_iq.py` | `db6908a967740f82ae535470af6dfec1417d8a480a81d4e004b57d78ab0ce6bd` | `ad1ce4bcded8a0c87bcc30a171c52b4415725eb1cb1d639cfe152372425a9897` | 是 |
| `export_phase1_clic_source_v_leo_features.py` | `4e7b375a5ad6ddeb494e6dc1c62606383de42edda9e4ec056cee952a3ce67cbe` | `84d4e12f154f36406b798f2e2d10be732e6d1301e211b8afa5076d2e6b10036e` | 是 |
| `evaluate_phase1_clic_source_metrics.py` | `d60ba165a511be8e881e8c8f0f2835a037a0cf7c199fbcd88e30e654fff464f7` | `e4eb7ca70e2d0dc1656a733f92359e21d41e4e60d52deb554ff3843e1683c3f9` | 是 |
| `launch_phase1_clic_source_metrics12_v3_20260816.sh` | `5146c48202ba5ec2141dfe20f6c1b33ce65f40588e61a16e80f76260c5710d8e` | 同左 | 是 |
| `smoke_phase1_clic_source_metrics_f1_v3_20260816.sh` | `32febf2aba43d5ae30fa6f8203d96f8686972020981bf6c519e0c272ecf2f809` | 同左 | 是 |
| `report.md` | `bbdc00edee3a4119e0499bb74d9dd3d98ceb6fd5a784f0f30eaeebd6f76a656e` | `8ea2ea81712888cfcb4a066029ade0bd33e6da8e7d77e9827a7dc244a9318f79` | 是 |

- 远端静态门：exact`CVS-RFFI/bin/python`临时cfile`py_compile`通过；三入口`--help`通过；两launcher`bash -n`通过；formal/smoke dry-run分别为25/3行；ManySig SHA、12组训练/terminal/clean与6组PAIR-v3输入存在且未读target/query/truth/性能；formal/smoke运行根、日志根和outer在启动前均为空。

## F1 smoke技术终态

- 唯一命令：`cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_source_metrics_20260816_v3_a98819f5/code/scripts/smoke_phase1_clic_source_metrics_f1_v3_20260816.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_metrics_20260816_v3_smoke_outer.out 2>&1 &`。
- `SMOKE_INVOCATION=1`，outer PID=`871012`，启动时间=`2026-08-16T17:29:14+08:00`，GPU=`0`，`RETRY=NO`。cache完成后，F1C forward在产生feature前确定性失败：`CLICSourceVFeatureExportError: PAIR-v3 proxy diagnostic fit_rows must remain zero`。该错误触发技术停止规则，未启动F1G、scorer或formal。

| partial artifact | bytes | SHA256 |
|---|---:|---|
| `F1_SHARED/source_validation_known_leo_weak.npz` | 41127936 | `fdc7d7cc00df80028fba432499658ee86ba22bcf993bcda83a7dd6bd1adb1a5c` |
| `F1_SHARED/source_validation_known_leo_weak.receipt.json` | 13739 | `cc698d6fdb3d982a615d1669c4c234d5e4b7cb560a904f37c70f5329ed69a5d4` |
| `F1_source_v_cache.out` | 574 | `0d76bdd398b9387a35fbef05292226559026815881b8b45caaaf3301218e24d6` |
| `F1C_CLIC12_source_v_forward.out` | 1140 | `f68460708e91b2505f90bfaad88f80d1f0b421b68933b95b0a0c0eeeb0a7bafd` |
| smoke outer | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

- cache receipt结构审计：`source_validation_row_count=16800`；日期为`2021_03_01/2021_03_08`；3个LEO场景各5600行；7RX各800行；每个TX/RX/day各300行；`single_leo_observation_per_physical_sample=true`、跨场景物理ID不复用、C/G共享received-IQ字节；`source_v_only=true`；`fit_rows=0`、`threshold_fit_rows=0`、`source_l_forward_rows=0`、`source_v_forward_rows=0`、`proxy_forward_rows=0`；`clean_source_runtime_access=false`、`target_access=false`、`query_access=false`、`selection_access=false`、`retry_access=false`。
- F1C/F1G feature与binding、formal cache/receipt、pair metrics和aggregate均未生成。smoke进程已退出，GPU0回到`1MiB/0%`，本地SSH客户端和N607 TCP22连接均清零；partial run/log保留且未覆盖。

## 最终结论

- `SMOKE_STOPPED_TECHNICAL_FAILURE / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT / RETRY=NO`。
- 所有性能字段均为`N/A`/未读取；本run不支持任何性能、候选晋级或Phase3声明。该PAIR-v3输入/策略故障需由主控另行完成本地修复、验证、独立复审并申请新的非覆盖run ID；本runner不重试当前run。
