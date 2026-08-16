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
