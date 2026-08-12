# Phase1 CLIC后冻结source clean v4预注册报告

## 状态与唯一修复

- 实验ID：`phase1_clic_postfreeze_20260812_v4`；当前`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`。
- v3已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`：12/12臂在任何NPZ前同一`source receiver aggregate drifts from split receipt`，0工件。
- 根因：source split receipt封存WiSig RX/day轴索引，导出行封存物理RX/day标签；v3错误地直接比较两个不同表示空间。
- v4唯一修复：严格验证receipt索引非空、规范整数、唯一、与重建`rx_keep/day_keep`集合相同且不越界；再通过同一WiSig轴解析物理标签，并与全部source-L/V导出行观测集合精确比较。G bundle使用解析后的物理标签做配置等价，真实v5无`split_info`时禁止旧格式回退。
- checkpoint、模型、ManySig、TX角色、split、fixed400、12臂矩阵、GPU映射和seed均不变。v4是新run，不覆盖/恢复v3。

## 冻结输入与运行合同

- 训练根`runs/phase1_clic12_20260812_v5`；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- 输出/日志根分别为`runs/phase1_clic_postfreeze_20260812_v4`、`logs/phase1_clic_postfreeze_20260812_v4`，必须预先不存在；outer写项目根，不得预建run/log。
- 固定12臂GPU映射`0,1,2,3,4,5,6,7,0,1,2,3`；formal launch=1，retry=`NO`。
- 成功要求12/12 NPZ，每份21120行（L3920/V16800/proxy400）、finite、source split/partition、物理RX/day、checkpoint/terminal/physical-order全部闭合；target/query/truth/role访问为0。
- 只标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`，不读取任何性能。

## 本地门

- 修复TDD：真实轴索引fixture使旧实现精确RED；修复后真实`CLEAN.export→G bundle`通过，重复索引、轴外索引和物理标签伪装3项均fail-closed。
- `ssr-gpu`postfreeze`135/135`、Phase1核心`190/190`通过；py_compile/diff-check通过。
- 独立审查`P0=0，P1=0，ALLOW`，确认真实v5缺任一物理标签字段均不能进入legacy分支。
- 待回填commit/archive/SCP/release/静态门/唯一launch/PID/GPU/12工件与SSH清理证据。
