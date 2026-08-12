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

## 落地、静态门与唯一启动

- 冻结commit：`e1d5d8dd30b5e087df0e38e39321f5c2df95cc85`；工作树除既有未跟踪`conversation_index/`外未改动。`git archive`严格取该commit，未在本地解包：`clean_commit.tar`，267509760bytes，SHA256=`751533E25840FFC2FAD6EFE186221CFC13173932F4C42D454844F05E8EDDA3BF`。
- SCP恰1次落到项目根`/home/szu2070436088/2510044040/CV-SincNet/clean_commit.tar`；远端bytes/SHA闭合；解压到全新`.phase1_clic_postfreeze_20260812_v4_e1d5d8dd.stage`后原子改名为`releases/phase1_clic_postfreeze_20260812_v4_e1d5d8dd`。v4run/log/outer在落地前均ABSENT，未预建run/log。
- 远端release物理SHA（git archive的CRLF物理态）为：exporter=`2894718306391b8c8e1ecdb90064dca780f5275a7dc337423eeb527dbe48f9d5`、bundle=`7404074f2a8394caf6120a6b95da47be5ed87437bfebbcfd6a7b137c70aa4405`、launcher=`d5a97deb2031cfca1de356ba4b26ece36e9ec0c448cd45e0b8cb3ab088c887ba`。exporter/bundle的CRLF归一化LF SHA分别等于canonical blob`D4935B5C07748D6F4A956E52E2505A46B8F572BF2C860E2B237F214863688087`、`669C636A3AC4BFE3C3AC7A7F6D884E547BA6F8876AB17DC0574E118F2877AC03`；launcher归一化前后等于canonical`D5A97DEB2031CFCA1DE356BA4B26ECE36E9EC0C448CD45E0B8CB3AB088C887BA`。ManySigSHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- 静态门PASS：release文件SHA/归一化等价、ManySig、`py_compile`exporter+bundle、`--help`两入口、`bash -n`launcher、`--dry-run`恰12行；dry-run中forbidden target/query/truth/role=0。真实F1C/F1G checkpoint+terminal只读validation均PASS（C/G绑定、`source_l_only=true`、checkpoint/receiptSHA闭合）。
- 唯一正式命令（`bash`调用，`nohup`，project-root outer重定向）于2026-08-12T20:55:29启动：
  `nohup bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_postfreeze_20260812_v4_e1d5d8dd/code/scripts/launch_phase1_clic_postfreeze_source12_v4_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_postfreeze_20260812_v4_outer.out 2>&1 &`
  - `FORMAL_INVOCATION=1`、`RETRY=NO`；outerPID=`2711176`。12childPID记录于`logs/phase1_clic_postfreeze_20260812_v4/pids_source12.tsv`，映射严格为`0,1,2,3,4,5,6,7,0,1,2,3`，不超过2/GPU。

## 首波、工件闭合与最终状态

- 首波时outer及12exporter均live，GPU仅本run的12child；约20秒内全部正常退出，无`Traceback`/`Error`/`Exception`/`FAILED`标记。outer0bytes属于正常无标准输出；12个per-armout各约5384–5386bytes，包含单行manifest与`out_npz`。
- 12/12`source_clean_proxy.npz`全部存在（每份约32MB），candidate为`F1C/F1G…F6C/F6G_CLIC12`。生产`evaluate_phase1_clic_postfreeze_pair._load_feature_npz`逐份重开PASS；每份21120行，角色计数`labeled_fit=3920`、`source_validation_known=16800`、`proxy_unknown=400`；`features=z_id`形状`(21120,160)`、`tx_logits`形状`(21120,4)`且全部finite。
- 自定义闭合核验PASS：所有`(tx,rx,day,eq,sig)`物理key全局唯一，L/V/proxy物理集合两两disjoint；manifest中`source_split_receipt`、`tx_partition_receipt`、source receiver/day轴、L/V/proxy physical-order SHA均与当前行序一致；checkpoint与terminal receipt实际bytesSHA一致；`source_only=true`、`clean_source_runtime_access=false`、`query_fit_access=false`、Uloader/forward=0、validation/proxy fit/threshold=0、`held_tx_loaded_by_training=false`、source-target receiver overlap=0；C/G`clic_enabled`绑定正确。
- 完成后短连接核验：outer/child均退出，GPUcompute-app为空，所有本地SSH/SCP进程与TCP22连接清零；未读性能指标、未重试、未清理非本run。最终状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。
