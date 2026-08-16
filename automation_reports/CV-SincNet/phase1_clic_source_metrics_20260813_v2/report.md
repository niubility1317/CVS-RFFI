# Phase1 CLIC源域指标补全v2报告

## 预注册与状态

- 实验ID：`phase1_clic_source_metrics_20260813_v2`。
- 当前状态：`LOCAL_VERIFIED / P1_REMEDIATED / INDEPENDENT_REVIEW_ALLOW / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT`。
- 目标：仅修复F1独立结构smoke的输出/路径合同，使它直接使用封存的正式F1输入绝对路径，并将技术缓存和单次C/G前向写入独立、不可覆盖的smoke根。不得改变数据、方法、阈值、fold、指标、GPU矩阵、sealed receipt或任何target/query边界。
- v1封存事实：`phase1_clic_source_metrics_20260813_v1`的唯一smoke已停止，`SMOKE_INVOCATION=1`、`FORMAL_INVOCATION=0`、`retry=NO`。hardlink镜像后的checkpoint绝对路径与terminal envelope的`selected_checkpoint_path`不一致，builder在任何缓存、feature、scorer或性能输出前以`CLICSplitExportError: CLIC terminal envelope selected checkpoint path drifted`失败。v1永久封存，绝不重试、覆盖或改标为formal。

## 冻结v2运行合同

| 项目 | 冻结值 |
|---|---|
| formal运行根 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_metrics_20260813_v2` |
| formal日志根 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_source_metrics_20260813_v2` |
| F1 smoke根 | `/home/szu2070436088/2510044040/CV-SincNet/runs/.smoke_phase1_clic_source_metrics_20260813_v2_F1` |
| F1 smoke缓存/feature叶根 | `<F1 smoke根>/phase1_clic_source_metrics_20260813_v2` |
| 正式训练/terminal输入 | 原始`runs/phase1_clic12_20260812_v5/F1{C,G}_CLIC12/{final_ssdg.pth,phase1_clic_terminal_receipt.json}`绝对路径；禁止镜像、hardlink替代路径或复制 |
| 正式clean输入 | 原始`runs/phase1_clic_postfreeze_20260812_v4/F1{C,G}_CLIC12/source_clean_proxy.npz`绝对路径 |
| PAIR与ManySig | 原始`runs/phase1_clic_source_pair_20260812_v3/F1_C_vs_G_pair.json`和`Dataset_WigSig/ManySig.pkl`；禁止target/query输入 |
| smoke范围 | 仅`fold=1`、`F1C_CLIC12`、`F1G_CLIC12`、共享F1 cache、C/G各一次串行forward；无scorer、无性能读取 |
| formal范围 | 与v1同构：`6 cache→12 forward→6 score→1 aggregate`；dry-run必须25行，禁止target/query/truth/prediction/package/retry输入 |
| GPU | cache/forward固定GPU0..5；每个formal GPU最多两个forward；score/aggregate仅CPU；F1 smoke使用GPU0且C/G串行 |
| 终止/重试 | 仅协议、哈希、覆盖、访问违规或预注册系统技术失败可停止；不得看性能停止；`retry=NO`；失败新run ID而非恢复 |

## 可追溯实施表

| ID | 来源 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| V2-01 | v1终端错误 | 以原始absolute checkpoint/terminal重开；技术smoke的caller声明必须逐字等于冻结N607 canonical project root，镜像绝对路径继续拒绝 | exporter测试、F1 smoke脚本 | P1 round1 local complete | mirror RED→GREEN路径测试 | 不放宽terminal envelope |
| V2-02 | v2接口 | `--technical-smoke`仅允许F1、F1C/F1G；training/checkpoint/terminal/clean/PAIR/cache/output必须逐项等于冻结canonical root下的run ID和文件名 | exporter、测试 | P1 round1 local complete | 正/负根目录合同测试 | 无flag独立父目录必须拒绝 |
| V2-03 | formal不变性 | 无flag保持训练/clean/cache/output同一formal`runs`父目录；v2只改run identity | builder/exporter/evaluator、v2 launcher、测试 | local complete | focused回归与formal dry-run | 不改数据/方法/矩阵 |
| V2-04 | 技术smoke | fresh roots后先共享cache，再串行C/G一次forward；run/log根用exact mkdir独占认领，PID/log用noclobber持有FD；不score、不产生性能 | 新smoke脚本、formal脚本、脚本测试 | P1 round1 local complete | PID/log race RED→GREEN、bash-n与smoke dry-run | `FORMAL_INVOCATION=0` |
| V2-05 | release | 新v2报告、hash、测试、Git提交；独立P0/P1 review由主控另派 | 本报告 | independent review ALLOW | `P0=0/P1=0/P2=0` | 实现作者不自审；本地门不代表N607已运行 |

## 计划命令与交接边界

- 本地验证环境：`ssr-gpu`，串行运行聚焦测试和受影响回归；随后执行`py_compile`、三个`--help`、`bash -n`、formal/smoke dry-run及`git diff --check`。
- future formal入口（仅在主控完成独立review并交给唯一runner后）：`nohup bash <immutable-release>/code/scripts/launch_phase1_clic_source_metrics12_v2_20260816.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_metrics_20260813_v2_outer.out 2>&1 &`。
- future smoke入口：`bash <immutable-release>/code/scripts/smoke_phase1_clic_source_metrics_f1_v2_20260816.sh`；smoke失败仍保持formal根、日志及outer为`ABSENT`。
- 本次作者边界：只做本地代码、测试、脚本、报告和Git版本化；不访问N607，不同步、不启动、不评分、不解释性能。

## v2本地实施与验证记录

- 实施状态：`LOCAL_VERIFIED / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT`。
- 新增严格`--technical-smoke`导出控制：仅接受`fold=1`和`F1C_CLIC12/F1G_CLIC12`，训练、terminal和clean必须位于原始正式`runs`父目录；cache与output必须同为`<runs>/.smoke_phase1_clic_source_metrics_20260813_v2_F1/phase1_clic_source_metrics_20260813_v2`。任何镜像训练根、错误fold/arm、无flag独立根或正式输出根复用均在输出前失败。
- 新增F1 smoke脚本：只执行一次共享`F1_SHARED`缓存，然后按`F1C→F1G`顺序各执行一次forward；不调用scorer、不读取性能、不创建formal根。smoke根或对应日志根已存在时先以退出码3拒绝覆盖。
- v2 formal launcher保持`6 cache→12 forward→6 score→1 aggregate`冻结矩阵，dry-run为25条命令；F1 smoke dry-run为3条命令。

### 独立review P1修复round1

- P1-1攻击RED：旧`validate_source_v_execution_roots`只检查同形目录和leaf名称；同形`mirror_only`可一路触达镜像checkpoint的`torch.load`，terminal路径和SHA仅与镜像自洽，未在任何输出前锚定当前项目的冻结formal输入。
- P1-1最小修复：`--formal-project-root`只被当作声明，technical-smoke必须逐字等于冻结`/home/szu2070436088/2510044040/CV-SincNet`；并对F1C/F1G的training、checkpoint、terminal、clean、PAIR、cache和output根执行原始绝对路径、冻结run ID与文件名逐项比较。相对路径、caller自报mirror或同形mirror均在checkpoint打开和输出发布前失败；formal非smoke分支不读取该控制。
- P1-2攻击RED：旧formal/smoke先存在性检查，随后`mkdir -p`和`>`创建PID/log；攻击者可在guard后占位并使formal返回1、smoke返回127，同时覆盖marker。
- P1-2最小修复：保留早期只读exists拒绝，但guard后以exact`mkdir`独占认领formal的run/log根及smoke的smoke/log/source leaf；PID和每个log以Bash`noclobber`独占打开并持有FD，formal按stage预先claim全部本stage日志后才dispatch。任何root、PID或log碰撞均退出3，不向攻击者marker继续写入；本作者自己的安全半成品可保留，绝不删除他人文件。

| 阶段 | 攻击/验证 | 结果 |
|---|---|---|
| RED | 同形mirror把自身写入`--formal-project-root` | 旧代码仍打开mirror checkpoint，随后以`source-V final checkpoint is unreadable`包装；未在输出前拒绝。 |
| RED | formal PID post-guard marker | 旧代码返回1而非3；`mkdir -p`和PID重定向路径可覆盖marker。 |
| RED | smoke log post-guard marker | 旧代码返回127而非3；cache log重定向路径可覆盖marker。 |
| focused GREEN | mirror authority、formal PID race、smoke log race | 串行`ssr-gpu`验证3/3通过，exit0。 |
| full GREEN | 本轮exporter和两份owned测试 | 先`py_compile`3个Python文件，再执行38/38测试通过，exit0。31个test definition加6-case和3-case参数化共38个收集/执行实例。 |

### RED→GREEN证据

| 阶段 | 证据 | 结果 |
|---|---|---|
| RED | v2 launcher聚焦测试初始失败 | 2项失败均为缺失F1 smoke脚本，子进程退出码127；该RED直接对应结构烟测入口缺失。 |
| GREEN | Git-Bash合规聚焦验证 | `py_compile`3个source文件、builder/exporter`--help`、7项v2根合同与launcher测试均成功；解释器为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。 |
| 回归修复 | 三份owned测试首次全量 | 45通过、7失败；失败夹具仍使用已永久停止的v1 cache根，在v2根合同之前即被拒绝。仅将两个测试fixture根替换为冻结v2运行ID，未放宽生产代码。 |
| GREEN | 三份owned测试最终全量 | `52 passed`，退出码0。 |

### 非性能静态证据

- 两份脚本`bash -n`通过。
- `launch_phase1_clic_source_metrics12_v2_20260816.sh --dry-run`精确25行；`smoke_phase1_clic_source_metrics_f1_v2_20260816.sh --dry-run`精确3行。
- 本轮`py_compile`覆盖`export_phase1_clic_source_v_leo_features.py`、`test_phase1_clic_source_metrics.py`和`test_phase1_clic_source_metrics_launcher.py`，随后两份测试文件全量38/38通过；无N607、SSH、SCP、sync、训练、实际smoke/formal运行、scorer或性能读取。
- launcher测试执行器显式调用`C:\Program Files\Git\bin\bash.exe`并使用`/e/...`路径；不再使用裸`bash`、`/mnt/...`或`WSLENV`。
- `git diff --check`通过。未执行N607、SSH、SCP、sync、训练、smoke实际运行、scorer或任何性能读取。

### 独立P0/P1复审

- 独立复审结论：`P0=0 / P1=0 / P2=0 / ALLOW`，仅代表本地发布门通过，不代表N607落地、运行或性能结论。
- canonical F1C/F1G路径正例通过；caller自报mirror与同形mirror均在checkpoint打开和输出发布前拒绝。
- formal与smoke的post-guard根/证据竞争均以退出码3拒绝，攻击者marker保持原字节且零dispatch。
- fresh复核再次通过`py_compile`、两份测试38/38、两份`bash -n`、formal/smoke dry-run 25/3和`git diff --check`。

### 本次变更文件

- `code/build_phase1_clic_source_v_leo_iq.py`
- `code/export_phase1_clic_source_v_leo_features.py`
- `code/scripts/launch_phase1_clic_source_metrics12_v2_20260816.sh`
- `code/scripts/smoke_phase1_clic_source_metrics_f1_v2_20260816.sh`
- `code/tests/test_build_phase1_clic_source_v_leo_iq.py`
- `code/tests/test_phase1_clic_source_metrics.py`
- `code/tests/test_phase1_clic_source_metrics_launcher.py`

独立P0/P1审查、任何N607落地和实际smoke/formal运行仍由主控与唯一runner另行处理；本报告不构成性能、发布或N607已落地声明。

### Git与哈希

- 本地提交：`503b07be7ed20359fa0a5f95eb8244f822173ab9`（`Add v2 source metrics smoke path`）。
- P1修复提交：`9f56e731d4443c5e1f396bf77afb8a877461cd9d`（`Harden v2 source metrics smoke path`）。

### P1修复round1最终源文件哈希

- `code/export_phase1_clic_source_v_leo_features.py`：`2b1367a2508a503795d4c193264615e67e348eedfc46c4bd731803ccd18341b7`。
- `code/scripts/launch_phase1_clic_source_metrics12_v2_20260816.sh`：`a7182ca1835a3caa87a4397d5b552c795796ded0e97f1e3d920c0f49ae3ca05f`。
- `code/scripts/smoke_phase1_clic_source_metrics_f1_v2_20260816.sh`：`0470ffb627bd7ee059d4a9f722a770346c6c35624ad1a9373205f59ee8f93b6c`。
- `code/tests/test_phase1_clic_source_metrics.py`：`815a9f2d51a1c6d0a6f863ea0b7dc2f1f6f6adba423687f51cbd18ec9f2476e9`。
- `code/tests/test_phase1_clic_source_metrics_launcher.py`：`440f5347b5fde93ced239c2bea8ea2a9f19b86a644569711992fd1fb5ebfef8e`。
- 该report的SHA256在task report中记录，以避免报告自引用哈希。
