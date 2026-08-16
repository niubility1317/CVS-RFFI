# Phase1CLIC源域指标补全v4预注册与可追溯报告

## 身份、状态与不可变边界

- 实验ID：`phase1_clic_source_metrics_20260816_v4`。
- F1smoke实验ID：`.smoke_phase1_clic_source_metrics_20260816_v4_F1`。
- 当前状态：`LOCAL_VERIFIED / REVIEW_ROUND2_ALLOW_P0_0_P1_0_P2_0 / SMOKE_INVOCATION=0 / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT`。
- 本轮性质：source-metrics发布工程的第二且最后一次具体修复轮；仅修复PAIR-v3合法proxy diagnostic的读取合同，不改变数据、科学方法或评分。
- 禁止：不读取或接触N607、target、query、truth或性能；不改训练、C/Gcheckpoint、scene、seed、channel、阈值、scorer公式、metrics或冻结矩阵；不恢复、覆盖、重标或重试v3。

## 上一运行封存事实与根因追溯

| 项目 | 封存事实 | v4处理 |
|---|---|---|
| v3运行ID | `phase1_clic_source_metrics_20260816_v3` | 永久保留，不覆盖 |
| v3终态 | `SMOKE_STOPPED_TECHNICAL_FAILURE / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT / RETRY=NO` | 不重试、不重标、不作为性能结果 |
| v3cache | 已合法产生`16800=4×7×2×300`held-V行，并保持C/G同一received-IQ、V-only和zero-fit | 不重建、不改变字节、场景或seed |
| 失败点 | F1C exporter在特征输出前拒绝PAIR-v3：`PAIR-v3 proxy diagnostic fit_rows must remain zero` | 仅修正reader对真实PAIR-v3输出结构的验证 |
| 已证实根因 | `compute_clic_proxy_diagnostic`将正的source-L几何拟合写入`diagnostic['fit']['fit_rows']`及`diagnostic['geometry']['fit_rows']`，而顶层无`fit_rows`/`threshold_fit_rows`字段；现有reader错误要求顶层字段为零 | 接受嵌套、绑定的一次source-L几何拟合；仍拒绝V/proxy或任何threshold拟合 |

## v4冻结合同

| 项目 | 冻结值 |
|---|---|
| formal运行根 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_metrics_20260816_v4` |
| formal日志根 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_source_metrics_20260816_v4` |
| F1smoke运行根 | `/home/szu2070436088/2510044040/CV-SincNet/runs/.smoke_phase1_clic_source_metrics_20260816_v4_F1` |
| F1smoke日志根 | `/home/szu2070436088/2510044040/CV-SincNet/logs/.smoke_phase1_clic_source_metrics_20260816_v4_F1` |
| 实现改动 | `export_phase1_clic_source_v_leo_features.py`的PAIR-v3proxy diagnostic验证、v4身份常量与机械launcher副本、对应TDD测试 |
| 保持不变 | 六fold、C/Gcheckpoint、`4×7×2×300`held-V、三LEO场景、seed、channel、source-L policy、proxy AUROC/u_gap、scorer与全部性能门 |
| smoke | 先运行唯一F1cache→F1C→F1G结构烟测；不scorer、不读取性能、不创建formal根 |
| formal | 仅在smoke技术artifact完整、local验证、独立P0/P1审查、Git提交和唯一runner交接后，执行`6cache→12forward→6pairscore→1aggregate` |
| retry | `RETRY=NO`；任何技术失败只能由主控申请全新的不可覆盖run ID |

## TDD与验证可追溯性

- reviewround1的唯一P1：`evaluate_phase1_clic_source_metrics.py`的完整PAIRscore入口曾错误要求真实producer从不输出的顶层`fit_rows`与`threshold_fit_rows`。该问题位于formal score_command必经路径，曾使v4为`LOCAL_NO_GO_PENDING_SCORE_ENTRY_REPAIR`；独立reviewer未发现第二项P1。
- 本轮入口级TDD真实完整score：PAIRfixture采用production-shape嵌套diagnostic，使旧scorer在顶层缺失时以`None`触发原有fit/threshold拒绝；GREEN只复用exporter已冻结的`_validate_pair_proxy_diagnostic`并转换其异常类型，不复制或放宽合同。
- RED测试先于生产改动：使用与`evaluate_phase1_clic_postfreeze_pair.compute_clic_proxy_diagnostic`一致的PAIR-v3diagnostic形状，顶层不含`fit_rows`和`threshold_fit_rows`，但`fit.role=source_L_only`、嵌套source-L`fit_rows`为正且等于`geometry.fit_rows`。旧reader应因缺失顶层字段错误拒绝该合法输入。
- 负向合同：`source_validation_known.fit_rows`、`proxy_unknown.fit_rows`、聚合`source_validation_fit_rows`、`proxy_fit_rows`以及任一嵌套或聚合threshold-fit字段非零必须失败；嵌套source-L拟合为零、非整数、角色漂移或与geometry不绑定也必须失败。
- GREEN最小修复：不再读取或要求顶层`fit_rows`/`threshold_fit_rows`；只接受正整数、与`geometry.fit_rows`一致的source-L几何拟合，并严格保持source_validation/proxy和所有threshold计数为零。AUROC/u_gap、PAIRpolicy、SHA与immutable reopen检查保持只读且不变。
- P1GREEN实现：完整score入口已调用同一exporter validator，并仅将`CLICSourceVFeatureExportError`转换为`CLICSourceMetricsError`；未复制、放宽或重解释diagnostic合同。aggregate的`_pair_core`仍校验本scorer写出的`proxy_readonly`输出顶层零字段，该输出与原始PAIR-v3producer不同，且入口RED继续显式断言该输出为零，因此不构成同构P1。
- reviewround1RED/GREEN运行证据：在唯一串行`ssr-gpu`wrapper中，受影响生产与测试文件`py_compile`通过；真实PAIR形状的完整score入口回归为`1 passed in 2.64s`；三份受影响自有测试文件全量为`81 passed in 24.99s`；wrapper退出码为0。随后两份v4launcher的`bash -n`通过，formal/smoke dry-run分别严格为25/3行，且与v3的identity-only机械替换逐字一致；`git diff --check`通过。结束后`conda.exe`、`python.exe`与`pythonw.exe`均为0。
- RED证据：v3N607F1C已以完全相同的生产reader在输出前确定性报错`CLICSourceVFeatureExportError: PAIR-v3 proxy diagnostic fit_rows must remain zero`；依照主控禁止临时回滚共享生产代码的要求，不在本地伪造回滚RED。
- GREEN证据：获得`CONDA_CLEAR`后，以唯一串行`ssr-gpu`wrapper依次运行生产`py_compile`、三个新增PAIR诊断合同和三份受影响自有测试文件。全部退出码为0；新增合同验证合法嵌套source-L拟合的接受、9种V/proxy/threshold非零拒绝以及3种source-L几何拟合绑定拒绝。
- 已完成静态检查：`git diff --check`通过；两份v4launcher的`bash -n`通过；v4formal/smoke分别与v3机械副本的identity替换结果逐字一致；launcherdry-run由受影响测试文件覆盖。测试结束后`conda.exe`、`python.exe`和`pythonw.exe`均为0。
- reviewround2独立终裁：`ALLOW`，`P0=0 / P1=0 / P2=0`；唯一的scorer合同P1已闭合。fresh受影响验证为完整score入口`1 passed in 2.64s`与三份受影响测试`81 passed in 24.99s`，均退出码0；这仅证明本地实现就绪，`SMOKE_INVOCATION=0 / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT`。

## 停止与证据规则

- 只因协议、访问、hash、输出覆盖、错误checkout或确定性技术异常停止；绝不根据accuracy、AUROC、floor或其他性能值停止、调参、选择或重试。
- v4仅在完整预测与独立评分artifact齐全后才可进入分析；技术完成不等于性能门通过、候选晋级或Phase3声明。

## v4发布与F1 smoke技术证据（2026-08-16）

- 唯一runner使用冻结commit`37d244be125c6e79aee35ff35c4a8dd6e889530d`创建LF-safe Git archive；本地archive为`35815871`bytes，SHA256=`ee279dca406378b021de2ec0f9d753e071136eab4028688e0ef187f4daf2d946`。SCP恰1次，远端bytes/SHA一致。release已atomic rename至`/home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260816_v4_37d244be`；stage、tar和release均未覆盖或删除。
- 远端STATIC：exact`CVS-RFFI/bin/python`的builder/exporter/metrics临时cfile`py_compile`通过；三入口`--help`通过；两launcher`bash -n`通过；formal/smoke dry-run严格为25/3行，且dry-run未出现target/query/truth/prediction/package/`--retry`；ManySig SHA、12组training-v5/clean-v4与6组PAIR-v3输入存在。release、formal/smoke run/log及outer根在启动前均ABSENT，release post-state hash闭合。

### Source-V身份闭合

- F1_SHARED receipt与feature binding共同封存的7个physical source receiver IDs为`1-1、1-19、14-7、18-2、19-2、2-1、2-19`；2个source day IDs为`2021_03_01、2021_03_08`。本身份集合与v3已封存clean-v4/source-V输入一致，未重建或改变物理数据。
- source-V receipt结构QA：`16800=4TX×7RX×2day×300`；三LEO场景各5600；两日期各8400；每个TX/RX/day各300；physical_sample_id全局唯一；三场景physical ID两两不交；`single_leo_observation_per_physical_sample=true`；C/G共享received-IQ字节；source-only、zero-fit、zero-threshold、zero-source-L/proxy forward及clean/target/query/selection/retry访问均为零或false。

### F1 smoke

- 唯一入口：`cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260816_v4_37d244be/code/scripts/smoke_phase1_clic_source_metrics_f1_v4_20260816.sh > /home/szu2070436088/2510044040/CV-SincNet/.smoke_phase1_clic_source_metrics_20260816_v4_F1_outer.out 2>&1 &`。
- `SMOKE_INVOCATION=1`，outer PID=`902611`，启动时间=`2026-08-16T18:28:32+08:00`，CWD为canonical project root，GPU=`0`，`RETRY=NO`。cache→F1C→F1G串行完成；未运行scorer，未创建formal root。

| artifact | bytes | SHA256 |
|---|---:|---|
| `F1_SHARED/source_validation_known_leo_weak.npz` | 41127936 | `fdc7d7cc00df80028fba432499658ee86ba22bcf993bcda83a7dd6bd1adb1a5c` |
| `F1_SHARED/source_validation_known_leo_weak.receipt.json` | 13739 | `b8aa907de0864281f36cc770dfa536f35a2eb08e6d2994e9bdb13d4e8f7a77f3` |
| `F1C_CLIC12/source_validation_known_leo_weak_features.npz` | 37711540 | `16ada3c984f758bb60d3efcaedc6a4014d8d82843dc5e0b0cd08606bbd1dee47` |
| `F1C_CLIC12/source_validation_known_leo_weak.binding.json` | 1887 | `0e2828467970776a8436f3f27a1d97438f22540fcfabc16cd4c70ce159f75bc4` |
| `F1G_CLIC12/source_validation_known_leo_weak_features.npz` | 37711544 | `280685f0565cd9ed12c091ab8da393b0fe1eb9a837f3c84e9f30c8e61bcd28f8` |
| `F1G_CLIC12/source_validation_known_leo_weak.binding.json` | 1887 | `2d210749805a7e6254bde52d9fb48d628c5011a31a204a5288765d26accce5d3` |
| `F1_source_v_cache.out` | 574 | `77e94d9115c402444099f454a93f4510d9915a18a1ffa2a75985ea4eaf02a7d5` |
| `F1C_CLIC12_source_v_forward.out` | 2569 | `af31c2f1af57da17f4c510b282cb6ce72abd6a49cf3170a719bfacdcb77de100` |
| `F1G_CLIC12_source_v_forward.out` | 2570 | `25d9ba7e2f2aa10b3370a5abe6882cd2fcb97fce71ca0967f15a50ffad7117a9` |
| smoke outer | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

- 独立结构QA未读取或输出任何性能字段：三份NPZ的16800行/finite/physical-order闭合，C/G binding的cache、PAIR、physical-order和zero-access/zero-fit/zero-threshold字段一致；6个预期artifact全部存在。smoke wrapper已退出，GPU0为`1MiB/0%`，本地SSH/TCP22已清零。
- 过程记录：一次早期健康检查曾无意输出包含封存proxy字段的完整binding日志行；其性能字段未被保存、比较、选择、停止或写入本报告，后续QA仅访问结构与权限字段。

## v4当前状态

- `SMOKE_PASS / SMOKE_INVOCATION=1 / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT / RETRY=NO`。
- 仅技术烟测通过；formal尚未启动。下一阶段必须使用同一已落地release按冻结6fold完整矩阵执行，不得选择性运行或读取性能作停止依据。
