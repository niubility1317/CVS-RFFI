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

- `FORMAL_ARTIFACTS_COMPLETE / SMOKE_INVOCATION=1 / FORMAL_INVOCATION=1 / METRICS_MECHANICALLY_RECORDED / RETRY=NO`。
- formal完整矩阵已完成技术artifact与独立评分封存；sealed source metrics与aggregate已只读机械记录。性能门结果不作科学解释、调参、选择或晋级判断，由主控处理。

## v4 formal最终技术QA与sealed source metrics（2026-08-16）

### Formal启动、进程与技术异常

- 唯一formal入口：`cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260816_v4_37d244be/code/scripts/launch_phase1_clic_source_metrics12_v4_20260816.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_metrics_20260816_v4_outer.out 2>&1 &`。
- `FORMAL_INVOCATION=1`、`RETRY=NO`；outerPID=`906842`；启动时间=`2026-08-16T18:34:03+08:00`；CWD为canonical project root；GPU映射为`F1..F6→0..5`。
- PIDTSV绑定：cachePID按F1→F6为`906939、906945、906951、906957、906963、906969`；forwardPID按F1C/F1G…F6C/F6G为`908106、908112、908118、908124、908130、908136、908142、908148、908154、908160、908166、908172`；pairPID按F1→F6为`909568、909574、909580、909586、909592、909598`；aggregatePID=`909608`。最终核验时上述run进程均已退出，仅作为历史PID证据保留。
- formalrun=`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_metrics_20260816_v4`；日志根=`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_source_metrics_20260816_v4`；outer为0bytes，SHA256=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- 最终只读核验时run进程已退出；GPU0–7均为`1MiB/0%`；日志文件数=26；`pids_source_metrics12.tsv`行数=26；技术异常token计数=`0`，异常行计数=`0`（`Traceback`、`RuntimeError`、`CUDA out of memory`、`unrecognized arguments`、`NaN`）；本地SSH/SCP进程与TCP22均清零。

### Formal结构与访问QA

- 6个sharedcache+receipt、12个C/Gfeature+binding、6个pairmetrics、1个aggregate全部存在。
- 每fold均满足总量`16800=4×7×2×300`、三scene各`5600`、TX/RX/day集合为`4×7×2`、每个TX/RX/day合计`300`；physical_sample_id全局唯一，三scene两两物理ID不交，`single_leo_observation_per_physical_sample=true`。
- 12份feature均为`16800×160`且finite；C/Gfeaturephysical顺序一致，并与sharedcache及binding中的physical-order/cache/receiptSHA闭合。
- 6份receipt及12份binding均为source-only；fit/update/proxy-forward/threshold计数为0；target/query/selection/retry访问为false；PAIR与aggregate均为`POST_TARGET_COMPLETION_AUDIT_NON_SELECTION`。6份pair的C/Gclean各16800行、每scene各5600行，决策计数闭合。
- E:\type10-7根镜像路径`/e/type10-7/automation_reports/CV-SincNet/phase1_clic_source_metrics_20260816_v4/report.md`在最终QA时不存在，未创建、未同步；仅更新本Gitworktree中的同一report。

### Formal主artifact逐项bytes/SHA256

| artifact | bytes | SHA256 |
|---|---:|---|
| `F1_SHARED/source_validation_known_leo_weak.npz` | 41127936 | `fdc7d7cc00df80028fba432499658ee86ba22bcf993bcda83a7dd6bd1adb1a5c` |
| `F1_SHARED/source_validation_known_leo_weak.receipt.json` | 13690 | `ef06ddfd254bacbabd618be91c1065a95db28bfb2d856d46d9177bcd0ab5cd57` |
| `F1C_CLIC12/source_validation_known_leo_weak_features.npz` | 37711540 | `79eaf8fb84c62dcf5cf251a549e225470b4ab25df79c161a60a43c928a868ed0` |
| `F1C_CLIC12/source_validation_known_leo_weak.binding.json` | 1838 | `a41c028438e5738d99d0f71a960548694ae081913a6efe81c1c31a3a15d1a7d6` |
| `F1G_CLIC12/source_validation_known_leo_weak_features.npz` | 37711544 | `89145e73c75b3c75e9e8b31e149e425a8bec161f5dfac94426ce2d1029a54f3f` |
| `F1G_CLIC12/source_validation_known_leo_weak.binding.json` | 1838 | `2d41bcb97dc578399494047e22997286da23590e37751b72f5b732561b1f90a0` |
| `F1_PAIR/source_metrics_pair.json` | 16963 | `bbcfd15ed6c0d1adc437452ed2b2986ce384b4be54e137e4955aa4a5010144f6` |
| `F2_SHARED/source_validation_known_leo_weak.npz` | 41127936 | `5f64ef6bf67cc7169ae7fcd91e9ff311a5c05beba9876363379af7fdd5eafe4d` |
| `F2_SHARED/source_validation_known_leo_weak.receipt.json` | 13690 | `de596dd2794b1a0b98cc2ad456a722ab9d4055081f5064854826a062f511d505` |
| `F2C_CLIC12/source_validation_known_leo_weak_features.npz` | 37711544 | `8fe9c785887b220634138bba06edb7767b2a7ecc5b380acb0f55b90cf755c2d8` |
| `F2C_CLIC12/source_validation_known_leo_weak.binding.json` | 1838 | `e65e56116c0e52fc2a45dccc7fd533079d98a1a02ec370a6f2cfe62ec2b0db5a` |
| `F2G_CLIC12/source_validation_known_leo_weak_features.npz` | 37711536 | `85e0e4f899ae981b7f5657bee8dd3a07d1196e7a2c9c2f0b0029ea689c6d0b02` |
| `F2G_CLIC12/source_validation_known_leo_weak.binding.json` | 1838 | `de945816ffa9111cf07de7bc9036a94cdfc48e6474fb0cdb61cd717b65cbfe72` |
| `F2_PAIR/source_metrics_pair.json` | 16855 | `1cbb62fcee96f94cf3b52f7b1badf9a71d987f4a1d9e1db3cd119d21a886edb9` |
| `F3_SHARED/source_validation_known_leo_weak.npz` | 41127936 | `1dc8a39ebef086e489fb467456fdc0fe7ce1e4886771ec0592a055aaf97b7dd3` |
| `F3_SHARED/source_validation_known_leo_weak.receipt.json` | 13630 | `71848e5ce545c242fcbfdd0eb425c626a5c946929de85180b71778a101fbd5ff` |
| `F3C_CLIC12/source_validation_known_leo_weak_features.npz` | 37711528 | `237450bf444ff8c0ba481804ba5106f250fe412aaa5376f8083c65c2e4e95763` |
| `F3C_CLIC12/source_validation_known_leo_weak.binding.json` | 1837 | `8bd45c6477c820d09c8e4790393b4f3decd71ab9be86e8caab1003632e345e10` |
| `F3G_CLIC12/source_validation_known_leo_weak_features.npz` | 37711532 | `17d4b11236cfc2895399cff81e0defaa7229a93deebb9592d5fd82d5e8e4412f` |
| `F3G_CLIC12/source_validation_known_leo_weak.binding.json` | 1837 | `c305ba0d48d1c7c94fab540280aa30af64947964edf9d4adc54e10b640702a56` |
| `F3_PAIR/source_metrics_pair.json` | 16893 | `7a4c266e464d0259c247a3b5314ae506222cd199324de66ef5cab841f29032d0` |
| `F4_SHARED/source_validation_known_leo_weak.npz` | 41127936 | `c45fa7e00e969803cbd6023f4f23cd04d2962bb9ac7c53960d79483b17bb6849` |
| `F4_SHARED/source_validation_known_leo_weak.receipt.json` | 13690 | `e5026e2b89bed61ec68c08705b6fe8862025b6c596ead4ce12ca8b80c97e0393` |
| `F4C_CLIC12/source_validation_known_leo_weak_features.npz` | 37711544 | `75e64fe49168d73795918cf0dc2bd459b8f0ce7a01acb796c84cd8e6ee974428` |
| `F4C_CLIC12/source_validation_known_leo_weak.binding.json` | 1838 | `829f3598ccea048f447301b21cc8b0cab13c70b4817d48262a741947c27ef2ca` |
| `F4G_CLIC12/source_validation_known_leo_weak_features.npz` | 37711544 | `d896d9727ef5b85a44844b38d41a3368f2b76de2316f54adb97286e75a58aa57` |
| `F4G_CLIC12/source_validation_known_leo_weak.binding.json` | 1838 | `63bcca5d360da2c65b70bdc39087a38a7ce3f121b4d8c778939e1ed3bfadbb40` |
| `F4_PAIR/source_metrics_pair.json` | 16970 | `e9e8faebf9fcc6e95fd656a48778fb4d372cce75dac1c5b0d9ed6d82a6ffb861` |
| `F5_SHARED/source_validation_known_leo_weak.npz` | 41127936 | `ffc327e01fc5e766ab1e754ae511ac263ac83e61eef04a1eed13999e6f625680` |
| `F5_SHARED/source_validation_known_leo_weak.receipt.json` | 13750 | `850758c90df6746c935e68953eb4dfa0a344014253f0961577496cdb40510576` |
| `F5C_CLIC12/source_validation_known_leo_weak_features.npz` | 37711544 | `8d851de2b95cd5c1fde6b4655110de44fca3af978c7b526e2bd87fb2b7f596e3` |
| `F5C_CLIC12/source_validation_known_leo_weak.binding.json` | 1839 | `067b488d7c25971dc34b517fd01e7f6a8ad507139e82049d6c3e21e211e9f122` |
| `F5G_CLIC12/source_validation_known_leo_weak_features.npz` | 37711544 | `2be6b51d61d4a03b08e7fe0bebc97bace23390995a1511d85357e233281e8442` |
| `F5G_CLIC12/source_validation_known_leo_weak.binding.json` | 1839 | `1207d29491574d7fa15775fc0aff4cee666cbf913be193e107a43bf436d7120f` |
| `F5_PAIR/source_metrics_pair.json` | 16931 | `1db18375d4d91db4b56f6f802cebf2ce0b5d34e20f6d348438848859e6eec61e` |
| `F6_SHARED/source_validation_known_leo_weak.npz` | 41127936 | `1a82048684f1e7731cb103cbaf25b5bee7829cacea2783d6cdfc04b4f8f84d41` |
| `F6_SHARED/source_validation_known_leo_weak.receipt.json` | 13690 | `ce4f420d0241d1b48862d2f3ffaad638270a78cffe035f956803e03d6c9c9d0e` |
| `F6C_CLIC12/source_validation_known_leo_weak_features.npz` | 37711536 | `472d36155273f56aab009f01229c155535f5e8c7710a89a2921056698837b6b0` |
| `F6C_CLIC12/source_validation_known_leo_weak.binding.json` | 1838 | `5ecf8dfc8e6e09643051b9611726c6ffa8834d3a39b20cebe33f594ddb6e2bf1` |
| `F6G_CLIC12/source_validation_known_leo_weak_features.npz` | 37711544 | `57c29fffae38a3cffeeae852c108a1ceccba37104f90396d139958477dc61a51` |
| `F6G_CLIC12/source_validation_known_leo_weak.binding.json` | 1838 | `04105155b9a3265f0cc63747ec34ba31c909c017e34782b39f5e522e31f450de` |
| `F6_PAIR/source_metrics_pair.json` | 17042 | `d33c975c9ffa96fb7bea1d32105666610f2de4a49c726a4c0cefd69eb428d06b` |
| `source_metrics_aggregate.json` | 8760 | `cbb89caefe5c5644fb7187dc8a3084b178232a7cc503df8e2ef63affedc39deb` |

### Formal日志/PID逐项bytes/SHA256

| log/artifact | bytes | SHA256 |
|---|---:|---|
| `F1C_CLIC12_source_v_forward.out` | 2471 | `e1d764a9beb3cca1041cf6b16960fbb917dd11066ef6e26e17a5859433115a16` |
| `F1G_CLIC12_source_v_forward.out` | 2472 | `7f2afc6c9ccb1e90cc872d6926e1f2888f91430249610de4dd6d26dab07bf058` |
| `F1_source_metrics_pair.out` | 16963 | `bbcfd15ed6c0d1adc437452ed2b2986ce384b4be54e137e4955aa4a5010144f6` |
| `F1_source_v_cache.out` | 476 | `f7b876d3e44543446c139c7462ace81ca5d3e4789b7004bef50d09074a32bf66` |
| `F2C_CLIC12_source_v_forward.out` | 2472 | `3969aead8a418e70059838ecaca2923b26945477bcf368766aae8939299ade67` |
| `F2G_CLIC12_source_v_forward.out` | 2470 | `b06ca61937cb99d8f750f4a3b13988a092ba8307c4a29095d7e42b1e6a2cb03a` |
| `F2_source_metrics_pair.out` | 16855 | `1cbb62fcee96f94cf3b52f7b1badf9a71d987f4a1d9e1db3cd119d21a886edb9` |
| `F2_source_v_cache.out` | 476 | `56f796db0c7904d5c40e38e9e24862be45ce1e46cea8f7c59891d63cf0f7bf11` |
| `F3C_CLIC12_source_v_forward.out` | 2468 | `fda4dbb588ac090eefc87f780839c8ebee9bb773ee596164bd0d87b88b717978` |
| `F3G_CLIC12_source_v_forward.out` | 2469 | `f888df99cf8e87d3187a994dc8e14fd9f2099039c4e99d51071732f1f8ab9914` |
| `F3_source_metrics_pair.out` | 16893 | `7a4c266e464d0259c247a3b5314ae506222cd199324de66ef5cab841f29032d0` |
| `F3_source_v_cache.out` | 476 | `e8ab907e15c8f8840f9817734fc67da8e91d8c239762696038d61398cdbc87a5` |
| `F4C_CLIC12_source_v_forward.out` | 2472 | `fdea71cd5ceea54ef00f9b3f253b73674c542ceea35c409a436922a242efca3d` |
| `F4G_CLIC12_source_v_forward.out` | 2472 | `14fd2fb036e77e61c60ee3d8103469358479b6b7290d157ba9aaddd7cd774912` |
| `F4_source_metrics_pair.out` | 16970 | `e9e8faebf9fcc6e95fd656a48778fb4d372cce75dac1c5b0d9ed6d82a6ffb861` |
| `F4_source_v_cache.out` | 476 | `64816425593eef2cc8ba089770832149fa0bcdb9480d1ce7d312c559b82d087b` |
| `F5C_CLIC12_source_v_forward.out` | 2472 | `3398b4238591bd91755a80efbeab5871e01f23f65a4d37a8a915002f48f3ba11` |
| `F5G_CLIC12_source_v_forward.out` | 2472 | `20be25ffb0f19781aab138f72e6f6db89c969ae7aff33ba7a175fa860547b447` |
| `F5_source_metrics_pair.out` | 16931 | `1db18375d4d91db4b56f6f802cebf2ce0b5d34e20f6d348438848859e6eec61e` |
| `F5_source_v_cache.out` | 476 | `8b22da9bd5c03fb23077ea9fe550f24f566f6a278f1f06e9bd2d43c5d9c322bb` |
| `F6C_CLIC12_source_v_forward.out` | 2470 | `1d939e980fbe15fee9c21106164fe5df9fcbb3b07bb6841167ec273d8790ab75` |
| `F6G_CLIC12_source_v_forward.out` | 2472 | `642c7c3c332b31a2e869e67c294941087df40ff9045065e1c59a6a551832afe0` |
| `F6_source_metrics_pair.out` | 17042 | `d33c975c9ffa96fb7bea1d32105666610f2de4a49c726a4c0cefd69eb428d06b` |
| `F6_source_v_cache.out` | 476 | `b0376d026529f51f73fc1bf2be1752f8d64cb5149e69e842b792465f528fa488` |
| `pids_source_metrics12.tsv` | 7167 | `b1748eeb1c2269048d7389663083a6fd1c522973324e6b016065aa528aef964d` |
| `source_metrics_aggregate.out` | 8760 | `cbb89caefe5c5644fb7187dc8a3084b178232a7cc503df8e2ef63affedc39deb` |

### Sealed source metrics同row表

以下表格机械转录pairartifact，未跨fold、跨arm或跨scene拼接；accuracy字段保留为比例，delta字段按scorer原始`pp`单位记录。

| fold | arm | clean overall | clean min-class | clean min-RX | clean min-day | proxy AUROC_unknown | proxy u_gap |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| F1 | C | 0.993036 | 0.987381 | 0.977917 | 0.992024 | 0.843887 | 775.254313 |
| F1 | G | 0.992976 | 0.986429 | 0.978750 | 0.991786 | 0.923042 | 1790.268729 |
| F2 | C | 0.988512 | 0.973095 | 0.967917 | 0.985476 | 0.887854 | 309.529917 |
| F2 | G | 0.991905 | 0.986667 | 0.968333 | 0.990357 | 0.722582 | 500.957766 |
| F3 | C | 0.992024 | 0.987619 | 0.965417 | 0.990952 | 0.927134 | 2051.729821 |
| F3 | G | 0.993393 | 0.992143 | 0.972500 | 0.993095 | 0.916887 | 1521.356039 |
| F4 | C | 0.992202 | 0.987143 | 0.976250 | 0.991667 | 0.600217 | 2866.658575 |
| F4 | G | 0.992440 | 0.987857 | 0.977083 | 0.991190 | 0.382748 | 212.405577 |
| F5 | C | 0.972500 | 0.930476 | 0.876250 | 0.967262 | 0.921065 | 710.533386 |
| F5 | G | 0.975119 | 0.931905 | 0.892917 | 0.970952 | 0.946883 | 683.093323 |
| F6 | C | 0.982083 | 0.966905 | 0.932083 | 0.979881 | 0.836032 | 1653.089189 |
| F6 | G | 0.982262 | 0.966429 | 0.931667 | 0.980595 | 0.789333 | 1338.032012 |

| fold | arm | scene | overall | min-class | min-RX | min-day |
|---:|:---:|:---|---:|---:|---:|---:|
| F1 | C | leo_clear_weak | 0.960536 | 0.932857 | 0.870000 | 0.957072 |
| F1 | C | leo_low_elev_weak | 0.932143 | 0.895000 | 0.806250 | 0.923555 |
| F1 | C | leo_rain_weak | 0.943036 | 0.897143 | 0.845000 | 0.934844 |
| F1 | G | leo_clear_weak | 0.962500 | 0.952143 | 0.866250 | 0.961566 |
| F1 | G | leo_low_elev_weak | 0.940536 | 0.918571 | 0.836250 | 0.939649 |
| F1 | G | leo_rain_weak | 0.942321 | 0.921429 | 0.842500 | 0.939448 |
| F2 | C | leo_clear_weak | 0.869107 | 0.610714 | 0.765000 | 0.861024 |
| F2 | C | leo_low_elev_weak | 0.844286 | 0.560714 | 0.737500 | 0.841624 |
| F2 | C | leo_rain_weak | 0.845000 | 0.555000 | 0.748750 | 0.841503 |
| F2 | G | leo_clear_weak | 0.904464 | 0.738571 | 0.781250 | 0.901776 |
| F2 | G | leo_low_elev_weak | 0.881250 | 0.685714 | 0.757500 | 0.877469 |
| F2 | G | leo_rain_weak | 0.869286 | 0.665000 | 0.766250 | 0.865832 |
| F3 | C | leo_clear_weak | 0.897321 | 0.775000 | 0.777500 | 0.887018 |
| F3 | C | leo_low_elev_weak | 0.869643 | 0.725714 | 0.767500 | 0.854891 |
| F3 | C | leo_rain_weak | 0.870000 | 0.745714 | 0.782500 | 0.864444 |
| F3 | G | leo_clear_weak | 0.901964 | 0.735000 | 0.783750 | 0.893684 |
| F3 | G | leo_low_elev_weak | 0.878036 | 0.680000 | 0.780000 | 0.866356 |
| F3 | G | leo_rain_weak | 0.868036 | 0.666429 | 0.790000 | 0.862994 |
| F4 | C | leo_clear_weak | 0.922143 | 0.835714 | 0.778750 | 0.916106 |
| F4 | C | leo_low_elev_weak | 0.903214 | 0.797857 | 0.750000 | 0.895560 |
| F4 | C | leo_rain_weak | 0.911429 | 0.813571 | 0.788750 | 0.909420 |
| F4 | G | leo_clear_weak | 0.913571 | 0.795000 | 0.772500 | 0.913514 |
| F4 | G | leo_low_elev_weak | 0.894464 | 0.744286 | 0.752500 | 0.890586 |
| F4 | G | leo_rain_weak | 0.903929 | 0.754286 | 0.787500 | 0.898188 |
| F5 | C | leo_clear_weak | 0.728750 | 0.448571 | 0.502500 | 0.721873 |
| F5 | C | leo_low_elev_weak | 0.691429 | 0.408571 | 0.506250 | 0.676673 |
| F5 | C | leo_rain_weak | 0.702143 | 0.456429 | 0.507500 | 0.699964 |
| F5 | G | leo_clear_weak | 0.716607 | 0.417857 | 0.508750 | 0.709993 |
| F5 | G | leo_low_elev_weak | 0.687500 | 0.395714 | 0.496250 | 0.663653 |
| F5 | G | leo_rain_weak | 0.700357 | 0.420714 | 0.507500 | 0.697079 |
| F6 | C | leo_clear_weak | 0.822679 | 0.712857 | 0.617500 | 0.797411 |
| F6 | C | leo_low_elev_weak | 0.791607 | 0.635000 | 0.618750 | 0.775994 |
| F6 | C | leo_rain_weak | 0.799821 | 0.649286 | 0.663750 | 0.773652 |
| F6 | G | leo_clear_weak | 0.825357 | 0.713571 | 0.632500 | 0.796361 |
| F6 | G | leo_low_elev_weak | 0.795893 | 0.647857 | 0.626250 | 0.774900 |
| F6 | G | leo_rain_weak | 0.796429 | 0.657857 | 0.640000 | 0.769725 |

### 非补偿门delta与封存verdict

`cleanΔ`列顺序为`overall/min-class/min-RX/min-day`；scene列顺序为`clear/low-elev/rain`；delta均为scorer原始pp单位。`floor`、`proxy`、`scene-equal`与`fold verdict`仅机械记录sealed gate字段，不作解释。

| fold | cleanΔ overall/min-class/min-RX/min-day (pp) | sceneΔ overall clear/low/rain (pp) | sceneΔ min-class clear/low/rain (pp) | sceneΔ min-RX clear/low/rain (pp) | sceneΔ min-day clear/low/rain (pp) | proxyΔ AUROC (pp) | proxyΔ u_gap | floor | proxy | scene-equal | fold verdict |
|---:|:---|:---|:---|:---|:---|---:|---:|:---:|:---:|:---:|:---:|
| F1 | -0.005952/-0.095238/0.083333/-0.023810 | 0.196429/0.839286/-0.071429 | 1.928571/2.357143/2.428571 | -0.375000/3.000000/-0.250000 | 0.449387/1.609364/0.460340 | 0.079155 | 1015.014416 | True | True | True | True |
| F2 | 0.339286/1.357143/0.041667/0.488095 | 3.535714/3.696429/2.428571 | 12.785714/12.500000/11.000000 | 1.625000/2.000000/1.750000 | 4.075235/3.584492/2.432916 | -0.165271 | 191.427848 | True | False | True | False |
| F3 | 0.136905/0.452381/0.708333/0.214286 | 0.464286/0.839286/-0.196429 | -4.000000/-4.571429/-7.928571 | 0.625000/1.250000/0.750000 | 0.666667/1.146542/-0.144980 | -0.010247 | -530.373782 | False | False | True | False |
| F4 | 0.023810/0.071429/0.083333/-0.047619 | -0.857143/-0.875000/-0.750000 | -4.071429/-5.357143/-5.928571 | -0.625000/0.250000/-0.125000 | -0.259268/-0.497336/-1.123188 | -0.217469 | -2654.252998 | False | False | True | False |
| F5 | 0.261905/0.142857/1.666667/0.369048 | -1.214286/-0.392857/-0.178571 | -3.071429/-1.285714/-3.571429 | 0.625000/-1.000000/0.000000 | -1.187980/-1.301989/-0.288496 | 0.025818 | -27.440063 | False | False | True | False |
| F6 | 0.017857/-0.047619/-0.041667/0.071429 | 0.267857/0.428571/-0.339286 | 0.071429/1.285714/0.857143 | 1.500000/0.750000/-2.375000 | -0.104969/-0.109449/-0.392717 | -0.046699 | -315.057177 | False | False | True | False |

| aggregate passed | global 18-scene equal-overallΔ (pp) | global scene-equal gate | floor limit (pp) | fold verdicts |
|:---:|---:|:---:|---:|:---|
| False | 0.434524 | True | -2.000000 | F1=True，F2=False，F3=False，F4=False，F5=False，F6=False |

以上sealed metrics仅作同row机械封存；不据此自行解释机制、选择arm、调参、停止或作晋级决定。
