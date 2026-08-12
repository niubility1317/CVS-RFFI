# Phase1 CLIC C／G predictor artifacts v2预注册报告

## 状态与目标

- 实验ID：`phase1_clic_predictor_artifacts_20260812_v2`。
- 当前状态：`LOCAL_VERIFIED / REVIEW_ALLOW / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- v1已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`：6／6fold在0工件前因detached环境的`python -m cvsrffi.phase1_clic_target_leo`解析失败。v2是新run，不覆盖、不恢复、不重试v1。
- v2科学输入、6fold、C／G顺序和输出合同与v1完全相同。首轮fresh review复现包内文件直跑会让`cvsrffi/logging.py`遮蔽Python标准库`logging`；现将C入口最小改为release顶层绝对文件`${CODE_ROOT}/phase1_clic_target_leo_cli.py`，该薄入口只调用同一个`cvsrffi.phase1_clic_target_leo.main()`，不改变方法、数据、状态或输出。

## 冻结输入、输出与资源

- 训练：`phase1_clic12_20260812_v5`；clean：`phase1_clic_postfreeze_20260812_v4`；source-LEO：`phase1_clic_source_leo_20260812_v4`；PAIR：`phase1_clic_source_pair_20260812_v3`。
- 输出：`runs/phase1_clic_predictor_artifacts_20260812_v2`；日志：`logs/phase1_clic_predictor_artifacts_20260812_v2`；启动前必须不存在。
- 预期：6份C descriptor、6份C train config、6份G bundle、6份日志和6行PID表。
- 6个CPU fold worker；CUDA禁用；每worker依次C再G。正式入口唯一一次`nohup bash <release launcher>`，retry=`NO`。

## 本地门与停止规则

- v5物理轴／WiSig身份修复已由fresh独立复审`P0=0，P1=0，ALLOW`；postfreeze`138／138`、Task5 core`190／190`。
- G target runtime新增直接可执行性修复：verified bundle在runtime加载时重开、验签并重建一次，后续每条IQ仍严格独立单forward；不缓存target输入／输出、不更新模型／阈值。新RED证明旧实现两行后重建计数由1升至5，修复后保持不变；postfreeze全量`139／139`。
- 启动前要求新launcher`bash -n`、dry-run精确12行（C6＋G6），C命令必须包含release顶层绝对文件入口且不含`-m`／包内直跑；禁止target／truth／score／role／cache／package参数为0。
- 首轮fresh review结论为`P0=0，P1=1，NO-GO`，唯一P1即包内文件的标准库遮蔽；其他G runtime缓存、TOCTOU、逐row独立与矩阵合同均通过。
- 修复后本地实测：顶层绝对文件入口`--help`通过；launcher dry-run精确12行（C6＋G6）；C顶层入口6行、包内直跑0行、`-m`入口0行。窄测试`2／2`通过；launcher SHA-256=`05B66AB77F5C4761DBE217896690E036E91AAA6752F13869E12109A46A79696D`，顶层入口SHA-256=`D24CF3F007260775F05B206BB565EE33A92A52B2A08DC661C7F0DBE567A0F6DB`。
- fresh复审commit`2e81d54e`结论：`P0=0，P1=0，ALLOW`。复审真实执行顶层入口`--help`、专用测试`2／2`、`bash -n`与12行dry-run；确认标准库遮蔽已消失、科学输入未变化、G runtime生产blob未漂移。
- 若至少2fold在完整C／G工件前出现同一确定性异常，立即封存系统性技术失败，不重试、不读取性能。
- 成功后仅做工件技术QA：C生产loader重开descriptor；G生产verify重开bundle；核fold／arm／local4／训练配置和所有SHA、zero-fit／update／threshold／selection。不得打开target cache或计算性能。

## 后续

- 12个predictor工件闭合后，立即进入同一target confirmation v2缓存的VALIDATED_ONCE收据、IQ-only package和12臂零适配预测。
- 最终评分必须同时报告三scene target-known DG、unknown拒识和域泛化，并只与训练数据及测试数据配置相同的合法ADV3B02原件比较；无需复用同一个封存目标包字节。

## N607落地、静态门、唯一启动与失败封存

- 冻结commit：`a837bcced4d9aee4aa70a3a8b4ab1e897f6cbf12`；本地dirty仅既有untracked target config/launcher/test及`conversation_index/`，均未纳入archive或修改。
- `git archive`严格取冻结commit，未本地解包：`clean_commit_215927227.tar`，267581440bytes，SHA256=`1F11E1C0FED70F707383EBA8738EB5EAB584C85ADFF4563F66BB7C813972C8C1`。SCP恰1次到项目根`clean_commit_predictor_v2_215927227.tar`，远端bytes/SHA闭合；全新stage解压后原子改名为`/home/szu2070436088/2510044040/releases/phase1_clic_predictor_artifacts_20260812_v2_a837bcce`，v2run/log/outer启动前均ABSENT。
- 远端release物理SHA：launcher=`05b66ab77f5c4761dbe217896690e036e91aaa6752f13869e12109a46a79696d`（冻结值闭合）；顶层C入口raw=`9416ef19826cfe5ae622ad69cefbafe9e4eac9db6b2983fccc7f640c55ffe64a`、CRLF归一化后=`d24cf3f007260775f05b206bb565ee33a92a52b2a08dc661c7f0dbe567a0f6db`（canonical冻结SHA）；G entry raw=`373eb769c8c1368b537723af7b50a61ad4fa6b2382ff2fef0ec28ec527cd3bdc`、归一化后=`279805019afe5300bdb5a858d8a1286fee3ff1709471b04f5429f845ecd5a113`。C/G`py_compile`、顶层C/G`--help`、`bash -n`均PASS；dry-run恰12行（C6/G6），禁止target/truth/score/role/cache/package参数flag=0。
- 顶层C真实F1输入只读validation PASS：重开F1C checkpoint/terminal/PAIR，source-only=true、target_artifacts_present=false，source class order=`20-15,20-19,6-15,8-20`；未写任何输出。
- 唯一正式命令于2026-08-12T22:01:20执行：
  `nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_predictor_artifacts_20260812_v2_a837bcce/code/scripts/launch_phase1_clic_predictor_artifacts12_v2_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_predictor_artifacts_20260812_v2_outer.out 2>&1 &`
  - `FORMAL_INVOCATION=1`、`RETRY=NO`；outerPID=`2756964`，foldPID=`2756969,2756970,2756973,2756975,2756977,2756979`，记录于`logs/phase1_clic_predictor_artifacts_20260812_v2/pids_predictor_artifacts6.tsv`。

## 系统性技术失败与只读根因证据

- 6/6fold先成功写出C descriptor及同名C train config，随后在G bundle真实模型重建前发生完全相同确定性异常：`ValueError: sample_rate too low or min_band_hz too large.`（release`code/model.py:215`，调用链`export_phase1_clic_deployment_bundle.py:_rebuild_real_model:593`→`_source_rule_from_clean_and_leo:1203`→`export_bundle:1304`），外层统一`CLICBundleError: strict real CLIC model reconstruction/state load failed`。满足至少2fold在完整C/G工件前同异常的系统性停止规则，不重试、不修代码。
- F1G真实checkpoint只读args根因字段：`sample_rate_hz=0.0`、`dataset='wisig'`、`wisig_out_len=256`；`input_len`、`sample_rate`、`min_band_hz`字段均不存在（未读性能）。该证据仅记录输入配置与构造异常的技术关系，不作科学/性能解释。
- 保留证据：6C descriptor、6C train config已生成；G bundle=0；6日志各3280bytes，SHA分别为：F1`30b72dd34d79174d4d881a7e50dcb81d18ad13703e4963b824a65231bebb5628`，F2`0c97eafba2b8a468357782cd6871400fc825f1432623d54c86fc5f1d786979d6`，F3`bab51fdbffa5f3ef06e0d6d69f68840f2af47385fb89cdc65be41a957a521927`，F4`0009de3920bd2ea14fb9adf34693e935ff7485bdc8169c1dace9ae2d089f2387`，F5`2aafaf5c2098fb33facfcf0289dbb5f5ba92fe4c45f56e421ad45bcf3c7b5378`，F6`7e307a67f70e844608163e3189be325cf26367f3e7270b99f29db6e9c2658b21`。outer=0bytes、PID rows=6、技术异常扫描12（6 Traceback+6 ValueError/CLICBundleError chain），无GPU进程，所有PID已退出。

| fold | C descriptor | C train config | G bundle | log bytes/SHA | technical verdict |
|---|---:|---:|---:|---|---|
| F1 | 1 | 1 | 0 | 3280；`30b72dd34d79174d4d881a7e50dcb81d18ad13703e4963b824a65231bebb5628` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F2 | 1 | 1 | 0 | 3280；`0c97eafba2b8a468357782cd6871400fc825f1432623d54c86fc5f1d786979d6` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F3 | 1 | 1 | 0 | 3280；`bab51fdbffa5f3ef06e0d6d69f68840f2af47385fb89cdc65be41a957a521927` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F4 | 1 | 1 | 0 | 3280；`0009de3920bd2ea14fb9adf34693e935ff7485bdc8169c1dace9ae2d089f2387` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F5 | 1 | 1 | 0 | 3280；`2aafaf5c2098fb33facfcf0289dbb5f5ba92fe4c45f56e421ad45bcf3c7b5378` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F6 | 1 | 1 | 0 | 3280；`7e307a67f70e844608163e3189be325cf26367f3e7270b99f29db6e9c2658b21` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。v2不重试；后续任何修复必须由主控另行本地验证、独立复审并分配fresh run ID。
