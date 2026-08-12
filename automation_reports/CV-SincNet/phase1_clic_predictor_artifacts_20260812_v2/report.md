# Phase1 CLIC C／G predictor artifacts v2预注册报告

## 状态与目标

- 实验ID：`phase1_clic_predictor_artifacts_20260812_v2`。
- 当前状态：`LOCAL_VERIFIED / FRESH_REVIEW_REQUIRED / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
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
- 修复后本地实测：顶层绝对文件入口`--help`通过；launcher dry-run精确12行（C6＋G6）；C顶层入口6行、包内直跑0行、`-m`入口0行。窄测试`2／2`通过；launcher SHA-256=`05B66AB77F5C4761DBE217896690E036E91AAA6752F13869E12109A46A79696D`，顶层入口SHA-256=`D24CF3F007260775F05B206BB565EE33A92A52B2A08DC661C7F0DBE567A0F6DB`；待更新后的fresh review重新裁定。
- 若至少2fold在完整C／G工件前出现同一确定性异常，立即封存系统性技术失败，不重试、不读取性能。
- 成功后仅做工件技术QA：C生产loader重开descriptor；G生产verify重开bundle；核fold／arm／local4／训练配置和所有SHA、zero-fit／update／threshold／selection。不得打开target cache或计算性能。

## 后续

- 12个predictor工件闭合后，立即进入同一target confirmation v2缓存的VALIDATED_ONCE收据、IQ-only package和12臂零适配预测。
- 最终评分必须同时报告三scene target-known DG、unknown拒识和域泛化，并只与训练数据及测试数据配置相同的合法ADV3B02原件比较；无需复用同一个封存目标包字节。
