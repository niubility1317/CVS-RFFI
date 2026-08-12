# Phase1 CLIC C／G predictor artifacts v2预注册报告

## 状态与目标

- 实验ID：`phase1_clic_predictor_artifacts_20260812_v2`。
- 当前状态：`LOCAL_VERIFIED / FRESH_REVIEW_PENDING / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- v1已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`：6／6fold在0工件前因detached环境的`python -m cvsrffi.phase1_clic_target_leo`解析失败。v2是新run，不覆盖、不恢复、不重试v1。
- v2科学输入、6fold、C／G顺序和输出合同与v1完全相同；唯一release入口修复是C改用release内绝对文件`${CODE_ROOT}/cvsrffi/phase1_clic_target_leo.py`。

## 冻结输入、输出与资源

- 训练：`phase1_clic12_20260812_v5`；clean：`phase1_clic_postfreeze_20260812_v4`；source-LEO：`phase1_clic_source_leo_20260812_v4`；PAIR：`phase1_clic_source_pair_20260812_v3`。
- 输出：`runs/phase1_clic_predictor_artifacts_20260812_v2`；日志：`logs/phase1_clic_predictor_artifacts_20260812_v2`；启动前必须不存在。
- 预期：6份C descriptor、6份C train config、6份G bundle、6份日志和6行PID表。
- 6个CPU fold worker；CUDA禁用；每worker依次C再G。正式入口唯一一次`nohup bash <release launcher>`，retry=`NO`。

## 本地门与停止规则

- v5物理轴／WiSig身份修复已由fresh独立复审`P0=0，P1=0，ALLOW`；postfreeze`138／138`、Task5 core`190／190`。
- G target runtime新增直接可执行性修复：verified bundle在runtime加载时重开、验签并重建一次，后续每条IQ仍严格独立单forward；不缓存target输入／输出、不更新模型／阈值。新RED证明旧实现两行后重建计数由1升至5，修复后保持不变；postfreeze全量`139／139`。
- 启动前要求新launcher`bash -n`、dry-run精确12行（C6＋G6），C命令必须包含release内绝对文件入口且不含`-m`；禁止target／truth／score／role／cache／package参数为0。
- 本地实测：`bash -n`通过；dry-run精确12行（C6＋G6）；C绝对文件入口6行，`-m`入口0行，禁止参数0行。launcher SHA-256=`B8BA95E7E5A7750111033667D1D49B4947905EEE5757E07E66EDF570D35C42D8`。
- 若至少2fold在完整C／G工件前出现同一确定性异常，立即封存系统性技术失败，不重试、不读取性能。
- 成功后仅做工件技术QA：C生产loader重开descriptor；G生产verify重开bundle；核fold／arm／local4／训练配置和所有SHA、zero-fit／update／threshold／selection。不得打开target cache或计算性能。

## 后续

- 12个predictor工件闭合后，立即进入同一target confirmation v2缓存的VALIDATED_ONCE收据、IQ-only package和12臂零适配预测。
- 最终评分必须同时报告三scene target-known DG、unknown拒识和域泛化，并只与训练数据及测试数据配置相同的合法ADV3B02原件比较；无需复用同一个封存目标包字节。
