# PAMF第三阶段：RX24-1冻结跨接收机clean拒识

- run_id：`20261029-phase3-development-clean-transfer-s2026092701-r01`
- group_id：`pamf-phase3-unseen-rx-clean-transfer`；类别：`diagnostic`；阶段：`Phase3-development-clean-unseen-receiver-frozen-transfer`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Isolated C0 clean864, same frozen source E51OE/MLP/threshold and original F closedbaseline; no DURA or target updates.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## C0 固定执行

actual code/config commit `1e1876db7d0f7b8f3724fe95acf3c8c8d5b64fef`；生产predictor与practical完全相同，仅新增C0配置和3个聚焦测试，3 tests/8 subtests PASS完整日志已保存。一次仅新增C0输入/95%边界P0/P1 PASS，不重复旧代码审查/全套测试。固定source E51 OE/MLP/原knownV阈值，无DURA，两字段null；与前一实用景同源状态，不因实用景结果选模型或改阈值。

只读同一批V864的独立clean public，U/formal/sourceIQ/bank0；C0合成2IQ实际checkpoint smoke先于cleanV，PASS后立即预测。完整C0固定后独立score连接private truth，按clean跨RX95%、known接收最大化/FRR5/原F同输入2pp保护报告。真实clean数据已VALIDATED_ONCE，复用不重验，不把同物理ID的clean/star当更多独立样本。新run/release/root不可覆盖，停止规则和命令见experiment.json；科学失败保留，不调参重跑。

## 实际预测与评分完成（2026-10-29）

已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，科学NO_PROMOTION。864记录全部固定预测后独立truth-last评分，主方法MLP保持，全部source状态与阈值更新0、U/formal读取0。实际checkpoint的2条合成IQ smoke在V读取前PASS，证据保存在完整predict.log和complete，不存在独立smoke.json，登记已改为真实artifact。

首次独立post已终态，未捕获运行期独立PID/CWD/GPU/log增长；CLI首行实际身份和完整日志/markers可用，但不冒充live证据，不为补观察重跑。当前快照其他GPU进程不当作本run；foreignstop0。实际代码`1e1876db7d0f7b8f3724fe95acf3c8c8d5b64fef`，同row完整报告见`docs/PAMF_PHASE3_DEVELOPMENT_CLEAN_TRANSFER_RESULTS_20261029.md`，评分及post见local_artifacts/pamf_phase3_development_clean_transfer_20261029_r01。目标仍ACTIVE，clean95/star90及known保护未同时达成。
