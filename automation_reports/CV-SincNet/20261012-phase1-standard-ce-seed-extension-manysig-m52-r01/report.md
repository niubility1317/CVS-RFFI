# 十三种网络纯CE E200追加四种子：CVS为残差融合版本

- run_id：`20261012-phase1-standard-ce-seed-extension-manysig-m52-r01`
- group_id：`phase1-standard-fullwidth-ce-baselines`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Matched historical E200 seed extension; 12 common architectures and residual-fusion CVS; native CVS excluded by user.

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

## 固定发布计划与验证

用户于 2026-10-12 确认其余 13 种网络各追加 4 个模型种子，取消原 CVS。CVS 默认为 residual_fusion。新增模型种子为 2026092705/06/07/08，原 01 至 04 的已完成分数仅用于最终统计合并，不用于模型选择。

每行从零训练 E200，共 10000 次更新，固定末轮权重；数据、架构、优化器和其余配置与原 E200 计划相同。固定 split seed392005，L_s/U_s/V=6300/56700/27000；仅 L_s 身份 CE，无额外损失、训练增强、U_s 或教师。沿用原 clean 测试 capsule 和物理包，不重新划分。每行完成即预测 168000 个相同包，预测完整后独立 truth-last 评分。当前已暴露历史代理基准，不声明新的独立盲测。

唯一 launch owner 为 experiment.json 中的 owner；精确发布命令、环境、CWD、run/log/output 路径与逐行矩阵见该记录。GPU 沿用最多 2 个总 worker、至少 12 GB 空闲的保守队列，计入其他 owner。新旧输出不覆盖。技术停止规则：任意两行技术失败后暂停尚未启动的源训练，已健康运行的行继续完成并评分；低分不触发停止，不自动重跑。

本候选唯一独立 P0/P1 审查发现合并前过早 ANALYZED 的问题，已修复为 MERGING_EIGHT_SEEDS → 成功 ANALYZED／失败 FAILED_COMBINE；针对性复核 PASS。7 项相关协议与回归测试通过。launcher 首步执行真实 checkpoint 的 synthetic no-query smoke，PASS 后直接启动。N607 普通账户只读 preflight 于本次任务通过；发布使用一份 release archive、一次本地/远端 SHA 对比及远端 compile。

代码/config 身份为本次 Git 提交（remote release_commit.txt），后续启动证据记录实际 OID。预期新增 52 行各有 source、prediction、scores 产物。仅在新矩阵完整评分且旧产物匹配时生成 combined_eight_seed.json/csv、combined_per_seed.csv、combined_paired.csv；包含完整 8 种子均值、样本标准差、最差 RX 和 CVS 配对差，保留负结果与退化种子。未完成时不得报告 8 种子结果。
