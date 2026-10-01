# GroupBarrier support 三阶段连接修复

本次只修复独立分析器的生产输出连接。support 拟合版本仍为 9518d46d2，query 预测版本仍为 73aa1b31f；未重新拟合、调整参数或改动任何预测。

独立分析 r02/source a3d0791 于 149.303 秒后 FAILED，未生成 summary 或表格；[终态证据](../automation_reports/CV-SincNet/20261001-phase2-d92-group-barrier-joint-support-m2-r01/evidence/analysis_runtime_1790874130510704500.json)和原 release、日志完整保留。错误 `Fixed prediction streams incomplete or duplicated` 的直接原因是原生 A 阶段保存 `CURRENT_LEGAL_OLD_HELD_ONLY_ORIGINAL_SIX_CLASS_COMPETITION`，B/C 保存外层 `support_oof` 或 `support_oneshot_proxy`；旧分析器把不同 scope 当作不同分组，因而拆开同一组 A/B/C。生产 probe 源码从真实拟合 commit 9518 到 a3d 没有变化。四 row 的脱敏 schema 盘点各有 A 440 条、四种 B/C 各有 OOF 90 条与 proxy 350 条；盘点未输出物理 ID、预测、score 或准确率。

限定修复只解释 stream A 的这一生产原生 scope，保留其原始标识，按生产 fold/trial 含义恢复外层分组。全部三阶段仍必须同 run、row、split、parent K、train K，并在各路径上闭合完整旧类 held 物理 ID；A 和所有 B/C 都必须存在且唯一。缺失、重复、跨 parent 或身份不一致不能放行。生产 probe 明确把同一个冻结地面 A 写入 R0 和 R_GROUP_BARRIER_seq 各自证据；如实读取该记录，不套用历史 Margin 报告的 R0 A 缺失，也不从其他实验补值。

root 用已核实 ssr-gpu 原生环境执行受影响分析器全文件：39 项合成 case 通过，原始证据 `pytest_native_activation_1790875265727460700`。生产集成测试真实调用 probe、Native GroundClassifierA、StateArchive 与 prediction_callback，使用六旧类、两新类、K=2 的合法合成输入；无真实产物或评分参与。测试同时保留既有完整/紧凑引用格式检查和失败负测。未据合成测试宣称真实分析已成功。

修复提交、push 并独立核对远端 OID 后，在原 run 下用新 release/output 执行预登记 r03 独立只读分析。保留 r01 与 r02 的失败产物，不重复旧 publisher，不干预健康 query。完整四 row/160 parent 支撑结果闭合后才报告 OOF 与单样本 proxy；query 仍待完整四 row/2400 parent 固定后独立 truth-last 评分。分析结果不回流方法或参数选择，goal 仍 ACTIVE。
