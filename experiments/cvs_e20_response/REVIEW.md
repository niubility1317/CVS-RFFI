# E20 响应身份监督：独立发布审查

结论：**PASS；剩余 P0/P1 = 0。**

审查日期：2026-10-12。审查者：独立子代理 `/root/stage2_release_review`。对应运行：`20261012-phase1-e20-response-supervision-manysig-m21-r01`。

## 范围与边界

对照 `analysis/finite_response_identity_next.md`、`analysis/e20_response_traceability.md`、本包实现及预登记的 21 个实际配置，检查来源、数据角色、目标执行、完整预算、冻结预测、独立评分、报告收尾、发布依赖、容量和输出保护。只修改本文件；未修改实现或登记，未访问真实目标数据、目标评分或远端，未使用 GPU、提交代码或启动实验。

## 关键结论

- 7 个固定 arm × 3 seed，各自从对应的无显式 CFO 纯 CE E20 父权重独立续训 20 epoch/4440 次更新，总轮次 40；严格加载父模型，重新建立优化器。预测严格核验自身 final40 checkpoint 的配置、预算和完整祖先，只加载自身状态。
- 有限作用网络固定继承 source-only finite3 的执行版本与对应 seed，验证原 E20 参考、完整诊断预算、PID80/20 拟合/留出分区，以及当前 L/U 与缓存的物理 ID 对齐。拟合统计和目标固定；U 不允许 TX 真值。该审查未将已见源 TX/RX 的诊断解释成未知域泛化或真实 RX 的因果逆变换。
- 四个响应任务使用实际浅层/中层均值与总体标准差。D 的参数、统计和响应目标冻结，学生输入保持真实可微路径。各角色先对四任务求均值，再对启用角色求均值；没有重复扩大角色权重。评估态前向恢复逐模块模式及 RNG，并核验缓冲不变。
- CONT_CE 实际关闭 U、peer、D 和附加前向；其他 arm 按登记启用额外 eval、L/U 响应或真实有限 IQ 操作。所有 arm 无 EMA、星地增强、辅助优化器和目标反馈。V 仅健康验证，固定 final，不参与选模。
- 逐 row 冻结后自动预测 clean，再由独立评分阶段连接 truth。完整矩阵要求 21 行、每行 168000 个查询。控制器使用唯一 owner、全局容量锁及预 CUDA reservation，每卡最多 3 个总进程，拒绝覆盖既有输出；失败或 UNKNOWN 不静默重启其他任务。

## 发现并关闭的 P1

| 问题 | 修复与定点核验 |
|---|---|
| `final_report.analyze` 将 JSON 的 `None` 转成字符串，与 CSV 的空字段不同，正常训练日志会使收尾失败 | 比较时将 `None` 映射为 `''`；完整 21 行合成日志含正常空 validation/gradient 字段，实际 analyzer 和 finish 均通过 |
| 发布器继承 `--remote-preflight`，但新预检入口不接收该参数 | 入口现显式接收该 flag；实际 configure 后 REMOTE 的模块与参数一致 |
| `evaluate` 的 CLI 位于辅助函数定义之前，模块方式执行全批 score 会遇到未定义函数 | CLI 移到文件末尾；读回确认位于全部辅助函数之后，实际评分函数完成逐行与全批合成核验 |

## 验证证据

1. **实际 CPU 源入口。** 用合成 L 和外部父加载夹具执行真实 `source.train` 到首个真实原生 CE objective：128 个 L 样本，有限 CE，U/peer/response 构造设置为失败哨兵且均未触发。在第一次 optimizer 更新之前有意终止，身份更新数为 0。仅外部数据读取和父加载使用夹具；实际原生前向和 CE 目标未 mock。该验证不冒充真实远端父权重验证。
2. **完整源日志。** 实际 analyzer 消费 21 行共 93240 条逐步 JSONL，以及 420 个 epoch 的 JSONL/CSV；检查完整计数和正常空字段。`SOURCE_COMPLETE` 后故意放置异常文本，验证只读取闭合的训练日志范围。
3. **全查询评分与收尾。** 在临时目录构造符合 6 TX × 7 RX × 4 day × 1000 的 168000 个合成物理查询，实际执行自身和父参照的来源校验、预测校验、单 row 评分、全 21 行评分以及 `finish`，未 mock 这些算法。得到 3528000 个预测、294 个主分层记录、84 个 day 记录、42 个父参照记录、18 组配对和 8 个 arm 汇总；逐项正确率与预设合成答案一致。缺少预测 complete 时在读取 truth 前拒绝。报告、全源分析、原登记 21 行终态及事件记录全部闭合。合成结果不是实验成绩。
4. **发布与登记。** 21 个登记配置与 `design.config` 完全相等。configure 后 67 个归档路径均存在；禁止导入未打包 experiment 包的隔离检查中，source/response/objective/preflight/evaluate/final_report/dispatch 七个入口均可导入。REMOTE 绑定新包的预检与调度模块，保留正确预检 flag。
5. **已有专项证据读回。** 本 run 的 `evidence/e20_response_cuda.json` 为 PASS：7 arm 原生有限前反向、响应项到身份的非零梯度、D 冻结、eval 模式/缓冲/RNG 保护、U 真值拒绝及 21 行严格自身 checkpoint 检查。没有重复执行 GPU 检查。

最终预检遥测的小修订已读回：主 L/U batch 各 16 与实际前向包数分开记录，RESP_LU 的 L/U 前向包数分别为 32，模型前向次数为 4；仅字段口径澄清，目标与预算未改变。

本结论限于上述发布正确性与数据权限；不代表已发布、已完成真实训练、已完成真实测试或已达到性能目标。真实远端预检、发布与运行仍由唯一主 launch owner 执行。
