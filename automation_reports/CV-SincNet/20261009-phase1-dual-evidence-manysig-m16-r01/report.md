# 旧门控＋cosine：保真与曲率双骨干、条件交互四seed研发

- run_id：`20261009-phase1-dual-evidence-manysig-m16-r01`
- group_id：`cvs-dual-complementary-evidence`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在固定强伪标签基线上检验容量、曲率互补输入和条件乘性交互；16个scratch固定对照，全量7视图truth-last测试。

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

## 设计与本地验证

详细设计见[设计说明](design.md)。4组×4seed固定比较，全部scratch E200×222，继承旧batch_neighbor伪标签、全U熵与cosine学习率。

CPU与正式FP32 GPU架构检查、E1/E80/E131/E132/E200原生合成训练集成、16×7完整合成评分闭环均PASS。初始GPU检查脚本的cuDNN TF32默认值与正式策略不一致，改用注册FP32后通过；模型及阈值未改变。证据见evidence。

正式性能尚未知，不能用合成检查或模块启用代替测试收益。主方案预期六视图均值增加至少2个百分点，但这是待证伪目标。

独立审查发现并修正正式incremental writer绕过新增遥测的问题。正式writer链合成验证通过，双骨干梯度、融合和资源字段实际落盘，原模型计算与旧训练配方不变。

## 发布与当前交接

RUNNING / VERIFIED：N607控制器PID2074109，正式训练代码提交a759963aa77150da25bd91fd8c031f7d06120cf4。独立PID/CWD/argv、实际配置、FP32策略、scratch来源及新增遥测核对通过。16行已发起，15行有epoch记录，1行等待GPU7显存余量；所有GPU总进程数3，未超过用户上限4。无训练失败记录。

本地只读观测与结果交付进程PID20076已启动；原PID68880仅因Windows远端路径分隔符问题被替换并核实退出。修正仅涉及本地观测与登记路径，远端发布代码、训练进程和参数均未改变。

后续自动执行：完成16行源训练→全矩阵源侧冻结→112组完整预测→独立truth-last评分→本地逐指标重算与完整报告→登记/Git交付。新机制性能仍未知，未宣称训练或测试完成。恢复时先读本run证据和实时进程，禁止重复启动。

## 2026-10-09 完整测试请求核实

SOURCE COMPLETE / TEST QUEUED / VERIFIED。16 行均完成 E200 和 44400 次成功更新，`source_matrix_frozen.json` 已固定全部对照；16 个 checkpoint 的实际配置、scratch 来源和数据契约预检 VERIFIED，预检未读取 query。原控制器 PID2074109 健康，已经从 source 队列进入 predict 队列，当前 16 行等待、0 行预测完成，无失败。

2026-10-09 16:27 香港时间独立读回：8 张 GPU 每张有 4 个多解耦训练进程，显存空余约 9.5 GB。当前没有低于 4 进程且满足原调度器 12 GB 空余条件的卡，因此测试尚未启动。保持已授权并发限制和健康训练，未重复提交、停止或热修改任何远端任务。

固定测试范围为 4 组×4 seed×7 视图，共 112 组预测，每视图 168000 个固定 query；包括 clean 和六种已登记 practical 代理视图。全部预测完成后，独立 CPU scorer 才连接 truth，输出 1568 条总体/RX/TX 指标及 448 条日期指标，并保留四 seed 统计和同 row 差值。不是仅测源域验证集，也不新增场景或更换 checkpoint。

本地观察器 PID20076 的实际命令行与原 observer.json 已独立核实，继续自动收取、复算和提交完整结果。本次仅完成训练/冻结/测试队列状态核实；测试和评分未完成，准确率暂无。证据见 [测试队列](evidence/test_queue_verified.json) 与 [冻结及预检读回](evidence/test_freeze_readback.json)。

## 立即测试交接

用户随后要求“直接启动”，测试由独立run `20261009-phase1-dual-evidence-test-now-manysig-m16-r02`接管并已运行。原空闲调度器已安全退出，源训练与权重不变。8个预测进程正在执行、8行待第二批；本次临时每卡最多1额外推理、总数5。最终完整结果见[立即测试报告](../20261009-phase1-dual-evidence-test-now-manysig-m16-r02/report.md)。原队列快照为历史状态，不代表仍等待容量。

## 测试已完成

ANALYZED / VERIFIED：独立测试run完成全部112组预测、1568条总体/RX/TX指标及448条日期指标，并完成独立复算。[完整结果](../20261009-phase1-dual-evidence-test-now-manysig-m16-r02/results/complete_summary.md)。之前的QUEUED/RUNNING记录属于历史状态。
