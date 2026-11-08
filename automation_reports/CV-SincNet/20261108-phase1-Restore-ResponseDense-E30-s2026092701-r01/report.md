# 源响应保持恢复器：原 dense 后验从零单阶段 30 轮

- run_id：`20261108-phase1-Restore-ResponseDense-E30-s2026092701-r01`
- group_id：`source-response-dense-posterior-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留响应保持版架构、源损失与训练预算；源训练后验从低秩改为原 dense FP64，以修复源微批残差超阈的技术故障。

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
# 实际启动

VERIFIED / RUNNING。N607 GPU 6 启动前已有 3 个计算进程、可用显存 19547 MiB；本任务增加 1 个训练进程，处于用户授权的总数不超过 5 个的范围。已独立核对进程、实际命令、工作目录、GPU 和日志增长。真实源样本与原冻结 Tri-SIG-TIME SSL 检查通过，均值与协方差头的响应梯度、物理分支隔离、训练／部署 dense 后验一致性和 E0 严格重载均通过。未读取 TX 标签、V、OE 或目标输入。

原失败运行不加载；本版从零单阶段 30 轮，完整 Main 两总体七视图预测完成后独立评分。当前尚无最终恢复性能或验收结论。完整实际启动证据位于远端本 run 的 `runtime_observation.json`。

## 完整30轮与原Main七景独立复算完成

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.29e-14，baseline判决差0。六景83.902976/82.429762/80.725000/82.444048/74.413095/65.862500%，1/6；clean两个总体净+149/-5，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整结果](analysis.md)
