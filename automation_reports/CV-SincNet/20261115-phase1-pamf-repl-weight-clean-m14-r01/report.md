# Tri-SIG-TIME 十类与六源损失权重及梯度诊断

## 本轮固定说明

用户按已曝光 clean 测试指定最大落后种子，按 cross_rx 与 cross_both 样本数加权：两设置均为 2026092701。十类原 Tri 82.591590%、PL 84.199446%；六源原 Tri 85.149306%、PL 86.428472%。这是目标反馈选择过的种子，本轮仅为探索，不能作为独立盲测或跨种子稳定性证明。固定全部候选及其 E50 终点，不利用本次 clean 结果继续调整其他候选、选 checkpoint 或择优重跑。

每设置同源数据契约及同 seed 原完整三视图固定重排、源训练无目标祖先的 E20，独立续训 10+20 轮。原模型和其他 owner 保持；配置／真实执行版本为发布 Git commit。七项逐一变化：总 U 6→3、6→1.5；浅中层对比乘 0.5；融合对比 0.05→0.025；外推 CE 重分配 0.25→0.125、差额回归普通 CE；风格概率 0.5→0.25；U/L 分组梯度范数上限 1→0.5。所有 candidate 均保持 0.95 阈值、原联合默认头和原损失骨架；实际 recipe 逐行记录在配置及登记表。低性能不停止其余行。

2026-11-15 19:33（香港时间）**VERIFIED / RUNNING**：发布代码 `bffc9472e8fe499ec75f2c46078350fddb546d9b`；归档传输和远端编译完成，14 个候选真实 E20 无查询 smoke PASS。独立核实控制器和 14 个训练 worker 的 PID、CWD、完整命令及 GPU，14 份 step 日志实际增长；自动 clean 预测与独立评分 28 阶段已排队，尚未产生测试结果。十类外推 0.125 变体的第 100 步实测记录了 37 个组件梯度字段和正确外推系数。U 真值读取 0；其他 owner 不受干预。

后续 **FAILED / PARTIAL_TECHNICAL_FAILURE**：14 行均完成并保存各自 E30（10 轮、2220 步），但段后源 V 代理的 `torch.inference_mode()` 与原生 `CoordinateSinc` 的动态频率张量版本计数不兼容，报 `Inference tensors do not track version counter`。全部 clean 预测及评分因依赖失败未启动，故无新测试结果。14 份 E30 完整 step 日志及状态已只读核实，原目录保留。修复以 `torch.no_grad()` 完成等价只读评估，并在独立 `r02` 中复用各行自身的 E30，重做 E30 V 代理后续训 E31–E50；不改变科学矩阵。

日志每步记录各参数组 L/U 投影前后范数、点积、冲突、限幅、L CE、源验证、U 接受／伪类别／置信分桶；每100步额外记录浅中层、融合对比、外推增量各组件与 L 的分组范数、点积和余弦。每段 E30/E50 后对全部原 V 只读计算 0.95 高置信预测准确率及逐类分解，不更新模型。V 准确率是有标签源验证代理，不是 U 的真实伪标签准确率；U TX 真值始终隐藏，无法计算真实 U 精度，不以 V 代称。

仅 raw clean：cross_date、cross_rx、cross_both。每行冻结后自动预测，其三份预测固定后独立 scorer 连接 truth；报告跨 RX 与双跨合并及逐 TX/RX/day，同时保留跨日。没有星地信道测试和 DURA 测试；clean 不证明星地性能。单 run 唯一 owner；仅 0/1/3/4/5/7 卡，计入其他进程每卡最多 5，核实空闲显存并预留初始化占用；不操作其他 owner。

- run_id：`20261115-phase1-pamf-repl-weight-clean-m14-r01`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

单种子十四个独立REPL变化；训练后自动raw clean预测与truth-last评分。

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
