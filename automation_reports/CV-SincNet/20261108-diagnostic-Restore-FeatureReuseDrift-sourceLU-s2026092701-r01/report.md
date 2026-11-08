# 生产恢复器特征复用：单步数值差异定位

- run_id：`20261108-diagnostic-Restore-FeatureReuseDrift-sourceLU-s2026092701-r01`
- group_id：`source-production-G-F0-feature-reuse-drift`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原生产G/F0 CPU复核未通过预设AdamW更新容差；同固定源L/U样本逐层记录误差及裁剪前后梯度，验证差异位置和可能原因，不修改训练。

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

## 固定的差异定位矩阵

前一生产 G 检查在原 rtol 1e-5 / atol 1e-6 的局部 AdamW 更新处失败，最大绝对差 4.3734908e-6；原日志保留，不以放宽容差宣布通过。本诊断固定原 source-only JointPrior E30、原冻结 F0、L_s/U_s 索引 0/6301 及 odd 轮次 1/3、全源先验、同一源模拟器 trace。仍只在 CPU2 nice10/CUDA empty 的独立 G 副本上比较，每个计算图一步新建 AdamW，不修改运行中或磁盘上的 G。

只新增逐层原容差超限计数、最大更新差的参数名及该坐标裁剪前／后梯度，并计算新建 AdamW 首步 epsilon 公式所预测的更新差用于诊断。原容差结果必须按实记录；MEASURED 或产物完整不表示等价通过。源/F0/G 身份不符、非有限值或后验残差 >1e-8 即停止。目标数据、TX/V/OE 均不读取。代码 commit 0b0c702e3ead128f2a1b7134352a874c9ca625f5；命令、路径、预期产物见 experiment.json。核心缓存验证 3 项通过，独立 P0/P1 审查 PASS；GPU、实际历史 AdamW 状态、全训练质量与真实加速均未覆盖。

## 差异定位完成

VERIFIED / MEASURED：两组损失及缓存目标特征一致，139 个有梯度 G 参数张量均通过原梯度容差；新建 AdamW 一步后分别有 16、18 个参数元素超限，最大差分别 8.827075e-6、1.316797e-5。最大差位置的首步 AdamW 公式预估与实际吻合，支持小梯度被归一化放大的解释；原数值等价判定仍失败。见 [analysis.md](analysis.md)。
