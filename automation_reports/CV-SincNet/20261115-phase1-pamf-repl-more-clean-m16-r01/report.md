# Tri-SIG-TIME 额外权重和机制 clean 对照

## 本批固定矩阵

用户要求多发布实验；十类、六源各用先前按曝光 clean 目标最大负差选择的 seed 2026092701，原始数据契约、固定重排 Tri E20 源 checkpoint、E21–E50 与 E50 原默认头保持。每设置固定 8 个单变量候选：总 U 权重 0.75 / 4.5；浅中层对比乘 0 / 0.25；融合对比权重 0；外推 CE 0 并把系数归还普通 CE；风格概率 0；ordinary/raw/mixed U paired KL 乘 0.5。阈值 0.95、L 目标、其余 U 机制和原梯度保护保持。16 行全部冻结并按预登记自动执行 raw clean 三测试集合与独立 truth-last 评分；不运行 DURA 或星地场景。

每步记录参数组 L/U 范数、点积、保护后 U 范数、冲突和限幅、L CE、源 V、U 接受率与伪类/置信分布；每 100 步采样浅中层对比、融合对比和外推 CE 差与 L 的分组梯度；E30/E50 全源 V 的 0.95 高置信预测准确率仅是源验证代理，U 真 TX 始终隐藏。`style0` 的 smoke 强制运行 hook，不代表正式训练有混合；须以实际训练日志中的 style 应用计数为准。不同 loss 标量和梯度冲突不构成独立因果证明。

只允许 GPU2、6，与正在运行的 r02 owner 的 0/1/3/4/5/7 不重叠，以免独立调度器争用同一张卡而竞态；每卡仍计入其他 CUDA/待初始化进程、最多 5 个并检查可用显存。固定候选和选中的种子均为探索，目标结果不回流调参或重排；健康旧任务和历史产物保持。

- run_id：`20261115-phase1-pamf-repl-more-clean-m16-r01`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

十类、六源各8个额外单因素候选，E20各自续训30轮，训练后自动clean测试。

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
