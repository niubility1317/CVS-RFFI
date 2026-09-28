# D92-MVKME冻结方法rx3透明重复基准

- run_id：`20260928-phase2-d92-mvkme-repeat-rx3-m4-r02`
- group_id：`d92-fixed-phase1-mvkme-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1与已验证received IQ及support/query；同一IQ的固定16视图经冻结编码器和核均值表征，仅用当前row support训练岭回归头；复用原D92预测。 技术恢复：修复既有Torch/NumPy双数组ABI的数值转换接口，使用Python数值列表中转；公式、精度、矩阵和预算不变。

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

Fixed original formula and data; one numeric tensor bridge technical repair after terminal rx3-r01. New rx3-r02 plus unlaunched rx1-r01; no target score read.

Recovered tensor bridge VERIFIED on real received export; keep run active, do not relaunch. No candidate scores have been read.

Both fixed cohorts launched under release70bd88dfc6a0864e09b8fa582d2aa48b34a741fb; one serial exporter per cohort, at most two concurrent frozen inference processes on GPU0. Do not inspect scores until both terminal.


<!-- MVKME_FINAL_ANALYSIS_20260929 -->

## 最终结果、核验与成本

当前结果为 SCORED / ANALYZED，证据状态 VERIFIED。上文启动或等待说明保留为历史记录。实际发布 commit：`70bd88dfc6a0864e09b8fa582d2aa48b34a741fb`。本批次 7236 条评分完整，混淆矩阵算术与拟合日志独立审计均通过。D92-MVKME-v1 未晋级，整体优化目标尚未完成；不得用单一批次或单元替代联合验收。

下表为候选减 D92，单位为百分点。前三列只含新类数大于 0 的任务，最后一列含 old-only，按预登记用于旧类保护条件。

| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部任务） |
|---|---:|---:|---:|---:|
| 1 | +1.586 | -1.013 | -0.220 | +1.867 |
| 5 | +6.599 | -1.147 | +0.896 | +6.580 |
| 10 | +3.159 | -1.669 | -0.556 | +2.715 |
| 20 | +0.744 | -1.176 | -0.805 | +0.133 |

全部 3600 次拟合中，K1 固定配置 900 次，support 内部交叉验证 2700 次。累计特征提取 461.034 秒，核心拟合 277.193 秒，query 打分 57.260 秒，预测序列化/写出 16.570 秒；任务总计时 358.838 秒另含函数调用与日志开销。各阶段计时不等于并行墙钟时间或星载硬件速度，不叠加已经包含的拟合计时。采用闭式岭回归与 support 交叉验证，没有梯度训练或迭代收敛结论。

每条物理观察执行 16 个固定视图，仍只计一个样本。分类头为 36864 至 159744 字节，固定 Fourier 矩阵为 262144 字节，两者数值状态合计 299008 至 421888 字节。Gram 临时矩阵、接收端派生 cache 和进程 RSS 单独记录在 artifacts.json，不混称持久状态或地面传输。

既有模型文件为 15,992,872 至 15,992,936 字节（约 15.99 MB），这是训练 checkpoint 文件包，不是经裁剪核实的最小推理包。模型是否已部署未知，新增模型传输字节记为 null；本轮新增源域 payload 为 0 字节，不读取地面摘要。query/source 拟合行数均为 0，Phase1 冻结。

四个 RX 按真实单元等权，rx3∶rx1 为 3∶1。复用目标此前已经评分，不是新增独立确认，也不支持未接触目标的泛化声明。结果不会反馈本候选选参或选择性重跑；所有失败结果与原始产物保留。

详见[产物索引](results/artifacts.json)、[本批次汇总](results/summary/report.md)、[拟合审计](results/fit_audit.json)。

Both cohorts terminal and complete9648-score/4800-fit analysis archived; no promotion and no relaunch. Figures visually verified.
