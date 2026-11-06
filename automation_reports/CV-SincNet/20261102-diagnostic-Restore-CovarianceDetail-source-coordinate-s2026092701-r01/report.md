# 冻结恢复器：源端I/Q坐标响应诊断

- run_id：`20261102-diagnostic-Restore-CovarianceDetail-source-coordinate-s2026092701-r01`
- group_id：`frozen-restoration-source-coordinate-response`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定final30/1fcb G与原3ff8 F0，源V每RX/day前32条共480，四个季度旋转输入后把输出反旋回原坐标。角度0复用f29缓存，只新增其余三个方向，不训练/融合/择角或读目标。

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


本项只检验同一冻结G的源端坐标响应，不训练新G、不新增Main读出或部署版本。固定1fcb最终30/7410/5481223与原3ff8完整Tri-SIG-TIME SSL。原f29源V每RX/day的原顺序前32条，共480；不读取TX或使用预测分数选择记录。七景固定四个季度旋转：先旋转received给G，再把恢复IQ逆变换回原坐标，之后才给F0并与原clean比较。保留F0相位敏感性，不假设其相位不变，也不融合或选择角度。

角度0的3360条原G/F0输出及480cleanF0参考逐值复用f29缓存。另三角各3360条，共10080新G/F0；source Ls2×四角检查另8条G/F0，为首个真实NN步骤，PASS立即继续全部矩阵。共13440源V读出、28单元、672个总体/RX/day/RX×day组，全部保存后独立CPU复算，无TX标签或目标访问。完整G/F0状态保持，失败保留且不自动重试。

5项聚焦检查和唯一独立P0/P1审查通过。只用GPU4/6，启动核实剩余显存≥12000MiB、已有compute≤2，新增1使总数≤3；一个本任务NN进程，不影响其他owner。一个Git代码/config身份、一次archive SHA和远端compile，实际source2及独立PID/CWD/cmd/CUDA/GPU/log增长核实；不新增签名、seal或数据重验。原数据句柄VALIDATED_ONCE沿用。

源V为已见五RX/三日期、历史曝光面板，只验证不选模。IQ/特征误差、Fclean判决一致率与真实TX准确率分开，源坐标误差不能解释成Main原因或信息上界。原Main六景3pp＋两个clean保持仍是性能主目标，默认DURA保持。

## IQ 坐标诊断已完成

480 条固定源 V 记录覆盖七景和四个 I/Q 坐标方向。恢复输出映回原坐标后交给原 Tri-SIG-TIME SSL。零度 3,360 条 G/F0 判决与 480 条 clean F0 参考复用原产物，新增 10,080 条 G/F0 输出，另有 8 条真实源 L_s 冒烟输出。独立复算覆盖 28 单元与 672 分组，指标差与原缓存变化均为 0。

旋转 180° 后，clean 的 IQ NMSE 从 0.000028 增至 0.188923，低档城区从 0.034477 增至 3.715660。低档城区的原 clean 判决一致率从 91.875% 降到 75.000%。这是同一源波形在可逆坐标变换下的实测差异。判决一致率以 F0 在原 clean 上的输出为参照，真实 TX 未读取。Main 差距的成因与理论信息上界仍待独立证据。

下一版将让物理算子和 IQ 细节路径满足四个直角旋转的等变关系。物理分支在同一坐标合并算子，求解一次高斯后验，再用共享细节分支修复 IQ。按原 L_s/U_s 从零单阶段训练 30 轮，并用同一冻结 Tri-SIG-TIME SSL 测 Main 七景。

[完整诊断](analysis.md) · [启动与退出核实](launch_evidence.json)
