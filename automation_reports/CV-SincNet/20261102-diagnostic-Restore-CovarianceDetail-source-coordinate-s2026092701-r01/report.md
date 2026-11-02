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
