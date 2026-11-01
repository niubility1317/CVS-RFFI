# 最终双分支恢复器与默认DURA：同原识别器实测延迟

- run_id：`20261102-diagnostic-Restore-CovarianceDetail-final-latency-s2026092701-r01`
- group_id：`restoration-final-CovarianceDetail-vs-defaultDURA-GF0-latency`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

父实验完整Main结束后，固定最终G30与同原TriSIGSSL F0，B1/B128比较rawF、DURA G、新G、DURA G+F、新G+F的同步wall/CUDAevent和增量显存。源only、0训练、0目标访问。

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


本项只补充最终冻结恢复器的实测计算成本，排在父实验完整Main七景预测及独立评分全部结束之后。开始条件只看技术完成，不看父成绩；父训练与测试保持原样。固定1fcb最终30轮G、原3ff8完整地面Tri-SIG-TIME SSL和默认47c264/R69 DURA，比较raw F、DURA G、新G、DURA G+F、新G+F，全部冻结、0更新，不读TX/V/OE/novel/target。

源输入仅为R07 draw1固定512面板（128 Ls＋384 Us）。B1取25个预定位置重复4次，B128取4个128块重复5次；每条路径每种batch先3次预热，随后B1各100次、B128各20次。按输入病例和路径均衡交错次序，所有timing保留，不挑最好一轮。共600次计时、30次预热，另5路径source2检查；每路径逻辑3049条，新G6098、DURA逻辑6098、F0实际9147。DURA图首次capture的额外内部前向未instrument，启动成本不称为0。

同步wall与CUDA event分别记录。模型加载、输入IO/上传、病例选择、首次图capture/预热及输出有限性检查在计时外；实际部署内部检查与DURA图输入拷贝在计时内。增量显存从当前路径前的resident基线算起；三模型、512输入及所有graph均驻留，不能把总allocated当作单模型独立进程显存。GPU型号、Torch/CUDA和实际执行路径一并报告；不是专卡测量，不将MAC换算为时延。

4项针对性检查及唯一P0/P1审查通过。真实source2全部路径是worker第一步，PASS继续全部计时；一次release archive SHA/远端compile，父CPU观察实际子PID/CWD/命令/CUDA/GPU和日志增长。权重和buffer逐值不变，无新训练/分类头/选模/恢复性能评分。旧DURA＋F50的4.531972ms不属于本版；本次会测同一原Tri-SIG-TIME SSL下的实际组合。默认DURA不变。

## 实际 CPU 等待队列

VERIFIED_CPU_QUEUE_ONLY：controller 1065879 的实际PID/CWD/命令及CUDA空环境已核实，尚未新增GPU worker，父Main仍未完成。archive SHA与远端compile通过，实际代码 `33cb43c2d4d387ab18498b451ddc1444070e997c`。真实最终checkpoint source2与全部测量仍待父实验完成，当前无时延结果。发布前重复进程扫描仅针对本延迟module；旧调用器状态文字NO_LIVE_OWN_GPU_WORKERS不能解释为父训练进程不存在，父GPU训练仍健康继续。
