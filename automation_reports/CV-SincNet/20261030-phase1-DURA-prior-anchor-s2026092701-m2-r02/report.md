# DURA两行冻结权重的零重训评估恢复

- run_id：`20261030-phase1-DURA-prior-anchor-s2026092701-m2-r02`
- group_id：`phase1-dura-prior-anchor-performance`；类别：`analysis`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

ReusebothoriginalE20/4940frozenweights;fixreaderconstructioninference_modeversioncounter;0newtraining;fullsource/target7viewsandindependenttruthlast

只读原r01的continuation/prior_anchor E20/4940、各63000×20曝光的final.pt，源checkpoint代码固定`44bb16ed08b29e7e4ee61c21e1e1287d18791c75`。源模型/权重/损失/预算不变，不重训，不移动或覆盖旧产物。原故障发生于postfreeze测速的F Sinc参数版本缓存，尚未进入正式prediction。

修复仅将评估函数外层构造环境从inference_mode改为no_grad，实际InferenceEngine仍FP32/TF32off、graph/reuse。四个精确回归先失败后通过，一次P1定点复审PASS。源权重commit与本恢复执行commit分别记录，不能改标源训练身份。

launcher第一步两条真实Ls与已冻结continuation/原F50 defaulthead无query smoke，随后按已登记资源自动补两行同卡成本、source1800×7和target168000×7，每组七预测完整后独立CPU评分。每行读取对应原sourcecomplete及严格customschema metadata，全部目标结果仅汇报，不回流原训练或单阶段候选。当前goal GPU worker合计最多2，其他健康owner保持。

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
