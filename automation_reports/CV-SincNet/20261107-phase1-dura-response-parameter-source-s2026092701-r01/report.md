# DURA 源微扰响应与参数梯度诊断

- run_id：`20261107-phase1-dura-response-parameter-source-s2026092701-r01`
- group_id：`phase1-dura-response-parameter-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；[源参数梯度结果](analysis.md)。

## 目的与对照

评估源合成微扰响应损失与源 IQ 保护损失对同一修正网络参数的梯度关系。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

在 32 条源记录 × 三视图的一个修正网络初始化上，四种源合成微扰的参数梯度余弦约为 -0.1365 至 +0.2810；未训练模型或访问目标。

## 交接

仅作为源代理继续探索物理响应保护，不代表已有新版本或正式识别收益；既有候选按预登记继续。
