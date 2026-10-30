# DURA 已知真实信道与独立源高斯先验恢复参照

- run_id：`20261031-diagnostic-DURA-known-action-Gaussian-s2026092701-r01`
- group_id：`phase1-dura-known-action-independent-prior`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

source62488拟合单一Gaussian二阶先验，排除固定512；cached01/20使用真实sourceaction和widely-linear插件噪声估计，与既有G输出配对；非部署/非上界。

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


## 最小研究说明

当前E30健康训练保持。本诊断只在源训练池，独立62488拟合一次全局高斯二阶先验，排除512固定probe；对原缓存01/20的512样本利用真实模拟action和插件噪声协方差求解一个固定参照，与旧noVar/default输出配对。没有target/V/TX读取、G加载或训练，不将源教师一致率称真实TX准确率。信号非高斯且AGC依赖实际噪声，故已知action＋高斯先验参照并不是真实条件Bayes最优，也不是准确率/误差上界；不从本诊断自动触发新训练。源码、矩阵、精确命令、环境/CWD/路径/CPU和技术停止规则见登记；现仅待发布。
