# PAMF-SUP source enrolled classification by path

- run_id：`20261119-diagnostic-satiq-pamf-sup-enrolled-head-s2026111907-r02`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Decompose native 160, extra full-rate 160 and fused 320 using fixed labelled L prototypes on known source V. Four matched checkpoints at 5000 and 20000 steps.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

在 46,785 个有标签源包上建立 3,103 个波束原型，均值和部分白化由 8,192 个源训练包确定。固定 49,814 个已知源验证包作为查询。下表是部分白化后的同星真假对 AUC，匹配对象为 L 原型，因此不能与使用 V 参考的 K16 指标混用。

| 模型与步数 | 原生 160 维 | 新增全速率 160 维 | 等权 320 维 |
|---|---:|---:|---:|
| 原 PAMF-SUP，5,000 | 0.5223 | 0.5125 | 0.5191 |
| 原型版，5,000 | 0.5882 | 0.5174 | 0.5769 |
| 原 PAMF-SUP，20,000 | 0.5160 | 0.5373 | 0.5340 |
| 原型版，20,000 | 0.6749 | 0.5228 | 0.6467 |

原型版在第 20,000 步的收益主要来自原生 160 维。与原生单路相比，等权融合的同星 AUC 下降 0.0282。原生／全速率／融合 Top-1 分别为 0.1847%／0.0783%／0.1887%，分类仍弱。两路不能仅按相同维度设为相同权重。

## 交接

N607 GPU 5 上的 PID 2009473 已退出且完整 JSON 已读回。四个冻结检查点、三种读出和三种维度对照均完成。第 1 版的 NumPy 索引兼容问题由 DLPack 修复。[原型训练](../20261119-diagnostic-satiq-pamf-sup-branch-prototypes-s2026092701-r02/report.md)
