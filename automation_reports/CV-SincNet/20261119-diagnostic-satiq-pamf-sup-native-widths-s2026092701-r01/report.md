# SatIQ PAMF-SUP original five-path width comparison

- run_id：`20261119-diagnostic-satiq-pamf-sup-native-widths-s2026092701-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Compare original 160 embedding with fused original-only nonlinear 256/320 and per-native-path 320 embeddings, on the same source and objective.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

四个试验只使用前 8,000 个原始 I/Q 点，即 320 μs 公共同步字段。这段波形降采样为原生 PAMF 五分支输入，无新增观测。GPU 实包运行和独立 CPU 前向均验证了输入长度。任意改变最后 3,000 点时输出逐值相同。四组共享的原生初值逐张量一致。

| 组 | 生成身份向量 | 可训练参数 | 物理 GPU | PID |
|---|---|---:|---:|---:|
| 原生 160 | 原融合表示 | 727,713 | 0 | 2021222 |
| 融合 256 | 原 160 加同一融合表示的 96 维非线性投影 | 1,091,681 | 4 | 2021772 |
| 融合 320 | 原 160 加同一融合表示的 160 维非线性投影 | 1,306,849 | 5 | 2022170 |
| 分支 320 | 原 160 加五条原生分支各 32 维投影 | 1,250,273 | 6 | 2022567 |

扩展坐标的向量范数设为原块的一半，等价于融合余弦中原块和扩展块分别占 80% 与 20%。所有组的原融合分类、同星错波束 CE、五路辅助 CE 和同包 KL 具有相同权重，五路留一原型 CE 的权重均为 0.5。各组同初值、同源顺序、同一分类头类别数 3,103。日志单独记录原生五路 CE 和原型 CE。匹配只用身份向量，分类用各自原生输出头。

## 交接

控制器 PID 2020881。四组均通过 GPU 实包前后向及源验证烟测，训练日志已独立核实增长。每组首轮 28,225 步，统一在 5,000、20,000 与 28,225 步做源验证。该矩阵专门比较原生特征宽度和投影形式。[原百轮](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md) · [留一原型结果](../20261119-diagnostic-satiq-pamf-sup-branch-prototypes-s2026092701-r02/report.md)
