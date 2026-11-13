# Phase2 第二组：C 地面主干接 POSTER 适应与 C 注册试运行

- run_id：`20261103-phase2-csource-poster-pilot-m1-r01`
- group_id：`phase2-group2-csource-poster`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Three fixed POSTER/C transplant pairs, one exposed-development capsule and seed, K5/N20.

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

## 2026-11-03 启动读回


POSTER 适配保留冻结的 C 源编码器，重置并训练两层稠密投影与六类头 10 个 epoch；投影维度按 C 编码器匹配，投影进入 C 注册。该项目移植与作者原 POSTER 架构不同。Hu 跨主干绑定仍待实现；ISSL 保持停止、CSIL 作者绑定缺失。

## 2026-11-03 试运行独立评分

三组固定预测和独立评分均已完成；P01/MIX01/1-1、单模型 seed、K=5、N=20。所有 26 类的 query 数均非零；自然 query 每组 26471 条。该单元已曝光，不能代表完整矩阵。

| C 地面主干与注册核心 | 自然 H | 旧类宏准确率 | 新类宏准确率 | 宏 F1 | 共同 query H |
| --- | ---: | ---: | ---: | ---: | ---: |
| mopc_hr | 0.00% | 0.00% | 12.50% | 2.72% | 0.00% |
| orthogonal | 0.00% | 0.00% | 23.95% | 11.85% | 0.00% |
| lora_triplet_knn | 15.90% | 32.47% | 10.53% | 13.35% | 15.82% |

| C 地面主干 | B 时间（秒） | C 时间（秒） | query 时间（秒） | B 判决迁移 |
| --- | ---: | ---: | ---: | --- |
| mopc_hr | 1.653 | 18.729 | 5.031 | reset_B_classifier_and_projection_inherited_by_MoPC |
| orthogonal | 1.170 | 5.009 | 0.530 | reset_B_classifier_weights_as_cosine_old_weights_bias_discarded |
| lora_triplet_knn | 1.255 | 0.029 | 1.360 | reset_B_classifier_replaced_by_native_LoRa_kNN_projection_retained |

三个任务均使用旧类目标标签 30 个、新类目标标签 100 个。时间来自真实运行，可能与其他任务共享 GPU；未测 MAC、FLOPs、能耗均保留缺项。结果未用于调整方法或筛选对照；与原版完整矩阵的平均值不可直接相减。
