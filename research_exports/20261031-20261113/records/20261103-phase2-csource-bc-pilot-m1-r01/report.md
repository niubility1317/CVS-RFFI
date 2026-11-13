# Phase2 第二组：C 地面主干接 B 适应与 C 注册的单种子试运行

- run_id：`20261103-phase2-csource-bc-pilot-m1-r01`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

15 fixed B/C transplant pairs, one exposed-development capsule and model seed, K5/N20; POSTER and Hu binding pending.

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


本试运行仅覆盖已绑定的 5 个 B × 3 个 C；POSTER、Hu 跨主干绑定仍待实现，ISSL 保持停止、CSIL 作者版本未绑定。LoRa 辅助旧类分类头由冻结源 gallery 的类别均值解析构造，不冒充原生 kNN 分类器；ProtoNet/RadioNet 到 C 的判决迁移差异按结果元数据记录。15 个预测固定前不进行独立评分，实验属于已曝光开发评估。

## 2026-11-03 试运行独立评分

五个 B 方法与三个 C 主干共 15 组已完成预测及独立评分。仅 P01/MIX01/1-1、seed2026092701、K=5、N=20 的已暴露开发样本，不代表完整矩阵。natural 与 common 预测评分绑定和每组 26 类非空均已核验。

| B 方法 | C 主干 | 自然 H | 旧类准确率 | 新类准确率 | 宏 F1 | common H | B/C/query 秒 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| protonet_cda | mopc_hr | 0.00% | 0.00% | 19.75% | 9.50% | 0.00% | 1.20/19.01/5.00 |
| protonet_cda | orthogonal | 27.29% | 44.65% | 19.65% | 23.13% | 27.28% | 1.06/5.91/0.88 |
| protonet_cda | lora_triplet_knn | 20.09% | 26.93% | 16.02% | 16.13% | 19.98% | 0.89/0.04/1.43 |
| mrior_sda | mopc_hr | 7.12% | 4.23% | 22.58% | 12.04% | 7.18% | 19.77/20.79/4.69 |
| mrior_sda | orthogonal | 26.51% | 41.78% | 19.41% | 23.51% | 26.51% | 6.93/5.30/1.06 |
| mrior_sda | lora_triplet_knn | 8.66% | 52.33% | 4.72% | 11.97% | 8.66% | 7.22/0.07/1.93 |
| dadda_sda | mopc_hr | 12.31% | 8.37% | 23.26% | 14.53% | 12.35% | 17.63/19.09/5.31 |
| dadda_sda | orthogonal | 27.25% | 42.08% | 20.14% | 23.36% | 27.25% | 5.70/5.25/0.90 |
| dadda_sda | lora_triplet_knn | 8.55% | 52.83% | 4.65% | 12.24% | 8.50% | 8.72/0.05/1.78 |
| twostage_sda | mopc_hr | 0.50% | 0.25% | 11.71% | 5.74% | 0.50% | 33.75/18.18/3.12 |
| twostage_sda | orthogonal | 27.09% | 56.76% | 17.79% | 24.74% | 27.19% | 9.46/5.17/0.54 |
| twostage_sda | lora_triplet_knn | 12.41% | 45.95% | 7.17% | 12.25% | 12.45% | 13.53/0.05/1.56 |
| radionet_ada_knn | mopc_hr | 0.89% | 0.45% | 29.21% | 16.46% | 0.90% | 1064.60/16.98/4.28 |
| radionet_ada_knn | orthogonal | 6.85% | 4.07% | 21.58% | 14.43% | 6.85% | 276.57/5.61/0.78 |
| radionet_ada_knn | lora_triplet_knn | 13.90% | 21.88% | 10.18% | 9.40% | 13.87% | 290.44/0.05/1.50 |

项目移植、辅助旧类头与 B 判决交接的差异由对应预测资源元数据记录；资源为实际运行时间，GPU 可能共享，未测 MAC、FLOPs、能耗保留缺项。结果不回流选模、调参或选择性排除。
