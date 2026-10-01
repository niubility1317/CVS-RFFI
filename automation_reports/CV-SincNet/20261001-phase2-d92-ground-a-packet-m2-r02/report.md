# D92适应前地面分类头：原生checkpoint类型兼容导出

- run_id：`20261001-phase2-d92-ground-a-packet-m2-r02`
- group_id：`d92-ground-a-source-packet`；类别：`analysis`；阶段：`Phase2-source-classifier-packet-export`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ARTIFACTS_COMPLETE；实际runtime为`0145234e381b5799306acfb3d36a4e17865c1814`。

## 目的与对照

从两个既有合规source-only scratch final200 checkpoint提取原始6×160 float32 CosFace分类头，显式绑定原头类别行序、s=30、eps=1e-4和raw feat_joint ABI。原模型及训练不变；只建立后续配对A测量输入，本run不读取任何源域样本、逐样本源特征、目标support/query，不训练、不推断、不评分。 原r01原生配置类解析失败；本r02仅使用已核实原training release/code解析SatViewStage、RC4Calibration。该依赖不构造encoder、不读取样本、不使用calibration属性，不复制到星载packet。

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

ARTIFACTS_COMPLETE/VERIFIED：r02实际runtime0145234，supervisor617603正常退出；两个源分类头包经独立loader、checkpoint/class/scale/eps/source lineage/文件统计读回核实，实际文件8737B与8735B，其中各3840B为原float32权重。两包已下载并核对metadata/complete及各文件字节。query/support/源样本/源逐记录特征0，encoder未构造/执行，A仍未评分。ground export程序内计时0.306050687s，进程峰值RSS450482176B；不含Python/Torch启动耗时，不代表星载资源或链路传输，实际星地增量传输N/A。

| 原模型seed | 权重B | metadata B | complete B | 完整包B | checkpoint加载秒 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 3840 | 4338 | 559 | 8737 | 0.09715832299843896 |
| 2026092702 | 3840 | 4337 | 558 | 8735 | 0.0371263330016518 |

这里只统计每模型实际三个packet文件；两个模型是实验重复，不要求部署同时传输两模型。原已部署encoder是否需要重复传输、协议开销、压缩/量化及实际星载资源未测。包生成不等于适应前准确率测量，下一步用固定A预测配对同物理旧held support并加入三阶段补充表。
