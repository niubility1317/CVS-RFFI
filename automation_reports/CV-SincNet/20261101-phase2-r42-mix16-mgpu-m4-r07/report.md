# R42 MIX 全 K/N 多 GPU 并发

- run_id：`20261101-phase2-r42-mix16-mgpu-m4-r07`
- group_id：`phase2-frozen-R42-group1-completion`；类别：`cvs`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

User-directed increase of R42 process concurrency across GPUs, leaving GPU7 unchanged; preserve all existing MIX predictions and in-flight workers.

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

2026-11-01 用户并发调整：VERIFIED TRANSFERRED: four real-source no-query smokes PASS; prior MIX completed units indexed and predecessor workers retained. R42 per GPU max GPU0/2/3/6=3, GPU1/4/5=2, GPU7=2 unchanged; minimum free6000MiB; other owner unchanged. All 672 MIX predictions fixed before independent scoring.

## MIX 全 K/N 评分与对照完成

VERIFIED MIX_ONLY：672/672 单元、139776 行预测冻结后独立评分完成。自然 query 同输入平均配对收益 S0Q0/S0Q1/S1Q0 均为16/16格≥5pp；S1Q1为12/16，K=1的四格未达标。共同query口径分别同为16、16、16、12格。逐单元258048条比较与384组汇总逐组核对通过；仅DURA、融合及单元达标数见 [完整对比报告](results_mix_kn/report.md)。单场景按用户要求暂停，不计为已完成。

资源profile按固定P01/MIX01/1-1/seed1/K5N20实测，GPU0采样孤立；profile查询核为CPU参考实现，不能冒充r07 CUDA实际预测延迟。MACs/FLOPs/能耗未测。MoPC及九原生独立评分已完成，但本报告仅R42与LocalRidge匹配对照。
