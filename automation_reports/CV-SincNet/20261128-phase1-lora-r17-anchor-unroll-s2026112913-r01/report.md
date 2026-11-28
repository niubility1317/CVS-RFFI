# LoRa 实增益锚点与 R17 展开耦合

- run_id：`20261128-phase1-lora-r17-anchor-unroll-s2026112913-r01`
- group_id：`phase1-lora-r17-anchor-unroll`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 5 执行。

## 目的与对照

冻结实增益与 R02/R07/R17；每轮物理求解后以三方向实增益更新实增益锚点，并将变化送入下一轮先验。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过；控制器 PID 3756110、训练进程 PID 3756125。实际 GPU、命令及真实 checkpoint 首步已核实，首步进程显存约 1330 MiB。执行提交 `a230f0b74143ff68434c0f9bcd1d1d1773de13d1`。状态事件已写入；索引更新遇共享锁，登记文件和事件保留。

## 结果与覆盖

训练后在 RX b200_1 固定 300 包 clean／low urban 预测，再独立 CPU 评分，排除前 1260 条源筛查包。E104 历史选模使用源 V，目标基准已有曝光。当前无本候选筛查结果，固定 DURA v1 的 LoRa 对照与目标七景待完成。

## 交接

恢复先核查本 run 的 `state.json`、PID 与日志，避免重复启动。
