# LoRa R17 三方向实频谱幅度融合

- run_id：`20261128-phase1-lora-r17-amplitude-gain-s2026112897-r01`
- group_id：`phase1-lora-r17-amplitude-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 5 执行。

## 目的与对照

冻结 LoRa 实增益、R02/R07/R17 与 E104，增加先验频谱实幅度方向，与伴随残差和 R17 差异方向共同训练。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验及远端编译通过，控制器 PID 3714484、训练进程 PID 3714496；已核实命令、GPU、真实权重首步及日志增长。执行提交 `6e5e76531c68b96add83736f6dc0abfcfb8dbfcb`。状态事件已写入；共享索引更新时遇到其他任务的锁，登记文件和事件保留。

## 结果与覆盖

训练后固定 RX b200_1 排除前 960 包后的 300 包 clean／low urban 预测，再独立评分。E104 历史选模使用源 V，目标基准已有曝光。当前尚无本候选筛查结果；DURA v1 的 LoRa 匹配对照及目标七景待完成。

## 交接

恢复先检查原 run 的 `state.json`、PID 与日志，避免重复启动。
