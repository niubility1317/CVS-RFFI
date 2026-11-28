# LoRa R17 三方向实频谱幅度融合

- run_id：`20261128-phase1-lora-r17-amplitude-gain-s2026112897-r01`
- group_id：`phase1-lora-r17-amplitude-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未达到预登记晋级条件。

## 目的与对照

冻结 LoRa 实增益、R02/R07/R17 与 E104，增加先验频谱实幅度方向，与伴随残差和 R17 差异方向共同训练。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验及远端编译通过，控制器 PID 3714484、训练进程 PID 3714496；已核实命令、GPU、真实权重首步及日志增长。执行提交 `6e5e76531c68b96add83736f6dc0abfcfb8dbfcb`。状态事件已写入；共享索引更新时遇到其他任务的锁，登记文件和事件保留。

## 结果与覆盖

训练完成 1500 步，源 L／U 各曝光 12000 次，接受 11058 次 U 伪标签；耗时 991.91 秒，峰值 CUDA 分配 766224384 字节，新增头 11555 参数。未读取 U 真值、V 或目标数据参与训练。

固定 RX b200_1 排除前 960 包后的 300 包 clean／low urban 预测后独立评分。low urban 直通、LoRa R17、实增益和候选正确数分别为 138、86、182、182；clean 分别为 299、295、299、299。相对实增益改对 0、改错 0，两个半区净变化均为 0，配对 bootstrap 区间 [0, 0] 包，clean 也与实增益相同。未达到逐半区净增 2 包的预登记门槛。

E104 历史选模使用源 V，目标基准已有曝光；DURA v1 的 LoRa 匹配对照及目标七景待完成。

## 交接

保留此版本及筛查结果；后续候选另行固定训练和源样本。
