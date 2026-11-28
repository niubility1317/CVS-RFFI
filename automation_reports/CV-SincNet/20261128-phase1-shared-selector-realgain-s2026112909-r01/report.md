# LoRa 与 WiSig 共用物理特征整包选择器筛查

- run_id：`20261128-phase1-shared-selector-realgain-s2026112909-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两数据集源训练已在 N607 启动。

## 目的与对照

同一逐包门控读取接收 IQ、实增益和 R17 恢复及 DURA 作用残差，选择一条完整恢复输出；合法源 U 的模拟 clean 配对仅提供波形误差比较标签。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `fee08276fe4b517713dbbc67f56a104ad34ce6e3`；独立 P0/P1 审查未发现直接缺陷，10 项本地测试通过。N607 归档传输校验和远端编译通过；控制器 PID 3680195，LoRa worker PID 3680268 / GPU 0，WiSig worker PID 3680269 / GPU 5，均已核实 CWD、命令、真实专家首步及非零有限梯度，日志增长到 50 步。登记原始记录与 PLANNED 事件已写入；发布时全局索引由其他进程持锁，索引更新稍后核对。当前候选避开上一 run 的 GPU 7/2，其他 owner 任务保持。

## 结果与覆盖

尚无本候选筛查或正式结果。两数据集使用相同整包选择规则、网络结构与训练目标，分别使用同契约的实增益和 R17 专家及独立训练权重；输出是所选专家的整包恢复 IQ。门控仅利用当前包和 DURA 作用残差，源 U 训练用模拟 clean 配对比较两专家波形误差，无 F 权重或 TX 标签；LoRa 实增益专家历史上有 E104 指导及伪标签，WiSig 专家 F-free。预登记源 V 面板已有多轮曝光，冻结预测后独立评分；目标集保留。LoRa 同包 R69 是合法源训练结构适配，原 WiSig 固定 DURA v1 checkpoint 身份单列。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
