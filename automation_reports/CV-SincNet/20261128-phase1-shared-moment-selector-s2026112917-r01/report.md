# LoRa 与 WiSig 共用增益不变矩选择器筛查

- run_id：`20261128-phase1-shared-moment-selector-s2026112917-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；已核实独立控制器和两数据集 worker。

## 目的与对照

同一整包门控结合实增益、R17 和物理作用残差；以模拟 clean 配对的相位及增益不变包统计特征误差构造无 TX 标签选择目标。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `b047cfe761595f4ebc432c7faf070c2070ceaa1e`。N607 preflight、单一归档传输 SHA 与远端编译通过；控制器 PID 3718024，LoRa worker PID 3718041 / GPU 7，WiSig worker PID 3718042 / GPU 1。已独立核查 CWD、命令、GPU 显存占用与真实权重首步；WiSig 日志增长至第 50 步。发布前 GPU 7、1 分别约 16.9、15.6 GiB 可用，资源动态分配，其他 owner 进程保留。一次独立 P0/P1 审查发现步数校验不一致，已修复并针对性复核通过；14 项本地测试通过。

## 结果与覆盖

尚无筛查或正式结果。共享选择器以源 U clean 配对中的归一化包络高阶矩、多个时延复自相关和八段频谱能量误差构造训练目标；不读 TX 或 F，推理只用当前包。LoRa 实增益专家历史继承授权 E104 指导与伪标签，WiSig 专家 F-free。此前源 V 面板已多次曝光，目标数据不用于此轮训练与筛查。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
