# LoRa 与 WiSig 教师指导同 seed 同预算无教师对照

- run_id：`20261128-phase1-shared-matched-mse-s2026112937-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；同 seed 同预算无教师对照正在两数据集源域训练。

## 目的与对照

与教师版同结构、seed、源 U 配对和预算，使用两专家源 clean IQ MSE 偏好生成二元目标；训练不读取识别器。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `f2f9186460a62e43a2004904cbf10cf5b638132f`。N607 preflight、单一归档传输 SHA 与远端编译通过；控制器 PID 3772282、LoRa PID 3772473 / GPU 7、WiSig PID 3772474 / GPU 1。独立核查 CWD、命令、显存、真实专家首步和有限非零梯度，日志增长。教师版使用 GPU 0/5，其他 owner 进程保留。一次独立 P0/P1 审查通过，14 项本地测试通过。

## 结果与覆盖

尚无筛查或正式结果。本对照与教师版使用相同 seed 2026112937、同门控及专家、源 U 抽样、模拟配对、优化器、每数据集 batch 和 1,000 步，只把训练目标改为源 clean IQ MSE 专家偏好；不读教师 F、U TX、V 或目标训练。不同 GPU 的实际浮点轨迹可能不同。LoRa 实增益父专家仍继承原授权 E104 指导与伪标签；WiSig 父实增益专家 F-free。既定源 V 筛查面板多轮曝光。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
