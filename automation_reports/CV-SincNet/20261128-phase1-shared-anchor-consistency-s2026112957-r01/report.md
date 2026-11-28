# LoRa 与 WiSig 共用锚点分布保持教师恢复筛查

- run_id：`20261128-phase1-shared-anchor-consistency-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两侧源域训练进行中。

## 目的与对照

在 2% 有界共同恢复训练中增加固定锚点识别分布保持 KL，并进行同包筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 c94fa567d61dad4023dd29dc9bf40a98275156d9；N607 控制器 3871361，LoRa 3871372/GPU 7，WiSig 3871373/GPU 1。两个真实权重首步和有限非零梯度通过；运行 PID、CWD、显卡及日志增长已核实。预检、单包 SHA、远端编译及一次独立 P0/P1 审查通过。

## 结果与覆盖

共用网络、2% 上限和损失，分别使用源 V 已曝光结果选定的 LoRa 实增益及 WiSig R17 锚点与独立权重。源 U 用户授权教师训练新增锚点识别分布 KL；LoRa W24 E104、WiSig Tri-SIG-TIME SSL F0 冻结默认头，LoRa 前向为原 scipy、反向使用 WideTracker 近似。预定 LoRa/WiSig 教师样本前向 12000/72000，额外祖先与反向成本另计；训练不读取 U TX、V 或目标。固定源 V 已多轮用于设计，筛查尚未完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
