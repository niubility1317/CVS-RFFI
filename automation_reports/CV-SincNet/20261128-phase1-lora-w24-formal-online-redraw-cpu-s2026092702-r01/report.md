# LoRa 独立噪声重抽训练的正式 Full 对照

固定原 W24 E104、默认分类头、原预处理和类别顺序。在原 15 台未见接收机、30,000 个物理包上，共同测试此前固定噪声训练的四分量头和独立噪声重抽训练的四分量头。两个方向头都与原冻结 Practical 父版组合，恢复只看正式实收 IQ，沿用固定四分量保范数读出。原边缘来自父版。匹配的旧对照包括直通、旧 DURA 半量、旧复响应半量和原长训 W24 DURA。 [训练登记](../20261128-phase1-lora-w24-online-redraw-directions-s2026112991-r01/report.md)

运行 `python -m experiments.pamf_lora_main.w24_formal publish --online-additive-redraw-cpu --repo . --context <原 Practical 源上下文> --ssh-config <N607 配置> --output <独占本地发布目录>`。固定执行提交、原 W24 环境与工作目录，N607 CPU 两推理和两个重放进程，单分片 worker。控制器只等两份合法末版权重及真实源 U 无目标检查。依次封存三个互斥 10,000 包的分片和全部七景预测。随后独立评分与同包分析接入真值，记录逐景、接收机与 TX 结果、CPU 直通对照、净纠错和工程开销。发布、预测和评分过程中不读取源操作算子。独占目录 `runs/20261128-phase1-lora-w24-formal-online-redraw-cpu-s2026092702-r01`。遇技术错误保留产物，其他进程继续。

验收固定六景正确 162,000／180,000、低城区 25,500／30,000 和 clean 28,566／30,000。正式分数只用于报告。
