# LoRa 跨频细化的正式 Full 配对测试

正式评测固定 E104、默认分类头、原预处理与类别顺序，使用 15 台未见接收机、30,000 个物理包和 clean 加六景 Full practical。两条恢复均采用同源训练的固定末步权重和实际实收 IQ。匹配的旧对照为直通、旧 DURA 半量、旧复响应半量和原长训 W24 DURA。报告两组逐景、净纠错、逐接收机、TX 类别、与门槛差距及单包时延。共享控制沿用冻结归档，另核对 CPU 直通判决。 [配对训练登记](../20261128-phase1-lora-w24-post-peer-refinement-s2026112991-r01/report.md)

由 `python -m experiments.pamf_lora_main.w24_formal publish --post-peer-refinement-cpu --repo . --context <既有 Practical 源上下文> --ssh-config <N607 配置> --output <独占本地发布目录>` 发布。同一次登记提交、原 W24 工作目录，N607 只用 CPU，两推理线程和两重放进程，按三个独占分片顺序执行。收到固定末步、校验来源和真实权重源 U 冒烟通过后直接预测。源域独立分数不参与放行。预测矩阵全部封存后，独立本地评分才接入真值。运行目录 `runs/20261128-phase1-lora-w24-formal-post-peer-refinement-cpu-s2026092702-r01`。保存 `smoke.json`、各片原始预测、覆盖与 CPU 直通对照、日志及本地独立评分。技术失败保留证据，不改其他在跑任务。

验收门槛固定为六景 162,000／180,000、低城区 25,500／30,000、clean 28,566／30,000。配对正式分数只用于报告。
