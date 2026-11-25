# LoRa 星地增强 W24 单种子无门控正式测试

运行 `20261125-phase1-lora-w24-formal-ungated-s2026092702-r01`。用星地增强 W24 E104（种子 `2026092702`）及原 40,000 步 W24 DURA。Full 和原生 Residual 共用同一恢复权重。独立直通将当前实收输入送给 W24。新恢复臂的 clean 和六景均调用完整恢复网络，中心按冻结模型的半量读出更新，首尾原生采样保留。模型输出全批交给 W24。Full 对照保留原冻结半量 DURA 和复响应半量，Residual 保留原冻结 R69 门控半量，分别按历史行为标示。

覆盖 15 台未见接收机、30,000 个登记物理包，两条路线各 clean 和六种 practical。每路线三个分片，各 5 台接收机和每台 2,000 包。Full 六景门槛 156,600／180,000（87%），低强度城区 24,000／30,000（80%）；两路线比较匹配原 DURA 的六景及 clean，并单列与直通的比较。Full 六景距同方案 clean 3 个百分点另列。先检查六个冻结分片、每景逐接收机全包调用计数和两路线 clean 直通一致性，随后由独立本机进程读取官方真值并评分。预期有逐景、逐 RX、净纠错和分层配对区间。

N607 普通账号，运行目录 `runs/20261125-phase1-lora-w24-formal-ungated-s2026092702-r01`，在已归档的原生 W24 环境运行。入口 `python -m experiments.pamf_lora_main.w24_formal publish --repo <checkout> --context <registered-LoRa-context> --ssh-config <config> --output <new-release-artifacts>`。本机独立评分入口 `python -m experiments.pamf_lora_main.w24_formal_score --ungated --predictions <frozen-matrix> --dataset <official-label-archive-root> --output <new-score-root>`。运行使用固定新根，优先有剩余显存的 GPU，最多同时两个本轮分片，每分片 8 个 CPU 工人。源 U 检查覆盖两路线的 clean 和受扰包，确认全部经过 G。来源、覆盖、数值或闭合失败时保留产物并停止本轮下游。预期输出 `smoke.json`、六份 `complete.json`、逐 RX 的盲预测和活动计数、独立 `scores.json`。 [数据协议](../PROJECT_PROTOCOL.md)
