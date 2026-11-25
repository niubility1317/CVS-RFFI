# LoRa 扩容 DURA 局部方差固定配对试验

固定扩容 DURA 的同一份 40,000 步源 U 权重、原旁路门和末步 0.5 回写。两路候选只有 CG 观测方差的时间轮廓不同。无新拟合、阈值更新或候选选择。源 U 的受控突发和错位对照已完成。源 V 和正式目标预测只读取当前包的六个 256 kS/s 实收前导窗。 [源 U 复核与参数来源](../20261124-phase1-lora-dura-capacity-source-m4-r01/report.md) · [实收方差代码](../../../experiments/pamf_lora_main/dura_variance_profile.py)

同一冻结 Tri-set 种子 `2026092702`。先做真实权重 source U 检查，然后同一批 900 个源 V 物理包按 Full 和 Residual 各跑 clean 和六种 practical。源 V 对照为直通、同路线原冻结 DURA 半量、原扩容半量和位置错移 128 点的局部方差半量。两路源 V 预测闭合后独立 CPU 读取源标签。源 V 正确数和误差供归因，目标试验不依赖其高低。

目标固定 15 台未见接收机、每台原始盲序前 200 个包。两条路线均使用同一个局部方差读出与原登记缓存，逐场景写直通和局部方差两行，共 420 个决策数组。目标全量固定之后，由独立本机评分器先逐包核对直通，并与同批原冻结 DURA、扩容半量和 `guard_half` 预测配对，再读取原始测试标签。记录 clean、六景、低城区、逐接收机和逐景正确数、纠错、改错、净变化，以及 1,600 次按接收机分层的物理包配对重采样区间。这个子集用于核查该机制的识别变化，正式接受条件仍使用原三种子 30,000 包矩阵。

N607 普通账号与已验证的项目 Python。命令 `python -m experiments.pamf_lora_main.dura_variance_profile_evaluate publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config>`。提交由推送后的 HEAD 固定，并保留在 `submit.json`。运行路径为 `${N607_PROJECT}/runs/20261125-phase1-lora-dura-local-variance-pilot-s2026092702-r01`，工作目录为同名 `-release`。真实权重检查优先使用较空闲 GPU，两路源 V 并行，目标为八个按现有接收机划分的 GPU 分片。源权重身份、旁路、真实识别器、视图契约、有限性或矩阵闭合失败时停止本链，保存已有产物。预期包含 `profile.json`、`smoke.json`、两份源预测与源评分、八份目标分片完成证据、`predictions/complete.json` 和独立 truth-last 的 `scores.json`。独立评分调用 `python -m tools.lora_dual_aperture_pilot_score {collect,score} --method local-variance` 及所对应的已登记参数。
