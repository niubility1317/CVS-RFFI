# LoRa PAMF W24 原配方与 DURA 原生 Full 配对测试

固定原配方 `phase_native24` 的种子 `2026092702` 和 E108 源域冠军，权重来自 `20261125-phase1-lora-pamf-tri-clean-nativewide98-m2-r01/F/phase_native24/best.pt`，原执行提交为 `e6dde7d40f2cc3dd7b21d83daa4bcbabac52d3ae`。该识别器源 V 为 11,781/12,000，既有独立 clean 为 28,146/30,000。

首轮比较 W24 直通、已有高相干半量的六中心窗写回，以及已有复响应半量的同样写回。恢复权重沿用同 LoRa 源数据契约冻结产物，无新增训练、校准或目标驱动选择。修正先从 256 点中心窗回到 1 MHz，再用固定 24 点平滑渐变写回已登记的六条内部 chirp。两个外侧 chirp 及每条 chirp 原生首尾 24 点保持实收 IQ。所有模式都用原版 W24 的八 chirp、384 中心点与每端 24 个原生点输入。

同一 15 台未见接收机和 30,000 物理包，测试 clean 与六景 native Full practical。对原始 8,192 点连续模拟，录制范围外以零处理，补偿数值滤波延迟。此视图单列，历史六中心窗成绩保留原口径。Residual 的原 256 kHz 快照缺少 W24 所需原生边界，留给后续单独登记。配对结果报告逐景、逐 RX、六景平均、clean、纠错和改错。

用 N607 普通用户和登记 LoRa 环境，发布入口 `python -m experiments.pamf_lora_main.w24_native_full publish`。实验根 `runs/20261125-phase1-lora-w24-native-full-s2026092702-r01`，GPU 顺序 0、3、5、7、2、6、1、4，每卡最多一个本 run worker，CPU 信道并发 12。发布后先做 4 个真实源 U 包的 W24 与恢复权重 no-query smoke、边界恒等和有限性检查。通过后进行全部 315 份盲预测，独立本机评分只在冻结清单和矩阵通过后接入原档案真值。契约、冒烟、数值或预测不全故障保留产物并停止本 run。
