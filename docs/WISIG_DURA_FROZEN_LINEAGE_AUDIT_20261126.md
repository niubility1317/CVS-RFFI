# WiSig 冻结 DURA 逐阶段诊断

运行编号 `20261126-diagnostic-wisig-dura-frozen-lineage-m4-r01`。对比原模拟器监督、旧 Adaptive 波形反拟合、Adaptive 直接算子与混合快估四条已完成的 DURA 谱系。统一读出共同 R02 和各组 R07、R17、R38、R44、R46、R65、R69 的 E20 固定权重。R69 识别数据复用各自正式实验的同包预测。其余阶段的输出使用相同三种子冻结 PAMF。

- 源域使用固定的 128 个 R07 源验证物理包，沿用原 R07 同参考探针抽样。用真实配对 clean IQ 衡量每阶段的 NMSE、按参考谱能量加权的相位误差及频谱幅度 NMSE，覆盖 clean 和六种 Practical。源诊断不读 TX 标签或目标数据。
- 目标诊断固定抽取 2,048 个 Main 物理包，其中 cross-RX 1,536 个、cross-both 512 个，分别使用种子 `2026112603`、`2026112604` 对物理索引抽样。覆盖默认 Full、加重 Full 的 clean 和六种 Practical。两档使用相同物理 ID。报告逐阶段准确率、Low Urban 和相对于同阶段其他谱系的配对百分点及 95% 区间，按物理包在两个划分内重采样。使用两个 Full 条件集中诊断本轮已知的差距。正式全量四组合成绩保持独立。
- R07 交换旧波形版和 Adaptive 直接版的信道作用与噪声方差输出，用共同粗恢复和原 MAP 求解。R17 交换两组冻结的 R07 上下文和各自的先验，沿用原迭代。交换只做诊断，无训练和参数拟合。
- N607 普通账户，原项目 CVS-RFFI Python，FP32、TF32 关闭，批量 96。最多两个具备足够显存的工作进程；在真实权重源 IQ 无 query 检查后执行。故障停止限于权重来源或物理样本不匹配、非有限输出、任一诊断无法合法闭合。三组正式训练与产物保持独立。

发布入口 `experiments.pamf_unrolled_restore.wisig_dura_lineage_audit publish`，远端工作目录 `releases/<run>-release`，输出 `runs/<run>`。输出包括 `smoke.json`、`source.json`、`predictions/{default,strong}`、独立 `scores.json` 和日志。两档预测全部封存后，单独的 CPU 进程才读目标真值。该诊断不用于重新选择权重和调整方法。
