# 四周期采样抖动估计器：本地合成反证预登记

状态：PLANNED；尚未启动正式矩阵。run：20261017-diagnostic-F-crossRX-jitterEstimator-synthetic-m1-r01。唯一 owner：codex/root/F-sampling-jitter-estimator-20261017。

本轮检验未知干净模板时，短前导是否足以把与信号斜率相关的抖动方向与增益、相位及白噪声分开。它只是既定 Ag／As 目标的机制反证，不产生识别性能证据，也没有定义可发布增强幅度。F＋跨 RX 同 TX／三 SIG_TIME／统一 Sinc／joint160／NoExplicitCFO／默认头联合训练／E20／原 U 保持。

方法为独立周期对的交叉矩。4×20 点中所有 3 种二分配对均使用：两周期之差外积的一半估计条件噪声协方差，另两周期的对称交叉乘积估计 z、jz 及周期导数方向，不向估计器提供干净模板。交叉 Gram 与右端矩在不重叠周期上计算，4 个非负方差通过固定矩方程求解。导数使用 20 点全部 FFT 频率，不选频带。归一化 Gram 最小特征值必须超过 70 包 Gram 的 Frobenius 标准误 5 倍；失败输出不可辨认。

完整固定矩阵为两种多音、单音、常量、零信号共 5 模板；9 情况（白噪声、增益＋白噪声、相位＋白噪声、三者混合、抖动＋白噪声、四者混合、纯抖动，以及 ±0.0001 cycles/sample CFO＋白噪声）；SNR 20／28／36 dB；每格 32 抽样，每次 70 包／4 周期／20 点。共 4320 格、302400 合成包、24192000 个合成复观察点。data seed=202610174001，同模板／SNR／抽样在 9 情况共享底层随机量；model／split／augmentation／support／evaluation seed 均不适用。逐项矩阵与数值以 experiment.json 为准。

每个多音／情况／SNR 组可辨认比例须 ≥95%；每个退化组拒绝比例须 ≥95%。零抖动组估计抖动能量占真实总创新能量的中位数 ≤2%、q95 ≤10%；含抖动组方差相对误差中位数 ≤20%、q95 ≤50%。非 CFO 组真实条件协方差相对误差中位数 ≤20%、q95 ≤35%。误差分位数按可辨认项计算并同时报告计数，拒绝率是独立必需条件；全部 32 项保留。所有组必须通过，不能挑情况或删难例。

CFO 为明确负例而非身份特征。其确定性周期间漂移计入总创新能量分母；不把漂移套成零 CFO 的精确协方差参数，所以 CFO 组不应用精确协方差误差门槛，但零抖动误归因与可辨认／退化拒绝条件仍适用。输入是一级线性误差合成，不能推定实际均衡 WiSig 满足这些假设。合成干净模板和真实系数只由生成及独立评分使用，estimate 仅接收观察数组。

任一科学条件失败，完成并保留所有格，关闭固定四周期抖动条件协方差来源入口；不换周期／包数／窗口／频带／不确定度／幅度／阈值／seed 或提供真模板救跑，不读取真实 IQ，不启动源实验或 E20。若通过，也只说明此合成估计器可用，另需合法源 RX／TX／day 依据、身份保护及同预算训练证据。联合目标未完成。

本地 22 项输入／配置负例及 11 项精确公式／退化反例通过；4 次无噪声估计器调用共 22400 合成观察点，真实 IQ／checkpoint／NN／更新均为 0。一次独立 P0/P1 审查另附，不追加报告审查。当前阶段没有模型，真实 checkpoint smoke 不适用；没有远端发布，N607 归档、GPU 与远端编译不适用。

命令：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_jitter_estimator17.run --output local_artifacts/F_sampling_jitter_estimator_20261017/run。CWD：当前 codex/cvs-iq-timing-e20-20261016 工作树。环境：已有 Windows 原生 Python／NumPy，CPU 2 线程；GPU 无。实际执行 commit、PID、CWD、Python／NumPy 与线程由 resolved_config.json 保存。独占输出根 local_artifacts/F_sampling_jitter_estimator_20261017/run 不得已存在；stdout 单独保留 run.stdout.log。

技术停止仅限当前自有运行的非有限值、配置／预算／形状／数学错误、输出冲突或无法启动；保留产物，不自动重试，不干预其他任务。无需新增数据权限或验证。预期产物：全部格充分统计与独立真值、配置／进度／完成、全量独立复算、同 run 报告／events／索引。正式真实 IQ／模型／NN／优化更新／教师／头拟合／target 均为 0。
