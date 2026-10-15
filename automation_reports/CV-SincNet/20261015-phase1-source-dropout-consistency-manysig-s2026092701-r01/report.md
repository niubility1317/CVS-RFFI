# 固定 SIG_TIME：随机失活一致性源端检验

状态：LOCAL_VERIFIED / PREREGISTERED，尚未发布或运行真实源数据。代码提交 `eb21026149b1d1d02ddf4b213408b37f782bf2f5`（初始实现 `e21bab7a0125edd9eaffa0f04f2c21c724dc6d7d`）。本实验只有一个 source-only 诊断 row；完整配置与逐行矩阵以同目录 `experiment.json` 为准。

## 问题和最小比较

现有固定权重 L720 隔离末端 dropout 回放曾测到 feature-feature area 噪声/clean RMS 31.9940%、残差160为25.2901%，仅说明该局部扰动敏感。它未复现完整随机训练，也未证明目标误差成因。一般方法依据是 [R-Drop](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)：约束同输入两个 dropout 子模型输出的双向KL。这里只检验该方向是否具备源端基础，不预言DG收益。

固定原DRO首SIG_TIME seed2026092701 final50 control，与此前源检验相同来源；不按目标成绩选父。每批用同一份 strong IQ（原.015噪声）做两次原生 train-mode 随机前向，计算两次默认 CE 的均值（smoothing.01）与 alpha=1 对称KL；两个预测均保留可微，不引入教师、投影或新头。另对相同输入做一次 eval 参考。测原 t1/t2/SIG投影与现有类别W的导数，以及实际 joint160 差异。原模型、buffers、类别W和optimizer均不更新；W导数测量不等于专门训练头。

源端规则提前固定：5批均有正KL，三骨干层KL导数各5/5非零，聚合KL对pairedCE导数余弦各≥0，模型状态保持。通过只允许另登记相同公式的02/03源复核；失败保留负结果，不扫系数、dropout或父权重。不会自动开始训练，不把局部导数兼容性称为可迁移特征或84%提升。

## 数据、成本和执行

复用已验证源契约：L6300 / U56700隐藏TX / V27000，训练池标注率10%，源RX1/3/4/6/8、day1/2/3、原split392005及NoExplicitCFO保持。只取既有固定前5个平衡L180 cohort：每TX/RX/day每批2包，共900唯一物理样本。U张量及U真值不读取；共享cache返回V后丢弃，V不前向、不求导。无目标IQ/统计/评分，无新数据验证。

预算0训练轮、0optimizer更新、0头拟合；2700源包前向、5梯度批、20参数层记录、40保存向量。真实checkpoint no-query smoke另用2个合成零IQ，不算真实L样本。普通N607既有CVS-RFFI运行时，FP32、CPU2线程、CUDA_VISIBLE_DEVICES为空；不占实验GPU、不改其他进程。

远端CWD `PROJECT/releases/cvs_source_rdrop_20261015_r01`；命令 `python -u -m experiments.cvs_source_rdrop.audit`。PROJECT沿用实际设计常量。独占输出 `PROJECT/runs/20261015-phase1-source-dropout-consistency-manysig-s2026092701-r01`，row `SOURCE_RDROP-s2026092701`。本地发布入口 `python -m experiments.cvs_source_rdrop.publish --output LOCAL_ARTIFACTS/source_rdrop_delivery_r01`，先自动preflight/no-query smoke，PASS后立即执行，无额外授权产物。

预期保存 resolved_config、actual_source_gradients.pt、analysis、source_qualification、独立保存产物复算、completion及日志。技术停止仅限本run输入/来源/权限不符、非有限值、固定cohort缺项、模型状态变化、复算失败或输出冲突；保留产物，不自动重试，不停止无关任务。单release archive校验、remote compile及发布后PID/CWD/argv/loggrowth按现有流程执行。

## 已有验证与未完成项

局部10负例、对称/零值/平移不变性和实际原生网络的两个随机joint160及四层导数检查PASS；状态、RNG及.grad不变，未读真实IQ或checkpoint。一次独立P0/P1审查发现继承检查器仍要求旧60向量，已定点修为40并复核PASS，未重复全面审查。真实父checkpoint无查询smoke、N607实际运行和源科学资格仍待发布后核实。
