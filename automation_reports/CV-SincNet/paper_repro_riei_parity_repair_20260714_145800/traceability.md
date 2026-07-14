# RIEI SPAWC 2023 Table III论文—代码一致性追踪

| ID | 论文依据 | 修复前实现 | 目标实现 | 状态 | 验证 |
|---|---|---|---|---|---|
| RIEI-P01 | WiSig信号先去除无信号段并做信道均衡；数据集专段未规定逐包RMS归一化 | `riei_original`硬编码`normalize=True` | 暴露RMS开关；严格候选关闭，保留开启对照 | verified | `cvs_data.py`接线；launcher dry-run |
| RIEI-P02 | Eq.(10a)–(10c)和Eq.(11)使用交替梯度步骤及中间FED更新；公式没有指定PyTorch优化器类型 | 固定Adam；FED在同一mini-batch中连续执行两个Adam step | 支持无momentum SGD和Adam消融；保持CE中间更新与固定分类器的MI/IE更新顺序 | verified | optimizer类型单测；既有交替更新单测 |
| RIEI-P03 | IEEE SPAWC 2023正文明确Table III统计最终10个epoch准确率均值与标准差 | 误称“期刊版”并固定last5 | 论文Table III正式口径改为last10；保留既有last5历史证据 | verified | 原PDF第4页实验段；新launcher dry-run |
| RIEI-P04 | Table I：6个发射机；每接收机14400个训练样本、4800个测试样本；数据随机划分后在Table III复用receiver组合 | 同一receiver的样本选择随source组合和RNG循环顺序变化；target另用独立随机流 | 每`(tx,rx,eq)`稳定分组seed，source取2400、validation/target复用同一后续800 | verified | synthetic split测试验证跨组合一致性及source-val/target-test一致性 |
| RIEI-P05 | ResNet1D-18 FED、三层FC EC/RC、`lambda_1=lambda_2=1.2` | 已实现 | 保持不变 | verified | 既有架构/损失单测 |
| RIEI-P06 | 优化器与loss reduction未完全公开，必须用同row受控实验定位 | 固定Adam+sum导致Table III第1行明显低于论文 | 8候选完整200epoch比较，以预定last5选型 | verified | 8份metrics共1600epoch、24份日志共6100行；P02=`80.12±0.58%` |
| RIEI-P07 | Table III最终证据必须覆盖论文全部12个receiver组合 | 发现阶段仅覆盖第1行 | 固定P02的SGD+mean+no-RMS+no-FN配置，运行12行确认 | verified | 12×200epoch自然完成、硬错误0；均值72.26%，MAE4.82pp，命中5/12，`NOT_REPRODUCED` |
| RIEI-P08 | Table III各receiver组合应建立在同一随机数据partition上 | 组合内顺序RNG导致同一receiver跨行样本漂移 | 完整12行以稳定全局partition和论文last10重新确认 | verified | 12×200epoch完成；均值72.72%，MAE4.34pp，命中6/12，`NOT_REPRODUCED` |
| RIEI-P09 | Eq.(2)–(8)把CE/MI/IE写为样本与receiver求和 | 完整12行采用单row消融胜出的mean | 保持其余协议不变，以`RIEI_REDUCTION=sum`运行完整12行论文字面尺度确认 | verified-rejected | 12×200epoch完成、硬错误0；均值68.75%，MAE7.92pp，命中1/12，显著劣于mean，拒绝sum作为支持尺度 |
| RIEI-P10 | 论文仅说明WiSig FED为ResNet 1D-18，未公开stem卷积核/stride/max-pool | 当前固定ImageNet式`kernel7/stride2+maxpool` | 在row3/6/10/12上受控比较现有`imagenet1d`与`short_stem1d`；其余协议固定mean最优配置 | verified | 8×200epoch、硬错误0；short四行MAE4.02pp低于image的7.91pp，且3/4行降低绝对误差，预注册门槛通过 |
| RIEI-P11 | Table III最终证据必须覆盖全部12个receiver组合；架构筛选不能替代正式矩阵 | short stem仅在4个诊断行验证 | 用`short_stem1d`和固定paper last10协议运行完整12行 | running | run`paper_repro_riei_table3_shortstem_confirm_seed1337_20260715_043000`；8个trainer健康运行至epoch12–14，硬错误0 |

## 声明边界

- 论文未明确给出优化器名称和总epoch数；Eq.(10)–(11)描述的是梯度更新顺序，不能据此断言作者使用了PyTorch SGD。
- 发现阶段只在Table III第1行比较训练动力学；最终论文结论必须用胜出配置重跑完整12行，不能用单row或目标域峰值代替。
- 目标域逐epoch曲线仅用于诊断；下一轮正式分数采用论文明确的最终10个epoch，禁止target-oracle选epoch。
- 当前反向审计：`verified=9`、`verified-rejected=1`、`running=1`、`pending=0`、`deferred=0`、`blocked=0`。short stem通过预注册的四行架构门槛，完整12行已启动但尚未完成，因此RIEI正式结论仍为`NOT_REPRODUCED`。只有完整矩阵达到`MAE≤3pp且至少10/12进入论文±2SD`才能改变声明。
