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
| RIEI-P09 | Eq.(5)、Eq.(7)和Eq.(8)把MI/IE写为样本与receiver求和 | 完整12行采用单row消融胜出的mean | 保持其余协议不变，以`RIEI_REDUCTION=sum`运行完整12行论文字面尺度确认 | implemented | launcher参数化、本地测试、`bash -n`与12-job dry-run；待N607结果 |

## 声明边界

- 论文未明确给出优化器名称和总epoch数；Eq.(10)–(11)描述的是梯度更新顺序，不能据此断言作者使用了PyTorch SGD。
- 发现阶段只在Table III第1行比较训练动力学；最终论文结论必须用胜出配置重跑完整12行，不能用单row或目标域峰值代替。
- 目标域逐epoch曲线仅用于诊断；下一轮正式分数采用论文明确的最终10个epoch，禁止target-oracle选epoch。
- 当前反向审计：`verified=8`、`implemented=1`、`pending=0`、`deferred=0`、`rejected=0`、`blocked=0`；最高风险为论文未公开ResNet1D-18具体结构、优化器和随机seed。全局partition修复仍未达到逐行阈值，下一轮先验证公式字面sum尺度。
