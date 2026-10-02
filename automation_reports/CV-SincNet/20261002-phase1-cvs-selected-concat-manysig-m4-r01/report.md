# CVS 固定身份网络：mid／low urban 拼接增强

状态：PLANNED，尚未启动或产生本次测试结果。网络固定为 `residual_fusion`，164225 参数。按用户“具体性能看测试集不是源域”的补充，使用既有clean测试较优的residual_fusion（四seed均值78.4543%）。此为BENCHMARK_INFORMED_FIXED_DESIGN，历史测试表现已影响用户指定结构，不宣称目标无接触的结构筛选或新的盲测。具体性能以本次冻结权重后的测试集准确率和 Macro-F1 判断，源域指标仅记录训练情况，不用目标测试进行后续调参、换权重或选择性重跑。

本次仅训练 `practical_mid`／`practical_low_urban`。E1 至 79 只做 clean CE；E80 至 90 在两场景之间均匀抽样，扰动概率 0.60；E91 至 200 仍只抽取两场景，概率 0.80。同一个 physical ID／epoch 对应固定随机增强，augmentation_seed 与 receiver_seed 都为 2027。high 的训练配置已移除。

每个 clean batch 生成同尺寸 satellite batch（未抽中扰动的样本保留），拼接成一个 batch 前向，再优化 clean CE＋0.68×satellite CE。拼接不增加 optimizer step 数；E80 后满批前向尺寸为 256，末批为 56。无一致性损失、域骨干、DAOT、RC4、PL、MixStyle、EMA 或其他辅助目标。

固定原物理划分：L／U／V=6300／56700／27000，U 不参与训练，source RX1、3、4、6、8，day1、2、3；ManySig 六类、equalized1、中心256点、单位RMS、25MHz。4个模型种子2026092701至2026092704，从零初始化，无历史权重、optimizer、teacher、EMA或原型继承。历史记录只提供设计与已固定对照证据。

AdamW，lr=0.0002，wd=0.0001，cosine至0.000001，batch128且保留末批。每轮50步、200轮，单模型10000步。完整FP32，cuDNN和matmul TF32关闭，无梯度裁剪；模型与loader RNG明确记录。本次相比历史practical基准还存在结构、显式loader及backend口径差异，不将差值单独归因于增强场景变化。

每行 E200 最终权重保存且源冻结标记核实后，由独立预测进程读取既有 VALIDATED_ONCE capsule；先固定全部4行预测，独立 scorer 再连接 truth。测试保留 clean 和3个既有 practical 场景（high、mid、low urban），即168000条clean及相同物理ID的一个satellite观测，卫星场景为互不重叠子集。high 是未用于训练增强的场景，不新建数据、不更换物理ID。它们不是每个样本各生成三个完整LEO视图。

报告4seed均值±样本标准差、每seed和RX的accuracy／Macro-F1及完整混淆矩阵，固定历史CVS-CE／CVCNN-CE作为描述性配对对照；负结果保留。既有benchmark已有公开结果，不称新盲测。无Phase2、SFT或新类注册。

保留完整文本日志、step JSONL、epoch JSONL／CSV，显示实际参数、clean／sat CE及权重、梯度及使用参数、LR、实际增强状态、源V／最差源RX、耗时与峰值显存；另在可丢弃clone上测资源，clone不回流训练。未启用项false，未测项null／N/A。

唯一launch owner，独占新run／输出，每GPU最多2训练任务且至少12GB空闲。不停止、重启或热改其他健康任务。失败保留产物，无性能停机或自动重试。初始checkpoint往返无query smoke通过后直接继续。

[逐行配置与测试预登记](experiment.json) · [只读实时preflight](evidence/preflight.json) · [聚焦验证](evidence/local_validation.json)。正式启动和完成状态以独立PID／CWD／argv／GPU／日志及artifact读回补充。
