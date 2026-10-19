# 冻结默认头的决策保真恢复

状态：PLANNED。用户目标为 low-urban 目标准确率至少 70%，其他五种 practical 接近原 F clean；Agent 暂将“接近”按差距≤2 个百分点报告。恢复 IQ 与同物理 clean 的差距、接入基座后的真实 TX 表现是独立评价线。目标基准已暴露，本轮不声称新的独立盲测。

## 唯一改动与匹配对照

前一特征约束仅使用 160 维 embedding 的 cosine 距离，源训练末轮均值为 0.00549；源 V clean 为 98.5%，mid-urban 91.277778%，low-urban 83.277778%，尚有身份保真缺口。新方案只加入原默认头的决策分布约束：`4*KL(softmax(detached clean logits/2)||softmax(restored logits/2))`，权重 1。原波形 IQ NMSE＋0.1 log power、特征 cosine 权重 1 及其余配方保持，不搜索温度或权重。

从原 R03 同一父权重独立分叉，绝不从目标测试后挑选的 feature 权重继续训练。只更新原 correction，重置 AdamW，20 轮／4940 步，batch 256，model seed2026092701、data seed2026101951；LR3e-5，247 步 warmup，cos 至3e-6，WD1e-4，clip1，FP32。每物理样本 observed→clean 与 clean→clean 等权，R02 的 .15 clean／.60 residual／.25 full 路由及每轮顺序不变，未生成的 clean 缓存槽不作为 damaged 读取。L/U 每组累计曝光126000/1134000，IQ例子2520000。

复用已完成 feature 实验的 frozenF_feature 作为同预算对照，不重复训练。两者父 R03、模型／数据种子、样本顺序、优化器及 E20／4940 相同。对照已固定且完整保留，不用目标分数挑选。原 F50 seed2026092701 的 features 和原默认头均冻结；clean teacher 分布分离梯度。F、头、coarse 优化更新为 0；恢复器训练不读取 L/U TX 真值。该系统利用了已有监督 F 的知识，称“识别器引导的自监督恢复”，不称整套系统完全无标签。

父 correction20＋新增20＝correction40；coarse20 单列，复合继承成本60个组件epoch，不能冒充总成本20。原 F50 配方与默认头不改。

## 源规则与自动目标验证

固定末轮，无 V earlystop／状态更新。源1800条 V 的 clean＋六种 practical 全部预测固定后，独立 CPU 读取 V 真值。相对固定 feature 对照，low-urban 与 mid-urban 各至少提高0.5个百分点，同时 clean 相对原 IQ 降幅≤0.1个百分点、六场景平均 NMSE≤1.05倍对照，才通过源规则。未通过则保留负结果，不访问此候选的新目标 IQ。

通过后控制器立即自动启动同一不可覆盖 root 内 target 阶段，无需再次指令：完整168000条、clean＋六种 practical、三套原 F50 默认头独立 argmax。三个 F 是已有读出 seed，不是三个恢复器训练 seed，也不集成投票。七视图的全部恢复 IQ 与三套 F 预测固定后，独立无 CUDA CPU scorer 才读目标 TX/RX/day 和评分用 clean。

目标分别报告真实 TX accuracy／Macro-F1、修复／损坏／净修复、逐 TX/RX/day、各 F 和均值、距同一 F clean 的百分点差；波形报告未作 oracle 对齐的 full256／prefix232／tail24 NMSE 与 log FFT power gap。clean 是独立测试视图及评分参照，不进入 practical 推断。保留前轮 clean 损伤和所有已知负结果，不将训练／源指标替代目标性能。

## 发布与技术停止

一个实际 code/config commit，一个独立 P0/P1 审查。21项聚焦检查覆盖 teacher 参考分离梯度、估计端梯度、只训练 correction、未生成缓存槽、checkpoint 权限／预算、未过源规则禁止 target、七视图／三 F／ID／类／IQ 完整性、提前 truth／paired-clean／状态更新。真实 R03＋原默认头两条 L_s smoke 是启动第一步，PASS 后立即继续；含新 posterior 经 native F 至 IQ 的真实非零梯度与新权重 roundtrip。

N607 普通账号、GPU0 每阶段最多一条本 run worker，总 owner 少于2且空闲≥12GB，CPU2线程、最多1条 scorer。source worker 完成退出后才可启动 target worker；不停止其他健康任务。新 root 不可覆盖，一次发布归档比较与远端 compile/import，启动后核实实际 PID/CWD/完整命令/GPU/日志增长。

数据／父模型／commit 不匹配、输出冲突、非有限、预算／预测失败属于技术停止，保留产物、不自动重试。科学源阈值失败属于负结果，目标不访问。具体命令、环境、路径、矩阵和预期产物见 [experiment.json](experiment.json)。
