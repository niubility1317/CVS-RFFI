# 固定 SIG_TIME 特征均值教师：首种子完整结果

状态 **ANALYZED / VERIFIED**。同架构对照 SIG_TIME 为 **83.6679%**，FEATURE_MT 为 **83.3619%**，差 **−0.3060 个百分点**。新增一致性确实优化了源 U 特征目标，但首种子没有目标泛化收益；不能据此宣称三种子结果或 84% 达成。

## 方法与比较边界

run：20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01；seed：2026092701；split：392005；ManySig clean 168000 个物理样本；readout：checkpoint 自带 ORIGINAL_HEAD。代码 373262b51f037f03e3238471b3aefb88e45fba5b，执行 6b23a5b033ee063dea97d6783d5634e220a1d93d。固定 SIG112→joint160、fθ 195041 参数和原默认 cosine 6×160/scale30 头，正常联合训练，另训、替换或重拟合头 0 次。

在原 33 label＋17 pseudo、batch_neighbor/own EMA .999/cos 下，将同一实际 U 包的 teacher weak 与 student strong 单位 joint160 特征作 half squared L2 mean；teacher stopgrad，eps=1e−8。所有实际 U 均进入该项，不用 U 真 TX、伪标签或特征可靠度阈值；系数 .1×clip((epoch−33)/5,0,1)。ZERO 同图权重 0，保持 sampler、前向和随机数消耗一致。原硬伪标签 UCE 返回对象及 λUCE=.16、λentropy=.01、L smoothing=.01 不变。无新增网络、投影头、预测器或模型前向。

理论动机来自 [Mean Teacher 原论文](https://arxiv.org/abs/1703.01780) 和 [作者实现](https://github.com/CuriousAI/mean-teacher)。原方法主要匹配输出预测，本实现是无额外头的部署特征变体，不能直接继承其性能保证。

## 默认头独立 truth-last 结果

|条件|clean 准确率|Macro-F1|最差 RX|最差 day|
|---|---:|---:|---:|---:|
|OLD_GATE_COS50|80.6685%|80.1734%|70.6167%|75.4524%|
|SIG_TIME|83.6679%|83.2448%|71.2458%|79.1929%|
|FEATURE_MT|83.3619%|82.9148%|70.4833%|78.7143%|

FEATURE_MT 对 SIG_TIME 纠正 544 个错误、引入 1058 个错误，净少 514 个正确决定。对旧门控＋cos 基线增加 2.6935 个百分点，其中同时包含 SIG_TIME 架构差异，不能归因于均值教师；机制归因以同架构的 −0.3060 个百分点为准。n=1，不计算样本 SD，也不把历史最高 seed 当作主结果。

三行先固定 prediction，再连接独立 truth；完整审计从保存的决定/CM 重计 504000 个默认决定、54 组指标，解析全部 150 个模型 epoch 和 22200 条新 step。新模型各 11100 次成功更新，历史基线另计 11100 次。全部 worker 自然退出，失败 0；未新增目标前向、头拟合或优化器更新。

## 源域证据与实际特征路径

原预登记源资格 PASS：最后五轮按 U 包数加权的一致性 discrepancy，ZERO=0.3292548832、FEATURE=0.2704788352，比值 0.8214876953；下降约 17.85%。最终 L CE 为 0.2476658864 / 0.2487854458，均≤.5；候选和 ZERO 的合法 L720 实际特征 L2 差为 2.0582449436。源资格只证明健康、目标被优化和真实特征改变，不能解释为 U 分类正确或 DG 有效。

每新模型实际 U 包次 963900、U feature steps 3774；原生门控累计接受 ZERO 15440、FEATURE 15348 包次，分别约 1.602% 和 1.592%，不是独立物理 coverage 或 U 准确率。两个新模型的原生梯度及 feature 梯度共独立重计 **7992 层记录、23976 个向量**；每套记录覆盖 33/34/50 各 222 批。有效 feature 梯度在候选 34/50 的 t1、t2、SIG 投影均非零，ZERO 为零；auxiliary 对 class W 梯度 None，原主损失仍正常联合更新默认头。

|第 50 轮对象|有效 feature/main 梯度范数比|余弦|非零批数|
|---|---:|---:|---:|
|t1 PW|0.012948|-0.025624|222/222|
|t2 PW|0.016210|-0.020170|222/222|
|SIG local projection|0.009464|-0.007835|222/222|

上述是分别计算的 main L128 与 U256 梯度，不是累计 AdamW 更新或已证实的因果冲突。两个模型各 193460 个 feature 参数元素和 960 个默认 W 元素发生数值更新；SIG 的 gain、local/output projection、output norm 五个 tensor 全部更新。真实 id_feat_joint 与独立 feature-only 导出逐元素一致，复制模型临时置零 W 不改变特征，分析不改 θ/buffer。这里只证明新增训练项进入可交接 fθ，不能证明 Phase2/new TX 可迁移收益。

## 完整曲线与成本

全部 50 轮曲线见 source_curves_and_clean_result.png，统计见 source_curve_summary.json。最后五轮与之前五轮相比，SIG/FEATURE 的 L CE 均值分别从 0.233980/0.234721 降到 0.228236/0.229319，源 V 从 98.1363%/98.1200% 到 98.1504%/98.1363%。V 已趋平台，CE 仍波动；固定 50 轮不等于证明充分收敛。V 只用于健康诊断，无梯度、拟合或目标选模。

2 新 scratch50：100 model-epochs、22200 更新；1 历史基线：50 epochs、11100 更新，分别报告。新模型原生日志 epoch 时长合计 4755.05 秒（ZERO 2394.83、FEATURE 2360.22），共享 GPU 下非受控速度比较。额外模型前向 0；诊断反传、向量保存及事后审计有成本，不能据此宣称零开销。

## 冻结决定与后续

首 seed 源资格扩展决定已在 ec8ea03f7810cb3eb1bcb8108cdb60b4053a59c9 提交并推送，随后才回收新的目标数值。固定 02/03 的配方和两臂不变；扩展发布 R01 因缺失依赖在任何训练前失败，技术修复后 R02 已启动，不以首 seed 负结果调权重、筛 seed 或停止健康任务。等待完整三种子后报告配对均值、SD、正向 seed 数和缺项。当前目标仍未达到；不增加结构分支、单独头训练或 Phase2 声明。
