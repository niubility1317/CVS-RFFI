# 冻结 F50 特征引导的匹配残差恢复

状态：RUNNING / VERIFIED。仅源域机制研究，无目标访问；性能最终以目标真实 TX 测试为准。用户尚在确认恢复网络目标测试沿用 clean-only，还是 clean＋六种 practical。该范围不影响源域训练，但未得到答复前不扩展目标场景。

## 方法与匹配对照

此前 R03 的损失只约束 IQ 重建与对数功率，没有直接保持原识别器的特征。本候选检验冻结识别知识是否能补充波形目标，不根据目标分数选择损失权重、预算、模型或阈值。两组从同一 R03 checkpoint 严格加载，重置 AdamW，只训练原 correction；coarse 始终冻结。两组每个物理源样本均包含 observed→clean 和 clean→clean 两个等权重例子。控制组使用 IQ NMSE＋0.1 log-power；实验组额外使用 1.0×原 F50 特征余弦距离。原 F50 与 checkpoint 内默认头不更新，也不重训或替换分类头。

实验组仅用 63000 条 L_s/U_s 的 clean IQ 生成固定原 F50 特征，训练中恢复输出经过同一冻结 F50 取得输入梯度，参考特征已分离梯度。损失不使用 TX 标签、TX logits、CE 或 KL；但借用了已由源域监督训练的识别器知识，因此应称“识别器引导的自监督恢复”，不能宣称整个系统完全无标签。控制组训练不计算 F 特征。

## 固定预算与数据

两组各追加 20 轮、4940 次更新，physical batch=256，每步对应 512 个相关 IQ 例子。LR 从 3e-5 预热 247 步，cosine 至 3e-6；WD=1e-4，梯度裁剪 1，FP32。两组均使用同一顺序和原 R02 缓存：q<0.15 为原 clean，0.15≤q<0.75 为 residual，其他为 full；另附每条 source clean 的 identity 项。未生成的 q<0.15 受损槽位不读取，不重新生成信道。累计每组 L/U 物理曝光为 126000/1134000，共 2520000 个 IQ 例子；名义比例 clean 57.5%、residual 30%、full 12.5%，报告实际数量。

原 R03 correction 已训练 20 轮，本次同一 correction 再训练 20 轮，累计 40；coarse 单独 20；复合恢复器沿继承链的优化成本为 60 个组件轮次，明确披露，不冒充总计 20 轮。原 F50 保持 50 轮，不增加训练。固定各组末轮 checkpoint，不据 V 早停、调权、更新状态或选 checkpoint。

## 预登记诊断与后续边界

首先真实 R03、原 F50 和 2 条 L_s 执行无 query smoke，检查输入梯度非零有限、F/coarse 无梯度和新 checkpoint 往返；PASS 后立即继续。各组完成训练后，用原 1800 条 V 的 clean＋六种 practical 生成 identity、父 R03、refined 三路默认头预测。每组全部 7 视图固定后，独立 CPU scorer 才接 V TX 标签与配对 clean，最多一条 scorer。逐 TX/RX/day 报告 accuracy、Macro-F1、修复/损坏/净修复、混淆矩阵和 NMSE。

源规则：实验组 low-urban accuracy 比同预算控制组至少高 0.5 个百分点，clean accuracy 相对原 IQ 降幅不超过 0.1 个百分点，六种 practical 平均 NMSE 不超过控制组的 1.05 倍。它只决定源机制是否值得扩大验证，不能替代目标性能结果。负结果保留，不调阈值、不挑种子或选择性重跑。本 run target_reads=0；后续目标范围与流程单独冻结登记。

## 运行与检查

N607 普通账号，GPU0 为 waveform_control、GPU1 为 frozenF_feature，每卡最多一条本 run 训练/预测 worker，初始化受共享容量锁、至少 12 GB 空闲显存与总 owner 进程少于 2 约束，CPU 每 worker 2 线程。各行独占输出，控制器无 CUDA。保留其他 owner 的进程、产物和代码，不向另一个对话发消息。

12 项聚焦检查通过，含分离参考梯度、参数归属、权限/预算负测、未生成槽位和七视图 truth-last 契约；一次独立 P0/P1 审查未发现阻断问题。N607 预检已核实普通账号、原 R03 source-frozen/commit、原 R02 cache 完整句柄、新 root 不存在及 GPU0/1 空闲。非法权限/父模型/代码、输出冲突、非有限、预算或预测缺失时保留产物并停止，不自动重试；一组技术失败不终止另一健康组。完整命令、环境、路径、矩阵与产物见 [experiment.json](experiment.json)。


## 启动核验

方法 commit：`d896e0552f1f17e7b421fb7d94d45af9256899b4`。发布归档只比较一次 SHA，远端 compile/import 通过；真实 R03＋原 F50 的 2 条 L_s 输入梯度、冻结参数和新 checkpoint 往返 smoke 通过后，控制器立即启动两组训练。

独立后态已核实两个实际 PID 的 CWD、完整 worker 命令、物理 GPU0/1 归属，以及日志从 0 增长到首轮记录。两组均完成 1/20 轮、247 次更新；控制组首轮总损失 0.023307，特征组 0.030273，其中 feature 项 0.006889。损失目标不同，不能直接以总损失高低判定优劣。两组本轮 L/U 曝光和 clean/residual/full 路由数量完全一致。

特征组的 63000 条源 L/U 固定参考特征已完成，生成用时 5.7205 s；F 未更新，不使用 logits 或 TX 标签。当前没有源 V 结果或新目标性能结果。GPU 运行和源域损失均不能作为性能改善证据。
