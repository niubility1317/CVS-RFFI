# 固定 SIG_TIME：R-Drop 源检验剩余两模型

状态：PLANNED，实际实现 `b0903780256454eaebcbd1f7da66af359a39712e`。保持 SIG_TIME 和既有默认分类头，首模型源检验 PASS 及不变扩展规则已在 `7e83b126a242daeda24bed94a8a319df1d2a8e9e` 上传。此处仅增加模型种子 2026092702/03 的源端零更新检查；首模型复用原结果，不重复前向。

沿用 [R-Drop 原文](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) 的同输入两个随机失活预测双向 KL，alpha=1。paired CE 为两次默认 CE 均值，smoothing=.01；同一 strong IQ 使用原 .015 噪声，两个预测均可微。另作相同输入的 eval 参考。检查 t1、t2、SIG 投影及现有类别 W 导数，模型、buffers、W、optimizer 均无更新。W 导数测量不构成单独训练分类头。

两个父模型固定为原 DRO 剩余种子运行 `20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s0203-m8-r01` 的各自 SIG_TIME final50，执行 `a283291279cb01b939325cf5ccba1d0a671b28c4`；加载前核对实际源契约和从零训练来源。不选择 Sinc 变体或按目标成绩挑父模型。

复用既有 VALIDATED_ONCE 契约，L6300 / U56700 隐藏 TX / V27000、split392005、标注率10%、源 RX1/3/4/6/8、day1/2/3。每模型使用与首模型完全相同的900唯一物理 L（5批180，每TX/RX/day每批2包），strong/dropout RNG2026101519+cohort。U张量和真值不读，V由共享缓存返回即丢弃，不前向。无目标访问。

新增预算：0训练轮、0optimizer和头拟合更新；2模型、1800模型样本观察、5400次包前向、10梯度批、40参数层记录、80向量。加首模型共3模型、8100前向、60层、120向量，但独立物理样本仍为同900包。真实checkpoint no-query smoke另用2合成零IQ。普通N607既有CVS-RFFI、FP32 CPU2线程、CUDA_VISIBLE_DEVICES为空，不占GPU。

各模型沿用首规则：5批KL均正；t1/t2/SIG的KL导数各5/5非零；聚合KL与paired CE导数余弦各≥0；状态不变。只有三模型全部PASS才允许另行登记正式训练研究，不自动发布。失败完整保留，不扫系数、dropout、父权重或阈值。源端通过不证明早期训练、U准确率、跨域提升或84%。

CWD `PROJECT/releases/cvs_source_rdrop_seeds_20261015_r01`；命令 `python -u -m experiments.cvs_source_rdrop_seeds.audit`。不可覆盖输出 `PROJECT/runs/20261015-phase1-source-dropout-consistency-manysig-s0203-m2-r01`，两个row与完整配置见 experiment.json。本地发布入口 `python -m experiments.cvs_source_rdrop_seeds.publish --output LOCAL_ARTIFACTS/source_rdrop_seeds_delivery_r01`。自动preflight/no-query smoke通过即继续。单发布包校验、远端编译和PID/CWD/argv/日志增长核实沿用既有流程。技术停止限本run输入/来源/形状/非有限值、状态变化、独立复算失败或输出冲突；保留产物，不自动重试、不操作无关任务。

预期产物：preflight、逐row resolved_config、实际源导数、analysis、独立复算，以及三种子 combined_three_seed_source_analysis、source_qualification、completion和日志。13个针对性负例、旧默认复算接口、2个实际原生网络梯度上下文已PASS；一次独立 adapter P0/P1 审查PASS。真实远端checkpoint和源运行待核实。
