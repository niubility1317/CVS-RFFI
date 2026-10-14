# 源L浅中层统计与身份保护特征扰动

状态 LOCAL_VERIFIED，尚未发布。run 20261014-phase1-source-style-profile-manysig-s01-m1-r01，普通N607 CPU诊断，无新训练或目标测试。实际矩阵/路径见experiment.json。

依据是完整源梯度证据：U信号进入t1/t2但较小，合法源RX敏感度不同；它不足以证明均值/方差都是域噪声。因此直接测量实际t1/t2每包通道均值/标准差，并作固定身份保护干预，不按目标分数选择层位或强度。MixStyle较低连续层的统计扰动依据官方实现（https://github.com/KaiyangZhou/mixstyle-release）；射频有效性是待检验推断。

只读固定首seed2026092701 OLD final50/11100，来源/角色/祖先经既有source_proof。完整合法L_s6300一次，batch128尾28共50批，shuffle seed=modelseed+95001。clean和styled各一次共12600包次，U/target读取0；V缓存materialize后丢弃，无前向或求导。NoExplicitCFO及内部补偿、默认cosine6×160/scale30不改；headfit/backbone/optimizer/EMA更新0，所有params/buffers须逐元素不变。

层位固定t1、t2。p=.5、alpha=.1、strength1，无扫描；每batch/layer私有NumPy种子modelseed+batch701+layer1709。同真实L TX/day，不同sourceRX和真实physicalPID，无候选则self、改变0；不按hidden U标签配对。强制eval特征hook只是冻结干预，不是训练，不据augmented source错误自动否定可学习增强。真实z必须joint160，原头只评估不拟合。

输出：每层channel mean/std维数、总中心化SS、同TX/day内RX均值差异SS和同RX/day内TX均值差异SS；描述性fractions不作独立因果分解，不称域捷径/feature rank/迁移增益。逐批实际donor、eligible/changed计数、z L2/cosine及源原头预测变化，原head不换。一个模型seed不能写成三seed性能。

CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_style_profile_20261014_r01；命令python -u -m experiments.cvs_source_style_profile.profile；不可变输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-style-profile-manysig-s01-m1-r01。代码/config为实际发布Gitcommit，release_commit.txt和resolved_config读取该身份。CPU，CUDA_VISIBLE_DEVICES为空，Torch/OMP/MKL/OpenBLAS2线程。

停止规则：任何来源/角色/shape/nonfinite/donor/state/count直接缺陷停止诊断并保留产物，无自动retry，不动其他owner。无目标成绩或微小源涨分筛选。预期preflight/submit/dispatcher日志、逐row resolved_config/results/completion、整轮completion或failure；原IQ和权重不上传Git。

本地6协议负例、保护配对、input梯度活跃和全局Torch RNG保持PASS。一独立候选P0/P1审查后，一Git代码/config提交、一个archive一次本地remoteSHA比较/compile、真实CPzeroIQnoquery smoke，PASS立即运行；随后独立PID/CWD/cmdline/CPU/log增长及终态核验。没有其他门槛。
