# 预登记：F 原始监督份额三组检验

状态 REGISTERED_NOT_RUN。精确矩阵、命令、环境/CWD、路径、seed、GPU及停止规则见 experiment.json。基座用户最新明确指定 F，全部scratch E20，不继承E50权重。

# F 基座下的泛化增强与星地增强研发

当前唯一基座采用用户指定 F：三路独立 SIG_TIME＋CENTER_25K＋CENTER_UNITS。历史 A–F 执行提交为 `09554c3a06c976c53efc4dbc61f5c501a1e7b06e`；实际实现位于 `experiments/cvs_rdrop_sinc_units`，三路投影共同进入原 joint160 和默认 cosine 头，不投票、不新增本轮部署分支。本轮复制其四个纯实现模块，不加载 E50 模型、optimizer、EMA、统计或 teacher，所有模型从零总 E20=13L+7U。

F 的实际数学：Fs=25 MHz，s=25 kHz，p=p0+s*q；Sinc lr=alpha*Fs/s、epsilon=epsilon*s/Fs；使用物理 Hz 梯度与全模型一次 clip5，再恢复 q 梯度；原点在内的物理衰减 q-=alpha*lambda*(q+p0/s)，Sinc AdamW 自带 decay=0。其余网络/默认头正常联合 AdamW/cos，NoExplicitCFO 保持。输入仍原归一化 IQ 2×256，任何新视图在 IQ 处生成，不修改参考响应、补偿、测试 IQ 或 practical。

## 已完成探索的结论

|证据|实际结果|不能据此声称|
|---|---|---|
|旧基座普通 practical paired CE|旧因子实验 B01−B00：low urban +20.1708pp，clean −2.8488pp|clean/urban 双提升；F 的收益|
|旧 bounded G 与 practical 联合 B11−B01|low urban 约 +0.0393pp，clean +0.0131pp|泛化增强已独立有效|
|旧普通增强五源 RX 训练|均值 +2.0296pp；RX6 −0.9259、RX8 −2.5926pp，仅 3/5 正|跨 RX 稳定性成立|
|冻结模型的 PA 去除干预|五 RX 输入保持率均值 +1.1266pp，准确率 +0.6002pp|删除 PA 一定改善训练|
|随后 NoPA E20 训练|RX1 比无增强 +3.0556pp，但比原 ordinary −3.3519pp，macroF1 −3.5120pp|NoPA 可晋级；冻结诊断等同训练因果收益|
|HQ 独立噪声|冻结 EMA 默认头源 L_dev720：low urban 64.8611%→70.9722%，0 更新|目标测试提升或训练成功|
|原始 SIG_TIME RDrop E20|三 seed clean 比 ORIGINAL +0.8272pp，2/3 正|IQ 泛化增强成立或星地同步获益|
|F 历史 E50|clean 84.2702%±0.2754pp，F−A −0.0698pp；最差 RX 均值 +2.1347pp|E20 性能或总体均值优于 A|

课程、间隔与对齐的历史增量/负结果保留；它们未解决 clean 代价，不作为本轮叠加项。旧源域训练、冻结干预与正式 target 指标保持各自语义，不能跨行拼接收益。旧 RAW_HZ raw-anchor 仅本地原型，未登记、发布或训练；本轮改为 F 后重新检查实际图。

## 唯一假设及最小否证实验

普通增强替代原始视图的错误迁移，可能通过固定原始 IQ 的监督份额减轻。先在合法源域独立验证，不先重启完整混合矩阵。固定 held RX1、seed2026092701；三组均从零 E20、4440 updates，L5040/U45360/seenV21600，held 源 V5400 仅 FINAL；held L/U 不进入训练。split392005、原数据角色与全部 6 TX 保持，U 真值隐藏。

定义 RD(v)=同一份 IQ 两次随机子网络前向的 mean CE(LS.01)+对称 KL(alpha1)。

|行|第一对|第二对|总 L 目标|
|---|---|---|---|
|A0|原始 x|原始 x|0.5 RD(x)+0.5 RD(x)|
|A1|G(x)|同一次 G(x)|0.5 RD(G)+0.5 RD(G)|
|A2，唯一主候选|原始 x|G(x)|0.5 RD(x)+0.5 RD(G)|

三组均每个 L 样本 4 次完整 CVS/default-head 前向。A0 不生成无用 G；A1/A2 每批只生成一次实际原 ordinary G，使用相同私有增强随机流、同一实际调度。G 保留历史正常 PA 等真实开关，仅为待验证的固定研究视图，不能逐项预称身份保持。每个 RD 对内两前向共享 IQ；没有 raw/G 跨视图一致性、特征对齐或新门控。全部组保留旧 U/ownEMA/oldgate 配方，完整对照实际 L/U 物理顺序。RDrop 是原训练正则，不称作 IQ 泛化增强。

A2−A0：held accuracy≥+1pp、macroF1≥0、seenV 均值损失≤0.5pp；A2−A1 的 accuracy 与 macroF1 各损失≤0.5pp，且预算、有限更新、覆盖、同物理序列全部成立。失败关闭本候选，不换主行、强度、RX 或 seed 救援。通过仅允许另登记原固定其余四源 RX，不能直接宣布 Ag/As 成功。跨五折预定均值≥+1pp、≥4/5 正、最差≥−1pp、mean macroF1≥0、每折 seenV 损失≤0.5pp，同时完整报告 A2−A1。

新成本 3×20=60 模型轮、13320 更新；每行 2273280 个 L 网络前向，是旧双前向研究的 2 倍。U/EMA/源验证/held/CPsmoke 分别按实际计数，旧研究成本不隐藏。真实 scratch checkpoint 无 query smoke 为 worker 第一步，PASS 即继续。

本轮 source-only，不访问 target、不生成 As。若源域收益通过固定五折确认，再独立登记 full-source B00/B10/B01/B11，匹配 F/E20、前向/监督尺度、U 配方与初始化规则。raw、G(x)、原 practical As(x) 从同一原始 IQ 分别生成，先不叠 As(G(x))。As 保持原 residual/no-EQ 接收处理依赖，不偷换弱信道。正式行冻结后自动 clean+六 practical prediction，再独立 truth-last；目标分数不反馈晋级。low urban>60%、clean 正增益与 suburban 差距≤2pp（探索目标）目前均未证明，跨 seed 未验证。

本地F实际图PASS：206初态张量与权威F构造一致，三投影/defaultW/gain/q梯度有限非零，物理clip/AdamW映射匹配，四前向两RD归一化及同一G复用正确，10负例通过。合计合成网络前向3584、增强输入512、合成optimizer更新12；0真实IQ/CP。两次本地fixture失败及修复如local_verification记载，不是远端实验失败。唯一限定P0/P1审查PASS；登记与compile通过。尚未发布，无新成绩。

## 实际启动已核实

VERIFIED / RUNNING。代码 44fc0bb689d03a65dc18a897b0e8e28324c7958a；三行PID/CWD/argv/GPU、scratchCP noquery smoke、日志增长、实际F三路/统一Sinc物理clip、E20=13+7、LS.01、NoExplicitCFO/默认头、ownEMA与source角色已独立核实。单archive transfer与remotecompile通过，未热改release/健康任务。尚无FINAL heldV成绩。
