# CVS 已知激励相对响应：源域预登记

状态 PLANNED，尚未发布N607。原物理L6300/V27000，U56700unused；普通身份CE、无增强、无域骨干；4个新scratchseed，E200×50、batch128、AdamW2e-4/wd1e-4/cosine1e-6，FP32无裁剪。177025参数，原残差164225；新增12800用于公共激励的相对复响应坐标，不是单纯加宽核心。

固定L-STF相关库320时移，80:160四20点周期，wrappedCFO校正、12tone相对响应、质量和显式相对CFO共29维；小MLP160通过零初始化门控接入原残差身份特征。同seed核心从零初值和全部初始logits严格相同，不继承checkpoint。仅参考分支有非退化flat复增益/相位归一化，原raw身份路径保留；不宣称全网不变、任意RX消除或TX硬件辨识。

受控合成先在100Msps形成TX三阶/IQ/记忆，再经信道和RX，12tone投影后25Msps/CFO/RMS；5TX配置×6RX/信道配置仅作冻结机制诊断，不用于增强或虚构身份准确率。保留TX/RX相同波形负例；不复现完整WiSig均衡/瞬态/噪声。

源矩阵共8记录：4新＋4原residual_fusion源控制，只读原指标。完整物理角色/表示/预算/scratch来源已实时核实，不加载权重、不读目标指标。固定四seed0.5V+0.5最差源RX性能最高优先，完全并列后才比较V/最差RX、MAC/参数和固定顺序。

新候选胜出后默认20261002-phase1-cvs-reference-clean-manysig-m24-r01：4新clean预测+20旧冻结预测复用，168000个原物理query、6TX、7RX，prediction全部固定后独立truth-last评分。原基线胜出时保留历史测试、不重跑历史、不测未选新候选。目标成绩不反馈调参、选择或重跑。

10模型/数学检查、13协议负测PASS，本地4一次性CPU模型/12CE更新PASS。初始三阶TX/RX反例bit-equality断言因复数乘法舍入失败，改为1e-12容差；数值相同波形证据保留。模型/数学独立P0/P1审查PASS；执行独立P0/P1审查PASS，无阻断。远端Torch2.1烟测、真实训练/成本与新测试均N/A。保护所有健康任务和历史产物。

[前瞻结构与数学边界](../../../docs/CVS_REFERENCE_RESPONSE_HYPOTHESIS_20261002.md) · [源控制实时核实](evidence/source_control_preflight.json) · [本地CPU烟测](evidence/local_cpu_smoke.json)。

## 远端启动独立核实

**VERIFIED / RUNNING**。immutable source commit `bc26714f58932bc7d50b393b4c8e9a3d5a6937e1`，dispatcher PID `1219624`；独立读取 `/proc` 的CWD/argv与实际release一致。N607 Torch2.1+cu121 CPU4个一次性模型、12次CE更新PASS（包括弱/零输入梯度及30组合冻结机制检查），没有正式IQ/旧权重/target访问。

| seed | GPU | workerPID | 已完成epoch | 总CE | 源V | 最差源RX | 已测response门控均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026092701 | 0 | 1219638 | 21 | 0.176064 | 95.6111% | 89.7037% | 0.053161 |
| 2026092702 | 1 | 1219727 | 21 | 0.180648 | 95.0111% | 88.3148% | 0.057868 |
| 2026092703 | 2 | 1219814 | 21 | 0.180431 | 95.6370% | 90.1667% | 0.057900 |
| 2026092704 | 3 | 1219963 | 19 | 0.195563 | 94.9741% | 87.4444% | 0.060311 |

以上为独立读取时的中途进度，不是最终选模或性能提升结论。4行实际配置为177025参数、L6300/V27000、50步/epoch，源V只读；日志持续增长，门控由CE从零初值开始实际学习。健康训练继续，不停止、重启或热修改。完整逐步/epoch日志、源分层、冻结机制及GPU成本尚待E200完成；clean新预测/评分尚未启动，仍N/A。

条件clean执行链已经补齐：reference_source重新核对4新+4旧源记录、完整scratch payload与实际response激活，只有真实source-selected新候选才允许query。52聚焦用例PASS（原47+本轮5），独立P0/P1审查PASS。复制模板的旧相位规范化标题、候选数、owner和参数文案已在生成正式报告前纠正；这不修改正在运行的源release，也不读取target。

[启动与实际进度读回](evidence/launch_readback.json) · [远端CPU烟测](evidence/remote_cpu_smoke.json)。下一步：低频核实同一4worker；E200完整日志分析/8源记录冻结；选中后默认clean4新+20旧truth-last；原基线胜出保留历史test、拒绝未选query。目标持续，尚未完成。
