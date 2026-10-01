# CVS 已知激励相对响应：源域预登记

状态 PLANNED，尚未发布N607。原物理L6300/V27000，U56700unused；普通身份CE、无增强、无域骨干；4个新scratchseed，E200×50、batch128、AdamW2e-4/wd1e-4/cosine1e-6，FP32无裁剪。177025参数，原残差164225；新增12800用于公共激励的相对复响应坐标，不是单纯加宽核心。

固定L-STF相关库320时移，80:160四20点周期，wrappedCFO校正、12tone相对响应、质量和显式相对CFO共29维；小MLP160通过零初始化门控接入原残差身份特征。同seed核心从零初值和全部初始logits严格相同，不继承checkpoint。仅参考分支有非退化flat复增益/相位归一化，原raw身份路径保留；不宣称全网不变、任意RX消除或TX硬件辨识。

受控合成先在100Msps形成TX三阶/IQ/记忆，再经信道和RX，12tone投影后25Msps/CFO/RMS；5TX配置×6RX/信道配置仅作冻结机制诊断，不用于增强或虚构身份准确率。保留TX/RX相同波形负例；不复现完整WiSig均衡/瞬态/噪声。

源矩阵共8记录：4新＋4原residual_fusion源控制，只读原指标。完整物理角色/表示/预算/scratch来源已实时核实，不加载权重、不读目标指标。固定四seed0.5V+0.5最差源RX性能最高优先，完全并列后才比较V/最差RX、MAC/参数和固定顺序。

新候选胜出后默认20261002-phase1-cvs-reference-clean-manysig-m24-r01：4新clean预测+20旧冻结预测复用，168000个原物理query、6TX、7RX，prediction全部固定后独立truth-last评分。原基线胜出时保留历史测试、不重跑历史、不测未选新候选。目标成绩不反馈调参、选择或重跑。

10模型/数学检查、13协议负测PASS，本地4一次性CPU模型/12CE更新PASS。初始三阶TX/RX反例bit-equality断言因复数乘法舍入失败，改为1e-12容差；数值相同波形证据保留。模型/数学独立P0/P1审查PASS；执行独立P0/P1审查PASS，无阻断。远端Torch2.1烟测、真实训练/成本与新测试均N/A。保护所有健康任务和历史产物。

[前瞻结构与数学边界](../../../docs/CVS_REFERENCE_RESPONSE_HYPOTHESIS_20261002.md) · [源控制实时核实](evidence/source_control_preflight.json) · [本地CPU烟测](evidence/local_cpu_smoke.json)。
