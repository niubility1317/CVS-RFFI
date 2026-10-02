# CVS 相对滤波能量保留：正式源实验预登记

状态 RUNNING，8 个源训练已核实健康运行，尚未证明性能提升。性能优先；普通 CE、无增强、仅身份骨干、原物理划分、clean-only。

[数学假设和完整执行矩阵](../../../docs/CVS_ENERGY_IDENTITY_HYPOTHESIS_20261002.md)。六个复数时间/行为 block 共用跨通道和时间的 RMS 分母，归一化本身保留滤波通道相对能量。后续学习尺度与径向门可改变比例；输入已有单位 RMS，未恢复绝对发射功率。received 能量包含 TX/信道/RX，不宣称唯一 TX PA 参数。

两个候选均 202553 参数：energy_equivariant 原输入精确直通，alpha=0；energy_half 全256点固定 alpha=0.5 波形校正。同一核心、深宽、参数结构，无新增学习参数、对齐标量或 conditioner。各四 seed2026092701..04，8个模型从零训练，无权重继承。

L6300/V27000/U56700unused、split392005、E200×50、batch128、AdamW2e−4/wd1e−4/cosine1e−6、完整FP32。实际配置/物理角色/来源按既有记录，目标不进入源执行。详细文本、step JSONL、epoch JSONL/CSV，实测CE/LR/全梯度/固定alpha前后和梯度N/A、V/最差RX、耗时/显存；记录实际六block归一化前后分数（每epoch末源batch和冻结公共干预）、90源单元与5TX×6RX公共干预。

8新源记录+4原残差源指标，四seed E200 mean(0.5V+0.5最差RX)最高优先，完全并列才比成本。物理误差只报告，不参与选模或停机。固定候选与源规则在新target访问前登记；全负结果保留。

新候选选中后默认 20261002-phase1-cvs-energy-clean-manysig-m24-r01：4新clean预测+20旧冻结控制，24行/168000query/6TX/7RX，独立truth-last评分。原残差胜出则保留已核实历史测试，未选新模型测试N/A，不重跑旧实验。目标评分不得回流结构/alpha/超参数/选模/选择性重跑；无LEO/SFT/新增类。

每GPU最多两训练任务且≥12GB空闲，仅使用剩余容量；唯一launch owner，不热改/停止/重启健康任务。59项定向源检查覆盖PASS；17项条件clean与17项前轮回归检查PASS；源与clean各一次独立P0/P1审查PASS；8个一次性CPU模型/24次CE验证PASS。未读取正式数据/历史权重/目标，不把合成检查称为识别提升。

[本地验证](evidence/local_validation.json) · [实时源控制与输出不存在](evidence/source_control_preflight.json) · [CPU检查](evidence/local_cpu_smoke.json)。

## 实际启动核实

VERIFIED：immutable release `4a9170b768bbb39e4c256d377fbe60ff294a2f0d`，dispatcher PID 1480567，8 个 worker 分别在 GPU0..7；CWD/argv/run-root/实际 resolved/config、每轮50步/6300样本和日志增长已独立读回。各实际 backend flags 与固定完整 FP32 一致，源数据角色相同、无 target/增强/域骨干。远端一次 CPU 8 模型运行/梯度/物理 smoke 完成，真实初始 checkpoint roundtrip 在各源 worker 内完成。当前仅训练进度，无最终源或新测试结论；不热改、停机或重启健康任务。

## 条件测试路径准备

本地已准备新候选冻结后的独立 clean 路径：source12记录重算、actual完整FP32/物理角色/scratch/E200/payload/strict模型加载核实后才读取query；仅选中4个新模型加20个旧冻结预测，共24行，同seed配对原残差及常见基准。energy17项（含两种真实模型 checkpoint 合成预测回读）与 fractional17项回归检查通过，独立P0/P1审查PASS。没有生成真实clean配置、访问target或修改正在运行的源release。[验证](evidence/conditional_clean_validation.json)。
