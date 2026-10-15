# SIG_TIME＋RDrop：同权重同输入算子梯度诊断

状态 PLANNED。本轮是冻结源模型上的机制诊断，不新增训练。包络 E20 四行已完整失败关闭，原版 RDrop 的 E20/E50 三种子正收益保留。新诊断要区分不同一致性算子的局部梯度方向与强度，不能把上轮退化直接归因于停止梯度或置信度。

固定来源为 20261016-phase1-SIGTIME-rdrop-margin-envelope-sourcefold-s2026092701-m4-r01 的 FULL-RX1-s2026092701 final20 student，训练代码 7827a9a6b04764e690ecaf90457666435017c824，自己从零20轮、4440步、fit RX3/4/6/8、day1/2/3、6TX、L5040/U45360hidden/V21600，final-only、自身EMA、0target、默认头联合训练。按原版控制的角色固定选择，不按分数排名。调用已有 source_proof 核对真实角色与祖先后才加载；其他 RX 契约或未知权重禁止。父训练成本20轮单列，本诊断更新为0。

从既有 L6300 carrier 的原生索引按 TX/RX/day cell 顺序取每格前2包，仅 fit RX3/4/6/8，144个唯一有标签 L；核对其原生 opaque ID 属于该父模型实际 L5040。U/V 只核对既有角色句柄，不读取 IQ/真值；held RX1、target、星地均不访问。数据不新划分、不重新验证或生成摘要签名，复用 VALIDATED_ONCE 源句柄。

同144输入测 CLEAN/NATIVE_AUG 两个上下文，后者按原 native build_aug_base_cfg、make_augmentor、configure_augmentor_for_epoch(min(20,13))，只生成一次普通增强给两次前向复用，无星地拼接。每上下文固定4个 dropout seed2026101601/02/03/04；增强 seed=对应seed+43210，在独立保留流中抽样，所以不消耗配对 dropout 流。四次抽样不是四个模型种子。

每次只计算两次原生 identity-only 前向，所有算子复用同一真实 logits/joint160 图。比较 CE（两次 LS.01 CE 均值）与已有 FULL/MEAN/CONFIDENCE/ENVELOPE，不改任何算子；只做 autograd.grad，记录 logits、joint160、t1.pw、t2.pw、SIG_TIME.local_projection、默认 W 的梯度 L2、对 CE/FULL 余弦与范数比，以及 CE＋算子总梯度对 CE 的关系。joint/logit 两视图共同／差分能量是正交代数分解，不等于共享编码器中心不动；不同视图 Jacobian、clip、AdamW moments 均未模拟。只测3处指定特征参数，不冒称全部骨干逐参数结论。

参数仍允许 autograd 以测量导数，但没有 optimizer/backward累计/EMA/头拟合或checkpoint写入。每次结束严格比对全部state张量、nonpersistent buffers、空parameter.grad和特征计数器；train模式仅用于随机dropout，随后恢复eval。CPU实际模型9协议负例、30记录、真实joint160梯度连通、浅中层/SIG/W有限非零梯度与state/gradslot不变 PASS，288合成包、0数据/CP/更新。worker首步自己的固定真实CP做2合成包无query smoke，PASS后立即继续。

成本：一个固定源模型，144个真实 L materialization；2contexts×4draws×2forwards×144=2304身份前向包次，另2个合成smoke包；40次autograd.grad、240个梯度记录，0优化更新、0训练轮、0另训头、0U/V/RX1/target IQ。不输出分类决定或性能指标，不做target scorer。CPU144均衡batch与历史GPU128训练批次不同，不能称历史训练步骤重放。

没有科学晋级阈值，不以该诊断挑alpha或救跑包络。结果只用于解释局部梯度，不能证明历史准确率差的原因、真实AdamW位移或通用特征能力；新机制若提出，必须另行固定方案与公平源训练验证。

唯一owner codex/root/SIGTIME-RDrop-operator-probe-20261016，ordinary N607 CVS-RFFI，CPU2/loader0，CUDA_VISIBLE_DEVICES为空、占用GPU进程0。CWD PROJECT/releases/cvs_SIGTIME_rdrop_operator_probe_20261016_r01；命令 `CVS-RFFI/bin/python -u experiments/cvs_rdrop_operator_probe/runner.py experiments/cvs_rdrop_operator_probe/configs/request.json`；不可覆盖root PROJECT/runs/20261016-phase1-SIGTIME-rdrop-operator-gradient-source-s2026092701-r01。一次聚焦新诊断P0/P1、一个实际Git提交、一次preflight、单archive SHA比较和remote compile；发布后核对PID/CWD/argv/CPU环境/日志增长或终态产物。技术错误保留输出、失败且不自动重发，不改变健康或其他owner。预期8批次、progress/completion/selected_L和stdout；原始物理ID、CP及日志不进Git，公开只存聚合。实际Git身份在release_commit.txt。

一次独立新诊断 P0/P1 审查 PASS，未发现 P0/P1。已核对源证明先于 CP 加载、144fitL opaque ID/标签边界、原生增强与独立随机流、冻结状态、CPU bootstrap 和不可变输出；未重复训练机制审查。见 review.json。
