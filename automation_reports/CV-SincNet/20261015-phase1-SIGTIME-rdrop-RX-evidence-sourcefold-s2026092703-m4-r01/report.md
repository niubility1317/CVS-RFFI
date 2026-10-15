# SIG_TIME＋RDrop：置信度分离与跨 RX 混淆证据

状态 PLANNED，四个固定 E20 源域行已登记，尚未发布。上一轮差分 RDrop 在 RX1 少错37但 mean CE+.042982，固定资格失败已关闭。本试不再使用差分投影、不复用其权重，也不修改其判据。当前只优化 SIG_TIME＋RDrop，既有 E50 三种子84.340079%里程碑保留，不据目标成绩改方案。

三种可执行方向：监督证据切空间投影（有通用梯度手术先例且局部约束不能保证实际 AdamW 行为，暂缓）；跨 RX 完整概率一致（可能强迫不同包置信度相同，暂缓）；本试的置信度分离与跨 RX 关系一致（优先）。新核心假设仅一个：可把自样本置信度一致与跨接收机身份混淆关系分开学习。

记 p 为六类概率，b=[p_y,1-p_y]，q 为排除真类后重新归一化的五类概率。KL 链式分解为 binary KL 加 (1-p_y)×conditional KL。本试二元项始终是同包两次 dropout 的双向 KL；关系项用各自 (1-p_y) 的停止梯度权重，交换同身份样本的 q。该关系项对真类原始 logit 的直接梯度为0，不保证训练后置信度、CE或AdamW位移不变。原 CE 与默认 W 始终正常联合训练，所有 KL 图也可更新 W；部署仍是原 joint160。

四条件：FULL 为精确原 RDrop；LOCAL 只分离关系权重梯度；SAME_RX 使用同身份同 RX 的关系；CROSS_RX 使用同身份其他 RX 的关系。每个身份选每个至少含2包的 RX 前2包，至少要有2个这样的 RX；同 RX 两包互换，跨 RX 将已选 RX 组按批内首次出现位置排序，将2M个索引中相隔M的位置配成一对（M为已选RX数）。其余样本保持自配对。每个有效anchor严格只有1个peer，两组配对均为对称置换，有效配对数量完全相同，关系矩阵行列和均为1，不抽额外随机数；数据集原生打乱决定批内顺序。RX 由实际 raw domain→compact map→rx_i 还原，并每步核对原 L batch 的 rx_i，不用domain除法猜测、不读U标签。

固定下一模型 seed2026092703，四行各从零20轮（13L+7原U）、222步/轮；不从前两源seed挑父。SIG_TIME/RAW_HZ/普通增强/LS.01/原cosine6×160/scale30/eps1e-4/AdamW lr.0002 wd.0001/旧门控自身EMA均不变。只复用增强后实际L128做第二次原生dropout前向，无增强重抽、无额外训练IQ。四RX有效角色L5040/U45360hidden/V21600，RX1在模型、U、EMA、calibration、cache和初始化中先排除，最终1260合法held L只在fixed E20读默认头。

科学资格在首次训练前固定：所有行final L CE<=.5；CROSS_RX selected anchor fraction>=10%；它在RX1的错误严格少于另外三组，mean CE均不高于对照。V只健康诊断，不选终点。PASS仅允许另行登记主实验；FAIL关闭固定配方，不扫alpha、seed、层或轮次救跑，也不访问target。

预算4×20=80model-epoch、17760更新，每行568320额外L身份前向、396368 U循环观察（物理U45360），共2273280和1585472；每行1260 held前向和1404额外L读取，其中144fit anchors未使用。另8合成包用于各own初始CP无query smoke。最终在同一次1260前向保存逐包CE、真类概率、真类margin，用于判断错误率与CE权衡，无另训头或新增评价前向。

CPU/CUDA各57负例、5资格否决、198相同初态、实际原CE/logits/RNG/增强后IQ复用、浅中层与SIG/defaultW有限非零梯度PASS；KL链式分解与原loss/gradient一致、binary项不变、关系真类raw logit梯度0、类/RX置换及配对质量守恒PASS。每设备1024合成包，0真实IQ/CP/优化/头拟合。这些是实现检查，尚无训练收益。

最小发布：一次聚焦P0/P1；Git提交推送及远端OID；一个新root；一次ordinary N607 preflight；单archive传输SHA比较和remote compile；worker首步own CP无query smoke后继续。八卡资源锁考虑全部owner/preCUDA，每GPU最多3进程、CPU2/loader0。技术错误保留产物、停pending，不重复Popen、不杀健康或无关进程；低分不重训。原始预测、CP与日志留远端，公开报告只保留聚合。

环境ordinary N607 CVS-RFFI，CWD PROJECT/releases/cvs_SIGTIME_rdrop_RX_evidence_source20_20261015_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_evidence20/runner.py experiments/cvs_sigtime_rdrop_evidence20/configs/matrix.json；唯一owner codex/root/SIGTIME-RDrop-RX-evidence-20261015；root PROJECT/runs/20261015-phase1-SIGTIME-rdrop-RX-evidence-sourcefold-s2026092703-m4-r01。执行commit在release_commit.txt；模型代码依赖b3ac442e2d33a5b315fb3251c30e0ad7083566bc，仅代码，不继承已评分模型。

文献边界：[R-Drop](https://arxiv.org/abs/2106.14448)已有同包双随机子模型KL；[DKD](https://arxiv.org/html/2203.08679v2)已有真类/非真类KL分解及质量权重耦合分析。本试不把该代数分解宣称为创新，具体待证贡献是平衡跨RX身份关系传输、自包二元置信度与配对质量控制，也不声称全球首创。只读原文已核实，有限检索不能排除全部先例。

发布前修正一个直接对照混杂：原草稿 CROSS 多peer平均、SAME单peer，现统一为单peer置换；发生在任何真实IQ或训练前，不改变预算/科学门槛。唯一初审PASS后仅请求该项定向复核，不重做未变全审。

独立next-run P0/P1初审PASS；单peer修复后只做一次该项定向检查PASS，覆盖2/3/4个有效RX的对称置换、身份/RX合法性、质量与RNG，无P0/P1。当前代码具备发布条件。
