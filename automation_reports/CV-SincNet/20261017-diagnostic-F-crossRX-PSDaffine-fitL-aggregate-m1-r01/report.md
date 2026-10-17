# 正仿射源 PSD 关系：最小源依据预登记

状态：已实现，真实汇总上的本模型拟合尚未运行。唯一问题是固定 source RX 对之间是否存在跨 TX/day 的正仿射谱收缩，不是识别训练或新增 A_g 成绩。

输入：复用 20261017-diagnostic-source-nonDC-spur-crossfit-manysig-m1-r01 已保存的 cell_PSD。原有效 L5040、RX3/4/6/8、六 TX、day1/2/3、每格70包；两窗 rectangular/Hann，2×6×3×4×256，共144均值谱。该源峰值报告历史已暴露，本轮仅读轴/schema与生成代码后锁定模型；不声称新盲测。只使用 cell_PSD，不用曾选择的峰、prominence、候选频率或谱峰晋级条件。U/V/heldRX1/target 不用，不打开新 IQ 或权重，不重验未变数据。

每个单元谱除256成为总和1的向量 P。固定模型 Pdst=aPsrc+b，0≤a≤1、b逐点非负且sum(b)=1−a；同一拟合内 a/b 对所有 TX/day 共享。对固定 a，将训练目标均值减a倍输入均值投影到质量1−a的非负单纯形；64次 golden-section搜索及两个端点求凸最小平方误差，1e−18同值偏向较大a。拟合仅使用其余5TX×2day的10个单元；全部18个同时留TX/day折、12个有向RX对、两窗共432拟合。保留全部频点，不移除DC、边缘或前导谐波。

对照：输入原谱（identity）、训练侧目标RX均值（不含被留TX/day）、同heldday下全部5个错误循环TX源谱。错误TX只作统计控制，不作样本负标签或身份分类器。

每个有向RX/窗固定条件：pooled SSE相对identity至少下降25%、相对目标RX均值至少下降20%；至少15/18折优于identity；每个TX及day合并风险不反向；正确TX相对最佳错误TX的SSE差至少为目标均值基准SSE的10%；全部折a在.5至.995，两个参考总风险均大于1e−12。六个无向RX对中的每对，必须有至少一个相同方向在两窗均通过；不得只挑部分RX晋级。

任一必要条件失败，关闭固定正仿射PSD模型，不换RX、频点、窗、系数范围、门槛或另选方向规则救跑，不生成IQ或自动E20。通过也只说明此统计关系可迁移，不能识别纯RX噪声、真实硬件来源或TX身份保持；仍须另外确定有界IQ算子、默认头源身份保护与匹配E20。不能把谱风险下降当作准确率。

物理动机及限制：不相关加性成分在期望功率谱上相加；归一化后可产生收缩与非负偏置，但逐包归一化、信号依赖噪声、传播/EQ/TX效应均可能破坏此近似。PSD无法单独确认零均值、相位、噪声来源或唯一的RX变换。本模型是可被源数据否定的统计假设，不是对当前数据的物理断言。[Cekic等原研究](https://arxiv.org/html/2002.10791v3)说明混杂因素需要信号模型；不把其CFO/信道实验移植为本模型证据。[SciPy CSD文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html)区分互谱估计；这里没有从PSD臆造互谱、独立性或真实接收机替换。

模型边界：无模型构造、头拟合、训练父权重、teacher或新损失。后续仍F＋跨RX同TX／SIG_TIME／统一Sinc／joint160／默认cosine头正常联合训练／NoExplicitCFO／原U／E20。真实同包映射问题保持独立，不是本模型前提。

本地原生Fpython/NumPy，CPU2，CUDA隐藏；CWD为本worktree根。命令：Fpython -X utf8 native_run.py Fpython -X utf8 -m experiments.cvs_psd_affine17.run；随后独立 recount 模块。唯一不可变输出 local_artifacts/F_PSD_affine_20261017/r01。Git commit为代码/config身份，执行时保存实际HEAD。没有N607发布或远端新运行；0模型阶段不加载checkpoint作无关smoke、不生成额外许可。

系统技术停止：输入原source角色/轴/schema不符、非有限、单纯形约束失效、输出冲突或实现失败时保存failure和原产物，不覆盖或自动重试；完整科学判定保留所有单元/方向/折。预期产物resolved_config、execution、folds.jsonl.gz（全a/b及SSE控制）、summary、completion或failure、独立KKT/算术复算及本报告/登记。新IQ/CP/NN/更新/teacher/target均0；主拟合与复算各读取一次既有汇总，实际CPU成本分列。默认逐行测试不适用0训练/0身份预测的本统计诊断，不冒称完成正式Phase1测试。

本地25个输入/权限反例PASS；1296个合成PSD拟合、1个闭式拟合及单纯形KKT检查PASS，真实源汇总读取0、真实IQ/CP/NN/更新/target0。一次限定P0/P1独立审查PASS：未发现会直接使本次汇总拟合错误、越权、破坏、无法启动或不能产生合法统计结果的缺陷；审查未读取实际源PSD值或执行真实拟合。完整联合目标ACTIVE，仍缺有效Ag、稳定As、四因子及交互、跨seed与正式clean＋六practical证据。
