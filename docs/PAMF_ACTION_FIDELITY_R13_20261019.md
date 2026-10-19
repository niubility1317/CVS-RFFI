# R13：接收作用与 prior 的源身份瓶颈诊断

持续目标不变：独立 TX 无标签恢复器使用全部合法 L/U，输出同一份 IQ 供原 F 的全部 PAMF 分支；正式 low urban≥70%，其余五场景与同 F/seed 原始 clean 差距≤5pp。R12 已完成而未通过身份验收，不调阈值或救跑。

本轮仅依据用户六类网络报告 R3/失败处置与 R06、R07、R10、R12 **源域**证据，诊断作用估计、噪声代理和学习 prior 的相互作用。R06 的 60 条源样本上已知作用改善平均 NMSE，但尚无对应原 F 身份证据；不能据此直接增加展开层。R10/R12 平均误差改善也未充分保护全部 TX。

固定原 R10 native 完整最后权重，不更新任何网络、归一化或优化器。原始 1800 条 source V、clean＋三郊区＋三城区复用 R07 已验证的 received 及匹配 action cache。新增五个已登记读出：学习 prior 本身；学习 prior＋已知作用/估计噪声；学习 prior＋估计作用/模拟器噪声代理；学习 prior＋已知作用/模拟器代理；coarse prior＋已知作用/模拟器代理。原 identity/coarse/R07/R10 预测复用，避免把 prior 共适应误当纯算子效应。

所有 MAP 均固定 6 CG 与原 SCALE²，噪声仅作软约束。known action 知道实际 AGC/lock，是 oracle 条件诊断，不是部署输入；simulator variance 是原生白输入等效实噪声方差代理，不是完整真实协方差。原学习 prior 是在估计作用下训练的，替换作用未必改善，因此 oracle 分数不是全模型理论上界。所有组合只作用于同一份 received，不读 clean 作为预测输入。

原 F50/defaulthead 冻结；五读出七视图预测完整保存并校验后，独立 CPU 才读取 V 标签与 clean 评分。报告每个场景、TX/RX/day、repair/damage、NMSE/频谱/前232后24点、配对 clean 特征。没有训练、target 或 source V 持久拟合，本轮候选均不得自动晋级正式测试。

预先固定解释规则：known action＋simulator proxy 相对原 native 的六场景平均 NMSE 至少−5%、low urban≥+1pp、mid urban≥+0.5pp、clean 相对原输入下降≤0.1pp、low urban 最差 TX≥−1pp时，支持优先改善作用估计。分别比较单独替换 action、单独替换 variance 以及 coarse oracle，区分估计瓶颈与学习 prior 共适应。否则不声称算子可解决当前身份目标，也不自动堆更多迭代。此规则只组织下一次源实验，不读取目标指标。

不可覆盖 run/release、一个 launch owner、普通账户 N607、GPU 池0/1/5/7择空闲卡；最多一个本 run GPU worker，计入所有 owner cap2/free≥12GB，共享 capacity4 锁至 CUDA PID 可见。入口先真实2条L_s/固定checkpoint no-query smoke，PASS继续；完成全部源预测后独立CPU评分。非法来源/角色、输出冲突、非有限或不完整预测时保留输出并停止本run，无自动重试，不动健康任务。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|A13-01|报告 R3|已知/估计作用与噪声分别替换，prior 共适应对照|experiments/pamf_action_fidelity/models.py|verified|精确解析 MAP 算例及同prior对照通过|不把 oracle 当部署或理论上界|
|A13-02|无监督优先|固定完整来源、0训练/TX/F训练信号、原源面板|runtime.py/runner.py|implemented|来源/角色负测及N607句柄核实，真实checkpoint smoke待入口执行|source V仅评分|
|A13-03|用户有效性要求|原F50/原头、全部预测先于truth|evaluation.py|verified|缺视图/读出/ID入口/状态负测，独立CPU路径通过|0新分类头|
|A13-04|郊区/urban|clean+六场景及TX/RX/day完整指标|evaluation.py|pending|实际同row汇总|源数据不冒充正式70%/5pp|
|A13-05|解释规则|固定源身份/NMSE门槛，0target晋级|design.py/evaluation.py|verified|阈值/最差TX/角色负测通过，CLI无target/train模式|不依目标调参|
|A13-06|最小工作流|单owner/GPU容量/真实smoke/独立P0P1/发布核实|runner.py/登记|implemented|9tests、compile/help、一次独立P0/P1及普通账户preflight通过；待实际post-state|无额外gate|
|A13-07|最终目标|合法可部署新网络、正式70%/5pp、多seed/独立物理族|后续源训练/正式测试|deferred|本轮oracle诊断不能证明|保持完整持续目标|
