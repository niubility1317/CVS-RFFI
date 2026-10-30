# PAMF 开放表征重设计：已知与辅助未知的合法分离

固定 run `20261030-phase3-star-crossset-metric-s2026092701-r01`。这是前一独立度量残差的机制修正，不替换已经固定的 Tri-SIG-TIME SSL/default6head 或 DURA v1.0。唯一主 arm 为 `S_SEP_TOPO`，匹配控制为 `S_SEP`；唯一主 readout 为 `learned_known_support`，其余分数仅作预登记诊断，不能按结果换主。

## 问题与固定假设

前轮 L 监督度量与 OE 软邻域损失只约束各自集合内部关系，没有显式约束 L 与 OE 的相对位置。共享网络限制了任意独立变换，但并不等价于学习已知与未知的区分。本轮直接把 OE 的合法 outside-six 成员信息加入 L anchor 的对比分母。该信息证明 L 与 OE 不同旧设备；不能证明任意两个 OE 来自不同设备，也不能证明 OE 覆盖所有未来未知。

OE 背景密度可能很低，即使 known 支持也很低，密度比仍可能偏大。因此预先把 absolute known-support density 固定为主 readout，并保留相同特征的 density ratio 作诊断。以上为结构假设，不把前轮负结果写成已证实的因果解释。

## 输入、标签与来源

只使用既有同 G10MIX01/P1、原 F0 的 L6300 与 OE6134 特征训练。L 为旧六设备各1050；OE 为21辅助设备，算法只能看到 outside-six 成员角色与公共 RX/day，OE 真 TX/K 隐藏。两 arm 都读取同一批 L192/OE64，模型初始化、L/OE随机流和2220步完全匹配。

known V27000（旧六设备各4500）仅在模型与聚合统计冻结后用于既定 known42 风险/阈值校准；source novel3150（19-15/19-16/7-1各1050）只在全部阈值冻结后预测，随后独立 truth-last。原 Us56700、本轮 target U/Q、正式 RX24-18 均不读取、不优化，历史 query 不转训练。四 source 角色互斥，共42584条既有物理记录。

源 RX1/3/4/6/8 对应1-19/18-2/19-2/2-19/3-19；日期2021-03-08/15/23均已见。L/novel 每TX×RX×day70，V300。source novel 有历史开发曝光，不是新RX、新TX盲测或未来注册观测。

全部沿固定 G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect，郊区/城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001，原一次 unitRMS/单 received 不变。P0是叠加星地信道后的 received，不是clean；本轮不生成新IQ/信道、不读P0/clean实值、不运行G。

L/OE 原 fused160 与四贡献来自 protected-cache actualcode `7dd90671968ddf58aab5a5b302b4d56371c8b9c9`；V/novel 的 u160/c640/defaultlogits 与固定原 F0 density reference 来自 contribution-density actualcode `edc28ad00baace5e4b9567fbf3391154bf6c34b4`。原 F0/sourcecontract392005、DURA祖先与现有匹配核验结论复用，不加载前轮任何学习后的开放网络或640密度权重。

## 网络、损失与预算

同前轮102688参数新网络：640→128 Linear→GELU→160 Linear，seed2026092701，最后层权重与bias精确零初始化，z=unit(u+g(c))。u为原F0 fused160单位向量，c为四贡献各自FP64单位化、拼接整体单位化后FP32的640向量。无0.1硬残差界、maxmember关系头、新身份头、EMA或目标状态。原F0/default6head冻结；训练容器不加载原F0，旧身份始终为原默认head argmax。

L每步192：六类×15公共RX/day格各2，再每类两个不同轮转格各1；extra域 `(2*step+c)%15` 与下一格，L RNG2026103017。OE每步64：15域各4，再从 `4*step mod15` 起连续四域各1；OE RNG2026103018 独立。step从0计，格内无放回、单batch不重复，两arm全部L/OE索引匹配。

新 cross-set 监督项仅以L为anchor，T=.1。候选集合为L∪OE，候选须同时不同RX且不同day。L候选与OE候选各占先验质量1/2；每个集合内部先按有效公共域等权，再域内按样本均分。因此candidate logit为 cosine/T + log(.5) − log(有效域数) − log(该域candidate数)。两个集合每个anchor均须有有效候选，否则报错。正样本只为L集合中同旧TX且有效的记录，目标先按有正样本的公共域等权，再域内均分；OE目标质量为0。L anchor loss为目标对全部候选学生概率的KL；anchor先按旧六类×15域格内平均，再格等权。L和OE投影均参与反向，teacher与原输入detach。没有OE-anchor硬负关系或OE–OE异设备假设。

`S_SEP` 只优化上述 cross-set KL；`S_SEP_TOPO` 优化 cross-set KL + OE软邻域KL（权重1）。OE项完全复用前轮 frozen-u teacher、跨RX且跨day、公共域等权、T=.1、teacher stopgrad 的软结构。它是弱结构约束，不是真实未知身份标签；通过匹配控制检验它是否帮助保持未知结构。

两arm fresh AdamW lr1e−4/wd1e−4/clip5、常量LR、固定2220更新，不早停或追加。每arm L426240/OE142080训练特征曝光，全部L/OE索引与真实loss/梯度/漂移记录。原F0仍E50/11100/232993，新native更新0；新4440总更新不写成原F0续训。合成梯度probe不更新优化器，不计入2220。

## 冻结读出、评分与推进

共同prepare第一步实际原F0 synthetic G10两行no-query smoke，PASS立即导出既有L/OE派生特征；新真实F0/G前向0。每arm训练终点严格重载真实新checkpoint，合成两行投影smoke后映射L/OE，拟合同族D160 known90 Gaussian和background32/25Lloyd/seed3019/covshrink.1。known统计与background统计均在读取V前冻结。

固定三个输出列：`fused160_density_ratio`（原F0参考，模型/阈值/预测逐值复用）、`learned_known_support`（主）、`learned_density_ratio`（诊断）。两个learned读出使用同一次冻结投影和同一known聚合统计；known-only状态保存known字段，ratio状态另含OE background字段，不用不同参数或不同拟合预算制造差异。全部列的旧身份一致，不另训或替换分类头。

每个learned读出在相同V上独立执行已固定known42/common-tail六predicted-class阈值规则。所有模型、聚合统计与两套learned阈值均固定后，才消费novel特征。独立CUDAempty CPU scorer验证保存的features/聚合统计能重建全部raw/margin后，唯一连接novel真值。报告宏A_K、R_U、H、C_K、同输入原F0闭集损失、AUROC/OSCR、逐TX/RX/day/scene/42风险组；保存主与固定F0、主与learned-ratio的配对纠错，并在矩阵结束后比较两arm。不能依据NV结果选readout、阈值或arm。

唯一主 `S_SEP_TOPO::learned_known_support` 必须通过known42和source整体宏R_U≥90%才另预登记跨未见RX星地验证；控制或诊断通过不替代主。source门槛不代表最终跨RX成功。本轮无target DA/discovery/REG，A_N/H_OW=N/A；完整目标保持ACTIVE：跨RX星地宏C_K/R_U≥90%、A_K较同输入原闭集下降≤2pp，主报A_K/R_U/H，继续争取更高拒识及未来无标签注册。

N607：prepare后最多两个本任务train/eval worker，每卡新增≤2、启动实际free≥4096MiB；独立CPU scorer一次最多1个。允许与外国owner共卡，不停止/修改/重启其工作。每row冻结即自动预测/独立评分，健康peer继续。非法来源/角色、非有限、输出碰撞、无法产生合法预测或系统技术失败保留产物且不自动重启；科学负结果不停止健康训练。唯一launch owner和不可覆盖root，一次代码commit、聚焦负测、一次直接P0/P1审查、最小登记、一次preflight/archive SHA比较/remote compile、PID/CWD/argv/GPU/log增长与独立终态读回。
