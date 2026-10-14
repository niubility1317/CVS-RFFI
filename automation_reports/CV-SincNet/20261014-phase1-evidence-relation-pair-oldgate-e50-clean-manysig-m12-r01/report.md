# 跨 RX 关系自蒸馏与 CVS 增量证据响应

本轮研究“给无标签样本提供什么表征监督”。全部合法 U 参与连续关系学习，既有 PAIR 门控伪标签 CE 和全局 U entropy 原样保留。新增目标不产生六类标签，不经 W，不新增分类器、投影器或部署参照查询。实际机制、矩阵、环境/CWD/命令/路径/stop rule 详见 experiment.json。

## 固定目标

180 个合法 L 个体参照由 ownEMA 刷新：每 TX/RX/day 2 包，每 222 U 更新刷新一次，共 17 次、3060 额外 L 前向。标签用于合法 L 均衡采样和原主 CE，U 隐藏 TX 始终不读。bank 与教师输出 stopgrad，允许最大 221 更新的版本间隔并记录年龄。

每 U 比较 120 个参照，各 TX 20 包，各 TX 的 day 构成为 7/7/6。B 使用全 5 RX，每 RX24；C/D 排除自身 RX，使用另4 RX，每RX30。每个参照 RX 分别 softmax，再平均损失，不把 RX 的目标机械拉成相同向量，不按 TX 聚合为标签。

教师/学生关系分别为 T=softmax(normalize(z_T)·normalize(a_T)/0.2)、S=softmax(normalize(z_S)·normalize(a_T)/0.2)。L_rel 为逐 U、逐 RX 的 KL(T||S)。学生使用原强噪声0.015与Dropout，教师原clean弱视图。温度与系数均在新源结果前固定，不锐化目标后再以其确定性自证。

真实 CVS 增量 z=a+b_PA+c_ref。对 PA/ref 分别用 alpha=0.5 得到 z-alpha·b；教师和学生均削弱同项证据。响应 D=(R(z)-R(z-alpha·b))/alpha 是有符号关系变化，不要求缺证据时仍输出完整答案。L_resp 使用 SmoothL1(beta=0.02)，sum参照坐标、mean RX/证据路径/U。B/C固定lambdaR=1；D另加lambdaE=1；A新增目标为0。四组参照刷新和计算匹配。

新增辅助重组的 z 与部署joint160数值一致，只对 PA/ref 融合增益 stopgrad；增量内部特征不断梯度。原CE仍正常联合训练 W、增益与骨干。只减可分离的 PA/ref实际增量，不把t_proj当作最终可加项。网络证据削弱不等于移除真实PA硬件效应，跨RX参照也不保证RX信息已消除。

## 矩阵与科学推进

| 组 | 新目标 | 固定问题 |
|---|---|---|
| A：ZERO | 0；相同参照/计算 | PAIR原配方匹配起点 |
| B：ALL_REL | 全源RX L_rel | 通用关系监督是否有用 |
| C：CROSS_REL | 排除自身RX L_rel | 跨RX组织是否提供增量 |
| D：CROSS_RESP | C＋signed L_resp | CVS增量证据响应是否提供增量 |

三个预定义modelseed2026092701/02/03。每row scratch50，33L+17U、11100mainupdates、3774辅助调用、963900U包观察；默认cosine头不变、无另训头。完整12模型600modelepoch/133200更新，无历史checkpoint训练继承。已有50epoch模型只作真实checkpoint smoke/sourceL诊断，不能视为新方法父权重。

先seed01 A/B/C完成固定源训练，D等待其源科学标准：finalLCE<=.5、相对A源V均值下降<=1pp、最差RX下降<=2pp，B/C非零关系比例>=.5、四预定实际aux/L特征梯度比均值>=1e-5。D完成同标准后才扩02/03。所有资格只读源信息并保存，目标评分不进选择。科学失败保留已完成行/缺项并停止pending；不凑完整矩阵，不调参补过。

## 验证、诊断和结果边界

核心负测PASS：同TX/day参照数量匹配、排除自身RX、教师/学生完全一致时损失零；完整joint相同而PA增量相反改变的代数反例，其relation项零但response项非零。真实native CPU检查 W/两增益的辅助梯度None，t1与参考投影梯度非零，实际特征参数变化；这是接线证据，不是泛化收益。一次独立P0/P1审查PASS，无阻断缺陷。remote首步严格加载实际同契约student/ownEMA，零IQ noquery smoke；用合法L180参照＋不重叠L720探针记录语义/信号诊断，不拟合参数或阈值。

训练在四预定更新7327/7548/8880/11100记录实际L/UCE/entropy/aux/R/E fθ梯度、aux-L夹角、t1/t2/t3/PA卷积/分支投影分组，原clipping和真实AdamW步后的参数变化。另记关系熵/样本关系差异、参照方差、PA/ref增量范数、损失median/max/top1%贡献，防止以覆盖率、标量loss比或末端变化代替骨干更新证据。发现塌缩不自动叠加其他正则。

每row final50固定后直接clean168000默认头预测，再fresh CPU独立truth-last评分；完整矩阵报告3seed均值/SD/配对差、各RX/day/TX和最差RX/day，负结果保留。B有效不代表跨RX成立；C有效不代表响应有效；只有D稳定优于C才支持CVS响应解释。源已见RX诊断不称未见RX泛化。本项目基准已暴露，不称新的独立盲测，不预告84%。

唯一owner codex/root/evidence-relation-20261014；普通N607账户，max3总进程/GPU含otherowners，>=12000MB空闲，不停健康旧任务。源码身份以release_commit.txt的Gitcommit为准，archive只核对一次运输SHA，remote compile和smoke通过后继续。错角色/继承/输出冲突/nonfinite/实际重组不一致/默认预测或评分技术失败停止pending，保留active和产物，无自动重试。

## 依据与创新边界

[Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) 研究样本关系传递；[ReSSL](https://arxiv.org/abs/2107.09282) 使用实例相似关系及弱增强动量教师；[PAWS](https://arxiv.org/abs/2104.13963) 使用有标签参照非参数构造类监督。本轮关系目标保留个体参照分布，不按TX求和形成类目标。关系蒸馏和参照本身不是新发明；待检验贡献是匹配跨RX组织＋真实CVS增量的有符号响应监督。[SimSiam](https://arxiv.org/abs/2011.10566) 对Siamese结构退化的讨论也不能作为本任务EMA必然不塌缩的保证。

用户最新材料是35f4dc861分支的未执行研究建议。这里按本工作树真实ResidualPhysicalHead/ReferenceCVS与默认头实现，不把材料的预期效果写成已验证事实。此前切向路线独立原样确认剩余种子；可信纠错完整源诊断41/41正确但量不足固定门槛，保留结果，不继续搜索接受规则。

当前状态：LOCAL_VERIFIED；尚无本机制实际训练/泛化结果。
