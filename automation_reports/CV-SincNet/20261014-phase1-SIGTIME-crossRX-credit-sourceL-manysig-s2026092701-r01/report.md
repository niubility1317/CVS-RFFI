# SIG_TIME中层停止排序与跨RX增量：源结果

源探针已ANALYZED / VERIFIED；实际学习信号存在，fixed final50状态的20个辅助增量全部未达到预定1e−6贡献容差，因此停止施压。不能将它解释为新训练增益，也不能为了放行降低容差。后续研究只检验从零训练早中期的同一机制。

Run `20261014-phase1-SIGTIME-crossRX-credit-sourceL-manysig-s2026092701-r01`，执行Git `cc11c16e7c10d6fcf12d3db8e179b0d49bd40964`。预声明SIG_TIME首seed2026092701 own final50，原scratch、空ancestor、11100原joint更新、完整L/U/V物理role契约与SOURCEexactmatch、final-only/no target training-selection及严格学生state均VERIFIED。原默认6×160 cosine头保持，无另训/重训/替换头，0新模型训练。原native一组188参数、72有moment状态、LR1e−6、WD1e−4、betas(.9,.999)/eps1e−8严格重建，没有调用Stage2两组或补造moment。

## 完整L6300实际特征

全部90cell各70包来自同一次已声明源forward；两特征各126000固定same-day/同ordinal关系，margin=.1、不mining。

| 特征 | 平均正负gap | 排序违反 | 未满足间隔 | hinge均值 |
|---|---:|---:|---:|---:|
| SIG_TIME pre-gain160 | .69974959 | 4,988 | 8,240（6.5397%） | .01210476 |
| actualjoint160 | .34134123 | 2,793 | 14,004（11.1143%） | .00811183 |

pre-gain的平均gap更高，同时极端排序错误也更多，不能只报均值认定优于整体特征。margin几何不等于默认分类准确率，相关关系不能作为独立模型seed。完整特征是已经参与原部署路径的sequence_pool.output_norm，非另造embedding或事后fitprobe。

## 实际参数梯度与临时更新

固定四cell ordinal0/17/35/52，各90可见真标签L。真实默认L CE .01283072/.00198656/.00751969/.00365227，main参数梯度norm .352487/.051897/.443266/.088207；普通SupCon为 .394558/.503993/.460913/.528778，显示无条件对比可能比主CE施压更大。这是当前eval状态的梯度，不是累计训练效应。

五anchorRX各 .1/FULLtriplets 的ORDER参数梯度norm范围 0.00125573 至 0.00550928，辅助W/gain均None，mainW正常非零。正负端点为当前source特征detach，不是teacher关系模仿。相同savedAdamW state、独立clip5两临时分支，辅助参数增量norm为 1.7284874e-07 至 4.4259754e-07。

两临时模型使用同pre-step默认W，在同批其他四RX各18包比较CE；这些包主CE已见，不是V或真正unseenRX。20组平均贡献范围 -7.8649061e-09 至 2.7111835e-08，全部低于1e−6，0/20接受。final50的LR仅1e−6，变化很小；这只能说明本次末期假设步应停止，不能外推所有早期step无用。固定容差保持。原simulate的`U_increment_norm`字段名被复用为本次auxincrement统计，实际0U读取或U梯度。

## 完整性、成本和限制

source role/provenance/defaulthead/noqueryrealcheckpoint smoke PASS；唯一P0/P1 review PASS，preflight/archiveSHA/compile/独立livePID-CWD-argv-loggrowth与terminal完整产物VERIFIED。live参数/buffer/gradient/optimizer/RNG保持，0actualoptimizer/EMA/head-fit/V梯度/U/target读取。

实际6300IQ读取，9540真实L网络前向＋2零IQsmoke、40临时AdamW步、CPU2线程/0GPU，处理15.5687秒。原sourcebank、summary、runtime/stdout保留不可变root，原权重不覆盖。独立freshCPU重读bank，重建252000固定关系、18组几何及20个固定accept决定/分母与serializedpreservation复算VERIFIED，0额外IQ/模型/优化器执行；没有重复参数梯度或临时更新回放。公开[几何](source_geometry.csv)、[贡献](source_credit.csv)、[梯度](source_gradient.csv)、[证据](source_result_evidence.json)。

当前仅原final50 eval图的L-only假设基本步，不能冒充原历史U联合更新或从零新训练。下一阶段最多一个正式训练矩阵，以相同SIG_TIME/default头、相同L/U与actualstep预算比较BASE、普通SupCon、停止ORDER与CREDIT。实际训练应在完整base loss（含当时合法U项）组装后评价更新；日程/第一seed源扩展规则及所有固定值在新target前声明，保留最终自动clean独立truth-last。

MLDG/Fish已有跨域更新/梯度思想，[机制假设与文献边界](../../../analysis/phase1_crossrx_increment_credit_20261014.md)明确本方案待验证点，不先声称首创或DG效果。旧原始观测/集合/PA/关系等负结果保留，SIG_TIME等正收益按各自配方保留，固定三seed84%目标active。
