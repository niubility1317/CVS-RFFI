# 协方差三组 F 直接识别正式确认

状态：ANALYZED_SINGLE_SEED。按用户最新撤下DURA配合路线，固定CovSel-U、CovRandom-U、CovSel-LU三份已完成E50/11100单视图完整PAMF，直接预测rawclean和六practical。新训练0，不读取G权重/恢复IQ，不新建恢复缓存。全部矩阵/命令/环境/CWD/输出/GPU/停止规则见experiment.json。

三候选按独立F-only源必要条件入选，依次rawclean少6/5/6条，raw三景总正确数多58/38/103条；fullmask三景少31条未入选。全部使用冻结final，非按目标排名；不把小源收益称泛化成功。旧含G判定保留，不追认通过。此次范围变更来自用户明确要求，发生在新covariance target访问前。

各行固定cross_rx126000和cross_both42000，共168000物理包×7raw视图；按正确数合并，单独cross_date30000不加入本次与严格SIG1对照匹配的正式矩阵。原监督/普通PL同SIG1三seed模型及已完成预测复用，seed01配对本轮；无对照重训/重复预测。原三seed和更完整目标范围仍未完成，不能仅凭此pilot宣布成功。

加载实际来源一致的E20+10+20/defaulthead后，先sourceL2无query smoke，PASS即连续预测。预测配置无truth/G句柄；每row两任务全部7视图固定后独立CPUscorer连接truth，评分配置不含source/model/G。保留逐TX/RX/day及receiverday的accuracy/macroF1/混淆。目标不回流调mask、权重、epoch、候选顺序或选择性重跑。

候选已完成一次独立P0/P1 source审查PASS；此处复用同一冻结加载和原truth-last评分，新增raw-only配置/权限边界5项测试PASS，不额外添加runner审查或数据hash gate。源码/config由执行Git commit固定，单releasearchive一次传输比较、远端compile和实际PID/CWD/argv/GPU/loggrowth核对。只在全局空闲卡启动，每卡一条ownworker；首个技术失败停止新启动、健康在跑自然完成，其他owner不操作。

最终至少检查rawclean≥85.981746%、raw六景>72.096197%，并按严格同SIG1PL/Sup的逐景和分群配对结果报告；原三seed等目标保持。DURA指标因用户最新范围撤下，不再用于本路线验收。

## r03 技术替代边界

本run替代r02，r01/r02均在target访问前停止并保留。使用三行实际source_cache/source_contract，不放宽检查；独立Fbootstrap增加release/code中原可信checkpoint的SatViewStage metadata模块，不加入G路径/权重/恢复IQ。已对真实同一checkpoint做CPU/sourceL2严格加载读出PASS，无target/optimizer/G访问；四项相关配置/独立bootstrap测试PASS。launcher仍以实际GPU/sourceL2smoke作为第一步后连续正式预测。

## 完整正式结果

VERIFIED。实际执行commit `ec73e83ae6c2f2c42f3b1dcc0ea90573ff54d3cf` 已提交/推送并核实远端OID，单archive比较VERIFIED、远端compilePASS。三份实际GPU/E50/sourceL2/defaulthead smoke均PASS，随后连续完成正式预测。第一次独立观察时三GPU worker存活且PID/CWD/命令/GPU匹配、日志已有跨RX七视图进展；后续完整14视图日志、prediction/score标记及终态全部读回，controller/六worker自然退出，无技术错误。r01/r02保持失败，不覆盖或改称成功。

seed2026092701，每模型cross_rx126000+cross_both42000按正确数合并168000，raw clean与六practical。三候选共3528000个冻结类别决策的形状、范围、视图/ID覆盖已核对，候选与两个已完成SIG1对照共12组physical-ID顺序完全匹配。全行预测固定后独立CPUtruth-last；此后仅解析已有scores/混淆和预测ID，没有重新读truth/IQ/模型权重，没有重复控制预测。评分日志的旧后端名称不改变实际raw-only配置。

| 方法 | raw clean（%） | raw六景平均（%） | clean相对PL（pp） | 六景相对PL（pp） |
| --- | ---: | ---: | ---: | ---: |
| SIG1-Sup | 85.139881 | 70.911806 | -0.351786 | -0.037401 |
| SIG1-PL | 85.491667 | 70.949206 | +0.000000 | +0.000000 |
| CovSel-U | 85.451786 | 70.981250 | -0.039881 | +0.032044 |
| CovRandom-U | 85.524405 | 70.965179 | +0.032738 | +0.015972 |
| CovSel-LU | 85.347024 | 71.011706 | -0.144643 | +0.062500 |

三候选均未达到原rawclean≥85.981746%和raw六景>72.096197%门槛，且各自至少一个practical场景较PL下降，不能宣称各方面强于PL。选择性U相对PL clean−0.039881pp/六景+0.032044pp；随机控制clean+0.032738pp/六景+0.015972pp；选择性L/U clean−0.144643pp/六景+0.062500pp。不能将随机组的最佳clean与L/U组最佳六景拼成一个模型。

本单seed没有选择规则稳定优于随机的证据；共享给L增加了raw六景的小收益，同时clean代价更大。相关一致性损失下降、U全覆盖和源域小收益均不足以证明目标泛化有效。选中项本来仍有TX关联，且固定mask的通道基底可能随训练改变；这些是解释假设，不作为已证实失败原因。

不扩本候选其他seed，不用目标结果调整mask/系数/epoch或选择性重跑。保留当前全部正负结果。后续按目标访问前已列出的第二方向，先以合法训练L验证跨TX/跨日的低维公共接收响应及其稳定性，再决定是否做有界输入校正；不使用DURA。新的训练输入仍只有clean IQ及星地衍生物，U TX真值隐藏、已知RX可用、默认头联合训练与单SIG_TIME保持。

此处为一个模型seed的完整正式确认，不是完整三seed目标；原目标保持active/full_goal_complete=false。[汇总](summary.csv)、[逐景](per_scene.csv)、[逐TX/RX/day等配对](paired_strata.csv)、[比较](comparisons.csv)、[结构化结果](results.json)。
