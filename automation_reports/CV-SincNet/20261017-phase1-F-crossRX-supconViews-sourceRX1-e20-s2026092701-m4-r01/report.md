# 原跨RX同TX对比项视图组织：两行已启动，尚无新性能结论

**RUNNING／VERIFIED。** [固定预登记](preregistration.md)、[四行矩阵与配置](experiment.json)、[本地验证](local_checks.json)、[唯一P0/P1](review.json)。实际执行代码提交 ed0ca91481f8c11a28cb8f4d4726a7c3d488720e，保持原配方运行，没有热改或重启。

Q0/Q_ord为既有sceneCycle B00/C_IID冻结源结果，仅复用；已核对完整实际config、全部L/U/V物理ID、seed、E20/4440、8L/pairweight1.25、默认头和权限。旧model/channel/source/design与c9执行commit无差异。新Q_sat/Q_mix均scratchE20，不加载控制或其他对话权重，不是续训。历史控制40modelE/8880更新与新增预算40modelE/8880更新分开记录。

## 实际启动及最新完整边界

一次普通账户N607预检、一次release archive传输SHA比较、remote compile均VERIFIED。dispatcher第一步真实匹配数据角色checkpoint的2合成包无query兼容smoke PASS后立即继续。两行实际PID／CWD／argv／GPU环境与NVIDIA进程独立核对，原dispatcher与两worker均存活。未干预其他owner。

| 新行 | GPU | 已完成 epoch | 累计更新 | As 对比输入 | ordinary 对比输入 | 原对比项数 |
|---|---:|---:|---:|---:|---:|---:|
| Q_sat | 0 | 10 | 2220 | 142080 | 0 | 4440 |
| Q_mix | 1 | 10 | 2220 | 71040 | 71040 | 4440 |

这是完整 epoch 边界的累计计数，运行期间继续增长。E6 对比路由启用已实际验证，两行原 PID/CWD/argv/GPU 与初次启动匹配，日志增长、无技术故障。每步 1024 个 L 前向样本、256 个 extra 前向样本及 128 次完整 As 生成/监督符合固定预算。两层/两 phase 的暂存 hook 均清理，新增网络前向和损失项均为 0；描述提取、暂存和路由仍有计算成本。该证据只证明机制按配置执行，不能证明 clean 或星地收益。

## 已验证与未验证

本地29个协议/科学反例、真实F合成执行通过：ordinary-route loss与旧控制完全相同，74组参数梯度逐一接近；相同scratch状态、八次输入、fullAs IQ和CPU/Python RNG；两条新行默认头联合更新、export160逐点相同。合成NN4108包/2次更新、128ordinary/512fullpractical生成与2次额外对比重算单列，0真实IQ/CP/target。本地最初无Torch环境导入失败后使用现有已验证Torch环境；保留已知NumPy桥警告，未安装/修改环境。

底座F＋跨RX同TX／三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认cosine头／原13L7U、oldgate、ownEMA、ordinary保持。每新行完整raw.75/ordinary.25＋extra.25As，8L/总pairweight1.25不变；原SupCon两层/两phase/温度/schedule/总权重不变，仅使用practical或64/64出现位置混合maps，没有新增loss项、NN或部署模块。额外map捕获/路由/descriptor计算仍有计算和内存成本，不能把0新增NN说成0计算；实际总体walltime、生成时间和原生CUDA资源记录待终态统计。

当前只有启动与执行匹配证据，未完成训练、源七视图预测或独立分析，没有新clean/low/suburban成绩。固定primary Q_mix、两新行完整final20后自动source RX1 clean＋六practical，每景1260；旧控制的17640预测复用，新行17640预测待生成。正式target未构造或连接truth。

Q_mix必须同时相对Q_ord/Q_sat达到low与urban均值各≥+1pp、clean/F1不降、mid不降、每practical/seenV下降≤.5pp；相对Q0 low≥+5pp/urban均值≥+3pp及相同保护。失败关闭固定视图组织，不调比例/层位/系数/scene概率/RX/seed/预算救跑。对比loss、anchors、梯度和源健康不能代替真实身份收益。

这属于As训练组织，不是新IQ泛化增强Ag；即使单seed通过也不证明稳定As或完整联合成功。有效Ag、匹配B00/B10/B01/B11和交互、跨seed、正式clean＋六practical、low>60%、suburban各景距自身clean≤2pp探索目标仍未完成，完整目标ACTIVE。原始预测/ID/权重/日志/runtime只留本地或所属run，未提交Git。


## 终态分析工具已准备

新增离线收集、逐包场景复算、全量轨迹/预测独立评分和图表工具，均不修改实际训练。39 个反例检查通过，完整合成路由 13320 步、源判决夹具 1260 条，0 真实 IQ/模型/前向/更新。已完成旧 Q_ord 的全部 8820 个源判决、4440 步及 20 轮轨迹独立复核；这属于旧控制分析，不是新行结果。新增描述统计在双方 clean 都正确的固定集合内报告 urban 修复/新增错误，不参与调参或晋级。原场景、Q_mix primary、固定判据、最终 E20 与 source-only 边界不变。


## E14 原半监督阶段已实际进入

两行原进程均完成 E13/2886 次有标签阶段更新，并进入原 E14 半监督。Q_sat/Q_mix 当前完整刷盘记录分别到 2989/2910 步，累计执行 103/24 次 U 更新。所有已刷盘记录的 U 真值可用计数、阶段错误及 EMA 更新计数错误均为 0。此为部分实时执行观察，E20 全预算尚未完成。终态独立分析新增对全部 4440 步原 13L＋7U 组织及 U 样本总数的复算，39 个反例 PASS；旧 Q_ord 完整轨迹独立核验为 2886＋1554 次更新、396368 次 U 样本、4440 次 EMA 成功更新。没有新增训练前向或改变规则，不从伪标签选择量推断识别收益。
