# 原跨RX同TX对比项视图组织：两行已启动，尚无新性能结论

**RUNNING／VERIFIED。** [固定预登记](preregistration.md)、[四行矩阵与配置](experiment.json)、[本地验证](local_checks.json)、[唯一P0/P1](review.json)。实际执行代码提交 ed0ca91481f8c11a28cb8f4d4726a7c3d488720e，保持原配方运行，没有热改或重启。

Q0/Q_ord为既有sceneCycle B00/C_IID冻结源结果，仅复用；已核对完整实际config、全部L/U/V物理ID、seed、E20/4440、8L/pairweight1.25、默认头和权限。旧model/channel/source/design与c9执行commit无差异。新Q_sat/Q_mix均scratchE20，不加载控制或其他对话权重，不是续训。历史控制40modelE/8880更新与新增预算40modelE/8880更新分开记录。

## 实际启动及首轮边界

一次普通账户N607预检、一次release archive传输SHA比较、remote compile均VERIFIED。dispatcher第一步真实匹配数据角色checkpoint的2合成包无query兼容smoke PASS后立即继续。两行实际PID／CWD／argv／GPU环境与NVIDIA进程独立核对，原dispatcher与两worker均存活。未干预其他owner。

| 新行 | GPU | 已完成epoch | 完整边界更新 | L NN样本 | As生成/监督 | stdout bytes |
|---|---:|---:|---:|---:|---:|---:|
| Q_sat | 0 | 1 | 222 | 227328 | 28416 | 6532→11703 |
| Q_mix | 1 | 1 | 222 | 227328 | 28416 | 6532→11703 |


表中L/As是完整epoch边界的实际计数，运行期间还在增长，不是最终累计成本。完整source_base_progress同步核对每步1024个L前向样本（8×128）、256个extra前向样本、128次完整As生成与监督，两个layer/phase暂存hook已清理。源IQ元数据覆盖全部5040合法L，无U真值或target。原对比schedule从E6开始，当前warmup的0次routing是预定状态；**E6实际As maps路由尚未远端验证**，不能把首轮CE成功当作机制有效。

## 已验证与未验证

本地29个协议/科学反例、真实F合成执行通过：ordinary-route loss与旧控制完全相同，74组参数梯度逐一接近；相同scratch状态、八次输入、fullAs IQ和CPU/Python RNG；两条新行默认头联合更新、export160逐点相同。合成NN4108包/2次更新、128ordinary/512fullpractical生成与2次额外对比重算单列，0真实IQ/CP/target。本地最初无Torch环境导入失败后使用现有已验证Torch环境；保留已知NumPy桥警告，未安装/修改环境。

底座F＋跨RX同TX／三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认cosine头／原13L7U、oldgate、ownEMA、ordinary保持。每新行完整raw.75/ordinary.25＋extra.25As，8L/总pairweight1.25不变；原SupCon两层/两phase/温度/schedule/总权重不变，仅使用practical或64/64出现位置混合maps，没有新增loss项、NN或部署模块。额外map捕获/路由/descriptor计算仍有计算和内存成本，不能把0新增NN说成0计算；实际总体walltime、生成时间和原生CUDA资源记录待终态统计。

当前只有启动与执行匹配证据，未完成训练、源七视图预测或独立分析，没有新clean/low/suburban成绩。固定primary Q_mix、两新行完整final20后自动source RX1 clean＋六practical，每景1260；旧控制的17640预测复用，新行17640预测待生成。正式target未构造或连接truth。

Q_mix必须同时相对Q_ord/Q_sat达到low与urban均值各≥+1pp、clean/F1不降、mid不降、每practical/seenV下降≤.5pp；相对Q0 low≥+5pp/urban均值≥+3pp及相同保护。失败关闭固定视图组织，不调比例/层位/系数/scene概率/RX/seed/预算救跑。对比loss、anchors、梯度和源健康不能代替真实身份收益。

这属于As训练组织，不是新IQ泛化增强Ag；即使单seed通过也不证明稳定As或完整联合成功。有效Ag、匹配B00/B10/B01/B11和交互、跨seed、正式clean＋六practical、low>60%、suburban各景距自身clean≤2pp探索目标仍未完成，完整目标ACTIVE。原始预测/ID/权重/日志/runtime只留本地或所属run，未提交Git。
