# coarse 与三步先验的无 TX/F 联合恢复 R23

状态：**RUNNING**。方法 `d43dbab326d0b1e2cb6fb7faa23dc9a3ec25a025` 已固定、推送并独立核对远端 OID。实际源checkpoint/defaultF入口与两条GPU worker的PID/CWD/argv/GPU/日志增长已核实；尚无完整源评估或新正式测试成绩。未改变正在运行的训练与选择规则。

本次只读核查使用当前所有已完成轮次，未读取TX/F/V/target，未改变优化器、权重或排序。核查范围如下，曝光次数不表示新增独立物理样本数：

|行|已核查轮次|prior更新|coarse更新|estimator更新|L曝光|U曝光|
|---|---|---|---|---|---|---|
|prior_only|8|1976|0|0|50400|453600|
|joint_coarse|5|1235|1235|0|31500|283500|

下面为当前设计与追踪。训练和预测仍按原入口串联，源资格失败则0target；退出检查在实际进程终止后完成，不把启动核实当作退出核实。


状态：RUNNING，真实入口与两条 GPU worker 已核实。持续目标为原冻结 F50 seed2026092701/defaulthead 下，正式 low urban ≥70%；其他五个受损场景分别 ≥79.6761904762%（原始 raw clean 84.6761904762%−5 个百分点）。restored clean 损伤单列。源域证据不能代替正式结果。

R22 的源域配对结果为：解析约束相对 IQ 控制 low urban +0.5556 pp、mid urban +0.1667 pp，但 mean6 IQ NMSE 增加 2.0001%，没有通过固定五参考资格。R17/R18/R19/R22 均固定 R07 coarse 和接收作用估计权重。R14 已知作用诊断不构成整个模型理论上界，亦没有支持单独追求作用估计精度。本轮检验另一个明确问题：在固定接收作用估计权重时，允许 coarse 适应最终三步恢复目标，是否改善源域观测与身份保真。历史 F 引导联合训练属于另一权限路线，不能混作本轮 0 F 训练对照。

两行 `prior_only` 与 `joint_coarse` 均从同一个合规 R07 完整上下文及同一个 scratch R17 三步不共享 Core9 初值开始，初始全部张量和输出完全相同。每步两次 warm CG、coarse 锚定，部署为一份恢复 IQ；原 PAMF 的所有视图继续从该 IQ 计算。`prior_only` 仅更新三个 prior；`joint_coarse` 还更新 coarse，学习率为 prior 的 0.1 倍。作用估计器参数在两行都冻结；joint 行梯度通过固定估计器关于 coarse 输入的导数、作用和方差、CG 返回 coarse。因此估计器权重固定不代表其预测的作用和方差在训练中固定，不称接收参数可辨识或物理正确。

全部原 6300 L_s/56700 U_s，原 R07 二十轮 received 缓存，batch256、20 epoch/4940 updates，model2026092701/data2026102001/cache2026101912。TX 标签、训练好的 F 权重/输出、源 V 拟合及 target 训练读取均为 0。只有逐包 IQ NMSE，clean target detach，无额外解析/FFT/身份/伪标签损失。允许的 clean→received 模拟配对须披露，准确名称为“TX 无标签模拟自监督”。F50 及原默认头只在训练冻结后评估，不重训或替换。

prior AdamW 2e-4→2e-5、coarse 2e-5→2e-6；共同 warm247/cos、WD1e-4、FP32/noTF32、固定末轮。prior 与 coarse 参数组各自 clip norm1，逐组报告裁剪比例。每行新的优化曝光 L126000/U1134000；作用估计器更新0，prior更新4940，coarse分别0/4940。同更新预算与相同部署不代表相同训练 FLOPs。顺序阶段为 coarse20+action20+本轮20；joint 行 coarse 累计40、action20、prior20，组件更新轮数不能混作新增独立数据或相同训练成本。

仅继承 R07（`20261019-phase1-restoration-learned-action-source-s2026092701-r07`，`c8ccdfdbd03ab5ad8828b1eb9210b99a664ebe7b`）及其 R02 coarse 来源，沿用已核实 original7/63/30、原物理 ID 和完整来源结论，不加载 R17/R19/R22 的训练后 prior，不扩大数据角色。R23 新 checkpoint schema 明确记录 coarse/estimator/prior 更新权限及实际更新数，拒绝旧 schema 或不一致来源。

固定源1800×七视图全部预测后独立 CPU truth 评分。五个既定参考 native/R15/R16/R17/R19 及原波形/身份/clean/最差 TX/其他场景保护不变；全部合格后按 low/mid 均值、最差 TX、最差 RX、名称排序，至多选一行。若无合格者则 0 target。若选定则自动完成原168000×七视图、原冻结 F50/defaulthead、全部预测固定后独立 truth-last。源/正式分别报告三郊区、三urban、clean、逐 TX/RX/day、修复/损坏、波形/特征/成本及缺项。已暴露基准不称新盲测；旧正式成绩不得反馈本轮结构或选择。

本地实现和聚焦检查后，实际代码 commit、一次独立 P0/P1、最小登记、新不可覆盖 release/run root、一次 N607 预检/归档比对/编译。launcher 首步真实2Ls/checkpoint/noquery smoke，PASS 后立即队列训练和条件测试。沿用池0/1/5/7、最多两条本任务 GPU worker、每卡一条、全 owner cap2/free12GB、共享容量锁直至 CUDA PID 可见。R22 原所属进程已自然退出仍须入口核实；不修改其他 owner。只有预登记系统技术故障停止并保留产物，不因低分重启或热改。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|J23-01|报告接收链逆问题|同完整初值/部署，仅 coarse 梯度权限变化|models.py/design.py|verified|初值/输出/逐包测试 PASS|严格实现本轮对照；非首次联合网络研究|
|J23-02|无监督首先正确|全 LU、0TX/F/target、clean仅训练目标|training.py/runtime.py|pending|负测/完整训练记录|模拟配对自监督，非无 clean 训练|
|J23-03|可达梯度|三 prior 和 joint coarse 真正更新，estimator不更新|models.py/training.py|implemented|真实梯度/优化器一步/完整更新计数|joint 包含固定估计器输入导数|
|J23-04|数据与继承权限|只继承原 R02/R07、schema隔离|runtime.py|verified|来源负测/真实checkpoint入口|不重验未变数据|
|J23-05|两阶段三层验收|五参考固定源选择及条件正式隔离|evaluation.py/controller.py|implemented|负测/全部预测后truth|无资格0target|
|J23-06|资源与最小发布|登记/commit/smoke/一次review/队列与退出|runner.py/controller.py/登记|implemented|发布/队列已核实；退出待实际结束|不增加其他门槛|
|J23-07|全场景身份保真|源及条件正式、逐组指标、成本/Git交付|report/summary|pending|独立评分与报告|不以NMSE代替TX归属|
|J23-08|逐细节归属|多seed、独立物理族、逐细节真实性|后续研究|deferred|尚无合格候选|当前不得宣称完成|

当前2 verified、3 implemented、2 pending、1 deferred、0 blocked。29项聚焦检查、compile/help、一次独立P0/P1及实际checkpoint入口、发布与两条worker运行证据已通过；退出及完整训练/源评估/条件正式结果仍待返回。最高剩余风险为 coarse 漂移、跨 RX 泛化与逐细节身份真实性。

实际参数：{"prior_only": {"total": 4665073, "groups": {"prior": 2507526}}, "joint_coarse": {"total": 4665073, "groups": {"prior": 2507526, "coarse": 1305958}}}。参数统计仅构造恢复器，未读取训练好的 F/TX/target。
非阻断记录：REJECTED_EXTRA_GATE——通用全仓库测试要求由项目聚焦负测协议覆盖，未添加全面测试或重复 review 门槛。

完整矩阵、命令、环境/CWD、路径、GPU及停止规则见[登记](experiment.json)。


控制行的完整源评分已返回：[prior_only局部报告](source_control_interim.md)。本矩阵仍RUNNING，未改变其他行或最终选择规则；该局部结果不冒充正式测试。
