# 九行冻结 RX 对抗模型的源域独立探针

最终目标仍是普通 cos、160 维身份表示的三主种子 clean 均值至少84%，尚未达成。本次是中间源诊断，0身份更新，不把源指标当作目标成绩。

前一固定矩阵的完整源记录显示：条件在线 RX 辅助头末轮三seed均值98.7748%，相对同公式DETACHED只低0.4798个百分点；活跃阶段encoder净域/基础梯度中位数.07360、夹角90.37°，不存在明确的严重反向冲突证据。因此不根据已知目标分数放大对抗强度或加冲突投影，先检验冻结特征能否被一个全新独立探针恢复RX。

全部3arm×3primary的固定FINAL30只读加载，来源、物理角色和训练预算匹配；不加载辅助对抗权重，不训练或选择身份模型。只扫描L6300/V27000，每个模型33300输入，共299700源模型前向；Ucache与target index/IQ/truth/预测/score都不打开。V只健康，无晋级门槛。

原样复用既有source健康探针：time_down时间均值/标准差96维、t2.pool192维、joint160原表示。每层1个无条件＋6个TX条件RX5线性分类器；每TX/RX/day内按物理ID排序mod3，fold0/1训练，fold2诊断，V只评估。标准化仅拟合训练L；100全批AdamW更新、lr.01/wd.001、seedprimary+31337，未据结果选择超参或轮次。总189探针/18900probe更新/567混淆；独立审计从私有rawsource重拟合的另18900probe更新单列，身份仍0更新。保存L/V概率、描述子、合法标签/RX/day/IDs/folds供全量重算，不提交这些原始源产物到Git。

同seed/层位/TX配对比较所有arm。身份参数、缓冲、模式、梯度与RNG严格保持；探针不写回、不部署。Lfold2只独立于探针拟合，身份此前见过全部L；V仍是已知源RX。可解码性下降只支持固定线性源探针结论，不证明全部RX信息、纯接收机硬件因果、未知RX或未来TX身份保留。

单CPU进程2线程，CUDA不可见，GPU新增进程0；一个ordinary-account owner与不可覆盖root，所有旧任务保留。一次代码/config提交与push、聚焦负例/真实9checkpoint无query首步、唯一P0/P1、最小预登记、一次N607 preflight、单archiveSHA、remotecompile、PID/CWD/argv/GPU/log检查。仅预登记技术失败停止，低指标不停止或重跑。命令、环境/CWD、路径和期望输出见experiment.json。明确source-only诊断，无target测试。

|ID|验收|状态|边界|
|---|---|---|---|
|D1|全部9固定来源、只L/V、无U/target|implemented|实际来源与完整扫描待核实|
|D2|同层位/同物理fold/同fixedprobe|implemented|189probe实际输出待返回|
|D3|Lfold2/V诊断与全部arm配对|implemented|V无fit/gate|
|D4|身份参数/缓冲/modes/RNG/梯度不变|implemented|实际冻结保护待返回|
|D5|rawsource全量重算、成本单列|implemented|完整独立审计待返回|

本地组件与注册检查 PASS：合成189探针/18900更新/567混淆，源角色/标签/元数据/配置/注册、detach、fold2不参与fit、输出不可覆盖，实际native96/192/160与身份状态/RNG保护均通过。唯一P0/P1发现发布清单缺已有漂移依赖，已补齐；仅一次定点复核PASS，无剩余P0/P1。compile与9row注册config一致性PASS。真实9checkpoint no-query与实际源probe尚未运行，不把合成检查冒充正式证据。

发布状态：未发布。普通账号只读preflight在本地SSH客户端启动权限层失败，未形成服务器SCP/解包/进程。本地准备与成果可交付，原run和其他任务未修改。真实smoke与完整源证据待连接恢复后由原owner继续同一固定矩阵。
