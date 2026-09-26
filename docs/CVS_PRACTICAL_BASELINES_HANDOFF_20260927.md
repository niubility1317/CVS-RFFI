# CVS外部对比实验准备状态

> 后续更新：用户已确认Phase1/2统一residual_noeq，并要求加入POSTER、RadioNet。后续实际矩阵与启动状态以automation_reports/CV-SincNet/20260927-phase1-baselines-practical-manysig-m5-r01/为准；下方保留本文件首次交接事实，不再代表当前阻塞状态。

## 本次范围

用户要求先运行CVS以外的方法，覆盖Phase1、Phase2域泛化、域适应和新类注册。Phase1使用新的practical LEO residual信道。协同实验属于另一篇论文。

## 当前事实

- 尚未启动本次正式训练、数据构建或评分；没有新的实验结果。
- N607普通用户SSH已通过只读预检；当时8张RTX3090空闲。启动前必须重新检查，不能把这次快照当作永久空闲授权。
- 已找到CVCNN、RIEI、DRIFT、receiver-agnostic训练入口，以及Phase2域适应、CSIL和MoPC-HR实现。STAR近期对比使用不同数据集，不作为CVS结果复用。
- 已从主仓库提交ff1cf0f46a028875960136eef1ab3c117c6e91f9创建独立分支codex/cvs-practical-baselines-20260927；主工作区原有修改未动。
- 本次仅完成公共训练器的source_only开关和伪标签遥测修正。各方法CLI、源域数据构建器和practical增强尚未完成接线，因此不能直接用旧CLI启动并声称隔离已生效。

## 已实现的准备修改

1. 伪标签损失不读取label/true_label计算precision。覆盖率、置信度和损失仍保留；隐藏标签评分须在独立评分阶段进行。
2. 公共训练器source_only=True时拒绝目标loader和测试回调，关闭训练期间测试，末轮保存last.pt，状态为SOURCE_TRAINED，不生成目标成绩。
3. 新增针对隐藏真值访问、目标入口拒绝、末轮权重保存的3项测试。先确认测试在旧代码失败，再修正通过。旧伪标签测试已按真值隔离协议更新。

验证：使用C:/Users/lh594/.conda/envs/ssr-gpu/python.exe运行test_baseline_source_only.py（3项）、test_baseline_pseudo_labels.py（5项）、test_baseline_training_behaviors.py（15项），共23项通过。测试中的模拟目标指标不是正式实验成绩。

## 待明确的技术口径

已向用户提出一个问题，尚待回答：所指新信道是否是已有residual_noeq（processing_route=residual、mode=post_sync、equalization_enabled=False），以及Phase2的support/query是否也切换同一信道，还是保留LEO_WEAK。

这影响数据和比较含义，不是再次索取已授权实验的执行许可。没有收到回答前，不静默将Phase2迁移到新信道，也不把尚未确认的信道配置登记为实际启动配置。

## 接续工作

1. 按确认的信道版本接入各方法。practical独立快照要求长度不超过1ms，必须显式记录采样率、物理sample ID及RX/day session ID；禁止以TX标签构造接收机硬件会话。
2. 源域构建必须遵守同一物理L/U/V角色契约。旧SSL构建把target days作为source holdout，遇到当前日期重叠会排空源数据；不能仅照抄旧参数。无标签训练对象应移除隐藏真值，不能只依赖损失函数不读。
3. 全部从零训练，不继承来源不明或目标污染的旧CVS权重。目标数据不进入训练或源域选模。Phase2衔接本次各方法的合规权重。
4. 明确原生方法与统一增强/PL扩展的命名和预算。Phase2分别登记冻结特征域泛化、目标支持集适应和新类注册，CSIL/MoPC原生权限单独披露。不得把协同融合混入本文。
5. 完成方法接线后的本地测试、一次独立P0/P1审查、逐row预登记，提交并验证push后发布。每GPU最多两个训练实验，一个run只有一个launch owner。
6. 启动后读回PID、CWD、argv、GPU、日志和resolved config，再报告RUNNING。预测固定后独立评分；不以目标成绩回流调参或选择性重跑。

## 路径

- 实施工作树：E:/type10-7/code/snapshots/cvs_practical_baselines_20260927_wt
- 主Git承载面：E:/type10-7/github_publish/CVS-RFFI-repo
- 已有总体设计：docs/CVS_PHASE12_PUBLICATION_EXPERIMENT_PLAN_20260927.md
- 新信道：code/leo_practical/channel.py、residual.py、batch.py
- 既有残差参考：E:/type10-7/code/snapshots/daot_practical_three_20260918_wt/experiments/adv3b02_xuc/configs/rc4_practical_residual_noeq_20260918.json

本文件是准备工作交接，不代替experiment.json/report.md/events.jsonl预登记，不表示任何run已启动。
