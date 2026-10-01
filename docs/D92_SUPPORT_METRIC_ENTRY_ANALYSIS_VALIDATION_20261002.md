# SupportMetric 联合入口与独立分析验证

状态：VERIFIED_FROZEN_SOURCE_SYNTHETIC_AND_INPUT_METADATA_NOT_PERFORMANCE。新候选 core 未改；此前75项已通过，本次只验证新受影响入口、分析器与控制，共116项，合计191项相关合成用例。没有新方法性能结论。

| 部分 | 通过用例 | 覆盖 |
|---|---:|---|
| 原独立 basis/metric/core | 75 | 精确秩及近秩亏、实际Gram、类均衡Fisher、SPD方向/KKT、完整head JVP和B→C |
| 新support入口 | 26 | 真实R0类别列、native A、实际B→C、一次row basis、四phase工作账、失败状态、详细日志 |
| 新独立分析器 | 43 | 四row/160parent/1800path全关闭后配对；归档U/Gram/Fisher/GGN/M/H/KKT回读、篡改/失败拒绝 |
| 新控制与发布 | 47 | 显式六项资源、exact与maximum预算、CPU2/root/no retry、marker/普通证据闭合、隔离源码依赖 |

数值由root串行使用实际激活ssr-gpu运行，NumPy2.2.6/SciPy1.15.3/Torch2.1CPU；无Conda CLI并发包装。已保留原始输出的逐字节gzip与环境/命令记录。Torch的已安装NumPy ABI初始化警告保留；native A实际路径以Python list构造tensor并通过测试，不声称全环境tensor↔NumPy互操作兼容。

隔离import37个模块、release46个文件；禁止np.load/torch.load并确认不加载encoder。只验证源码导入与配置存在，不能证明远端训练性能。四个row的只读元数据preflight全部VERIFIED，support缓存、source-only scratch来源、ground包和已验证capsule匹配；无特征值、query或源样本读取，独占新输出尚未存在。没有重做VALIDATED_ONCE数据验证。

root限定检查新producer的row basis工厂/失败保存、实际B→C、四phase账，新control的次数/资源预算与独占发布，以及新analysis的全矩阵先闭合后join；无未解决P0/P1。独立配置审查另见[预登记审查](D92_SUPPORT_METRIC_PREREGISTRATION_REVIEW_20261002.md)，不重复既有core全审或增加泛化、签名链和审批门槛。

方法使用实际UᵀU与预测Fisher定义M，保留RMS类CE的GGN，解dᵀMd≤0.25方向；真实RMSCE/Armijo接受与二次模型下降分别记录。固定一条方法，初始η=1，最多12次折半，不以成绩选择参数组合。数学出处及原创组合范围见[坐标度量推导](D92_PROTO_FRAME_GGN_COORDINATE_SCALE_NOTE_20261002.md)与[实现计划](D92_SUPPORT_METRIC_JOINT_IMPLEMENTATION_PLAN_20261002.md)。

实验固定old6、K1/5/10/20×new0/2/5/10/20，两个固定模型与两个cohort，4row/160parent/1800path。每row字典仅由地面Q构建一次，各prep绑定和实际Gram计算单独测量；有效秩与名义五坐标分开记录。所有算法、资源和seed角色均逐行预登记。合成验证不能保证新旧类准确率、遗忘、query泛化或星载资源优势。

实际配置/原始证据与后续状态见[实验记录](../automation_reports/CV-SincNet/20261002-phase2-d92-support-metric-joint-support-m2-r01/experiment.json)。完整head/JVP严格区间包络仍未建立；当前仅有限float64子问题与字典几何证据。旧query按自己的不可变source commit继续，不读取部分成绩，也不热改或重启。
