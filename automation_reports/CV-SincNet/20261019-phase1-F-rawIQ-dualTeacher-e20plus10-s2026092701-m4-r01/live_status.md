# 实际运行状态

RUNNING，尚无本轮最终性能结论。N607普通账户提交与运行均已核实：

|row|GPU|PID|首步真实父CP smoke|已记录本段epoch|
|---|---:|---:|---|---:|
|NOAS_ADAPTER|6|2318182|PASS|1/10|
|S1|7|2318268|PASS|1/10|
|S2|4|2318352|PASS|1/10|
|S3|排队|尚未创建|待执行|0/10|

Dispatcher PID2318162，PID/CWD/完整argv/CUDA设备与预登记一致，三个worker均在实际GPU进程表中。源训练日志与stdout持续增长，已保存第一轮完整epoch记录；无failure，无源矩阵决定，无新正式评分读取。新增参数实际4256、模型总参数904978，head_fit_updates=0，FP32/TF32off。

实际代码/config commit：`28d14761575b080077fde038cc96baedd4d982cc`。归档/运行版本：`6ea2f71f6654f4ceeadd460905a9d719b5dc1b89`。单发布包本地/远端SHA比较VERIFIED，远端compile VERIFIED；push后远端branch OID与本地一致。

原F/defaulthead保持学生冻结规则，新增raw/As在线正确数取已有logits、不额外NN；训练态读出不当作冻结拟合或目标结果。每GPU本run最多一worker，S3在空位出现后由同一dispatcher自动启动；不改健康任务或配置。

后续自动完成固定FINAL30→源资格读出→七视图预测→源决定固定后独立truth-last评分→完整汇总。已完成旧控制直接复用。源3条件和预算已经冻结，不因中途曲线或测试结果调整。

独立运行证据在私有artifact的postlaunch.json、progress002.json、progress003.json；监控只读本run状态、source日志和完成标记，不读取target分数。
