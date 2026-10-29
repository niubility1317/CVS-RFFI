# Phase2 MoPC-HR 两版本技术修复与队列恢复

用户授权：2026-10-29 检查并修复两个失败 MoPC-HR 实验、确认 RadioNet 是否正常、恢复受阻任务。原失败 run 和所有部分产物保留；本轮修复使用新不可覆盖运行根。

## 原因与修复

两个失败均为 `_mopc_register` 的非有限训练项。B 结束和 `register_new_classes` 初始化均使用 eval，C 未恢复 train；六层 BN 因此一直使用冻结运行统计进行大幅 SGD 更新。作者公开训练循环每轮启用 train（[原始训练器](https://github.com/xmuLdz/MoPC-HR/blob/main/MoPC_HR_trainer.py#L128)）。该实现遗漏可用原 E200 checkpoint 和原 support 复现；不是读取 query 或目标成绩后调整超参数。

修复在每个 C session 优化前启用 train。BN 只由当前新类 support 更新，参考模型／原型抽取／最终 query 均 eval。保持 B10、C每session20、batch16、SGD0.01/momentum0.9/wd0.0002、全部损失与prototype公式、seed/信道/支持集不变；不加梯度裁剪、不减学习率、不缩短预算。

回归测试先证明旧实现未启用训练模式，再确认每session训练与最终query BN不更新、B/source不变。实际源checkpoint＋同一首单元原support在CPU复现两个版本非finite；修复后2版本×P0/P1×K1/5/10/20×N2/5/10/20共64格完整预算全部finite。该复现仅加载合法support和源文件，不读query/私有truth/评分，不作为性能结果。

## 接续边界

修复只重做 mopc_paper、mopc_public_code 的目标预测与评分，独占新根；旧版本即使局部预测完成也保留原身份，不混入修复版。新 run 的7728预测任务和3864评分任务恢复两方法全部原矩阵，首个两路K/N闭合后才扩展；独立评分不反馈调度。

统一调度器精确核对并接管旧CPU父调度器，健康GPU worker（包括RadioNet）保持原PID、命令、CWD、代码和输出。已完成非MoPC任务与全部源权重资格结果复用；后续非MoPC任务仍用原release/runtime。源数据、DURA同一缓存和原核心release保持。只新增一个真实checkpoint无query smoke，PASS立即继续修复pilot。

GPU0–7本轮核心＋外部＋修复合计每卡2个，其他owner另计；接入前实际剩余显存至少5000MiB。CPU scorer合计最多一个。不停止/重启其他owner。

## RadioNet 检查

原 RadioNet triplet 首seed PID2600850、GPU6保持。完整结构化训练记录显示前8轮loss全部有限；第9轮从16897步推进到19585步，同PID CPU时间和日志字节增长。每轮3304350个正样本对、25815次更新，完整30轮共774450次更新/seed，耗时长源于原预算。源训练target_contact=false；不因零triplet loss或耗时长缩减预算/重启。ADA/kNN、ADA+FT和DF版本已有首单元完成产物，后续保持队列原规则。

## 运行与停止

数据p2_min_v1/VALIDATED_ONCE和原483capsule/9660split直接复用，不重建数据。当前源契约、4modelseed、三物理池、23episode/池、7RX、P0/P1、K/N、source replay和truth-last不变。

运行命令、环境/CWD、commit、远端路径、逐行继承与预期产物以同run `experiment.json` 为准。唯一owner为 codex/root/phase2-external-mopc-repair-20261029。相同数值指纹经一次本地修复后若再次发生，保留产物并停止受影响版本后续派发，不盲目重试；健康任务继续。修复方法目标成绩不参与停止、重跑或选模。

当前状态：VERIFIED RUNNING。一次最小独立P0/P1审查与启动后PID/CWD/argv/GPU/日志证据回填于本报告。

最小独立P0/P1审查：PASS；MoPC support-train/query-eval及恢复矩阵四项聚焦检查通过，无额外P0/P1。

## 启动后独立核验（2026-10-29T08:16:26Z UTC）

VERIFIED RUNNING。执行 commit `924bd939d3386a108332b65af9fea5ab30d302ee`，单 archive SHA传输比较与远端编译PASS，真实checkpoint无query smoke PASS。控制器2742828接管旧CPU2597281；旧CPU已退出，健康GPU worker重启0。原11589项BLOCKED已恢复，新队列当前FAILED=0/BLOCKED=0；GPU0–7本轮各2，共16，其他owner另计。

两个MoPC版本均已在N607完成首个P01/MIX01/RX1-1/seed2026092701全部K/N的P0/P1冻结预测（各24row/路径）并完成独立评分；只核验产物状态，未读取分数用于调参。原两个失败与旧partial保留于r01，不能与新实现混合。完整后续矩阵持续运行，非全部实验完成。

RadioNet仍为原PID2600850/GPU6、原r01代码/CWD/命令，未重启。最新源训练第10轮，第8449步；全部1894条结构化记录loss有限，target_contact=false，继续原30轮×25815更新预算。
