# BranchInteraction完整support诊断解释

VERIFIED：8个模型/接收机组合全部完成，共4800个任务；其中K1的1200个任务仅作数值检查，K5/10/20共3600个任务执行物理support内OOF。独立终态读回显示supervisor及子进程均已退出。

下表仅来自合法support内部留出，不是query准确率。新旧类任务含6个旧类及2、5、10、20个新增类；old-only另行报告。单位：百分点。

|对照|K|Δ旧类|Δ新类|ΔH|预定逐K条件|
|---|---:|---:|---:|---:|---|
|linear|5|+0.840|+1.743|+1.582|通过|
|linear|10|+0.642|+1.962|+1.621|通过|
|linear|20|+0.813|+2.522|+2.030|通过|
|energy_control|5|+0.358|+0.867|+0.757|通过|
|energy_control|10|+0.375|+1.102|+0.888|通过|
|energy_control|20|+0.683|+1.614|+1.368|通过|

预登记要求在每个K分别相对linear和energy_control满足H、新类准确率严格提高，旧类退化不超过1个百分点。完整结果全部通过，且联合任务旧类均值也全部提高。该结论仅支持冻结候选进入已授权的完整重复基准，不是自动晋级或独立泛化验证。K1没有可用的独立类内留出证据。

固定公式仍为KB+KA+KB*KA；保持原五块单view特征、ridge1、当前任务support一次正式拟合，不从此诊断调整参数。对照energy_control排查仅增大核能量的解释，但不能据support结果推断query改善。

实际runtime commit：`9cbebdb292d6fd3c54d029e9113360a12eea64c6`。墙钟44.062秒，32400次分解，每臂562800条物理OOF评分记录。CPU四lane、各两BLAS线程；不加载checkpoint，不访问query/源域样本/原型包，新增源域传输0字节。

完整分层和状态见results/support_summary/summary.json、by_k_newcount.csv、by_model_seed.csv、by_receiver_scene.csv及evidence/readback_1790669216.json。已有输出均保留，禁止重复启动或覆盖summary。

下一步：以同一固定公式和原矩阵复用BranchRidge正式缓存，逐样本推理；全部预测固定后由独立scorer评分，完整报告相对BranchRidge与原D92的结果。新实验尚未启动。
