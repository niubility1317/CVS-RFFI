# Orthogonal 组合接口支持集诊断

现有 clean 评分显示 30 个不同 B→Orthogonal pilot（六种 B 各 5 个，K5/N20）旧类宏准确率均为零。以 DADDA、P01/MIX01、RX1-1、seed 2026092701 为例，10332 条旧类 query 全被判为新类；在旧类内限制决策时旧类宏准确率为 94.89%。该结果提示新旧分数竞争失衡，不能单凭结果证明具体运算根因。已披露目标成绩，不作为调参或盲测证据。

诊断只读取同一 capsule/seed 的原 MoPC 完整队列中六种 B 的实际保存 `B-K5.pt` 和 clean 旧/新 support；pilot 本身未保存 B，另一次训练所得 B 即使 seed 相同也不能声称是 pilot 的同一个物理状态。重建生产 Orthogonal 使用的原桥，比较固定的仿射头权重余弦方案与另行标记的旧 support 原型余弦候选，在 C 注册前后记录旧 support 的新旧竞争、旧类限制正确数及 50 轮 loss 首末。后者是项目移植候选，不是论文原版，不能用其冒充原有结果。

采用 CPU 单进程、每次一个 profile、线程数 1；命令 `clean_orthogonal_support_diagnostic --runtime <原队列 hybrid runtime> --job <原队列 0000-hybrid-profile job> --k 5 --n 20 --device cpu`，六种 profile 为 DADDA、MRIOR、Poster、ProtoNet、Tweak、TwoStage。只输出汇总统计和 loss，无 query、truth 或预测文件，不修改任何正式运行或已有结果；N607 GPU 和其他 owner 保持。输出 `runs/20261116-phase2-orthogonal-bridge-support-cpu-r01`，注册报告记录 CPU 数值核与原 GPU 正式运行不同。
