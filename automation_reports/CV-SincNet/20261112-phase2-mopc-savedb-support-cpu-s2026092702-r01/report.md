# MoPC 保存 B 的 CPU 源／支持诊断

状态：PLANNED_NOT_STARTED。拟比较 r15 失败任务 `protonet_cda-s2026092702-c0195` 的实际 r11 B-K10 状态和原 K10/N20 support，在 CPU 上执行原优化路径及显式梯度裁剪 1.0 的诊断变体。变体更改优化器更新，不称作者仓库原版。两次各自重新加载同一保存 B，不拟合新的 B；原始 seed、阶段预算、loss、guard 保持。只读源权重、源 L_s 和任务 support；不读取 query/truth，不生成预测、不评分、不重发正式任务。CPU 与原 GPU 数值内核和随机数不同，未复现也不证明 GPU 故障已修复。

代码与配置以本报告所属提交为准；固定 release `releases/20261112-phase2-clean-preparation-r02` 中的诊断候选，CPU 单进程、1 线程，源 `releases/20261109-phase2-bsource-savedb-mopc-m4-r15-release/runtime.json` 和该 run `jobs/00002.json`，标准输出作为本机不可覆盖诊断证据。首个非有限按原 guard 停止，保留已有输出；两个变体独立记录完成与否和更新次数。所有源状态、旧 partial、claim 和评分屏障不变。

复核发现 r15 实际使用 RegistrationConfig 的 paper 默认，而此前准备的诊断错误绑定 public_code。修正诊断显式选择 `paper`，仅裁剪阈值不同；固定本报告所属提交的新诊断 release，不使用旧 r02 执行。诊断后仍单独研究用户指定的 public_code 版本，两种语义不能混称。

CPU 独立诊断已结束，无 query/truth/预测。原 r15 paper 语义及 FP32 guard 下，无裁剪完成 320 更新、最大 loss 11924.7432；裁剪阈值 1 完成 320 更新，320 次均裁剪，最大 loss 10.6312。CPU 未复现原 GPU 故障，裁剪改变更新；不得将该结果标为仓库 public_code 语义的修复。诊断数据和日志保存在 `runs/20261112-phase2-mopc-savedb-support-cpu-s2026092702-r01/diagnostic.log`。
