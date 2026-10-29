# Phase2 第一组 PAMF 技术修复发布

Run：`20261030-phase2-group1-concat-midPL-threepool-m4-r02`。替代 r01 中失败的 PAMF 链；10 个外部方法继续使用 r01 的原代码、配置和输出。本轮科学矩阵、三物理池、7 RX、4 seed、K/N、DURA 四格、目标数据权限及全部正式预算保持。

故障事实：r01 PAMF 完成 E20/4440 原生更新，但 CRIC_calls、extra_L、satellite 和 PL 均为 0。实际加载的 20261027 发布目录中原生 trainer 缺少监督 hook。该 E20 是错误配方，不作为正式 PAMF，也不作为替代训练的父权重。原产物及失败资源成本保留。

修复：显式绑定既有冻结 BASE 的支持 hook 的训练器，检查实际导入路径及调用能力，每次真实更新检查完整监督 hook 计数。PAMF 从零重训 4 seed，各 50 epoch/11100 updates，20＋10＋20 原优化轨迹，E26 开普通 PL0.95/权重1；保持源星地拼接0.68、完整监督机制、无EMA与原默认头。新增技术检查不改变科学目标。

只停止经 PID/CWD/argv 核实的 r01 CPU 调度器；健康 GPU worker 保持 PID、release、runtime、输入输出和显卡。统一控制器收养它们，已完成工作读取原 completion 复用，其余外部任务继续在 r01 release 执行。r02 只启动 PAMF，旧队列不恢复。两版合计每 GPU 最多两个本任务训练/预测 worker，其他 owner 另计，实际剩余显存至少6000 MiB；CPU scorer 总计最多1。原源增强服务及固定缓存共享，不重复生产。

发布前聚焦回归、一次独立 P0/P1 review、单次 archive SHA 传输核对、远端编译和真实源 checkpoint smoke。新增 smoke 运行四次 CPU 真实优化，确认监督 hook、卫星损失和 PL 反传；仅诊断时立即开 U，正式中期日程不变。smoke 权重不作初始化，不访问目标或 query。

A/B/C 继续分阶段实测 CPU RSS、CUDA allocated/reserved、训练/适应/注册/推理时间、优化步、实际更新参数和持续状态大小；共享资源环境如实标记。PAMF 资源与 r01 外部资源原位汇总，失败开销单列，不计作有效训练成本消失。

唯一配置：`configs/pamf_phase2_group1_recovery_20261030.json`，科学矩阵引用 r01。入口 `python -u -m experiments.pamf_phase2_practical.group1_recovery --runtime <release>/runtime.json`；实际 commit、环境、路径、owner 和命令见登记。每预测单元完成后独立 truth-last，分数不影响后续运行。技术失败阻止相应后继；相同修复指纹再次发生时不盲目重启。状态：PREREGISTERED，启动须独立读回。
