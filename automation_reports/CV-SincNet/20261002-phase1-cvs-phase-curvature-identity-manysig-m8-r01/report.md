当前接续状态：RUNNING。2026-10-02新对话核实原发布已成功，补齐启动登记；8个原进程健康继续。以下保留发布前历史记录。

# CVS六层相位曲率残差：源实验预登记

状态 LOCAL_VERIFIED，尚未发布。两种固定延迟(1,4)/(2,4)各四seed，共8行。六个复数FIR输出后、原共享归一化前加入延迟平衡三阶差分，每块一个零初始化全局tanh系数；保留原输入两个相位记忆系数、通道、深度、频域、读出和分类头。202561参数，比当前控制增加6，初始函数与own-scratch控制一致。CE唯一，无增强，仅身份骨干，原物理划分，scratch E200×50，完整FP32。控制仅读取四份adaptive源元数据，不加载控制权重。

[数学、通信、物理及RFF设计](../../../docs/CVS_FEATURE_PHASE_CURVATURE_20261002.md)。72项聚焦检查、8个公共模型24次CE更新通过。新算子对恒幅线性相位delta为零，对相位曲率有明确响应，局部仿射相位协变且delta有界。该性质不等于完整网络CFO不变；系数不是TX硬件参数，源/公共证据不证明唯一TX/RX分离。尚未证明识别性能提升。

固定源排名：最高四seed mean(0.5V+0.5最差源RX)，只用E200，完全并列后比较成本。若新候选胜出，冻结后默认4新clean预测＋32固定控制，共36行/288总体与RX记录，全部预测固定后独立truth-last评分；否则复用当前控制既有测试，未选候选clean记N/A。测试只clean，不追加LEO/SFT/support/新类。目标结果不回流结构、延迟、epoch、seed或选择性重跑。

保留详细文本、10000步JSONL、完整与紧凑epoch JSONL/CSV。每一步记录六系数实际梯度/raw/tanh前后值，保留原两系数日志；每轮末源batch检查六个实际FIR曲率输出、前缀mask、修正幅度、12项输入和六共享归一化。最终保存90源TX×RX×day分层、公共通信链与实际资源。ConvLinear MAC不覆盖新增逐元素运算，新增6参数不能推出总计算量相同。公共数值响应只报告，不参与排名或停机。

[本地检查](evidence/local_validation.json) · [公共执行](evidence/local_cpu_smoke.json) · [公共汇总](evidence/public_probe_summary.json)。正式源训练、源冻结和条件clean测试尚未完成。

发布前一次实际源P0/P1审查 PASS，无阻断项。独立3项定点检查、10模块及collector注入/publisher远端脚本编译通过。源preflight VERIFIED：完整四份adaptive控制角色/来源/预算/FP32一致，用户/主机正确，八张GPU空闲，磁盘5.5T可用，新release/run/log无碰撞。尚未正式启动。[独立审查](evidence/independent_review.json) · [源preflight](evidence/preflight.json)。

正式发布与运行 VERIFIED：release `d5a34a105348195c29601c97df248cc4e68fec24`，dispatcher PID 1852325；独立读回8个worker真实PID/CWD/argv/GPU、实际202561参数/完整FP32/alpha0/CE唯一/全参数梯度与六个FIR曲率输出测量，进度E22至E27。远端8公共模型24次CE smoke PASS。完整E200源冻结及条件clean尚未完成，不称识别性能提升。[启动读回](evidence/launch_readback.json)。

接续研发：条件clean集成已完成，77项新旧协议合成检查通过，独立46项检查及P0/P1审查PASS。实际目标预测仍待固定E200源选择。已完整分析5组40模型8000轮源证据，进一步预登记8模型×9冻结源V消融，区分曲率执行与实际识别贡献；不改变当前源排名。
