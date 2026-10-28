# 论文补充队列并行度调整

用户要求继续派发既有任务，每张卡最多 3 个进程。原科学矩阵、配置、训练和预测输出、既有权重、truth-last 评分及技术停止规则保持；已有工作进程不停止或重启。新调度器只接管原 run，先暂停并核实旧调度器、识别现有 worker，停止旧调度器后才派发；未知或不完整的既有进程和输出阻止接管。每张 GPU 总计算进程不超过 3，外部用户占用的卡跳过，单个独立 CPU scorer。继续沿用原 source/checkpoint smoke 和科学 worker 执行版本。

- 原 run：`20261027-phase1-pamf-paper-completion-m119-r03`；原科学代码 commit `c9ebb6db301815159710b4c5cc461c9fd808dc55`。
- 环境：N607，`CVS-RFFI` Python，原 release 和 `runs/20261027-phase1-pamf-paper-completion-m119-r03` 工作目录及输出根。接管控制脚本另用提交版本单文件发布。
- 本次目标卡为 0、3、4、5、6、7；1、2 当前有其他用户，继续避让。显存至少留 12000 MiB、系统可用内存至少 32 GiB；计入卡上其他进程和尚未进入 CUDA 的本机 worker。
- 接管脚本入口：`python parallel_takeover.py --runtime <原 release/runtime.json>`；执行后核实原 owner 退出、接管进程、每个 worker 的命令/GPU、日志增长及原 `pipeline_state.json`。原数据、配置和产物不移动、不覆盖。
- 任一任务失败时停止新派发，保留其余运行进程完成；预测完整后才独立评分。检查真实 GPU、CPU、内存和磁盘条件，单纯增加进程数不保证加速。

首次接管在暂停旧 owner 后，扫描非实验的用户系统进程环境时遇到 `PermissionError`，自动恢复旧 owner，未派发新 worker。修复为先按命令和本 run 配置路径过滤，再读取候选 worker 的环境；保留失败日志并以新文件名发布修复版本。

第二次接管已停止旧 owner，但在确认其退出时，进程退出瞬间读取旧 owner 的 environ 遇到 `PermissionError`；新 owner 退出且未派发任务。独立核实原 owner 已退出、5 个原 worker 存活，另一 worker 已产出 `SOURCE_FROZEN`。修复为只用进程状态确认存活，并提供仅在原 owner 确实不存在时的显式恢复入口，重新核对现有进程与产物后继续原队列。
