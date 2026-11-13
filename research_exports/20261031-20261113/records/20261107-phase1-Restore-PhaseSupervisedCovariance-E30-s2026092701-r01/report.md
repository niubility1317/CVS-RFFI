# 主路相位独立监督的恢复器，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-PhaseSupervisedCovariance-E30-s2026092701-r01`
- group_id：`explicit-source-canonical-main-phase-recovery`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在源均值定义的 FIR 相位坐标中，独立预测包级主路相位，用原源模拟真值监督圆周角差。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


对原源训练缓存固定 512 条 × 20 次的统计显示，相干参考归一化重合度中位数 0.0441，最小值 0.0009144。原首 tap 只有 1526/10240 次是最强 tap。使用相干参考固定 FIR 相位，并以圆周误差监督主路。 源训练模拟统计

## [计算环境] 主路相位监督版排队已核实


## 30 轮源训练完成，等待正式预测

VERIFIED / SOURCE_TRAINED。已完成固定30轮、7410步，189000次L_s和1701000次U_s曝光；30轮顺序、draw、计数、有限损失和目标组成复核通过，最大组成差2.87e-9。训练终态退出码0，最终权重文件存在，源TX和目标训练读取均为0。观察时尚未提交任何Main预测，原控制器在线且没有训练子进程；正式七景及独立评分未完成，不能据源训练指标判断目标识别性能。下方早期“训练待执行”为历史排队记录。 源训练审计

## 已训练模型的剩余预测转常规 GPU3

复用已完成原30轮/7410步及原权重，剩余cross_rx、cross_both七景和独立评分使用原科学release。仅在旧控制器CPU空CUDA、无子进程、两个预测均未开始时交接；GPU3总数最多3，准入前至少12000 MiB。不重训、不读取目标成绩作调度依据，保留原输出和日志。

控制代码：`6e6be6f3db354e7809ed186d490c634ed4add999`；科学代码：`36b13adad5b091bf1f8f4aa7934c403e3b50d8eb`。命令：`[已省略路径] -u -m experiments.pamf_phase3_restoration.phase_supervised_remaining_dispatch --root [已省略路径] --legacy-release [已省略路径] --cache [已省略路径] --commit 6e6be6f3db354e7809ed186d490c634ed4add999`。

## 剩余正式预测已实际启动

VERIFIED / RUNNING。复用原30轮训练和最终权重，剩余跨接收机预测已在常规GPU3实际启动；启动时已有2个计算进程、17128 MiB可用，总数上限3。原科学代码/CWD、实际进程/GPU及原源样本检查已独立核实。旧队列控制器仅在无子进程、无预测产物时交接；没有重训或重复预测。首景日志增长已观察，尚无完整Main结果。 运行核验

## 首个预测场景及日志增长核实

VERIFIED：原跨接收机 clean 预测日志已增长并完成126000条，实际预测进程仍在线；其他六景和跨日期总体仍按固定矩阵执行。预测完成不等于已评分。

## 完整30轮与原Main七景独立复算完成

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.2e-14，baseline判决差0。六景80.766071/78.182738/75.944643/77.827381/65.753571/56.278571%，0/6；clean两个总体净-453/+323，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 完整结果
