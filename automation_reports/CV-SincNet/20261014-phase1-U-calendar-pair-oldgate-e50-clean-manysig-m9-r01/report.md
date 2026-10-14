# 等 U 预算训练时序：当前教师与真实优化更新

用户最新报告要求先检验 U 的训练时机，再推进 PA 证据重组。关系＋响应三种子相对 ZERO 均值下降 0.5849 个百分点，当前不把它作为已证实机制。正在运行的时间结构×关系矩阵保持冻结计划，用结构内差异判断机制贡献；COMPACT、LONG_SILU 与 Sinc 居中坐标的历史正收益独立保留。

## 固定候选与预算

| 条件 | U 激活轮次 | 标签专用轮次 | L 总曝光 | U 总曝光 |
|---|---|---:|---:|---:|
| TAIL | 34 至 50 连续 | 33 | 1420800 | 963900 |
| MID（预定主条件） | 17 至 33 连续 | 33 | 1420800 | 963900 |
| SPREAD | 17、19、……、49 | 33 | 1420800 | 963900 |

每行从零训练 50 轮、222 步/轮、11100 次实际 AdamW 更新。L 每步 128 包，所有 50 轮均参与原 CE；每个 U 激活轮将原 56700 包完整遍历一次，前 221 批每批 256 包、最后 124 包，共 17 次。三个固定种子 2026092701/02/03，共 9 新模型、450 模型轮次、99900 更新。日历在本 run 任何 target 访问前固定。MID 使用用户给出的第 17 至 33 轮例子；SPREAD 固定在中后段隔轮，不从目标结果选择起止点。

原 CVS 默认架构、PA/frequency/reference、NoExplicitCFO 身份输入与内部补偿保持，默认 ResidualPhysicalHead 6×160、scale30 随骨干正常联合训练。没有另训、替换或冻结后重训分类头。原 PAIR、旧置信/时间/强视图一致门控、ownEMA=0.999、所有 11100 次 EMA 更新、strong noise=.015/Dropout、λU=.16、全 U entropy=.01、L smoothing=.01、AdamW lr=.0002/wd=.0001/clip5、原 50 轮 cosine 最低 .000001 均保持。没有新增关系目标、Sinc 改动、训练信道增强、参数重加权、优化器重置或后期教师注入。

三组真实初始 191 个 state tensor 和 CPU RNG 相同，实际 native 构建→NoCFO→安装入口与预测构建一致，参数 844754。原 CUDA 非确定性设置保持；改变 U 日历会改变后续强视图和 Dropout 随机流，因此不承诺不同日历的完整轨迹逐位匹配。本次是用户要求的已暴露基准探索，不声称新的独立盲测。

## 直接测量 U 的训练作用

不使用“U 阶段学习率总量”替代其实际贡献。在第 1、6、12、17 个 U 轮的第一个真实 batch，复制当前参数、AdamW moments、step、所有 param group 选项、学习率和 weight decay，独立模拟两步：

`Δθ_U = AdamW(θ,s,g_base+g_U,clip5) − AdamW(θ,s,g_base,clip5)`。

这里 `g_U` 明确包含原 `.16 selected pseudo CE + .01 all-U entropy`，`g_base` 为实际总损失减去这两个 U 项。两条临时分支各自进行全局 clip5；保留 AdamW 的 None 与显式零梯度语义。真实模型仍只做一次原更新，临时总梯度分支与真实更新读回比较。按 t1/t2/t3、PA、frequency/projection/fusion、reference、增益和默认 W 记录梯度、裁剪、U 增量和更新比值；不把一步差异解释为整段训练的精确累计因果贡献。

TAIL 测量步 7327/8437/9769/10879；MID 为 3553/4663/5995/7105；SPREAD 为 3553/5773/8437/10657。每模型 4 个真实状态测量点、8 次临时优化步骤、0 次额外实际参数更新。

在上述 U 轮开始前，用**当时的 ownEMA**对原完整源 V 27000 包做 weak 预测、用当前学生 identity-only train/Dropout 做 strong 预测。PAIR 元数据排列和诊断随机种子固定，旧 gate 函数原样执行。预测和门控先固定，V 标签仅用于下游 accuracy、ECE15、分阶段 coverage/precision、RX/day/TX 统计；不拟合阈值或模型，也不读取 U 真值。保留逐包 tensor 供回算。每模型增加 216000 包次 V 前向，9 模型共 1944000 包次，另列成本。诊断保护模型参数、buffer、gradient、train/eval 标志和训练 RNG，不引入 EMA 更新。源 V 当前阶段代理不能证明隐藏 U 的实际精度，不能用最终教师回填早期结果。

## 源条件与自动测试

先运行 seed01 三个日历；三行 final50 训练 L CE 均≤.5，MID/SPREAD 相对 TAIL 的源 V 均值下降≤1 个百分点、最差源 RX 下降≤2 个百分点，才放行其余 6 行。这是预先固定的健康条件，不以微小 V 涨分选择日历，不按 target 排序或重跑；已启动健康行自然完成。

每行固定 final50 后自动默认头 clean 168000 包预测，prediction 完整固定后由 fresh CPU 独立 truth-last scorer 评分。不因其他行尚未完成而等待整批。全矩阵 1512000 决定、162 个分组混淆矩阵完整复算。比较 MID−TAIL、SPREAD−TAIL、SPREAD−MID，报告全部种子、均值/SD、worst RX/day、源收敛/U 覆盖/当前教师和实际优化增量。预定主条件 MID 的固定三种子 clean 均值≥84%仍为未达成研究目标；代码或诊断 PASS 不代表性能收益。

## 最小发布

- 数据：复用已验证原 ManySig 契约和物理 IDs，L6300/U56700 隐藏 TX/V27000，source RX1/3/4/6/8、day1/2/3、6 TX。原 clean capsule/split 复用；仅测试 clean，Phase2 不在本轮范围。
- run：`20261014-phase1-U-calendar-pair-oldgate-e50-clean-manysig-m9-r01`，owner `codex/root/U-calendar-20261014`。
- 代码身份：不可变 release 中 `release_commit.txt`，提交并推送后独立比较远端 OID。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_U_schedule50_20261014_r01`；环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；命令 `python -u -m experiments.cvs_u_schedule50.dispatch`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-U-calendar-pair-oldgate-e50-clean-manysig-m9-r01`，独立且不可变。
- GPU：普通账号，每 GPU 最多 3 个总实验进程，计入其他 owner，至少 12000 MB free；不停止、重启或热改健康旧任务。
- 首步真实 checkpoint no-query smoke：只读取已核实同契约历史默认 E50 student/ownEMA，严格加载后零输入检查；不继承到训练。独立临时合成 AdamW smoke 验证远端运行时，不冒充实际训练状态测量。
- 技术停止规则：本 run 的来源/角色不一致、输出碰撞、非有限实际训练、L/U/更新预算错误、临时总更新与真实更新不符、无法合法预测或未处理 CUDA/data/artifact 错误。保留产物，无自动重试，不处置其他任务。源健康只控制未启动种子，target 分数不得触发停止/调参/重跑。
- 产物：原 source args/契约/初始化/完整 step 与 50 轮日志、calendar execution、4 份当前教师 V tensor/summary、4 次 actual U update summary、final checkpoint/freeze、无拟合特征诊断、逐行预测/score/completion、source expansion decision、队列和总报告。

本地六种非法日历负例、默认旧日历兼容、三组真实 native 初始化一致、真实模型上的观察→backward→clip→AdamW 更新及完整 27000 包合成 V 状态保护已通过。唯一候选 P0/P1 审查、一次 archive 比较/remote compile、首步真实 checkpoint smoke 和发布后 PID/CWD/argv/GPU/log-growth 按项目最小流程执行，无附加审批、封印或 hash 链。

## 后续创新机制

下一项独立主机制为同 TX、同 day、跨 RX 的 PA **实际增量证据重组**：`z̃_i=a_i+(1−α)b_i+αb_j+c_i`，对称训练，用停止梯度的现有 W 与 PA/ref 增益计算真实 L 标签 CE；PA 特征保留梯度，主 CE 仍正常训练默认头和增益。首先设置原始/同 RX 重组/跨 RX 重组三个匹配对照，检验是否迫使 PA 学到可跨接收机使用的身份证据。它不是同包硬件反事实，不假设不同采集包同底层波形。先仅用 L，源学习信号和固定比较通过后才考虑原已接受 U 与 L donor；不新增全 U 强制伪类门控。

第三项是同参照 RX 的有停止条件语义间隔，达到 margin 后停止该约束，并与 SupCon 控制比较。第四项分支更新约束仅在已有有用辅助信号时推进。当前尚未实施/发布这两项；不因关系蒸馏失败而只增大温度、权重或扩充判定链。

唯一独立 P0/P1 审查 PASS，无直接阻断问题。当前未发布、无本轮性能结论；首步真实 checkpoint smoke 随发布立即执行并在 PASS 后启动队列。
