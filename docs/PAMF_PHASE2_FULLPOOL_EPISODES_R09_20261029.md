# PAMF Phase2 同池重采样与逐套星地叠加 r09

2026-10-29。按用户最新更正执行：所有套数复用同一实际目标样本池；每套每 RX/TX 随机抽 K 条 support，其余全部作为 query；下一套重新抽 support、重新叠加星地信道。r08 的跨套物理互斥、固定小 query、历史 compact 排除及容量选类规则被本版替代。源域数据角色与既有模型身份保持。

## 1. 每个任务的数据规则

设某个目标 RX/TX 四日实际记录数为 M。第 e 套先对每条原始记录重新叠加一次 practical residual，得到本套固定 received；再以独立随机排列抽取 K 条 support，query 为该类全部剩余 M−K 条。support 抽样与信道无关，不按场景、SNR、质量或模型表现挑选。

例如 M＝5,000，K＝1/5/10/20 时，每类 query 分别为 4,999/4,995/4,990/4,980。每个任务内部 support 与 query 互斥；跨套可重复使用原记录，也可从 query 变为 support。每套都对完整池重新叠加信道，不能只重抽 support 而沿用前套 received。

同一套内四个 K 使用嵌套 support，且共用同一 received，以配对比较标签数。K 较小时的 query 可以在 K 较大时成为 support，因此四个 K 的 query 数不同；不同 K 的模型和目标状态必须独立初始化。跨方法、模型 seed、P0/P1 使用相同套号及相同 support/query；模型 seed 不额外重抽数据。

每个 episode/K/model seed 从相同匹配的冻结源模型开始，重置目标优化器、BN、记忆、原型及适应状态。只允许本任务内部 B→C 继承；不同套和不同 K 不串接目标状态。MILKR 的 B 与 C 仍按各阶段当前 support 独立重拟合，不把它描述为继承 B 核参数。

## 2. 实际容量与类别

直接使用已验证的 Full eq 原始前 256 点，不裁出伪增量窗口，不追加均衡器。float64 N×256×2 转为 float32 N×2×256，随后沿原 practical 实现逐样本单位 RMS 和 residual/post_sync/no additional EQ。源 RX 与目标 RX 不重叠。

目标池为原六旧类＋原 MILKR 二十新类、7 个目标 RX、4 个日期的全部记录，共 **554,823 条不同物理记录**。不再因互斥配额不足而另选新类。每个 RX/TX 的 M 为 590 至 6,262；即使 K＝20，最小 query 仍为 570。旧六类的最小 M 为 4,404。

|目标 RX|26 类四日记录数|
|---|---:|
|1-1|79,805|
|14-7|79,075|
|2-1|79,304|
|20-1|79,189|
|7-14|80,202|
|7-7|78,215|
|8-8|79,033|
|合计|554,823|

|K|N_new＝20 时七 RX support 总量|同一套七 RX query 总量|
|---:|---:|---:|
|1|182|554,641|
|5|910|553,913|
|10|1,820|553,003|
|20|3,640|551,183|

K 是每个 RX/TX 的标签数，不是七 RX 合计的标签数。每个 RX 是独立域适应任务，不合并七 RX 的 support。N_new 为 0/2/5/10/20，到达顺序恢复为原 MILKR 二十类顺序；旧类与已到达新类的 support 在 N 增长时保持一致，query 仅加入已注册类的补集，未来类别不进入该阶段。

旧类顺序：14-10、14-7、20-15、20-19、6-15、8-20。

新类顺序：11-4、3-13、12-20、5-20、20-12、17-11、14-14、2-17、4-1、19-13、19-8、2-19、8-1、14-20、13-3、1-18、20-7、7-7、2-3、18-12。

源侧维持 5 RX、D1 至 D3、L_s＝6,300、U_s＝56,700（TX 隐藏）、V＝27,000。日期为 2021-03-01/08/15/23；D0 是更早的未见日期，不写成未来时间外推。Full 扩容不改变源训练，不自动重训 Phase1。源 checkpoint 继承仍须满足原数据契约；本次更正不授权其他研究读取目标 query。

## 3. 套数、信道与存储预算

保留 23 个数据任务单元，把原纯场景的 support 重复也改为每套重新生成 received。

|用途|套号|套数|每套原池记录|累计重新叠加观测|
|---|---|---:|---:|---:|
|综合主实验|MIX01 至 MIX08|8|554,823|4,438,584|
|low suburban|LS01 至 LS05|5|554,823|2,774,115|
|mid urban|MU01 至 MU05|5|554,823|2,774,115|
|low urban|LU01 至 LU05|5|554,823|2,774,115|
|合计||23|同一原池|12,760,929|

使用已固定的六分量 G10 综合模型；本次只改变样本复用规则，不重新决定场景概率。综合六景每条记录独立按固定概率抽样，再抽几何、衰落、噪声；RX/day 的 receiver seed＝2027 保持。pure 三种场景按原生 low suburban、mid urban、low urban 的 practical residual 参数运行。跨套 scene/channel seed 不同，同套跨 K 相同。概率是已声明的覆盖几何参考，不称为实测卫星任务占比。

23 套×7 RX＝161 个 received bank；每 bank 有 4 K×5 N＝20 个 split，共 3,220 个 split。received 仅保存一次，约 26.1 GB（十进制），query 用索引补集读取，不为每个 K/N/方法复制 IQ。原始 ZIP 复用服务器已上传且 CRC 验证的 20 个文件，共 34,378,471,511 字节。数据生成使用最多 8 个 CPU worker，每 worker 的数值库线程数为 1，CUDA 不可见，不占 GPU。

这些是 554,823 条原始记录上的 23 次随机信道与 support 任务；12,760,929 条观测不能称为同等数量的独立物理样本。后续计算预算必须按大幅增加的 query 量计入预测成本，不能直接沿用 r08 小 query 的耗时估计。此作业不启动任何 B/C 拟合。

## 4. A/B/C 与 DURA 接口

每个 bank 保存打乱的 received 和匿名观测 ID；当前阶段的索引先选出已注册类别，再排除当前 K support。A 只读该 K 对应的 query，不打开 support 标签或 support IQ。B 只读旧六类实际 K support 及其补集 query。C 只读当前 6＋N 类实际 K support 及其补集 query。完整 bank 不能直接送入模型或提前导出未来类别特征。

真实 TX/day/原始记录位置和信道信息位于 capsule 外的 builder_only/score_only，query 真值不进入方法读取器。本轮 builder 离线使用真值组织合法任务，但不加载 F/G，不拟合、不预测、不评分。后续身份代码必须使用本版分阶段读取器，不能复用全 bank 读取的旧导出入口。

P0 是本套固定 received；P1 必须由同一 received 经冻结 DURA v1.0 得到，不重新抽信道，不扩大 support，也不读取 query 标签。此次先制作 P0 和任务索引，DURA 缓存尚未制作；原生方法、组合扩展及 B/C 状态衔接沿已登记的总实验表定义，不因本次数据更正自动增加算法行。

## 5. 评价与结果口径

由于 query 是各类全量补集，各 TX/RX/day 数量不完全相同。主表采用每类等权准确率、Macro-F1，以及旧类/新类 macro 准确率的调和均值；总体 micro 准确率及实际分母同时报告。不同方法必须在同一 split 上配对比较。可从冻结预测计算跨 K 公共 query 的补充结果，但主结果始终使用各 K 的完整补集。

先在相同模型 seed/RX/episode 上求配对差，再区分模型 seed 变异与给定原池下的信道/support 变异。23 套不是 23 个独立物理数据集，不能将 episode 与 model seed 相乘后当成同等独立样本计算置信度。跨任务记录复用不授权把已有 query 分数反馈到阈值、带宽或超参数。

## 6. 本次执行与验收

run：`20261029-phase2-fullpool-episodes-s2026102901-r01`。配置：`configs/pamf_phase2_fullpool_episodes_r09_20261029.json`。最小登记及命令见同 run 的 `experiment.json/report.md`，实际代码 commit 由发布器从已推送 Git HEAD 写入远端 runtime 和提交记录。

服务器 CWD 为同 run 的独占 release；入口为 `python -u -m experiments.pamf_phase2_practical.episodic_runner --runtime runtime.json`。先建立七 RX 原始池，再受控并行生成 161 bank。预期输出 pool manifest、每 bank received/IDs/stage_rows/actual-K split/标签文件、capsule manifest、独立 score sidecar、catalog、builder_report。

任一输入数量/格式变化、池容量不足、当前任务角色重叠、输出碰撞、非有限 received 或 worker 失败即停止本 controller 所拥有的子进程，保留 partial，不修改其他 owner 的任务，不自动改配方覆盖重跑。5 项聚焦检查已通过，包含全补集、跨 K 角色变化、跨套重采样、A 不读标签/未来类别，以及真实 practical 算子的跨套不同信道。发布后以实际 PID/CWD/argv、日志增长和产物独立回读核实启动；数据全部完成须有最终 builder_report/catalog，提交 PID 不能代表完成。

r08 已成功完成 539,448 条物理记录的旧版数据构建，保留全部输出；其互斥分配与重选类别因用户本次更正而被替代，不作为本版输入，不记为技术失败。历史结果保持原定义，不改标为 r09。

## 7. 当前执行状态

已在 N607 启动本版 CPU 数据制作，执行 commit `42e6dca573508f511d2f33861ac7dbd58eaeee45`。归档传输与远端编译 VERIFIED，controller/worker 的实际路径、命令和日志已独立回读。当前 RUNNING；最终 161 bank 和 catalog/builder_report 尚未完成。身份实验继续暂停。
