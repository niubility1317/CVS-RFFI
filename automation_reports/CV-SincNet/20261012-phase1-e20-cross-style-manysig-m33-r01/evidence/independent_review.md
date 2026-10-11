# 运行与协议独立 P0/P1 审查

- 结论：**PASS，未解决 P0/P1 为 0**。
- 日期：2026-10-12（Asia/Hong_Kong）。
- 审查者：`stage2_release_review`。
- 独立审查范围：本包 `design/model/objective/source/multilevel/style/warmup/preflight/evaluate/dispatch/publish` 及教师来源、概率缓存、登记配置。
- **独立性边界：审查者实现了 `analyze_source.py` 和 `final_report.py`，这两文件不在本次独立审查结论范围内。** 它们的作者验证另述于末尾。
- 未修改被审运行代码，未访问远端、目标预测或成绩，未启动实验。

## 核对结果

1. 固定 11 臂各三种子，共 33 行，登记 config 与代码逐项一致。每行严格加载同种子原生纯 CE＋旧门控＋cos E20，再重置优化器追加 20 epoch／4440 次主更新，固定 final，累计 40 轮；总 146520 次主更新、660 个增量模型轮。CE 不创建 U 迭代、教师模型或辅助头。
2. 非 CE 方案均为另外两个原 E20 的固定概率均值、ALL soft KL，排除学生自身。原生来源、合法 U 物理 ID、隐藏真值及缓存概率校验沿用已审路径；checkpoint 保存学生及教师完整来源。预测只执行学生，不读取 U 概率缓存。V 为 eval/no_grad 健康验证，不选轮次或方案。
3. Fishr 每活跃四步使用独立 A/B 两批 L60，合计 999 步、119880 次 L 使用、1998 次前向；各批内 PID 唯一，允许批间重叠。分层有限总体修正与双批中心化内积一致，保留两侧身份特征梯度和负值。OLD_PAIR 使用相同两批的旧平方项均值。Fishr 中头参数 detach；辅助 CE 使用 detach 特征，仅训练头。
4. 五个 Fishr 臂才创建辅助头，预热 200 步。之后固定 12 对 L 校准、1440 次样本模型读出、24 次前向，身份与头均零更新，恢复状态、模式及 RNG。两层系数由初始旧双批梯度参照确定，训练中只乘预设 ramp，不依据当前批重新缩放；层和固定除以 2，未启用层为零。
5. L_REPLAY 与风格臂每个活跃主步骤额外计算 L 身份 CE；风格位于所声明真实浅/中层。供体同 L 真身份/day、跨 RX、不同物理 PID，批内重复 occurrence 合法。额外前向保留普通 dropout 并隔离 RNG；L_REPLAY 不安装 hook，与其他风格臂形成额外前向/监督对照。STYLE-only 无辅助头。U 风格分支实际关闭。
6. 共享容量锁包含现有 GPU 所有者及本队列预约，每 GPU 最多 3 个总进程。教师缓存完成后训练，逐行同 PID 串联训练、冻结、clean 预测和独立 truth-last 评分。每行 168000 条、整批 5544000 条，33 行及逐层分层计数由完整性检查闭合。既有输出拒绝覆盖，未知启动不自动重试。
7. run/release/output/owner 独立，预登记保留来源、固定比较、已暴露基准背景及禁止目标反馈。发布归档含新包及全部 55 项依赖，远端 preflight/dispatch 绑定新入口。

## 证据

- `e20_cross_fishr_core_cpu.json`、`e20_cross_fishr_core_cuda.json`：有限总体及有限差分核验、两特征路径梯度、头无 Fishr 梯度、辅助 CE 无身份梯度、同种子各臂相同校准、状态/模式/RNG 不变、旧双批精确平均、负值保留和无当批重标定通过。
- `e20_cross_style_core_cpu.json`、`e20_cross_style_core_cuda.json`：11 臂实际 native 前向/反向有限、早期纯 CE、调用数、来源缓存负测、CE/SSL 与既有对照等价、L 主身份梯度及路由通过。
- `e20_cross_style_integration.json`：33 行预算、严格父模型载入和本包 checkpoint 往返、错误来源拒绝通过。
- 审查者额外只读核对：登记 33 个 config 完全相等；修复后的 55 项运输路径均存在，preflight/dispatch 字符串绑定正确。

## 已关闭问题

关闭 1 项 P1：初版 `publish.configure()` 没有把 `experiments/cvs_e20_cross_style` 本身加入运输列表，远端新入口会缺失。实现者已显式加入；定点调用 configure 后核实新包和 55 项依赖均在列表并存在。

## 自有报告工具的作者验证

以下仅为作者验证，不属于上述独立审查：两报告脚本已适配 33 行和 L_REPLAY；33 config 的机制计数与 source 一致；11 臂初始校准正负检查、有符号统计、合成完整 33 行最终报告和 41 项配对（含交互）、SOURCE_TRAINED 后不读取 stdout 的边界检查通过。最终报告单独判断预登记主方案 JOINT 是否达到 84%，其他达标固定对照另列。

本结论不证明性能改善、84% 达成或远端实际运行完成。真实来源预检、训练、完整预测及评分由唯一发布 owner 的运行证据确认。
