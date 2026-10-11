# E20 独立种子教师源域诊断：独立审查

- 日期：2026-10-12。
- 审查者：`/root/stage2_release_review`。
- 结论：**PASS，剩余 P0/P1 为 0**。
- 范围：本包 design/run/audit/checks、DESIGN、预登记及既有源输入读回与本地检查证据。未读取本批实际统计结果或目标产物。

## 已关闭问题

原 `run.analyze` 只拒绝已有 completion；若 analysis 已写成而 completion 写入失败，重入可能覆盖部分产物。当前同时拒绝已有 `analysis.json` 和 `completion.json`，并在计算前以独占创建的 `analysis_started.json` 阻止并发或失败后的盲目重入。原始文件及部分产物保留，本项关闭。

## 审查结果

- **来源与权限：** 固定三个原生 E20 的既有 L/V parent 读出，来源为已经完成的 source-only error audit。已读回的 metadata 确认逐 seed 原 checkpoint 路径、VERIFIED/EXACT_MATCH、scratch 无祖先、20/4440、无训练 target 接触及源角色数量。读取真实 source contract 的 L_s/V PID 集合与预测核对。E20 是用户指定且已暴露的开发基线，不能宣称全新盲测。
- **输入边界：** 反序列化采用 `weights_only=True`；文件本身包含 parent 和历史源终点，但统计入口只取得 L/V 的 parent，不消费 endpoints。未打开 U、IQ、模型 checkpoint、目标预测/真值/成绩；没有新模型前向、梯度、教师拟合、参数更新或 GPU 进程。远端入口只有源文件读回及下载。
- **物理配对：** 每个角色要求非空唯一 PID、六类有限归一概率及合法整数 metadata。跨 seed 严格比较 PID 顺序、标签、RX、day；L/V PID 不相交。输入样本数为每 seed L6300、V27000，复用共 99,900 条源样本模型读出，不把组合统计当作新独立样本。
- **教师定义：** 每个学生的 LOO2 只使用另两个 seed 的均匀概率均值；ALL3 为三个固定模型的均匀概率均值，显式计为一个教师。未选择种子、拟合权重/阈值或进行候选排名。
- **门控与分母：** 阈值固定 0.95；一致门控同时要求两位独立 peer 的 argmax 相同且各自置信度达标。纠正、破坏、持续错误、学生高置信错误修复、教师改判准确率与 selection risk 使用对应子集。coverage 明示采用当前角色/分层总样本数，cohort 数量另外报告；空子集 count 为 0，相关比例为 null。
- **校准和上限：** ECE 使用 15 个固定等宽置信分箱，最后一箱包含 1；NLL 的 `1e-12` 概率下限明确记录，无校准拟合。错误交集、条件错误率和至少一个模型正确的 oracle 按固定三模型逐样本计算，oracle 显式标为不可部署的诊断上限。
- **输出保护：** 下载目录独占创建，已存在目录拒绝复用覆盖；下载完成先核对文件字节数，分析前检查完成标记和源角色 PID。诊断固定 CPU 执行，三个模型共同构成一个确定性分析任务，与登记的一行矩阵一致。

## 验证与边界

已读取 `local_artifacts/e20_peer_audit_core_checks.json` 的 PASS 证据：固定三 seed 配对、LOO 排除学生、ALL3 单教师、手算纠错/破坏/高置信错误、双 peer 门控及改判风险、错误交集/oracle、ECE15/零概率 NLL、空集 null、输入和 RNG 不变、poison endpoints 未访问，以及 13 项无效输入拒绝检查。

审查者另执行源码 AST、登记与代码 config 精确比较，核对本 run `evidence/source_input_readback.json` 中三个实际源来源及新增检查所需字段；均通过。未重复运行已通过的核心检查，未访问远端、下载或打开实际预测张量、读取目标分数、启动诊断或修改实现。仅撰写本文件，不提交 Git。

本结论表示当前离线源统计实现通过直接正确性审查，不表示诊断已经完成。L/V 均来自原模型见过的源 RX；结果不能证明 U 真值质量、未知 RX 泛化、训练收益或 84% 目标达成。
