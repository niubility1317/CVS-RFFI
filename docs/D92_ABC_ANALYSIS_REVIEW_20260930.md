# D92三阶段描述性分析的实现与验证

日期：2026-09-30。用途是补齐用户已明确的旧类适应、注册后旧类下降、新旧类差距口径，不修改方法、训练、预测、原scorer、实验状态或运行门槛。[10／1／3个百分点是理想目标](D92_ADAPTATION_REGISTRATION_TARGET_20260930.md)，允许在现有方法上逐步改善。

## 实现

- [身份投影器](../tools/collect_d92_adaptation_registration_identities.py)：先核实两个已评分run的实际启动配置与完成记录；只从既有capsule读取物理ID和split元数据，从冻结DG预测投影身份字段，核对模型、checkpoint及来源。远端只读，输出到新的本地目录。解析原预测JSONL整行会接触其中数值字段，但立即丢弃；不计算、导出或向方法提供分数，也不访问truth、IQ成员、特征值、源样本或checkpoint权重。
- [物理配对模块](../tools/d92_adaptation_registration_pairing.py)：纯函数按物理ID与support标签对应的实际类名校验配对；支持顺序变化，忽略DG记录偶然附带的K/support seed。A为冻结DG，B为当前方法的新增0类拟合，C为当前方法当前完整注册表拟合。它验证身份关系，不声称C继承B状态。
- [三阶段汇总器](../tools/summarize_d92_adaptation_registration.py)：完整来源和物理配对检查后才读取既有scores；输出A_old、B_old、C_old、C_new、B−A、B−C、平均逐单元绝对新旧差与H。两cohort按实际任务数3:1合并；H和绝对差先逐单元计算再平均。保留完整K×新增类数、各K、模型与RX/scene分层。理想状态仅描述，不决定启动、晋级、淘汰、选择或重跑。

新旧都低但差距小不称为高准确率；B−A包含启用support及替换分类头的收益，不单独归因于原分类头的参数更新。现有B/C从头独立拟合的机制不构成违规，也不包装成持续学习状态继承。

## 检查证据

主任务在已验证的 `E:/type10-7/local_envs/ssr-gpu` 中串行运行：

| 测试模块 | 通过数 | 工具完成证据 |
|---|---:|---|
| `test_d92_adaptation_registration_pairing.py` | 28 | exec chunk 357485，exit 0 |
| `test_collect_d92_adaptation_registration_identities.py` | 12 | exec chunk 15ab84，exit 0 |
| `test_summarize_d92_adaptation_registration.py` | 36 | exec session 42172，最终exit 0 |

合计76项。覆盖乱序、类列重排与标签映射、DG无关K/seed、缺失/重复/错误物理身份、来源与完整矩阵、两阶段读取顺序、数值口径、全4800单元与输出拒绝覆盖。没有为报告测试执行任何训练或真实query预测。

branch_local_core独立只读审查了非其编写的物理配对模块和身份投影器，未发现未解决P0/P1；主任务审查了local_env_reconcile编写的汇总器，未发现未解决P0/P1。开发与独立审查未读取实际query分数；实际旧基准分析由主任务单独执行，结果不回流方法工作者。

本记录证明工具正确性检查已完成，不证明任何方法达到理想性能。实际身份提取和结果报告以各自输出为准。现有LocalMargin support运行及其冻结runtime保持不变。
