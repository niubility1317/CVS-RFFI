# 固定 SIG_TIME 接收机风险训练：02/03 条件准备

**当前状态：LOCAL_VERIFIED / PLANNED_SOURCE_CONDITIONAL / NOT_LAUNCHED。** 第一种子尚未满足完整源资格，当前不能发布扩展。实现 `8522fb75680c7b51c7d3d631f67c1b2b212eba8b`。

仅适配已登记的三臂 ERM、源RX等权、源RX动态风险加权，机制数值、字典、旧门控/ownEMA/cos、默认头正常联合训练方式和数据角色完全保持。两个固定modelseed2026092702/03，各三新scratch50/11100；另复用两个旧门控+cos历史基线。第一种子不重训，不择优挑臂或挑seed，不产生新科学假设。

发布前唯一源资格沿用首运行 `20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s01-m4-r01` 的 source_qualification.json：原固定规则必须PASS；源决定先保存并commit/push，再回收首run新numeric target摘要。源FAIL则本矩阵不启动，不调公式/判据/学习率或系数。适配器的preflight/dispatcher/train会读取原sourcePASS；这里没有新的参数扫描或目标选择。当前第一种子仍使用不可变执行 `977d4fb609436bac23621ef8e32c2017495acef7`，本地上下文提取不改远端归档或健康进程。

六新模型成本300模型轮次/66600原生更新，额外训练前向0；另2历史基线100轮/22200更新计入历史成本。与保留第一种子合并：9新450轮/99900更新＋3历史150轮/33300更新；12行、2016000默认头clean决策、216指标。剩余本矩阵8行1344000决策/144指标，不冒充完整三seed，不能单独完成84%目标。

每新模型仍33label＋17pseudo、L6300/U56700隐藏TX/V27000仅健康，原始物理ID/协议/split不改；正常默认cos头及joint160导出不变。每行固定final50后自动clean168000预测，全部声明预测固定后独立truth-last评分；剩余所有行完成，无目标反馈中途停止或重跑，不新增source筛选门。源观察保存 source_observations.json 并引用原firstsource资格，不期待不存在的新source_qualification.json。

本地检查PASS：12配置/资格负例，6个实际native上下文仅seed/run/candidate/output差异，三臂每个seed真实初态逐项一致，实际t1/t2/SIG及正常W联合梯度和原U loss对象返回。相同NaN的未启用诊断元数据按原有适配器对称比较，未修改实际参数。唯一新适配器P0/P1 PASS无发现；未读真实IQ/权重或远端第一源资格来做准备。

发布时普通N607既有CVS-RFFI/FP32，每worker CPU2线程；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_receiver_dro_seeds_20261015_r01`，命令 `python -u -m experiments.cvs_receiver_dro_seeds.dispatch`；owner `codex/root/receiver-dro-seeds-20261015`，不可复用输出根 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s0203-m8-r01`，日志 `<run>/logs/<row>.log`。共享GPU0..7按现行调度分配，所有owner每GPU最多3总实验进程且发布时至少12000MiB空闲，保护健康任务。实际release才执行一次普通预检、单归档SHA比较、远端编译及真实first SIG固定final50的2样本synthetic no-query smoke，PASS后直接启动；训练从零，不继承该CP。

停止规则及预期产物见experiment.json；仅run-owned技术失败停止，保留产物不自动重启。当前不创建远端输出、不占GPU。原始数据/CP/预测/PT/日志不入Git，必要聚合结果和实现上传并独立核实远端OID。
