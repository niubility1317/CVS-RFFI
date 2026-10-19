# 首次时间池化的信息诊断

状态：PLANNED。上一目标轮为progress：六行联合W实验训练/七景独立评分与完整重计完成，源规则全部FAIL；最终性能目标仍未达到。

本轮只做合法源L的不拟合参数诊断。共同B0 FINAL20及既定FULL_As_JOINT FINAL30均按原严格来源加载；后者依据已固定源比较作为全量适应参照，不依据正式成绩选择。二者不续训，不替换或拟合分类头。U始终隐藏且不读取，V不参与本诊断，目标IQ/truth不可访问，不增加正式预测。

已核实路径：Sinc/HF/非线性basis→time_fuse(Conv/GN/ReLU)→time_down(AvgPool2)→t1→t2→SIG_TIME。当前t1/t2适配器在首次平均池化之后。令偶/奇时间点为g0/g1，m=(g0+g1)/2、d=(g0−g1)/2，则g0=m+d、g1=m−d，t1仅接收m，不能恢复一般d。该代数事实不证明必要身份信息已丢失，也不证明raw旁路会有效。

固定同一既有R02 bank的L1800诊断物理样本，读取raw、同L新mid/low各2draw，共9000观察/模型；2模型共18000NN，CPU2、FP32、无反向、0更新、0新信道模拟。1800来自6TX×5源RX×3day×20；实际模型seed仍为1。20个L样本在每TX/RX/day内按固定诊断seed20261019821随机分为两半，各10，所有模型/观测共享划分。

每包保存无参数描述子phi(v)=[mean_t(v),mean_t(v²),mean_t(abs(v))]。比较v=m与v=d；相同描述子维数3×48=144，不拟合分类器。每个RX/day内，对每TX分别计算两半均值，中心化后取跨半内积作为TX条件可复现变化S；除以同RX/day全包描述子的总离散量T。S可为负，不截零。汇总为sum(S)/sum(T)，同时保留15个RX/day分层及固定默认头源准确率。该比率是描述子散布分析，不是识别率、唯一信息量、互信息或物理上限；两半、信道draw和RX/day不充当模型seed。

事前解释：若两个low draw在B0及FULL上均有正的被消去描述子跨半TX变化，可支持“首次池化消去了可复现的TX相关时间差异”的局部假设，仍不能证明这些差异是其他分支没有的必要信息；只能作为后续raw IQ补充通路因果比较的动机。若不成立，保留负结果，不因诊断不显著变更描述子、split或挑draw。不得依据任何正式分数改变本分析或训练计划。

一行CPU诊断，唯一owner codex/root/F-prepool-source。不可变输出PROJECT/runs/20261019-phase1-F-prepool-source-s2026092701-r01；发布后首个真实模型前向即验证输入/层位/池化重构并直接继续，无额外smoke token。技术停止仅限来源/输入/shape/数值/输出冲突，保留产物，不影响健康任务，不自动重试。预计产物：config、source-only results、无参数描述子、NN/数据角色/模型来源记录及完成或失败。

| ID | 来源 | 要求 | 目标 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|---|
| P01 | 目标三/七、附件九 | 仅合法源L/来源严格/无头拟合 | runner | pending | 既有own_model及source bank | 不用目标 |
| P02 | 附件三/七 | 实际首次pool前后与可逆分解 | probe | pending | 真模型hook/reconstruction | 不宣称必要信息丢失 |
| P03 | 目标七 | 无参数散布、固定两半、逐RX/day | statistics | verified | 解析小例与负测 | 非分类器 |
| P04 | 目标二/八 | 一行CPU/固定18000NN/无新模拟 | runner/publish | pending | 实际计数/后态 | 无新模型和正式测试 |
| P05 | 最小流程 | 一次review/commit/单归档/后态 | publish/report | pending | 实际证据 | 不增加额外gate |
| P06 | 目标一/八 | 完整结果/负结果/Git/边界 | report | pending | 完整读回 | 总体目标保持未完成 |

本地解析小例、常量零散布、负散布保留、四项负测及真实pool模块hook检查PASS；0完整模型NN/0真实CP/0IQ/0target。所有模块及发布内嵌脚本语法通过。
