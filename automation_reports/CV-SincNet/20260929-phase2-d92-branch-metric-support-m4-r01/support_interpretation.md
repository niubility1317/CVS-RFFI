# BranchMetric完整support诊断结论

**本候选未通过全部预设条件，不进入query重复基准，不替代已有BranchInteraction。**物理OOF支持“多条support下学习类内度量”，但单样本代理不支持全面提升。不能只引用通过的OOF部分而隐藏代理失败。

## 覆盖与类别数量

8个模型×cohort行全部完成，共4800个父任务；其中真K1的1200个任务仅做数值检查，3600个K5/10/20任务做完整三折物理OOF，另穷尽42000个内部单样本anchor。实际标准分解21600次、proxy分解42000次，总63600次。每臂proxy重复held出现7879200次，这不是独立样本量。

旧类固定6类，新增类数0/2/5/10/20，总注册类数6/8/11/16/26。父任务每个注册类均有K条合法support，总数为(6+新增类数)×K。各K×新增类数包含240个父任务：4模型×4接收机×3场景×5support draw。含新类平均对2/5/10/20四档等权，不含0新类任务；旧类单独任务完整保留在分层CSV。每个单元先算H，再平均，不能用平均旧/新准确率反算平均H。

OOF中的parent K=5/10/20分别对应每折实际train K=3或4、6或7、13或14。单样本代理每次实际train K=1，但数据仍来自相应父任务已经获准的support集合；不能改称正式K1任务，更不能让真实K1借用较大support池。每个父任务先平均其全部anchor，再平均父任务。

## 预设比较结果

下表为候选减去冻结interaction_ridge，单位为百分点，均为含新类任务的support指标。

| 诊断 | parent K | Δ旧类 | Δ新类 | ΔH |
|---|---:|---:|---:|---:|
| 物理OOF | 5 | −0.226 | +1.730 | +1.340 |
| 物理OOF | 10 | −0.059 | +1.918 | +1.440 |
| 物理OOF | 20 | +0.098 | +2.093 | +1.539 |
| 单样本代理 | 5 | −0.665 | +0.109 | −0.131 |
| 单样本代理 | 10 | −0.736 | +0.284 | −0.031 |
| 单样本代理 | 20 | −0.834 | +0.265 | −0.094 |

物理OOF在每个parent K相对ridge及kernel_ncm均满足H/新类提高和旧类退化不超过1个百分点。相对kernel_ncm的H增益依次为+4.418/+6.889/+8.819个百分点，这支持本诊断中类内度量确有额外贡献，而不只是把ridge换成最近中心。

单样本代理中candidate与kernel_ncm分数精确相等，符合R=0的数学退化；三个parent K的H差值全部为负，因而B条件全部失败。旧类均下降约0.67至0.83个百分点，新类小幅上升，不能称新旧类全面改善。差值是有限support上的描述，不代表统计显著性或正式query结论。下一项研究可以依据这些support证据分析单样本判决，但本版本不改参数、不选择性重跑，不将K分段拼接当作本次已验证方法。

各新增类规模的完整数据见[按parent K×新增类数的CSV](results/support_summary/by_parent_k_newcount.csv)；全部模型、RX×场景、support seed及old-only分层与两对照差值见[汇总报告](results/support_summary/report.md)和[机器可读汇总](results/support_summary/summary.json)。没有遗漏失败分层或未完成任务。

## 协议、成本与证据

本轮只读已经绑定的support原始特征缓存，不读query/truth/source样本，不读取或继承其他方法的适应头。固定Phase1，runtime不重新加载checkpoint或前向编码。设计者没有接收query成绩；本诊断本身无query输入，但既有研究已经存在透明重复基准，不能将整体研究宣称为从未接触测试信息。

新增source数据、地面统计和模型传输均为0B。诊断墙钟236.176秒，所有父任务累计probe调用898.251秒，最大进程RSS136491008B；这些是N607上包含对照和重复诊断的CPU成本，不是单次卫星部署成本。可部署numeric state最大3178464B是固定C26/K20布局，不包含Python对象、原cache或模型包；本轮诊断没有持久保存部署头。

runtime为7099da7ab85170034de90d41c2471c4572860207；分析代码为af8d1f6149f6e761a8cc315cd478ee629607853b。独立终态证据[evidence/readback_1790672161.json](evidence/readback_1790672161.json)确认8行完整、无live children；[分析执行记录](results/support_summary/analysis_execution.json)与下载汇总均VERIFIED。所有原始trace/log留在N607原run路径，不覆盖历史产物。独立新数据验证仍未完成。
