# D67交叉拟合registry-consistent连续行堆叠追踪

## 历史动机

D62是当前联合最强开发基座，但after82.22%、min-after53.33%和forget10.56pp不足。D65把after提升到86.11%、min-after提升到70%、forget降到6.11pp，却把new压到59.33%。D64说明registry扩图重写旧几何会扩大遗忘；D66说明ground int8共享可靠性只能带来很小的旧类保护并交换新类floor。第四轮不继续pair图、冻结强度、ground尺度或角色offset，而检验两个合法仿射专家的类别身份无关连续堆叠。

## 数学锁

对D62和D65的每个匿名类行，仅从train support估计正/负score中心与尺度，将两专家行编译为标准化仿射`z62_c,z65_c`。四折physical-rank cross-fit中，每折held两个rank且不参与专家拟合、中心或尺度。正/负标签采用各0.5总权重，闭式求解：

```text
d_i = z65_i - z62_i
alpha_c = clip(Sum w_i d_i (t_i-z62_i) / Sum w_i d_i^2, 0, 1)
h_c = (1-alpha_c) z62_c + alpha_c z65_c
g_out,c = center62_c + scale62_c * h_c
```

`t=+1/-1`；分母退化时`alpha=0`，映回后严格恢复D62原始行尺度。所有类、before/final、old/new都使用同一公式；只知道Stage2-B已存在类数以合法构造D65的冻结协方差生命周期，不在query读取角色。full support重算标准化后把所有行中心化并编译成单一FP32→int8/FP16仿射状态。

## 非重复与边界

- 不同于D62的离散Pareto行替换：D67没有TP/FP阈值或atomic gate，主机制是连续解析权重。
- 不同于D65：D67不会让全部类强制使用旧类covariance；D65只是每个类可选的同构专家之一。
- 不同于D26/D27/D36：没有new-group bias、按角色offset、IRLS或learned query gate。
- 不同于D66/D19/D25：本轮不读取ground组件作为拟合输入；D66的负增量已独立保留，不能为迎合机制叙述而强行加入。
- 不访问outer-held/query、clean/source、role Oracle、真实batch类数、quota或global assignment；不设置class/scene/receiver特例。

## 预注册实验与判门

development cell固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold，实际K8。复用D18`VALIDATED_ONCE/p2_min_v1` enrollment-only support。105行矩阵必须完成，并报告七候选、三场景、11类、15fold、alpha分布、专家标准化/风险、量化、资源和协议闭包。

相对D62必须在总体、场景、floor、遗忘、混淆上无交换伤害并严格改善至少一个after/forget/joint/floor指标；量化变化和margin flip必须为0。失败停止整条连续D62/D65堆叠路线，不扫描fold数、alpha温度、ridge或阈值；成功也先第二development seed，不直接125。

## 实现状态

已新增纯数学core、独立probe和两组测试。D67专项9/9、D42–D67完整链313/313通过；主工作树验证用时78.6s。实现保持四折held/train交集0、每个support row exact-once、闭式`alpha∈[0,1]`、K≤4 exact D62 fallback和单一affine query状态。真实105行尚未执行，不能从测试推断性能。

## 首次真实运行与PostRun-R1

真实runner已完成105/105行并写出完整基础artifact，随后在metadata前因预估fit记录30、实测60而退出。原因是INT8与matched FP32目标路径各自执行before/final；每次D67 fit含92个nested D62 component记录，所以真实闭包应为60/5,520。原执行脚本SHA为`5a6baa86b29f44b7553c4d81cd898ee152ce2b52688fab09bddd131700a97872`。

对既有输出调用原脚本只读verifier已通过105行、30条目标candidate row、60个fit audit、240个cross-fit partition和query0；alpha范围0～0.216726、均值0.025459。PostRun-R1只修计数并增加既有输出封存模式，不改变公式、support、预测、资源或已有artifact，也不重新运行实验。完整性能在封存后另行解析。

首次封存尝试因当前主工作树与执行用干净worktree的换行字节SHA不同而在metadata前fail closed。封存器现只从`executed-probe-script`所属root读取D62、D65和D67 core并核对原candidate lock，不再错误使用当前工作树helper字节；SHA检查本身未放宽。

## 最终性能与路线关闭

PostRun-R1已从原执行root成功封存：105/105行、60个D67 fit、240个cross-fit partition、5,520个nested D62 component fit，query/clean/source/role/quota/global assignment均为0。完整摘要SHA为`6c8349aed767f2d468e953d9d4d195e86aa99cbef6ca4e93cdc372cf52b60592`。

D67 INT8总体B/A/N/H/F/J为92.78/82.78/83.33/82.16/10.00/26.67，min-B/A/N为80.00/53.33/73.33，混淆22/11/14。相对D62，A+0.56pp、F−0.56pp，但N−1.33pp、H−0.47pp、new→old+3，floor无净改善；不满足无交换判门。三场景A/N/H分别为clear91.67/96.00/93.57、low80.00/74.00/75.45、rain76.67/80.00/77.45。

final阶段支持内风险由D62的0.532406降至堆叠的0.524892，但D65专家风险为4.139319，`alpha`均值仅2.906%；支持代理改善没有迁移到outer联合目标。D67按预注册不读取ground int8组件；D65的低遗忘来自冻结旧决策几何，不能归因于地面原型。真实ground路线D66仍为负向，因此不把ground组件强行并入D67。

最终状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D62继续为联合最强基座，D62/D65连续行堆叠路线关闭；不扫描alpha/fold/温度/ridge/阈值，不运行第二seed或125。

最终在显式激活的`ssr-gpu`环境运行D42–D67完整测试链315/315通过，用时80.9s；pytest exit0后的Windows临时链接清理权限告警不属于测试失败。
