# 三种子探索的模型原生掩码补齐

状态：ANALYZED_SOURCE_ONLY。用户明确探索要三个model seed，当前四协方差方法完整固定为01/02/03各一份。01模型/掩码/完成预测复用，02/03需各补四模型；既有PL/Sup控制均复用，不重训。本run只补前置mask统计，不称模型训练或三seed正式结果已完成。

每个NN的特征通道坐标不同，因此02/03分别严格加载自己m201 scratch SIG1 ordinaryPL E20/4440，在全L6300 raw/三H上拟合原相同自然支持集：敏感度top25%、RX/day条件TX关联bottom50%、按比值排序至多10%；随机maskseed2026102623、同尺寸/full控制不变。01的94/294项不能照搬。mask记录modelseed/sourceprefix/contract；无合格项则记录该seed方法未定义，不能回填或替换seed。

全部原物理角色及sourceIQ一致，L-only统计的父E20仍含原13L7LU，不能称纯监督父模型。此处不读U/V/target，不加载DURA权重/恢复IQ。只复用冻结星地信道数学代码来保持与原fit一致的H，namespace/物理seed/hardwareseed不变，不新增恢复器依赖。

02/03各实际E20/sourceL2/defaulthead smoke PASS即完整50批6300L及18900H，检查F状态不变。01只读取原mask元数据复用，0新F前向。每卡一条ownworker，只用全局空闲卡、free≥10000MiB；最多2fit，其他owner不操作。两个聚焦seed/source负测试PASS；唯一针对seed基底绑定的P0/P1检查PASS，不重复审查旧目标/报告。

唯一矩阵、执行命令、路径、环境/CWD、预算、GPU与停止规则见experiment.json。三seed完整学习/正式评价的用户最新要求见[当前计划](../../../docs/goals/PAMF_SIG1_THREE_SEED_FONLY_20261027.md)，不能再依据seed01一行目标分数决定是否补另两个seed，也不允许用源分数过滤差seed。

## 实际补齐结果

执行commit `e42fbfc30764087daae28303c5e39194d170b490`（包含本诊断代码），单archive比较VERIFIED、远端compilePASS。两个对应真实E20/sourceL2/defaulthead smoke均PASS；各完整50批L6300/三H18900，末批28，全部100条结构化记录和stdout已解析，0致命标记。模型参数/注册buffer未变。01掩码仅复用，没有新模型前向或训练。

| model seed | time_down 选择项 | t2.pool 选择项 | 新拟合耗时 |
| --- | ---: | ---: | ---: |
| 2026092701 | 94 | 294 | 复用 |
| 2026092702 | 112 | 282 | 34.423s |
| 2026092703 | 91 | 362 | 34.712s |

相同选择规则给出不同数量，不能复制01的通道坐标。随机控制逐seed匹配实际数量，full仍1128/4560项。三份mask均已有合法原生E20基底支持。首次独立观察时两个fit及controller已退出；完整终态/源码PID-CWD记录/结果已读回，未捕获存活快照，没有为此重跑。stage完成不等于三seed模型训练或正式性能完成。

接下来按固定四方法×三seed补齐8模型，分别固定9批训练L/U梯度共同系数；01alpha1/权重/完成读出复用。所有训练完成行自动rawclean+六景，不以source分数过滤seed，不用DURA。[实际掩码汇总](summary.json)。
