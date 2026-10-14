# 可信门控内有界权重：源代理结果

源代理已完成，统计和保存梯度独立复算 VERIFIED。原门控接受 867/6300 包（13.7619%），867 包均正确；uniform 与 weighted precision 均为 100%。这次没有精度提升空间，不能证明权重改善伪标签正确率。加权确实增强同一时域参数对象的伪标签梯度，但未开展此机制的真实训练或目标测试。

Run 20261014-phase1-source-trusted-weight-manysig-s01-m1-r01；实现 051de1a177819c0f8a9cfe5aba0fd09e7cefef6d；实际执行 c84b17396bc0861b8856e867d17aab54206436c5。固定原 moment-area SIG_TIME seed2026092701/final50/11100及其ownEMA，来源执行512c10a267b5e654cfd2c983ab1debdeebc7f626，加载前核对source roles/physicalIDs/scratch无继承污染、默认头额外fit关闭，无目标训练/选模。最低预定义seed作为与前两源诊断相同的固定anchor，未按目标成绩选权重。

原teacher confidence＋physical batch_neighbor window2＋studentstrong一致门控不变。h=1−p_student,strong(teacher伪类别)；raw w=stopgrad(.75+.5×clip(h/.1,0,1))；lambdaU=.16，按已接受样本权重和归一化CE。raw界[.75,1.25]，effective归一化权重理论界[.6,1.6667]，二者不能混称。scale/bounds在观测前固定，无扫描。L标签只用于原合法physical metadata、clean监督CE及形成mask/weights后的精度统计，不输入新难度计算。

完整L6300固定shuffle2026101411/Dropoutseed2026101412，一次25batch（24×256＋156）；teacherweak/studentclean/studentstrong共18900包前向。源观测耗时14.946秒（不含发布/加载前检查）。CPU2线程/CUDA为空；无U IQ/真值、V forward或目标访问，既有Vcache materialize后discard。student.train仅回放原Dropout/strongnoise.015，无identity BN；原student/EMA所有state逐值保持，optimizer/head更新0，额外分类头拟合0。

| 梯度对象 | 元素数 | Uniform λU/L 均值 | Weighted λU/L 均值 | 两种U梯度余弦均值 |
|---|---:|---:|---:|---:|
| t1.pw.weight | 3456 | 0.243982 | 0.356582 | 0.996384 |
| t2.pw.weight | 6912 | 0.240009 | 0.348997 | 0.995777 |
| t3.pw.weight | 9216 | 0.227731 | 0.320118 | 0.994168 |

表中为25batch逐批比值/余弦的算术均值，不是合并全数据梯度，也不是实际epoch33/34/50训练轨迹。每层只比较同一参数对象，参数量不同，不以跨层norm排名解释重要性。75层批记录及15个overall/RX/day/source-class统计完整保存在results.json，保留accepted/hits/有效权重分母。15项有接受样本的精度均100%；接受样本平均h=0.01461584，观察raw范围[0.75010610,1.25]。

L每physical cell70包，实际U每cell630包，邻包可见率不同；同一Lproxy用于监督和伪标签，与真实L128/U256分离采样不同。13.7619%覆盖和100%精度只属于本源代理，不能外推隐藏U。梯度方向基本保持，主要观测到贡献增大；不能据此断言改善身份信息、新TX迁移或目标性能，84%目标尚未达成。

本地stopgrad/原mask/归一化梯度/空accept/11配置负例/真实native2zeroIQ PASS，一次独立直接P0/P1 PASS无发现。一次release archive本地/remote SHA比较及remotecompile成功；真实checkpoint无查询2zeroIQ smoke PASS。独立readback核对PID2248792/CWD/argv正确，日志0→212bytes，completion字段精确匹配；私有source_weight.pt重读复算15统计及75梯度记录VERIFIED，无新增IQ forward。

运行入口 python -u -m experiments.cvs_source_weight.probe；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_weight_20261014_r01；保存产物 /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-trusted-weight-manysig-s01-m1-r01/SOURCE_WEIGHT-s2026092701。原IQ、checkpoint、梯度数组与大日志不入Git。原发布第一次因未同步HEAD在本地包装前停止，无远端写入；同步提交并核对remoteOID后发布一次，未重复源forward。完整固定方案见preregistration.md。

后续真实训练若落实，保留旧门控、默认头正常联合训练、scratch总计50轮、33＋17，预先固定实际33/34/50的L/U特征参数梯度记录。当前源代理不能替代这一验证；8个健康层级实验保持原方案。
