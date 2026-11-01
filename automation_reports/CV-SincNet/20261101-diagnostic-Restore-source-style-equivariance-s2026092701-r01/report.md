# 冻结恢复器源域成对变换稳定性

- run_id：`20261101-diagnostic-Restore-source-style-equivariance-s2026092701-r01`
- group_id：`restoration-source-paired-transform-stability`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原Ls128Us384×20draw×5固定变换，冻结DURA/PHY/FullFGrad，比较G(Ty)与T G(y)对Tclean的保真；0更新/TX/V/OE/target，不作native或Main因果声明。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 固定源端成对变换诊断

原Ls128/Us384固定512包×20现成R07 draw，五种固定T为identity、+90°相位、0.7/1.4增益、反射单lag FIR（beta=.25exp(jπ/3)，除sqrt(1+.25²)）。冻结默认DURA、PHY/8198及FullFGrad/940和原Tri-SIG-TIME SSL/3ff8/defaulthead。比较实际G(Ty)与参考T G(y)，两者都对T clean计算IQ、F0特征和判决一致率；另报F0(Tclean)相对F0(clean)的变化。全部五T/三G/两路径/各源场景保留，identity共享输出，0更新/TX/V/OE/novel/目标/统计拟合。source_validation仅复用严格模型loader/配置映射，不调用V worker或读V。

这是合成源端稳定性检查，不是native信道重模拟，不证明TX身份保持、Main因果解释或信息恢复极限。FIR一般不与原H交换；若先验分布不具变换对称性，非等变性本身也不是Bayes最优性反例。不根据head判决选择T或训练候选。F0原源尺度只读，变换前后参考分别计算，避免混淆恢复器与F0敏感性。

153600条G请求行；F0读出276480+clean参考2560=279040条，Ls2真实smoke另6条F0行。按请求行数计，内部三视图/图捕获工作另在worker耗时内。第一神经工作为实际Ls2 checkpoint smoke，PASS立即全矩阵。原N607 CVS-RFFI Torch2.1+cu121、FP32/TF32off；独占CWD/路径/commit/命令/停止规则见登记，GPU4/6 UUID余量>=12GB/原compute<=2/只新增1个本任务NN/单卡总<=3，其他owner不改。4针对性检查和一次独立P0/P1通过；实际源smoke和结果待运行，完整Main性能目标及默认DURA保持。

## 完整执行与独立复算

VERIFIED / ANALYZED。Ls2实际checkpoint smoke及512包×20draw×5T/三G/两路径全部完成，0更新/TX/V/OE/目标。独立汇总差0、所有判决/配对与完整源面板核对，identity差异精确0、phase/gain参考IQ不变性误差≤2.39e-7；全状态保持、所有进程退出。T改变clean先验/F0表现，不能直接据等变差异推出模型缺陷、Main因果解释或硬等变改造收益。详见analysis.md。
