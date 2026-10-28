# PAMF第三阶段：跨接收机星地冻结开集拒识首轮

- run_id：`20261028-phase3-pamf-open-set-s2026092701-r01`
- group_id：`pamf-phase3-open-world`；类别：`cvs`；阶段：`Phase3`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定三视图PAMF，source L原型与V的99%已知保留分位数；五种拒识分数、2目标RX、旧6TX和严格未见2TX、三种历史固定practical，零训练/适应/注册。

## 数据、seed与模型来源

实际逐行矩阵、六类seed、路径和资源以experiment.json为准。当前三视图PAMF学生由同契约scratch E20→自身E30→自身E50产生，11100更新；旧TX为14-10、14-7、20-15、20-19、6-15、8-20，源RX[1,3,4,6,8]、D1–D3，L/U/V=6300/56700/27000。首轮固定model seed2026092701，不按target结果选seed。

本轮从既有VALIDATED_ONCE固定practical capsule的query角色中取目标RX[0,2]、旧6TX和严格未进入该F祖先的11-4/3-13。三景为practical_high、practical_mid、practical_low_urban，每RX×TX×scene原query30条，共1440条。只复制原received，不重叠信道，不把同一物理记录的多场景重复当独立样本。公共目录仅opaque IDs、IQ和无角色manifest；原ledger仅builder/scorer访问，predictor配置不含未知类表、truth或source cache。

这是历史曝光基准复用，不声称新盲测。发现/适应数据池暂未发布，历史query保持测试身份；本run不完成无标签适应、发现或注册。后续Phase3目标及分工见../../../docs/goals/PAMF_PHASE3_OPEN_WORLD_20261028.md。

## 规则与科学判定

source L_s特征只用于冻结类中心，全部source V clean分数只用于每种最终knownness分数的全局99%保留分位数。MSP、maxlogit、margin、energy(T=1)、cosine五种规则均在target前固定，接受时统一使用原default-head类别；不按目标TX/scene选择阈值，不读source U，不拟合新头，不加载DURA。source V阈值可能在目标域不适用，须如实记录。

主指标未知TX等权宏平均拒识≥95%、每景宏拒识≥95%；旧类宏误拒≤5%、含误拒的已知正确率相对相同IQ的原闭集基线下降≤2pp。报告逐RX/scene/TX、AUROC/OSCR及95%统计区间，不把全拒绝、空分组或阈值扫描最优点当部署达标。固定阈值结果不反馈本候选调参/选模/选择性重跑，不根据此target表替五种基线选“主方法”。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

执行顺序为真实checkpoint无query smoke→source-only校准→固定received转换→全部预测→独立CPU评分。smoke PASS后自动继续，无人工smoke许可。新root不覆盖已有产物，本pilot最多一个GPU worker；每卡本任务可新增两个worker但不干预其他用户。低科学性能不触发停止；错误契约/角色、输出冲突、无法合法预测或执行失败触发本run技术停止，保留partial，先独立核对后才能重发新run。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 本地验证与版本

实际代码/config commit：`1ff583b84b137e9ba1166a667b70f40ffd559134`。43项Phase3针对性测试通过；独立P0/P1审查PASS，无阻断问题。协议负测、固定模型接口和模拟完整校准/预测/评分已检查。真实checkpoint与N607执行依赖由launcher首步无query smoke验证，不将本地fixture当实际性能。原有其他任务的暂存改动已逐字保留。

## 正式首轮结果

VERIFIED/ANALYZED。真实checkpoint smoke PASS，1440条预测全部固定后独立truth-last评分完成，五个基线全部未达科学目标。执行器正常结束，未做技术重启；其他用户进程保留。完整逐方法/场景/RX/TX分析见`docs/PAMF_PHASE3_OPEN_SET_PILOT_20261028.md`。目标模式仍ACTIVE；不得按本目标表调阈值或选主方法。
