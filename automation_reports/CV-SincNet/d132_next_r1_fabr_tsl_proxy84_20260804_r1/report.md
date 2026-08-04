# D132 NEXT-R1 FABR-TSL proxy84实验报告

## 当前状态

|字段|值|
|---|---|
|experiment ID|`d132_next_r1_fabr_tsl_proxy84_20260804_r1`|
|日期|2026-08-04|
|operator|主agent（Sol/high）；科学研发与复核为Terra/max；冻结后唯一runner为Luna/max|
|状态|`DESIGN_FROZEN / IMPLEMENTATION_PENDING / NOT_RELEASED / NO_NEW_PERFORMANCE_RESULT`|
|目标|一次必要的84行六臂source-held筛选，同时验证FABR域适应、TSL相对历史D92机制和联合替换|

## 已有真实性能与本轮假设

D130完整proxy84表明CSPAR-2的K5 DA为`ΔH=-0.556pp、总正确数-9`，SRDH-2为零效应；Lite160虽`ΔH=+0.164pp、总正确数+14`，但`A_held_proxy=-0.529pp、F_retained=-1.270pp`，因此没有可晋级正收益。D131只完成393条partial prediction，因5个K1精确并列和2个primary160零行技术退出，没有truth、score或性能结果。

本轮唯一假设是：Phase1-only Fisher锚定的单block rank2功能残差能在不写checkpoint的前提下改善support/query公共表示；同一signed-pre-ReLU160表示上的TSL类对称对角EB头能删除full covariance与role分裂，同时不牺牲retained、held-proxy和最低类。full288/selector、D131补丁链、D62 row splice、CSPAR、SRDH、RDCE及重复125均不进入本轮。

## 冻结前矩阵与比较

|维度|冻结值|
|---|---|
|fold|7个receiver×6个seen-class LOCO=42|
|K|`{1,5}`，K1为K5 support物理样本前缀|
|原子row|84|
|逻辑臂|`R0Q/R0F/R0L/R1Q/R1F/R1L`|
|表示|同一真实forward的signed-pre-ReLU160；不得读取full288/aux|
|K1|F/L逐logit alias Q；任一top tie为`TIE_UNRESOLVED/NO_PERFORMANCE_RESULT`|
|K5主比较|`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`|
|通过条件|每个主比较均`ΔH_proxy>0`且总正确数增加，并且`ΔA_retained>=0、ΔA_held_proxy>=0、ΔF_retained>=0`|
|失败处置|完整负结果立即关闭，不调参数、不重复矩阵|

## 版本、文件与本地验证

当前Git分支为`codex/next-r1-fabr-tsl-20260804`，设计起点commit为`b989cd0c`。本次先更新：

- `docs/STAGE2_RD_GOAL_20260731.md`：吸收D131复盘及独立复核的2个P0、3个P1；
- `analysis/next_r1_fabr_tsl_traceability_20260803.md`：增加表示、并列、FABR曲率、顺序/量化、TSL API和资源追踪；
- 本报告及根目录同路径镜像。

尚未修改科学代码，尚未执行真实checkpoint smoke，尚未同步N607。设计差分复核、实现和测试完成后在此补充commit、hash、测试命令及结果。

## 预注册资源与协议检查

- `p2_min_v1`；复用匹配`VALIDATED_ONCE`的既有received-IQ，不重复数据验证；
- query零fit、零update、零selection；不得读取clean/source、query truth、role、quota或global reassignment；
- FABR只封存Phase1-only的INT8 rank2基、FP16 scale、2×2 Fisher几何和冻结常数；Phase2仅support闭式拟合2个系数；
- TSL只接收`support_z160/support_labels/registered_classes`；Phase1先验封存量化误差、正方差、hash和margin-slack；
- 分开保存FABR support/query forward成本与head拟合、wire字节、query MAC、墙钟和瞬时工作集receipt。

## 发布前仅保留的硬门

1.独立差分复核已确认`P0=0、P1=0、DESIGN_FROZEN`；
2.实际Git方法入口与聚焦协议负测通过；
3.真实checkpoint-derived received-IQ no-query smoke通过，并完成一次K1 no-truth liveness scan；
4.不可覆盖run/output路径、本地commit、N607预检和资源记录。

不要求重复数据验证、通用发布平台、额外签名层、论文叙事、D62/D92/SVRN复跑或125矩阵。

## N607发布字段（冻结后填写）

|字段|值|
|---|---|
|本地文件与hash|待实现|
|同步目的地|待冻结|
|server command|待冻结|
|Conda/Python环境|`ssr-gpu`，具体解释器待预检|
|CWD|待冻结|
|log/output|不可覆盖新路径，待冻结|
|PID/GPU|待runner落地后记录；每GPU不超过2个训练实验|
|expected artifacts|84行prediction、完整manifest、表示/并列/量化/resource receipt、独立truth score|
|技术停止规则|P0协议/安全错误，或至少2个不同row在prediction前出现相同确定性异常指纹；不得按性能早停|
|fresh retry authority|默认无；若系统性技术失败，保留artifact并由主agent重新冻结新run ID|

## 完成后结果表

|candidate|机制|receiver/class/K|A_retained|A_held_proxy|H_proxy|F_retained|总正确数|state/fit/query资源|结论|
|---|---|---|---:|---:|---:|---:|---:|---|---|
|`NEXT-R1 FABR-TSL/r1`|单block rank2 FABR＋类对称TSL|待84行完成|—|—|—|—|—|—|`NO_NEW_PERFORMANCE_RESULT`|

当前推荐：完成唯一候选的设计差分复核后立即实现并发布84行必要矩阵；不启动D92 Lite125。
