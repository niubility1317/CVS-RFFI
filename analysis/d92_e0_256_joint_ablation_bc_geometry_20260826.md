# D92 E0当前256维B0/C3/几何联合消融

## 最小预登记

- `run_id`：`d92_e0_256_joint_ablation_bc_geometry_20260826`
- 当前状态：`LOCAL_VERIFIED`
- 运行代码提交：`8d4c97f0357e9b866d22a9fa800b2b424371a63e`
- 本文是Git承载的预登记镜像；执行报告位于`E:\type10-7\automation_reports\CV-SincNet\d92_e0_256_joint_ablation_bc_geometry_20260826\report.md`。

本实验在真实紧凑`identity160+FFT96=256`维D92 E0路径上，联合检验B0稳健中心、C3等权协方差与几何路径的条件效应。其余模块、数据、切片、F3储存状态和评分器保持FULL设置。没有`F0`FP32状态对照臂。

## 为什么是12臂而不是16臂

B0和C3分别是二元开关，但D0和D2不是两个能同时删除的模块：它们是同一几何模块的互斥控制路径。因此把几何写为三水平因素`G∈{FULL,D0,D2}`，与`B∈{FULL,B0}`和`C∈{FULL,C3}`交叉，得到`2×2×3=12`个可执行臂。`P2-256-J-B0-C3-D0`是全部兼容移除的端点；`D0+D2`不构成一个合法物理设置。

|G|B|C|逻辑臂|
|---|---|---|---|
|FULL|FULL|FULL|`P2-256-FULL`|
|FULL|B0|FULL|`P2-256-B0`|
|FULL|FULL|C3|`P2-256-C3`|
|FULL|B0|C3|`P2-256-J-B0-C3`|
|D0|FULL|FULL|`P2-256-D0`|
|D0|B0|FULL|`P2-256-J-B0-D0`|
|D0|FULL|C3|`P2-256-J-C3-D0`|
|D0|B0|C3|`P2-256-J-B0-C3-D0`|
|D2|FULL|FULL|`P2-256-D2`|
|D2|B0|FULL|`P2-256-J-B0-D2`|
|D2|FULL|C3|`P2-256-J-C3-D2`|
|D2|B0|C3|`P2-256-J-B0-C3-D2`|

固定输入为接收机`3-19`、`K10/new5`、method/support/query/new-class draw种子`7282101/7282201/7282301/7282401`和`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。数据协议为`p2_min_v1`，重用前一真实256维v2run的`VALIDATED_ONCE`绑定、`capsule_id`和`split_id`；本次仅变化方法配置，不重建received-IQ。

每个臂只用Phase1bundle和当前目标域合法support拟合；query不参与任何拟合、选择或全局重分配。每臂、每场景独立预测封存后，独立truth-last scorer再给出B-old、A-old、New、H、F、min-old与min-new。结果按三场景等权平均；每个几何档`g`的中心—协方差交互定义为`I_BC(g)=m(B0,C3,g)-m(B0,FULL,g)-m(FULL,C3,g)+m(FULL,FULL,g)`。

## 数值封存约束

所有臂使用F3封存状态，仍只可声明`storage_compression_only`，因为`integer_kernel_used=false`。若FP16偏置因类别共有的仿射logit常数超界，则注册端去除这一共同常数并验证support argmax不变；不保存FP32旁路，也不在query端补回。若共同常数去除后仍不能有限封存，该row按直接技术失败处理。

## 本地核验和发布边界

- 新联合目录、固定12臂计划、执行器组合、row执行和release回归均已通过。
- 7个新增组合均已由合法support生成256维F3状态；`B0+C3+D0`的共同偏置稳定化和`B0+C3+D2`的无需稳定化路径均已测试。
- 聚焦回归共107项通过；独立P0/P1审查为`P0=0`、`P1=0`，并确认紧凑D42运行时文件已受该运行提交跟踪。
- 已在外部报告根生成不可覆盖的12row预登记计划，且其`formal_launch_authority=false`；目前没有N607进程被启动。
- 正式N607发布前仍须写入实际Git提交、完成只读N607资源/路径preflight、release归档的一次本地/远端SHA比较和远端编译。
- 直接技术停止规则仅包括数据/边界错误、错误checkout、输出覆盖、重复确定性预预测异常、prediction闭合缺失、独立scorer错误或FP16状态不可表示；低性能不停止。

该单seed困难切片只用于筛选联合效应方向，不能在结果出来前或之后自动宣传为多seed、全接收机、全规模或实星硬件结论。
