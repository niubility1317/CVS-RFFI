# D92 E0真实256维B0/C3/几何联合消融实施计划

> **For agentic workers:**按测试先行执行；本计划对应用户已明确授权的“设计并发布实验验证”，不构成额外审批门。

**目标：**在不改变D92 E0其余模块、数据、K-shot、接收机、LEO场景或query边界的前提下，测量B0、C3与双几何/交叉拟合控制路径的联合效应。

**结构：**将稳健中心视为二元因子`B∈{FULL,B0}`，将任务均衡协方差视为二元因子`C∈{FULL,C3}`，将几何路径视为三档因子`G∈{FULL,D0,D2}`。`D0`和`D2`是同一几何模块的互斥控制路径，因此不能构成物理可执行的`D0+D2`组合；`B0+C3+D0`是“所有兼容消融同时生效”的端点。

**技术栈：**Python、NumPy、现有D92注册执行器、N607的`CVS-RFFI`Conda环境、独立truth-last scorer。

**关联报告：**`docs/D92_METHOD_COMPLETE_REPORT_20260727.md`；`analysis/d92_e0_256_module_ablation_hard11_20260826_v2.md`。

## 全局约束

- 协议固定为`p2_min_v1`，重用`VALIDATED_ONCE`的v2数据绑定；不重建或重验received-IQ。
- 固定接收机`3-19`、`K10/new5`、method/support/query/new-class draw种子`7282101/7282201/7282301/7282401`和`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`三场景。
- 所有support统计只使用当前目标域合法support；query仅逐样本预测，先封存预测再由独立scorer连接truth。
- A0、S0、量化配置、Fisher配置、冻结Phase1bundle、状态压缩和评分器保持FULL设置。没有FP32对照臂，也不作整数kernel加速主张。
- 运行状态只可为`LOCAL_VERIFIED→LANDED→RUNNING→ARTIFACTS_COMPLETE→ANALYZED`；低性能不是停止理由。

## 冻结矩阵

|几何G|中心B|协方差C|逻辑臂|是否新执行|
|---|---|---|---|---|
|FULL|FULL|FULL|`P2-256-FULL`|是，作为同提交控制|
|FULL|B0|FULL|`P2-256-B0`|是|
|FULL|FULL|C3|`P2-256-C3`|是|
|FULL|B0|C3|`P2-256-J-B0-C3`|是|
|D0|FULL|FULL|`P2-256-D0`|是|
|D0|B0|FULL|`P2-256-J-B0-D0`|是|
|D0|FULL|C3|`P2-256-J-C3-D0`|是|
|D0|B0|C3|`P2-256-J-B0-C3-D0`|是|
|D2|FULL|FULL|`P2-256-D2`|是|
|D2|B0|FULL|`P2-256-J-B0-D2`|是|
|D2|FULL|C3|`P2-256-J-C3-D2`|是|
|D2|B0|C3|`P2-256-J-B0-C3-D2`|是|

主要输出为每臂、每场景的B-old、A-old、New、H、F、min-old和min-new。对任意指标`m`，报告三场景等权平均`bar(m)`。在每个几何档`g`下，B0与C3的交互量定义为：

$$
I_{BC}(g)=m(B0,C3,g)-m(B0,FULL,g)-m(FULL,C3,g)+m(FULL,FULL,g).
$$

其中`I_{BC}(g)`衡量B0和C3同时启用后、不能由两个单独变化线性相加解释的部分。几何路径的条件效应以同一`B,C`状态下的`m(B,C,D0)-m(B,C,FULL)`和`m(B,C,D2)-m(B,C,FULL)`报告。单seed结果只用于筛选交互方向，不作普适或因果晋级声明。

### Task1：先写目录与矩阵的失败测试

**文件：**
- 新建：`tests/test_stage2_e0_256_joint_ablation_catalog.py`
- 修改：`tests/test_build_full_ablation_plan.py`

- [x] 写测试，断言12个臂、7个新联合臂、三档几何因子、无`D0+D2`逻辑臂、所有新臂与`P2-256-FULL`的差异键恰为其声明的2或3项。
- [x] 运行上述测试并确认它因联合矩阵尚不存在而失败。

### Task2：实现冻结联合臂目录与计划构造

**文件：**
- 修改：`code/cvsrffi/stage2_ablation_factory.py`
- 修改：`code/cvsrffi/full_ablation_spec.py`
- 修改：`code/scripts/build_full_ablation_plan.py`

- [x] 定义7个新的`P2-256-J-*`逻辑臂及其精确中心、协方差、几何覆盖。
- [x] 添加`e0_256_joint_screen`计划模式；`--arms t1`仅接受上述12臂的固定顺序，并固定为`3-19/K10/new5`和既定种子。
- [x] 修改目录验证，使历史单因素目录仍要求一个差异键，而联合目录要求声明差异键与有效配置差异完全一致。
- [x] 运行失败测试，确认通过。

### Task3：先写执行器组合语义的失败测试

**文件：**
- 修改：`tests/test_stage2_ablation_executors.py`

- [x] 为`B0+C3+D0`与`B0+C3+D2`写真实support拟合测试，断言均为256维、只读取support、没有query拟合、且审计记录同时标出plain-center、equal-covariance和对应geometry配置。
- [x] 运行该测试并确认因组合执行器尚不存在而失败。

### Task4：实现组合执行器

**文件：**
- 修改：`code/cvsrffi/stage2_ablation_executors.py`

- [x] 从冻结resolved config读取中心、协方差和geometry档位，而非把联合ID伪装为单一D0或D2路径。
- [x] 使用`build_d92_fit(...,apply_ground_center=False)`实现B0；C3使用D81等权Ledoit-Wolf路径；D0或D2通过D62builder替换实现；保留FULL路径不变。
- [x] 把新增逻辑臂接入既有F3存储压缩状态，但继续标记`integer_kernel_used=false`。
- [x] 运行组合执行器测试及既有256维回归，确认通过。

### Task5：本地最小发布核验

**文件：**
- 修改：`analysis/d92_e0_256_joint_ablation_bc_geometry_20260826.md`
- 新建：`automation_reports/CV-SincNet/d92_e0_256_joint_ablation_bc_geometry_20260826/report.md`

- [x] 运行聚焦协议负测、12臂计划构造测试、真实checkpoint无query smoke和一次独立P0/P1正确性审查。
- [x] 生成不可覆盖的12row预登记计划；记录提交、命令、环境、输入/输出、GPU、技术停止规则和预期工件。
- [x] 只stage本计划、代码、测试和正式报告；提交、推送并核对远端OID。

### Task6：N607发布与闭合

**文件：**
- 更新：`automation_reports/CV-SincNet/d92_e0_256_joint_ablation_bc_geometry_20260826/report.md`
- 更新：`analysis/d92_e0_256_joint_ablation_bc_geometry_20260826.md`
- 更新：`docs/D92_METHOD_COMPLETE_REPORT_20260727.md`

- [x] 在N607完成一次资源/路径preflight、release归档单次本地/远端SHA比较和远端编译。
- [x] 启动12臂run并验证PID、CWD、命令行、GPU映射和日志增长；不干预非本run任务。
- [x] 每行预测闭合后，独立scorer连接truth并生成同row结果。
- [x] 计算全部条件主效应与`I_BC(G)`，报告结果为单seed筛选证据；提交、推送并核对远端OID。

## 实际闭合

- 实际N607运行：`d92_e0_256_joint_ablation_bc_geometry_20260826`；发布提交：`05048a24652087b7e9b83f694b74bae01b109063`；联合实现提交：`8d4c97f0357e9b866d22a9fa800b2b424371a63e`。
- 12个逻辑row对应12个独立物理执行，全部预测和独立truth-last评分为`PASS`；失败row=0、alias row=0。
- 本地回收的`same_row_summary.json`已由汇总器校验，主结果、条件效应和证据边界写入`analysis/d92_e0_256_joint_ablation_bc_geometry_20260826.md`、Markdown技术报告及HTML技术报告。
- 结果报告发布提交：`b39436b6894379dfbe294a1f00cacaeb08865396`；该12臂矩阵仍是单seed、单接收机筛选，而非fresh confirmation。
