# SF-TAPFT-RSE R0–R3实验报告

## 最小预登记

- run ID：`stage2_sf_tapft_rse_r0r3_rx20_1_s392002_20260828_r1`。
- 科学问题：在固定E0许可边界下，support-only强度选择、双视图一致性或双delta聚合能否降低K=10估计方差，并在新目标域上优于E0。
- 数据协议：`p2_min_v1`、`VALIDATED_ONCE`；旧6类、每类10条独立物理support；support/query物理ID不相交；不注册新类。
- 目标域：`rx_20_1`、seed713103、`leo_clear_weak`；最大独立Query为每类20条、共120条。
- 矩阵：R0=E0；R1=两折×两次cross-fit选择step/alpha；R2=双视图与`lambda_view=0.05`；R3=两个每类8条子集delta均值+全support 30步低学习率收尾。
- 条件行：R4=R1+R2，仅当R1或R2相对R0单独满足科学门槛时发布；否则不运行。
- 固定E0边界：target head+`t3.norm(weight,bias)`；禁止早层Norm、类别bias、HardPair、Adapter、full t3、frequency/domain更新和EMA。
- 选择网格：`step={250,350,450,520}`，`alpha={0,0.25,0.5,0.75,1}`；R1全程只读support及其固定视图。
- 科学门槛：相对R0，BA不低于R0且至少一个候选提高；floor不下降；NLL不恶化超过0.02。低性能不触发技术停止。
- 资源合同：可训练元素≤2048、delta≤16KB、cache≤4MB、CUDA allocated≤256MiB、reserved≤384MiB；warm-resident median≤60s。缺少逐模块MAC/功率传感器时明确记录`NOT_CAPTURED`。
- 本地代码与配置：待实现后补记；本地环境`ssr-gpu`，工作目录为当前Git worktree。
- N607输入：ADV3B02 CORE90 checkpoint、已验证Phase1 bundle、seed713103 predictor package；远端run root为`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_rse_r0r3_rx20_1_s392002_20260828_r1`。
- GPU：preflight后按每卡最多两个训练任务分配；不得触碰无关进程。
- 停止规则：仅协议/query泄漏、错误输入或checkout、输出覆盖、无prediction闭合、launcher级故障或至少两行同一确定性预prediction异常。
- 预期artifact：每行`selection.json`、clean-single bundle、delta bundle、GNU time、GPU采样、truth-blind predictions；全部prediction闭合后由独立scorer连接truth。

## 实现与本地验证

- `target_only_progressive_adapt.py`新增RSE固定许可检查、双视图相位旋转与JS损失、单轨迹snapshot、support-only稳健风险、alpha插值、类别均衡子集和共同anchor delta平均。
- R1每个fold只执行一条完整E0轨迹；验证集建立一次H6 prefix cache，21个状态仅运行suffix。full-support提交再次运行完整520步轨迹，并直接读取被选择step的同轨迹snapshot，避免截短fast-tail改变学习率轨迹。
- R2把原始视图与0.05rad全局相位视图批量缓存；CE读取两视图，JS约束两视图，LOO-proto只读原始物理视图，不把同一物理样本的增强副本当成独立K-shot。
- R3两个子模型共享由完整60条support构建的target-head锚点，各自使用每类8条的确定性平衡子集；只平均共同注册的`t3.norm`和head delta，随后在完整support上用`lr_norm=1e-5`、`lr_head=5e-5`收尾30步。
- 新增资源审计：有效视图数、cache构建view-equivalent full forward、suffix forward/backward、head optimizer step；缺少逐模块MAC基线时FBE保持`NOT_CAPTURED`。
- 本地执行`py_compile`和55项聚焦回归，全部通过；4行矩阵解析回读为`MATRIX_OK`。独立P0/P1初审发现并修复R1截短轨迹错配与R3 clean-single时序配置失效；仅针对原R3问题的定点复审为PASS。
- 根目录`E:\type10-7`不是Git仓库；本报告镜像到本Git承载面，正式提交只包含本轮RSE文件。

## 结果

实验尚未发布。
