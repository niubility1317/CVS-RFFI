# D34-FCLER冻结碰撞局部注册实验

## 登记

- 实验ID：`d34_fcler_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`IMPLEMENTATION_IN_PROGRESS`。
- 目标：保留D33验证有效的FAST Fisher零梯度Stage2-B，完全冻结旧类score前缀；Stage2-C只在support证明存在混淆的old-new局部边上注册int8新类原型，优先消除注册遗忘和floor类侵入。
- 比较：Z0、C0、历史B3、D33-FAST负对照、D34-A/B/C；7候选×3个LEO_weak场景×5个独立held-rank折=105行。
- 边界：本轮仍是开发support-only screen，不打开query，不作正式receiver/seed性能声明。若D34不能同时优于D33-FAST的old、new、H、forgetting和floor，则不进入正式矩阵。

## 机制锁

同一已接收LEO_weak IQ只生成一行288维`[z160,FFT96,RF32]`拼接特征。FAST Fisher仅用旧类support闭式得到冻结旧头`g_i(x)`；Stage2-C不得修改旧log-diagonal、旧centroid或旧score：

`S_i(x)=g_i(x), i∈Y_old`。

每个新类用独立support生成mean/medoid shrink原型`p_j`，最终只保存int8中心、FP32 scale和inverse norm。碰撞图`E_ij`只由新类support在冻结旧头上的top-old频次生成。对旧锚`i`，所有相邻新类共享安全offset：

`alpha_i=max(0,max_{x∈G_i,j:E_ij=1}[z(x)^T p_j-g_i(x)]+1e-4+lambda*u_i)`。

逐样本先取冻结旧头winner`i*`，然后对全部注册类给出有限score：

`S_j(x)=z(x)^T p_j-alpha_i*`，若`E_i*j=1`；否则`S_j(x)=g_i*(x)-2`。最终只做逐样本all-registered argmax，无query角色、标签、真实batch类数、配额或全局分配。

|候选|每新类碰撞边|新类原型|不确定度保护|
|---|---:|---|---|
|D34-A|top-1|mean|`lambda=0`|
|D34-B|top-2|25% medoid shrink|`lambda=0.5`|
|D34-C|达到top-old频次80%覆盖的最小top-1至top-3|25% medoid shrink|`lambda=1`；低于旧类median accuracy的类再×2|

K1只使用单个独立物理support作为原型，不制造LOSO或派生shot；Fisher走identity安全分支，`u_i`使用K10 method lock的固定0.05。K5/K10只从本K可达support重算闭式状态；统一arm和超参数只能在开发K10锁定。

## 选择与门禁

每折分别执行old LOO侵入审计和new LOO注册审计。候选排名顺序为：旧score prefix逐bit不变→旧support逐类/floor non-degradation→old LOO零侵入→最差场景joint floor→new最低类LOSO→H→new总体→更少碰撞边/MAC。若新类不存在同时满足旧类安全与正new LOSO margin的边，必须输出`UNREACHABLE_COLLISION_EDGE`，不能被总体均值掩盖。

D34成功也只解决注册抗遗忘；D33-FAST注册前旧类held总体82.22%，冻结旧头不会自动达到92%。若注册层验证成功，下一轮必须保留D34 Stage2-C并单独增强Stage2-B旧头。

## 协议与执行计划

- receiver`20-1`、seed`713101`、K10、5个新类、3个LEO_weak场景；复用D33同一密封support包，不新增数据准备。
- 每个physical support只有一个已叠加LEO_weak的IQ观测；三个数学描述来自同一IQ，support multiplicity/view均为1。
- query rows/labels/features均为0；无clean/source、角色Oracle、真实batch类数、quota、global assignment或dense query图。
- 预计0 optimizer step、活动状态远低于50k；必须同时报告平均/最坏碰撞degree、MAC/query、适配MAC、延迟、状态、head显存口径和相对identity-only单qKNN变化。
- 本地验证后先Git提交，再执行直接N607 preflight、live inventory、最小文件同步、SHA闭合和唯一输出检查；计划GPU0单任务，输出`runs/d34_fcler_20260718/output/support_screen_v1`。

## 完成后回填

待回填本地测试、Git提交、远端命令/PID/GPU、105行完整日志、逐候选/场景/类矩阵、碰撞边与不可达类、资源审计、support/query清单、artifact SHA、selection/RECEIPT及下一轮判定。
