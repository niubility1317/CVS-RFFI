# D66地面域可靠性残差开发探针

## 1.执行前登记

- 实验ID：`d66_ground_domain_reliability_residual_probe_20260719`。
- 时间：2026-07-19；operator：Codex。
- 目标：真正使用不可变Phase1地面int8域×类聚合知识，同时避免旧类专属anchor导致的新类塌缩；检验共享地面域可靠性变换能否在D62基础上同时改善旧类域适应与新类注册。
- 比较目标D62：before92.78%、after82.22%、new84.67%、H82.62%、forget10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold，实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 enrollment-only support，不重验数据。
- 根目录`E:\type10-7`不是Git仓库；版本化实现和本报告镜像位于`E:\type10-7\github_publish\CVS-RFFI-repo`。执行前Git HEAD为`51e375ada1ffcd56516b01dce88dd0b5b359d937`；工作树存在大量不属于本轮的既有修改，本轮只暂存D66精确路径。

## 2.机制与历史边界

D66从84个有效的地面域×旧类int8聚合单元计算每个z160坐标的类间身份方差`B`和同类跨域漂移`W`，固定`r=(B+eps)/(B+W+2eps)`、`s=sqrt(1+r)`。z160使用共享尺度`s`，FFT96/RF32恒等；D62全部支持拟合在共享坐标执行，再把系数编译回原坐标。对旧类、新类和未来query没有不同公式，query零额外MAC/state。

历史停止项：D19/D25/D36旧类专属anchor中心融合、独立半径似然、角色offset/IRLS、D30 old-old DALI、旧anchor Procrustes/transport及query batch统计。D66不复用这些机制，不持久化反量化ground bank，不读取clean/source样本或query。

当前组件manifest标记`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，所以本轮严格限定为开发support内部held-rank诊断，formal/query/performance claim和125权限均为false。组件必须只读，入口/出口SHA均应为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`。

## 3.预注册门

- 完成七候选×三场景×五折=105行，query/clean/source/role/quota/global assignment访问均为0。
- 相对D62总体、三场景、逐类floor、遗忘、混淆和量化不得交换伤害，并至少严格改善after、forgetting、joint或任一floor。
- 必须报告七候选、三场景、11类、15fold、地面尺度统计、FP32/int8量化、训练/适配MAC、状态、延迟和完整artifact。
- 失败即状态`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并停止本路线；成功也先完成D64–D66三轮回顾，不直接启动125。

## 4.待完成实现与运行信息

待补：本地变更、验证命令、Git提交、干净worktree、精确运行命令、环境、输出路径、运行时、完整结果与下一实验建议。
