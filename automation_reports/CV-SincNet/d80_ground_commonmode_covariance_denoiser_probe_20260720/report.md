# D80地面公共域模态协方差去噪实验报告

|字段|值|
|---|---|
|实验ID|`d80_ground_commonmode_covariance_denoiser_probe_20260720`|
|候选|`ground_commonmode_covariance_denoiser`|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
|目标|把地面压缩原型仅作为类对称接收机噪声协方差先验，联合改善旧类域适应和新类注册|
|开发单元|receiver`20-1`、new5、seed`713101`、K10(actual K8)、3场景×5fold|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 预注册机制

D80从84个地面int8域×类中心中，只提取每个domain在全部ground类上共同出现的接收机扰动。逐类减去域均值后跨类平均，SVD形成公共域投影`P`；不把任一ground类中心、类ID或类分数送入target分类。

把公共扰动解释为`Sigma_g=I+P`，其无参数精确逆为`I−0.5P`。对D62最终仿射头闭式编译：`W'=W(I−0.5P)`、`b'=b+0.5WPmu_support`，即只衰减target中心附近的公共接收机噪声方向。相对D62仅增加这个类对称、中心保持、0步闭式协方差去噪；不改变数据、D62支持集拟合、注册表或query规则。

## 协议与资格边界

- 复用D18 `VALIDATED_ONCE/p2_min_v1`；single-LEO_weak、support-only、query独立全类argmax。
- clean/source/query truth/role/quota/global assignment访问0；target-old/new使用完全相同公式。
- ground组件84 cell、逻辑状态25,428B，但当前`formal_phase2_eligible=false`、`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，所以本轮强制development diagnostic。
- 失败不扩展第二seed、125或N607；不扫描rank、投影系数0.5、中心倍率或类权重。

## 性能门与必报指标

相对D62要求`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格改善；`old→new/new→old/new→wrong-new`不得以一侧伤害换另一侧收益。完整报告必须给出同row`B/A/N/H/F/J`、逐场景、全部逐类旧类遗忘和新类准确率、mean row floor、三类混淆、INT8/FP32变化、outer hash变化、机制审计与资源。

## 版本与运行计划

`E:\type10-7`根不是Git仓库；预注册、实现、测试、追溯和完整报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`，本目录保留镜像。实现先在clean detached worktree完成；`ssr-gpu`窄测试通过后，在锁定D18开发单元本地`cuda:0`运行105行。当前不连接N607。
