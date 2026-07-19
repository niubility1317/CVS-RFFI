# D79中心化地面切向旋转实验报告

|字段|值|
|---|---|
|实验ID|`d79_centered_ground_tangent_probe_20260720`|
|候选|`centered_ground_tangent_worstclass_top2_margin`|
|状态|`PREREGISTERED_NOT_RUN`|
|数据单元|receiver`20-1`、new5、seed`713101`、K10(actual K8)、3场景×5fold|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

D79只修复D78的类先验漂移：优化和部署均使用`DeltaW(x−mu_support)`，等价编译`Delta b=−DeltaW mu_support`。在support均值处各类残差严格为0；ground仍只提供共享域切向，不产生类别分数。

当前ground组件84 cell、25,428B，但`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，因此本轮仅为development diagnostic。相对D62要求`A/N/H/min-A/min-N`不退化、`F`不升且至少一项严格改善，混淆不得交换伤害；失败不扩展第二seed、125或N607。

`E:\type10-7`根不是Git仓库；代码、追溯和完整报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`，本目录保留根镜像。实现、测试、版本、精确命令、PID和完整结果在运行前后补录。
