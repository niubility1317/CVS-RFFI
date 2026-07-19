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

## 实现与验证

- core SHA256=`1260de735e64aa32f11061be82ec2ca26e4c90e5590725d76d83299696848388`；probe SHA256=`aa2b0b4aee74eac0d551bf0387b2c87cfa2560df877924e579ca353b60499f68`。
- `ssr-gpu`下py_compile通过；D79专项6/6、D77-D79相邻专项24/24通过。
- D79 core仅包装锁定D78优化：先对全support减均值，D78返回`DeltaW`后解析生成`Delta b`；probe在D78已测D62/量化脚手架的单次compile点注入bias，再把全部D78兼容字段规范化为D79 artifact。
- clean detached worktree：`E:\type10-7\code\snapshots\d79wt`；本地`cuda:0`运行，不接触N607。

## 运行锁

精确运行参数与D78报告第7节完全相同，仅作以下三项字面替换，其余D18 capsule/seal/policy/authorization、D22 component、class binding、device、mode和candidate-set逐字不变：

|参数|D78|D79|
|---|---|---|
|入口|`probe_d78_ground_tangent_worstclass_margin.py`|`probe_d79_centered_ground_tangent.py`|
|arm|`--d78-arm ground_tangent_worstclass_top2_margin`|`--d79-arm centered_ground_tangent_worstclass_top2_margin`|
|probe/output root|`d78wt`及`d78.../ground_tangent_worstclass_top2_margin`|`d79wt`及`d79.../centered_ground_tangent_worstclass_top2_margin`|

预期105行、30个target fit、1,080个D62 component fit、每target row8个OOF LDA＋20个D78优化步；query0。独立stdout/stderr保存在D79报告根，禁止覆盖。
